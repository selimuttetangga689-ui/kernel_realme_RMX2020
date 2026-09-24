// SPDX-License-Identifier: GPL-2.0
/*
 * Copyright (C) 2019-2023 Sultan Alsawaf <sultan@kerneltoast.com>.
 *
 * Reclaim-first / one-victim-at-a-time rework, 2026.
 *
 * Design summary (see ANALYSIS_AND_NOTES.md for full rationale and the
 * verification performed against this 4.19 vendor tree before writing
 * any of this):
 *
 *   vmpressure high
 *        |
 *        v
 *   direct VM reclaim (try_to_free_pages), N rounds scaled by pressure
 *        |
 *        v
 *   recheck NR_FREE_PAGES against target
 *        |
 *        +---- target met ----> done, no kill
 *        |
 *        v
 *   pick ONE victim (protected UIDs excluded, oom_score_adj primary,
 *   size secondary)
 *        |
 *        v
 *   kill it, wait for reap, do one more light reclaim pass, recheck
 *        |
 *        +---- target met, OR estimated recovered size met target, ----> stop
 *        |     OR per-cascade kill cap hit
 *        v
 *   next victim (bounded by cooldown at the cascade level, not between
 *   individual victims within a single still-critical cascade)
 *
 * The "estimated recovered size" and per-cascade kill cap bounds exist
 * because relying on real free-memory recovery alone (memory_target_met())
 * has no natural floor if reclaim can't keep up -- a real device hit this:
 * it looped through victims until it SIGKILL'd zygote itself, causing a
 * bootloop. See scan_and_kill_one_at_a_time() for details.
 *
 * Protected applications are identified by Android package name in
 * userspace (PackageManager), resolved to a UID there, and pushed into
 * this driver as a UID list via a module parameter. The kernel never
 * deals with package names, and protected UIDs remain fully reclaimable
 * (reclaim/swap/zRAM) -- they are only excluded from the kill path.
 */

#define pr_fmt(fmt) "simple_lmk: " fmt

#include <linux/bitops.h>
#include <linux/cpumask.h>
#include <linux/cred.h>
#include <linux/delay.h>
#include <linux/freezer.h>
#include <linux/gfp.h>
#include <linux/kthread.h>
#include <linux/mm.h>
#include <linux/moduleparam.h>
#include <linux/mutex.h>
#include <linux/oom.h>
#include <linux/sched/mm.h>
#include <linux/slab.h>
#include <linux/sort.h>
#include <linux/string.h>
#include <linux/swap.h>
#include <linux/uidgid.h>
#include <linux/vmpressure.h>
#include <linux/vmstat.h>
#include <uapi/linux/sched/types.h>

/*
 * Built-in (non-loadable) module parameters only get a sysfs entry
 * under /sys/module/<name>/parameters/ if the registered param name
 * contains a '.' -- param_sysfs_builtin() in kernel/params.c splits on
 * it to infer "module.param" for code that never went through the real
 * module loader. Without this, module_param()/module_param_cb() still
 * compile and work internally, but silently get NO sysfs presence at
 * all (no error, no log -- exactly the "No such file or directory"
 * symptom). This must come before every module_param*() call below
 * that isn't already using its own prefix (minfree, at the bottom of
 * this file, sets its own "lowmemorykiller." prefix and is unaffected
 * by this).
 */
#undef MODULE_PARAM_PREFIX
#define MODULE_PARAM_PREFIX "simple_lmk."

/*
 * RAM-dependent defaults.
 *
 * Match the Redmi 10C donor: one kernel binary detects total RAM when
 * Android lmkd writes lowmemorykiller.minfree. Devices above 3 GiB use
 * the 4 GiB+ tuning; devices at or below 3 GiB use the 3 GiB tuning.
 */
static const char *slmk_ram_class __read_mostly = "unknown";

#define SLMK_DEFAULT_RECLAIM_ROUNDS_HIGH 3
#define SLMK_DEFAULT_RECLAIM_ROUNDS_CRITICAL 6
#define SLMK_DEFAULT_MAX_KILLS_PER_CASCADE 8
#define SLMK_DEFAULT_KILL_COOLDOWN_MS 100
#define SLMK_DEFAULT_NO_SWAP_GRACE_MS 500

/*
 * RMX2020 (Snapdragon 680-class) big cores. CPUs 0-5 are the little
 * cluster; CPUs 6-7 are the two big cores. This is only ever used as a
 * *preference* for the work this driver's own reclaim thread does
 * (direct_reclaim_pass(), which drives shrink_page_list() -> the ZRAM
 * swap-out/compress path via zcomp_stream_get()'s get_cpu_ptr() -- see
 * slmk_apply_reclaim_affinity() and the comment above it); it never
 * touches kswapd, other kernel threads, or any unrelated task's
 * affinity, and is bounds-checked at runtime against the CPUs actually
 * present before use (see slmk_apply_reclaim_affinity()).
 */
#define SLMK_DEFAULT_RECLAIM_CPU_MASK (BIT(6) | BIT(7))

/*
 * Bounded extra time to let an in-flight kill's async reap/compress
 * work land before this cascade concludes reclaim "failed" and moves
 * on to another victim. See the comment above slmk_post_kill_grace().
 */
#define SLMK_DEFAULT_POST_KILL_GRACE_MS 150
#define SLMK_POST_KILL_GRACE_POLL_MS 25

/* The minimum number of pages to free per reclaim, and the recheck target */
static unsigned short slmk_minfree __read_mostly;
#define MIN_FREE_PAGES (slmk_minfree * SZ_1M / PAGE_SIZE)

/* Kill up to this many victims per reclaim */
#define MAX_VICTIMS 1024

/* Timeout in jiffies for each reclaim */
static unsigned short slmk_timeout __read_mostly;
#define RECLAIM_EXPIRES msecs_to_jiffies(slmk_timeout)

/*
 * vmpressure levels at which we act. HIGH triggers a reclaim-only pass
 * (never kills). CRITICAL triggers the full reclaim -> recheck -> kill
 * cascade. Both are plain vmpressure percentages (0-100), the same
 * metric the original driver already consumed; no PSI is used since
 * CONFIG_PSI=n on this build.
 */
#define SLMK_PRESSURE_HIGH 85
#define SLMK_PRESSURE_CRITICAL 98

/* Number of direct-reclaim rounds to attempt before giving up on reclaim */
static unsigned int slmk_reclaim_rounds_high __read_mostly =
	SLMK_DEFAULT_RECLAIM_ROUNDS_HIGH;
static unsigned int slmk_reclaim_rounds_critical __read_mostly =
	SLMK_DEFAULT_RECLAIM_ROUNDS_CRITICAL;
module_param(slmk_reclaim_rounds_high, uint, 0644);
module_param(slmk_reclaim_rounds_critical, uint, 0644);

/*
 * Minimum time between the start of two kill cascades. This only gates
 * whether a *new* cascade is allowed to kill; it does not stop a single
 * still-critical cascade from killing more than one victim in a row,
 * since each of those kills is followed by a real memory recheck.
 */
static unsigned int slmk_kill_cooldown_ms __read_mostly =
	SLMK_DEFAULT_KILL_COOLDOWN_MS;
module_param(slmk_kill_cooldown_ms, uint, 0644);
static unsigned long last_kill_jiffies;

/*
 * How long to wait for swap capacity to come back before killing, if a
 * critical cascade starts while total_swap_pages == 0 (see the comment
 * in run_reclaim_cascade()). Polled in small steps so we can bail out
 * immediately if memory recovers or swap returns sooner.
 */
static unsigned int slmk_no_swap_grace_ms __read_mostly =
	SLMK_DEFAULT_NO_SWAP_GRACE_MS;
module_param(slmk_no_swap_grace_ms, uint, 0644);
#define SLMK_NO_SWAP_GRACE_POLL_MS 50

/*
 * Bounded reclaim grace given after a kill whose reap could not be
 * confirmed before moving on to another victim within the same
 * cascade. Runtime tunable; see slmk_post_kill_grace().
 */
static unsigned int slmk_post_kill_grace_ms __read_mostly =
	SLMK_DEFAULT_POST_KILL_GRACE_MS;
module_param(slmk_post_kill_grace_ms, uint, 0644);

/*
 * CPU affinity preference (bitmask) for this driver's own reclaim
 * kthread only -- see slmk_apply_reclaim_affinity(). A custom
 * kernel_param_ops is used (rather than a plain module_param()) so
 * that changing this after boot re-applies immediately to the
 * already-running thread instead of requiring a reboot, per the
 * runtime-tunability goal for this experiment. Writing 0 disables the
 * preference entirely and leaves the thread's affinity at whatever the
 * scheduler default is.
 */
static unsigned int slmk_reclaim_cpu_mask __read_mostly =
	SLMK_DEFAULT_RECLAIM_CPU_MASK;
static struct task_struct *slmk_reclaim_task;
static DEFINE_MUTEX(slmk_reclaim_task_mutex);
static void slmk_apply_reclaim_affinity(struct task_struct *tsk,
					 unsigned int mask_bits);

static int slmk_reclaim_cpu_mask_set(const char *val,
				      const struct kernel_param *kp)
{
	unsigned int mask;
	int ret;

	ret = kstrtouint(val, 0, &mask);
	if (ret)
		return ret;

	slmk_reclaim_cpu_mask = mask;

	/*
	 * Re-apply to the live thread if it has already started. This
	 * can block (set_cpus_allowed_ptr() may wait on a stop_one_cpu()
	 * migration), which is fine here: module param .set callbacks
	 * run in sysfs-write (process) context, not atomic context.
	 */
	mutex_lock(&slmk_reclaim_task_mutex);
	if (slmk_reclaim_task)
		slmk_apply_reclaim_affinity(slmk_reclaim_task, mask);
	mutex_unlock(&slmk_reclaim_task_mutex);

	return 0;
}

static int slmk_reclaim_cpu_mask_get(char *buf, const struct kernel_param *kp)
{
	return scnprintf(buf, PAGE_SIZE, "0x%x\n", slmk_reclaim_cpu_mask);
}

static const struct kernel_param_ops slmk_reclaim_cpu_mask_ops = {
	.set = slmk_reclaim_cpu_mask_set,
	.get = slmk_reclaim_cpu_mask_get,
};
module_param_cb(slmk_reclaim_cpu_mask, &slmk_reclaim_cpu_mask_ops,
		&slmk_reclaim_cpu_mask, 0644);

/* Maximum number of UIDs that can be marked protected at once */
#define SLMK_MAX_PROTECTED 64

struct victim_info {
	struct task_struct *tsk;
	struct mm_struct *mm;
	unsigned long size;
	/*
	 * Captured under rcu_read_lock() in find_victims(), since
	 * task_uid()/__task_cred() dereferences an RCU-protected pointer.
	 * By the time kill_single_victim() runs, we are no longer inside
	 * an RCU read-side critical section, so the UID must already be a
	 * plain snapshot rather than re-derived from tsk at that point.
	 */
	kuid_t uid;
	/*
	 * tsk/mm are pinned with get_task_struct()/mmget_not_zero() in
	 * find_victims(), which then releases task_lock() immediately.
	 * This differs from the original driver, which instead kept
	 * task_lock() held on every candidate from find_victims() all the
	 * way through to the kill loop, relying on that lock (not an
	 * actual refcount) to keep tsk/mm alive in the meantime. That was
	 * safe there because the whole batch was processed and *fully
	 * unlocked* before the single wait_for_completion_timeout() call
	 * at the very end. It is NOT safe here: kill_single_victim() now
	 * calls wait_for_completion_timeout() once per victim, and
	 * task_lock() is a spinlock -- it raises this CPU's preempt_count
	 * regardless of which task it's locking. With several *other*
	 * still-unprocessed candidates' task_locks left held (because
	 * find_victims() locks a whole batch up front), the very first
	 * per-victim wait hits "BUG: scheduling while atomic" and panics.
	 * This was confirmed on-device via a captured pstore panic trace
	 * (simple_lmk_reclaim_thread -> wait_for_completion_timeout ->
	 * schedule, immediately after the first "Killing ..." log line).
	 * Using real refcounts instead -- the same technique
	 * mm/oom_kill.c itself uses before doing anything that can sleep
	 * -- removes the held-lock-across-schedule() problem entirely.
	 * Every victim_info that reaches this struct must eventually be
	 * released via mmput(mm) + put_task_struct(tsk), whether or not it
	 * ends up being killed (see scan_and_kill_one_at_a_time()).
	 */
};

static struct victim_info victims[MAX_VICTIMS] __cacheline_aligned_in_smp;
static struct task_struct *task_bucket[SHRT_MAX + 1] __cacheline_aligned;
static DECLARE_WAIT_QUEUE_HEAD(oom_waitq);
static DECLARE_WAIT_QUEUE_HEAD(reaper_waitq);
static DECLARE_COMPLETION(reap_done);
static __cacheline_aligned_in_smp DEFINE_RWLOCK(mm_free_lock);

/*
 * Only one victim is ever "in flight" (killed but not yet confirmed
 * reaped) at a time, since victims are now processed strictly
 * sequentially. This replaces the old victims[]/nr_victims scan that
 * the reaper thread used to track multiple simultaneous victims.
 */
static struct mm_struct *active_victim_mm;
static bool reclaim_active;

static atomic_t needs_reclaim = ATOMIC_INIT(0);
static atomic_t reclaim_critical = ATOMIC_INIT(0);
static atomic_t needs_reap = ATOMIC_INIT(0);

/* Protected UID table, populated by userspace after package->UID resolution */
static kuid_t protected_uids[SLMK_MAX_PROTECTED];
static unsigned int nr_protected_uids;
static DEFINE_RWLOCK(protected_lock);

/* Debug/statistics counters (see simple_lmk_stats_get) */
static atomic_t stat_reclaim_attempts = ATOMIC_INIT(0);
static atomic_long_t stat_pages_freed = ATOMIC_LONG_INIT(0);
static atomic_t stat_kill_count = ATOMIC_INIT(0);
static atomic_t stat_protected_skips = ATOMIC_INIT(0);
static DEFINE_SPINLOCK(last_victim_lock);
static char last_victim_comm[TASK_COMM_LEN];
static unsigned int last_victim_uid;
static int last_victim_adj;
static unsigned long last_victim_kib;

static bool is_uid_protected(kuid_t uid)
{
	bool protected = false;
	unsigned int i;

	read_lock(&protected_lock);
	for (i = 0; i < nr_protected_uids; i++) {
		if (uid_eq(protected_uids[i], uid)) {
			protected = true;
			break;
		}
	}
	read_unlock(&protected_lock);

	return protected;
}

/*
 * Safe, overflow-proof comparator. The previous implementation used
 * "rhs->size - lhs->size" on unsigned long sizes, which can wrap and
 * produce an incorrect sign when the subtraction underflows.
 */
static int victim_cmp(const void *lhs_ptr, const void *rhs_ptr)
{
	const struct victim_info *lhs = (typeof(lhs))lhs_ptr;
	const struct victim_info *rhs = (typeof(rhs))rhs_ptr;

	if (rhs->size > lhs->size)
		return 1;
	if (rhs->size < lhs->size)
		return -1;
	return 0;
}

static void victim_swap(void *lhs_ptr, void *rhs_ptr, int size)
{
	struct victim_info *lhs = (typeof(lhs))lhs_ptr;
	struct victim_info *rhs = (typeof(rhs))rhs_ptr;

	swap(*lhs, *rhs);
}

static unsigned long get_total_mm_pages(struct mm_struct *mm)
{
	unsigned long pages = 0;
	int i;

	for (i = 0; i < NR_MM_COUNTERS; i++)
		pages += get_mm_counter(mm, i);

	return pages;
}

/*
 * Populate victims[] with candidate tasks, sorted primarily by
 * oom_score_adj (highest/least-important first, mirroring Android
 * semantics) and secondarily by size within each adj bucket. Protected
 * UIDs are excluded from the candidate set entirely here, before
 * anything ever reaches victims[] -- protection is therefore enforced
 * once, at the earliest point, rather than re-checked later.
 *
 * Unlike the previous implementation, callers no longer re-sort the
 * resulting candidates globally by size across adj buckets. Mixing size
 * into the ordering across buckets meant a lower-adj (more important)
 * process could be killed ahead of a higher-adj one just because it was
 * larger, which contradicts the requirement that adj strictly dominates
 * size in victim selection. Keeping the adj-bucket order intact and
 * only recovering real memory via a per-kill recheck (see
 * scan_and_kill_one_at_a_time) is both simpler and safer.
 */
static unsigned long find_victims(int *vindex)
{
	short i, min_adj = SHRT_MAX, max_adj = 0;
	unsigned long pages_found = 0;
	struct task_struct *tsk;

	rcu_read_lock();
	for_each_process(tsk) {
		struct signal_struct *sig;
		short adj;

		/*
		 * Search for suitable tasks with a positive adj (importance).
		 * Since only tasks with a positive adj can be targeted, that
		 * naturally excludes tasks which shouldn't be killed, like init
		 * and kthreads. Although oom_score_adj can still be changed
		 * while this code runs, it doesn't really matter; we just need
		 * a snapshot of the task's adj.
		 */
		sig = tsk->signal;
		adj = READ_ONCE(sig->oom_score_adj);
		if (adj < 0 ||
		    sig->flags & (SIGNAL_GROUP_EXIT | SIGNAL_GROUP_COREDUMP) ||
		    (thread_group_empty(tsk) && tsk->flags & PF_EXITING))
			continue;

		/*
		 * Exclude protected UIDs from the candidate set. Protection is
		 * UID-based (resolved from a package name in userspace), never
		 * based on task_struct->comm, and never touches VM reclaim --
		 * only the kill path is affected.
		 */
		if (is_uid_protected(task_uid(tsk))) {
			atomic_inc(&stat_protected_skips);
			continue;
		}

		/* Store the task in a linked-list bucket based on its adj */
		tsk->simple_lmk_next = task_bucket[adj];
		task_bucket[adj] = tsk;

		/* Track the min and max adjs to speed up the loop below */
		if (adj > max_adj)
			max_adj = adj;
		if (adj < min_adj)
			min_adj = adj;
	}

	/* Start searching for victims from the highest adj (least important) */
	for (i = max_adj; i >= min_adj; i--) {
		int old_vindex;

		tsk = task_bucket[i];
		if (!tsk)
			continue;

		/* Clear out this bucket for the next time reclaim is done */
		task_bucket[i] = NULL;

		/* Iterate through every task with this adj */
		old_vindex = *vindex;
		do {
			struct task_struct *vtsk;
			struct mm_struct *vmm;

			vtsk = find_lock_task_mm(tsk);
			if (!vtsk)
				continue;

			/*
			 * Pin tsk/mm with real refcounts so we can safely
			 * release task_lock() right away instead of holding
			 * it across this whole batch (see the comment on
			 * struct victim_info for why that matters here).
			 */
			vmm = vtsk->mm;
			if (!mmget_not_zero(vmm)) {
				/* mm is already being torn down; skip it */
				task_unlock(vtsk);
				continue;
			}
			get_task_struct(vtsk);

			/* Store this potential victim away for later */
			victims[*vindex].tsk = vtsk;
			victims[*vindex].mm = vmm;
			victims[*vindex].size = get_total_mm_pages(vmm);
			victims[*vindex].uid = task_uid(vtsk);

			task_unlock(vtsk);

			/* Count the number of pages that have been found */
			pages_found += victims[*vindex].size;

			/* Make sure there's space left in the victim array */
			if (++*vindex == MAX_VICTIMS)
				break;
		} while ((tsk = tsk->simple_lmk_next));

		/* Go to the next bucket if nothing was found */
		if (*vindex == old_vindex)
			continue;

		/*
		 * Sort victims within this adj bucket in descending order of
		 * size, so that if this bucket alone isn't enough, the largest
		 * same-importance processes are still tried first.
		 */
		sort(&victims[old_vindex], *vindex - old_vindex,
		     sizeof(*victims), victim_cmp, victim_swap);

		/*
		 * Stop when we are out of space or have gathered enough
		 * candidate pages by estimate. This is only a bound on how many
		 * candidates we enumerate; the actual number killed is decided
		 * later against real free-memory state, one victim at a time.
		 */
		if (*vindex == MAX_VICTIMS || pages_found >= MIN_FREE_PAGES) {
			/* Zero out any remaining buckets we didn't touch */
			if (i > min_adj)
				memset(&task_bucket[min_adj], 0,
				       (i - min_adj) * sizeof(*task_bucket));
			break;
		}
	}
	rcu_read_unlock();

	return pages_found;
}

static void set_task_rt_prio(struct task_struct *tsk, int priority)
{
	const struct sched_param rt_prio = {
		.sched_priority = priority
	};

	sched_setscheduler_nocheck(tsk, SCHED_RR, &rt_prio);
}

#if defined(CONFIG_HZ_100) || defined(CONFIG_HZ_300)
#define SLEEP_DURATION_MS 30
#else
#define SLEEP_DURATION_MS 28
#endif

/* ---------------------------------------------------------------------
 * Direct VM reclaim (reclaim-first / swap+zRAM-first)
 * ---------------------------------------------------------------------
 *
 * try_to_free_pages() is the same entry point the page allocator's slow
 * path uses (mm/page_alloc.c). It internally sets may_swap = 1 and
 * may_unmap = 1 in its scan_control (mm/vmscan.c), so a GFP_KERNEL call
 * here already lets anonymous pages be unmapped and pushed through the
 * normal LRU reclaim -> swap_writepage()/zram path -- the same path
 * kswapd and direct reclaim already use for every other allocation on
 * this system. No zRAM- or process-specific compression call is made
 * here; we only ask the existing VM reclaim/swap machinery to run.
 *
 * Because CONFIG_NUMA is not set on this build, numa_node_id() and
 * node_zonelist() resolve to the single system node/zonelist, matching
 * how try_to_free_pages() is invoked from __alloc_pages_slowpath().
 */
static unsigned long direct_reclaim_pass(void)
{
	struct zonelist *zonelist = node_zonelist(numa_node_id(), GFP_KERNEL);

	return try_to_free_pages(zonelist, 0, GFP_KERNEL, NULL);
}

static bool memory_target_met(void)
{
	return global_zone_page_state(NR_FREE_PAGES) >= MIN_FREE_PAGES;
}

/*
 * Prefer the big cores (CPU 6-7 on RMX2020) for this driver's own
 * reclaim kthread, which is the thread that calls direct_reclaim_pass()
 * both from run_reclaim_cascade() and from slmk_post_kill_grace()
 * below. try_to_free_pages() -> shrink_page_list() -> pageout() enters
 * the ZRAM swap-out path synchronously on whichever CPU is running at
 * the time; zcomp_stream_get() (drivers/block/zram/zcomp.c) then takes
 * that CPU's per-CPU compression stream via get_cpu_ptr(). There is no
 * separate compression worker in this tree's zcomp -- compression runs
 * inline in the caller -- so the *only* safe, targeted way to steer
 * this driver's own reclaim-triggered compression onto the big cores
 * is to steer the calling thread itself. This is deliberately scoped
 * to that one kthread:
 *
 *   - kswapd's own background reclaim is untouched and keeps its
 *     normal (unpinned) affinity, so ordinary allocator-path reclaim
 *     is not perturbed and little cores are not starved of it.
 *   - No global scheduler policy, cpuset, or other task's affinity is
 *     touched.
 *   - The reaper thread (simple_lmk_reaper_thread) is left unpinned:
 *     __oom_reap_task_mm() unmaps already-resident pages and does not
 *     itself invoke the compressor, so pinning it would not help and
 *     would only reduce its scheduling flexibility.
 *
 * mask_bits is validated against cpu_possible_mask at call time rather
 * than assumed, so an out-of-range mask (wrong topology, or a bad
 * value written to the runtime tunable) safely falls back to leaving
 * the thread's affinity untouched instead of silently doing nothing
 * useful or, worse, being applied against nonexistent CPUs.
 */
static void slmk_apply_reclaim_affinity(struct task_struct *tsk,
					 unsigned int mask_bits)
{
	cpumask_var_t mask;
	unsigned int cpu;

	/* 0 means "no preference"; leave the thread's affinity alone */
	if (!mask_bits)
		return;

	if (!alloc_cpumask_var(&mask, GFP_KERNEL))
		return;

	cpumask_clear(mask);
	for_each_possible_cpu(cpu) {
		if (mask_bits & BIT(cpu))
			cpumask_set_cpu(cpu, mask);
	}

	if (!cpumask_intersects(mask, cpu_possible_mask)) {
		pr_warn_ratelimited("reclaim_cpu_mask 0x%x matches no CPU present on this device (nr_cpu_ids=%u); leaving reclaim thread unpinned\n",
				    mask_bits, nr_cpu_ids);
		goto out;
	}

	cpumask_and(mask, mask, cpu_possible_mask);

	if (set_cpus_allowed_ptr(tsk, mask))
		pr_warn_ratelimited("failed to set reclaim thread CPU affinity to 0x%x\n",
				    mask_bits);
	else
		pr_info("reclaim thread affined to CPU mask 0x%x\n",
			cpumask_bits(mask)[0]);
out:
	free_cpumask_var(mask);
}

/* ---------------------------------------------------------------------
 * Kill path: exactly one victim at a time, with a real recheck between
 * each kill.
 * ---------------------------------------------------------------------
 */
static void publish_active_victim(struct mm_struct *mm)
{
	write_lock(&mm_free_lock);
	active_victim_mm = mm;
	reclaim_active = true;
	write_unlock(&mm_free_lock);
}

static void clear_active_victim(void)
{
	write_lock(&mm_free_lock);
	active_victim_mm = NULL;
	reclaim_active = false;
	write_unlock(&mm_free_lock);
}

static void record_last_victim(const char *comm, kuid_t uid, int adj,
				unsigned long kib)
{
	spin_lock(&last_victim_lock);
	strscpy(last_victim_comm, comm, sizeof(last_victim_comm));
	last_victim_uid = from_kuid(&init_user_ns, uid);
	last_victim_adj = adj;
	last_victim_kib = kib;
	spin_unlock(&last_victim_lock);
}

/*
 * Returns true if the victim's reap was confirmed (reap_done completed
 * before RECLAIM_EXPIRES), false if we gave up waiting and proceeded on
 * a timeout. See slmk_post_kill_grace() for why the caller cares about
 * this distinction.
 */
static bool kill_single_victim(struct victim_info *victim)
{
	struct task_struct *t, *vtsk = victim->tsk;
	struct mm_struct *mm = victim->mm;
	kuid_t uid = victim->uid;
	unsigned long kib = victim->size << (PAGE_SHIFT - 10);
	bool reaped_confirmed;

	pr_info_ratelimited("Killing %s (uid %u) with adj %d to free %lu KiB\n",
			     vtsk->comm, from_kuid(&init_user_ns, uid),
			     vtsk->signal->oom_score_adj, kib);

	/*
	 * Capture everything we log/record now. Unlike the original
	 * driver, task_lock() is NOT held here -- find_victims() already
	 * released it after pinning vtsk/mm with get_task_struct()/
	 * mmget_not_zero() (see struct victim_info). Those two references
	 * are what keep vtsk and mm valid through the rest of this
	 * function, including across wait_for_completion_timeout().
	 */
	record_last_victim(vtsk->comm, uid, vtsk->signal->oom_score_adj, kib);

	/* Make the victim reap anonymous memory first in exit_mmap() */
	set_bit(MMF_OOM_VICTIM, &mm->flags);

	/*
	 * Publish this mm as the single active victim for the reaper thread
	 * and simple_lmk_mm_freed(). This happens after MMF_OOM_VICTIM is
	 * set, which guarantees MMF_OOM_VICTIM is visible before the mm can
	 * ever enter exit_mmap(), same invariant the original driver relied
	 * on -- the mm is additionally pinned here by our own mmget_not_zero()
	 * reference rather than by a held task_lock().
	 */
	publish_active_victim(mm);

	/* Accelerate the victim's death by forcing the kill signal */
	do_send_sig_info(SIGKILL, SEND_SIG_PRIV, vtsk, PIDTYPE_TGID);

	/*
	 * Mark the thread group dead so that other kernel code knows,
	 * and then elevate the thread group to SCHED_RR with minimum RT
	 * priority. The entire group needs to be elevated because
	 * there's no telling which threads have references to the mm as
	 * well as which thread will happen to put the final reference
	 * and release the mm's memory. If the mm is released from a
	 * thread with low scheduling priority then it may take a very
	 * long time for exit_mmap() to complete.
	 */
	rcu_read_lock();
	for_each_thread(vtsk, t)
		set_tsk_thread_flag(t, TIF_MEMDIE);
	for_each_thread(vtsk, t)
		set_task_rt_prio(t, 1);
	/* Signals can't wake frozen tasks; only a thaw operation can */
	for_each_thread(vtsk, t)
		__thaw_task(t);
	rcu_read_unlock();

	/* Allow the victim to run on any CPU. This won't schedule. */
	set_cpus_allowed_ptr(vtsk, cpu_all_mask);

	/* Wake the reaper thread so it starts working on this mm */
	atomic_set(&needs_reap, 1);
	smp_mb__after_atomic();
	if (waitqueue_active(&reaper_waitq))
		wake_up(&reaper_waitq);

	/* Wait until the victim dies or until the timeout is reached */
	if (!wait_for_completion_timeout(&reap_done, RECLAIM_EXPIRES)) {
		pr_info_ratelimited("Timeout hit waiting for victim to die, proceeding\n");
		reaped_confirmed = false;
	} else {
		msleep(SLEEP_DURATION_MS);
		reaped_confirmed = true;
	}

	reinit_completion(&reap_done);
	clear_active_victim();

	atomic_inc(&stat_kill_count);
	last_kill_jiffies = jiffies;

	/*
	 * Give reclaim a chance to fold the pages the victim just released
	 * back into the free lists / zRAM accounting before the next
	 * recheck, rather than only relying on exit_mmap()'s own unmapping.
	 */
	direct_reclaim_pass();

	/* Release the references find_victims() took on our behalf */
	mmput(mm);
	put_task_struct(vtsk);

	return reaped_confirmed;
}

/*
 * When kill_single_victim() could not confirm the reap before its
 * timeout, we do not yet know whether that kill actually helped --
 * exit_mmap() and the reaper's __oom_reap_task_mm() are asynchronous,
 * and the immediately-following memory_target_met() recheck in
 * scan_and_kill_one_at_a_time() can easily still read stale (pre-reap)
 * free-page counts. Treating that as "reclaim already failed" and
 * moving straight to another victim is exactly the over-kill pattern
 * seen in testing (repeated "Timeout hit waiting for victim to die"
 * followed by another kill in the same cascade).
 *
 * Give it a short, hard-bounded window instead: keep running light
 * direct_reclaim_pass() calls (which also folds in whatever the
 * now-dying victim has released so far) and re-checking the real
 * target, rather than immediately re-entering the kill loop. This is
 * strictly bounded by slmk_post_kill_grace_ms (a handful of
 * SLMK_POST_KILL_GRACE_POLL_MS steps) and always exits promptly the
 * moment memory recovers, so it cannot become an unbounded reclaim
 * loop and does not block indefinitely.
 */
static void slmk_post_kill_grace(void)
{
	unsigned int waited = 0;

	while (waited < slmk_post_kill_grace_ms && !memory_target_met()) {
		direct_reclaim_pass();
		msleep(SLMK_POST_KILL_GRACE_POLL_MS);
		waited += SLMK_POST_KILL_GRACE_POLL_MS;
	}
}

/*
 * Hard ceiling on how many processes a single cascade will kill,
 * regardless of what memory_target_met() says. This is a last-resort
 * safety net: see the comment in scan_and_kill_one_at_a_time() for why
 * it exists.
 */
static unsigned int slmk_max_kills_per_cascade __read_mostly =
	SLMK_DEFAULT_MAX_KILLS_PER_CASCADE;
module_param(slmk_max_kills_per_cascade, uint, 0644);

static void scan_and_kill_one_at_a_time(void)
{
	int nr_found = 0, i;
	unsigned long recovered_estimate = 0;
	unsigned int killed = 0;

	find_victims(&nr_found);
	if (unlikely(!nr_found)) {
		pr_err_ratelimited("No eligible (unprotected) processes available to kill!\n");
		return;
	}

	for (i = 0; i < nr_found; i++) {
		bool confirmed;

		/*
		 * Stop as soon as EITHER:
		 *  - real free memory has recovered,
		 *  - the cumulative estimated size already meets the target, or
		 *  - the hard per-cascade kill cap has been hit.
		 */
		if (memory_target_met() ||
		    recovered_estimate >= MIN_FREE_PAGES ||
		    killed >= slmk_max_kills_per_cascade)
			break;

		recovered_estimate += victims[i].size;
		confirmed = kill_single_victim(&victims[i]);
		killed++;

		/*
		 * If we don't yet know whether that kill actually freed
		 * anything (timeout, not a confirmed reap), give reclaim a
		 * short bounded chance to catch up before the next loop
		 * iteration's memory_target_met() check decides we need
		 * another victim. A confirmed reap already ran an extra
		 * direct_reclaim_pass() with real freed memory behind it
		 * (see kill_single_victim()), so it goes straight to the
		 * next recheck without this extra wait.
		 */
		if (!confirmed)
			slmk_post_kill_grace();
	}

	/* Every candidate was pinned by find_victims(); release any that
	 * were not consumed by kill_single_victim(). */
	for (; i < nr_found; i++) {
		mmput(victims[i].mm);
		put_task_struct(victims[i].tsk);
	}

	if (killed >= slmk_max_kills_per_cascade)
		pr_warn_ratelimited("Hit per-cascade kill cap (%u); stopping this cascade early\n",
				    slmk_max_kills_per_cascade);

}

/* ---------------------------------------------------------------------
 * Reclaim -> recheck -> kill cascade, driven by vmpressure
 * ---------------------------------------------------------------------
 */
static void run_reclaim_cascade(bool critical)
{
	unsigned int rounds = critical ? slmk_reclaim_rounds_critical
					: slmk_reclaim_rounds_high;
	unsigned long freed = 0;
	unsigned int i;

	atomic_inc(&stat_reclaim_attempts);

	for (i = 0; i < rounds; i++) {
		freed += direct_reclaim_pass();
		if (memory_target_met())
			break;
	}
	atomic_long_add(freed, &stat_pages_freed);

	if (!critical || memory_target_met())
		return;

	/*
	 * total_swap_pages is the total capacity of all currently-active
	 * swap devices; it is 0 whenever nothing is swapped on at all --
	 * notably including the brief window a zram-resize/recompression
	 * tool (e.g. a Magisk module) does swapoff -> recreate zram ->
	 * swapon. swapoff itself is synchronous and forces every page
	 * that was in zram back into RAM immediately, which is a genuine,
	 * if self-inflicted and short-lived, memory pressure spike -- and
	 * direct_reclaim_pass() cannot push anything into swap while this
	 * window is open, so memory_target_met() may simply be unreachable
	 * through reclaim alone until swap comes back. Rather than treat
	 * that exactly like a normal critical/no-swap-ever device (where
	 * killing immediately is correct), give it a brief chance to
	 * resolve itself first.
	 */
	if (total_swap_pages == 0) {
		unsigned int waited = 0;

		pr_warn_ratelimited("No swap capacity active during a critical reclaim cascade (total_swap_pages=0); waiting up to %u ms for it to return before considering a kill\n",
				    slmk_no_swap_grace_ms);
		while (waited < slmk_no_swap_grace_ms &&
		       total_swap_pages == 0 && !memory_target_met()) {
			msleep(SLMK_NO_SWAP_GRACE_POLL_MS);
			waited += SLMK_NO_SWAP_GRACE_POLL_MS;
		}

		if (memory_target_met()) {
			pr_info_ratelimited("Memory recovered without killing while waiting for swap to return\n");
			return;
		}
	}

	if (time_before(jiffies,
			 last_kill_jiffies + msecs_to_jiffies(slmk_kill_cooldown_ms))) {
		pr_info_ratelimited("Kill cooldown active, skipping kill this cycle\n");
		return;
	}

	scan_and_kill_one_at_a_time();
}

static int simple_lmk_reclaim_thread(void *data)
{
	/* Use maximum RT priority */
	set_task_rt_prio(current, MAX_RT_PRIO - 1);
	set_freezable();

	/*
	 * Publish ourselves so the slmk_reclaim_cpu_mask module param can
	 * re-affine us live if it is changed after this point, then apply
	 * whatever preference is currently configured. See the comment
	 * above slmk_apply_reclaim_affinity().
	 */
	mutex_lock(&slmk_reclaim_task_mutex);
	slmk_reclaim_task = current;
	mutex_unlock(&slmk_reclaim_task_mutex);
	slmk_apply_reclaim_affinity(current, slmk_reclaim_cpu_mask);

	while (1) {
		bool critical;

		wait_event_freezable(oom_waitq, atomic_read(&needs_reclaim));
		critical = atomic_cmpxchg(&reclaim_critical, 1, 0);
		atomic_set(&needs_reclaim, 0);
		run_reclaim_cascade(critical);
	}

	return 0;
}

/* ---------------------------------------------------------------------
 * Reaper thread: reaps the single active victim's mm, if any
 * ---------------------------------------------------------------------
 */
static struct mm_struct *get_active_victim(bool *retry)
{
	struct mm_struct *mm = NULL;

	*retry = false;

	read_lock(&mm_free_lock);
	if (active_victim_mm && !test_bit(MMF_OOM_SKIP, &active_victim_mm->flags))
		mm = active_victim_mm;
	read_unlock(&mm_free_lock);

	if (!mm)
		return NULL;

	/* Linux 4.14 uses mmap_sem directly; mmap_read_trylock() was
	 * introduced later. The read lock still serializes us against
	 * exit_mmap(), which takes the write side before tearing down VMAs. */
	if (!down_read_trylock(&mm->mmap_sem)) {
		*retry = true;
		return NULL;
	}

	/*
	 * Check MMF_OOM_SKIP again under the lock in case this mm was
	 * already reaped by exit_mmap(). The active victim is pinned by
	 * victim->mm until kill_single_victim() drops it, so the read lock
	 * protects the VMA walk while __oom_reap_task_mm() runs.
	 */
	if (test_bit(MMF_OOM_SKIP, &mm->flags)) {
		up_read(&mm->mmap_sem);
		return NULL;
	}

	return mm;
}

static void reap_active_victim(void)
{
	while (1) {
		struct mm_struct *mm;
		bool retry;

		mm = get_active_victim(&retry);
		if (!mm) {
			if (retry) {
				/* Wait one jiffy before trying to reap again */
				schedule_timeout_uninterruptible(1);
				continue;
			}
			return;
		}

		/*
		 * Linux 4.14's __oom_reap_task_mm() returns void rather than
		 * the bool used by the donor 4.19 tree. We already hold
		 * mmap_sem for reading, so the VMA walk is serialized against
		 * exit_mmap(). Mark the mm reaped after the walk completes.
		 */
		__oom_reap_task_mm(mm);
		clear_bit(MMF_OOM_VICTIM, &mm->flags);
		set_bit(MMF_OOM_SKIP, &mm->flags);
		up_read(&mm->mmap_sem);
		return;
	}
}

static int simple_lmk_reaper_thread(void *data)
{
	/* Use a lower priority than the reclaim thread */
	set_task_rt_prio(current, MAX_RT_PRIO - 2);
	set_freezable();

	while (1) {
		wait_event_freezable(reaper_waitq,
				     atomic_cmpxchg_relaxed(&needs_reap, 1, 0));
		reap_active_victim();
	}

	return 0;
}

void simple_lmk_mm_freed(struct mm_struct *mm)
{
	/*
	 * Victims are guaranteed to have MMF_OOM_SKIP set after exit_mmap()
	 * finishes. Use this to ignore unrelated dying processes.
	 */
	if (!test_bit(MMF_OOM_SKIP, &mm->flags))
		return;

	read_lock(&mm_free_lock);
	if (reclaim_active && active_victim_mm == mm)
		complete(&reap_done);
	read_unlock(&mm_free_lock);
}

static int simple_lmk_vmpressure_cb(struct notifier_block *nb,
				    unsigned long pressure, void *data)
{
	if (pressure >= SLMK_PRESSURE_HIGH) {
		if (pressure >= SLMK_PRESSURE_CRITICAL)
			atomic_set(&reclaim_critical, 1);
		atomic_set(&needs_reclaim, 1);
		smp_mb__after_atomic();
		if (waitqueue_active(&oom_waitq))
			wake_up(&oom_waitq);
	}

	return NOTIFY_OK;
}

static struct notifier_block vmpressure_notif = {
	.notifier_call = simple_lmk_vmpressure_cb,
	.priority = INT_MAX
};

/* ---------------------------------------------------------------------
 * Module parameters: protected UID list and debug statistics
 * ---------------------------------------------------------------------
 *
 * Protection is configured as "protected_uids=UID1,UID2,...". The
 * kernel never resolves package names; a userspace component (running
 * with the ability to query PackageManager, e.g. a small init service
 * or a privileged system_server hook) resolves the package names the
 * device owner wants protected into UIDs and writes the resulting list
 * here whenever it changes (app installed/removed, user switched,
 * etc). See ANALYSIS_AND_NOTES.md section E for why this boundary is
 * where it is.
 */
static int protected_uids_set(const char *val, const struct kernel_param *kp)
{
	kuid_t new_uids[SLMK_MAX_PROTECTED];
	unsigned int count = 0;
	char *buf, *pos, *tok;
	int ret = 0;

	buf = kstrdup(val, GFP_KERNEL);
	if (!buf)
		return -ENOMEM;

	pos = buf;
	while ((tok = strsep(&pos, ",")) != NULL) {
		unsigned int raw;
		kuid_t kuid;

		tok = strim(tok);
		if (!*tok)
			continue;

		if (kstrtouint(tok, 10, &raw)) {
			ret = -EINVAL;
			goto out;
		}

		kuid = make_kuid(&init_user_ns, raw);
		if (!uid_valid(kuid)) {
			ret = -EINVAL;
			goto out;
		}

		if (count == SLMK_MAX_PROTECTED) {
			ret = -ENOSPC;
			goto out;
		}

		new_uids[count++] = kuid;
	}

	write_lock(&protected_lock);
	memcpy(protected_uids, new_uids, count * sizeof(*new_uids));
	nr_protected_uids = count;
	write_unlock(&protected_lock);

	pr_info("Updated protected UID list (%u entries)\n", count);
out:
	kfree(buf);
	return ret;
}

static int protected_uids_get(char *buf, const struct kernel_param *kp)
{
	unsigned int i;
	int len = 0;

	read_lock(&protected_lock);
	for (i = 0; i < nr_protected_uids; i++)
		len += scnprintf(buf + len, PAGE_SIZE - len, "%u%s",
				  from_kuid(&init_user_ns, protected_uids[i]),
				  i + 1 < nr_protected_uids ? "," : "");
	read_unlock(&protected_lock);
	len += scnprintf(buf + len, PAGE_SIZE - len, "\n");

	return len;
}

static const struct kernel_param_ops protected_uids_ops = {
	.set = protected_uids_set,
	.get = protected_uids_get,
};
module_param_cb(protected_uids, &protected_uids_ops, NULL, 0644);

static int simple_lmk_stats_get(char *buf, const struct kernel_param *kp)
{
	char comm[TASK_COMM_LEN];
	unsigned int uid;
	int adj;
	unsigned long kib;

	spin_lock(&last_victim_lock);
	memcpy(comm, last_victim_comm, sizeof(comm));
	uid = last_victim_uid;
	adj = last_victim_adj;
	kib = last_victim_kib;
	spin_unlock(&last_victim_lock);

	return scnprintf(buf, PAGE_SIZE,
		"reclaim_attempts=%d\n"
		"pages_freed=%ld\n"
		"kill_count=%d\n"
		"protected_skips=%d\n"
		"last_victim=%s uid=%u adj=%d freed_kib=%lu\n",
		atomic_read(&stat_reclaim_attempts),
		atomic_long_read(&stat_pages_freed),
		atomic_read(&stat_kill_count),
		atomic_read(&stat_protected_skips),
		comm[0] ? comm : "(none)", uid, adj, kib);
}

static const struct kernel_param_ops stats_ops = {
	.get = simple_lmk_stats_get,
};
module_param_cb(stats, &stats_ops, NULL, 0444);

/* Initialize Simple LMK when lmkd in Android writes to the minfree parameter */
static int simple_lmk_init_set(const char *val, const struct kernel_param *kp)
{
	static atomic_t init_done = ATOMIC_INIT(0);
	struct task_struct *thread;
	unsigned long total_mb;

	total_mb = totalram_pages >> (20 - PAGE_SHIFT);
	if (total_mb > 3072) {
		/* 4 GB+ variant, matching the donor kernel. */
		slmk_ram_class = "4GB+";
		slmk_minfree = 128;
		slmk_timeout = 200;
	} else {
		/* 3 GB or lower variant, matching the donor kernel. */
		slmk_ram_class = "3GB";
		slmk_minfree = 256;
		slmk_timeout = 200;
	}

	if (!atomic_cmpxchg(&init_done, 0, 1)) {
		thread = kthread_run(simple_lmk_reaper_thread, NULL,
				     "simple_lmkd_reaper");
		BUG_ON(IS_ERR(thread));
		thread = kthread_run(simple_lmk_reclaim_thread, NULL,
				     "simple_lmkd");
		BUG_ON(IS_ERR(thread));
		BUG_ON(vmpressure_notifier_register(&vmpressure_notif));
	}

	pr_info_once("Detected %lu MB RAM (%s), setting minfree to %hu MiB with timeout of %hu ms\n",
		total_mb, slmk_ram_class, slmk_minfree, slmk_timeout);
	return 0;
}

static const struct kernel_param_ops simple_lmk_init_ops = {
	.set = simple_lmk_init_set
};

/* Needed to prevent Android from thinking there's no LMK and thus rebooting */
#undef MODULE_PARAM_PREFIX
#define MODULE_PARAM_PREFIX "lowmemorykiller."
module_param_cb(minfree, &simple_lmk_init_ops, NULL, 0200);
