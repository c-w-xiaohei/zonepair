//! The executor's shared context (RFC-0003-zonepaird §4.2): `Shared` carries the
//! mirror layout, the authoritative `state.json` table behind a lock, the
//! per-sid wake-up cache (`Notify`), the runtime suspension overlay
//! (`Suspended`), and the fail-closed latch from `state.corrupt-*`. It has
//! no executor knowledge of its own — workers, waiters and the materialize
//! executor all import this layer.

use anyhow::Result;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use crate::sync::persist::fs;
use crate::sync::persist::state::{self, CommitLane, Corruption, SessionRecord, State};
use crate::sync::state_machine::BProbe;
use crate::transport::Base;

/// §6.2 — single-task logical budget. One retry sequence may span this
/// much wall time before the task degrades (`network` for pulls, the
/// attempt's reason class for publishes). Owned here (the shared
/// executor context) so `wait` and `pool` both draw it without
/// depending on each other.
pub const TASK_BUDGET: Duration = Duration::from_secs(300);

/// Where a reconcile pass reads and mutates session records. `State`
/// implements it in tests (single-writer ownership); `Arc<Shared>`
/// implements it below so a pass commits one record at a time without
/// holding the state lock across file IO.
pub trait SessionStore {
    /// All sids the record table knows — unioned with the mirror scan.
    fn sids(&self) -> std::collections::BTreeSet<String>;
    /// One record, cloned (`None` = sid unknown to the table).
    fn get(&self, sid: &str) -> Option<SessionRecord>;
    /// Mutate one record and commit the whole table in the same
    /// transaction: the crash window stays one-record-wide.
    fn mutate_commit(
        &mut self,
        sync_dir: &Path,
        sid: &str,
        f: &mut dyn FnMut(&mut SessionRecord),
    ) -> Result<()>;
}

/// `watch` wake-up cache per sid (§7): a convenience for in-process
/// waiters (`Waits::wake`), a cache never a truth source — dropped and
/// rebuilt freely.
#[derive(Default)]
pub struct Notify {
    map: Mutex<HashMap<String, tokio::sync::watch::Sender<()>>>,
}

impl Notify {
    /// Wake every receiver parked on `sid`. GC's the channel when no
    /// receiver is listening (a stale sender costs nothing anyway — this
    /// just keeps the map bounded).
    pub fn signal(&self, sid: &str) {
        let mut map = self.map.lock().unwrap();
        if let Some(tx) = map.get(sid) {
            if tx.receiver_count() == 0 {
                map.remove(sid);
            } else {
                let _ = tx.send(());
            }
        }
    }

    /// A receiver for in-process waits. `wait.rs` polls disk as the
    /// authority; this just shortens the polling interval.
    pub fn subscribe(&self, sid: &str) -> tokio::sync::watch::Receiver<()> {
        self.map
            .lock()
            .unwrap()
            .entry(sid.to_string())
            .or_insert_with(|| tokio::sync::watch::channel(()).0)
            .subscribe()
    }

}

/// Runtime suspension overlay — in-memory only, exactly like the RFC
/// wants it: neither kind of suspension is a `state.json` field.
/// `manual` is rebuilt from reconcile verdicts each pass; `pull` holds
/// pull-path degradations until a fresh pull/hydrate intent re-arms the
/// sid or a healing pull / clean verdict converges it.
#[derive(Default)]
pub struct Suspended {
    /// Reconcile-derive suspensions (orphans, B-loss, field corruption).
    pub manual: HashSet<String>,
    /// Pull-path degradations (§6.3 — no persistent slot).
    pub pull: HashSet<String>,
}

impl Suspended {
    pub fn contains(&self, sid: &str) -> bool {
        self.manual.contains(sid) || self.pull.contains(sid)
    }
}

/// Everything an executor needs that is not the task itself: the mirror
/// layout, the authoritative table behind a lock, the wake-up cache, the
/// suspension overlay, and the fail-closed latch from `state.corrupt-*`.
pub struct Shared {
    pub mirror: PathBuf,
    pub sync_dir: PathBuf,
    pub journal_dir: PathBuf,
    /// The authoritative table. Mutations go through `commit_state` /
    /// `commit_state_lazy`, whose lock holds are memory-only (mutation +
    /// generation bump, µs-scale); the disk write is the commit lane's
    /// writer thread (cd-state-cpu spec §4.2).
    pub state: Arc<Mutex<State>>,
    /// The write lane — `Some` only for `open` (the writer's open).
    /// Read-only opens spawn no writer and register no lane: production
    /// read-only callers (`wait_for_publish`) never commit, so the
    /// absence never surfaces; a hypothetical commit on one falls back
    /// to a direct write, race-free exactly because no writer exists.
    lane: Option<Arc<CommitLane>>,
    /// Per-sid `watch` wake-up cache.
    pub notify: Notify,
    /// Runtime suspension overlay.
    pub suspend: Mutex<Suspended>,
    /// `state.corrupt-*` latch (§8 fail-closed): executors suspend.
    pub fail_closed: std::sync::atomic::AtomicBool,
}

impl Shared {
    /// Open `<mirror>/.sync/` (0700) and load the authoritative table.
    /// Corruption latches `fail_closed` and is returned for the caller's
    /// loud log line.
    ///
    /// This is the WRITER's open: it mints the commit lane and spawns the
    /// writer thread (spec §4.2 — "boot 早期建立"), so no daemon wiring
    /// step exists: `startup::boot`'s `Shared::open` IS the spawn point.
    pub fn open(mirror: &Path) -> Result<(Arc<Shared>, Option<Corruption>)> {
        let sync_dir = state::ensure_sync_dir(mirror)?;
        fs::ensure_dir(&sync_dir.join("journal"), 0o700)?;
        let out = state::load(&sync_dir)?;
        shared_of(mirror, sync_dir, out, true)
    }

    /// `open` for short-lived read-only callers (`wait_for_publish` in the
    /// adapter): the `.sync` tree is never created and `state.json` is read
    /// through `load_readonly` — a schema violation still latches
    /// `fail_closed` and reports a `Corruption`, but the file is never
    /// renamed; quarantine is the writer's privilege (RFC-0003-zonepaird §8.3).
    /// No writer thread: this process never writes `state.json`.
    pub fn open_readonly(mirror: &Path) -> Result<(Arc<Shared>, Option<Corruption>)> {
        let sync_dir = state::sync_dir(mirror);
        let out = state::load_readonly(&sync_dir)?;
        shared_of(mirror, sync_dir, out, false)
    }

    pub fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Read the table under the lock — short critical sections only.
    pub fn with_state<R>(&self, f: impl FnOnce(&State) -> R) -> R {
        f(&self.lock())
    }

    /// Durable mutation commit (spec §4.2): the lock hold carries only
    /// the mutation + generation bump (µs-scale, no IO); the caller then
    /// blocks on the writer thread's ack — lock already released — until
    /// the generation's bytes are durable on disk. Semantics for callers
    /// are unchanged ("commit returns = durable"); what changed is the
    /// wait no longer holds the lock nor burns the caller's thread on
    /// fsync. A laneless `Shared` (read-only open) writes directly: no
    /// writer exists, so there is nothing to race.
    pub fn commit_state(&self, f: impl FnOnce(&mut State)) -> Result<()> {
        self.commit_impl(f, true)
    }

    /// Fire-and-forget mutation commit (spec §4.2): same lock discipline
    /// as `commit_state`, but the caller returns after the generation
    /// bump — durability follows within one writer pass (ms-scale), and
    /// the accepted loss window is a process crash inside it, whose cost
    /// the frequency table adjudicates per call site.
    pub fn commit_state_lazy(&self, f: impl FnOnce(&mut State)) -> Result<()> {
        self.commit_impl(f, false)
    }

    fn commit_impl(&self, f: impl FnOnce(&mut State), durable: bool) -> Result<()> {
        let lane = self
            .lane
            .clone()
            .or_else(|| state::lookup_lane(&self.sync_dir));
        let Some(lane) = lane else {
            // No writer in this process (read-only open): the pre-D2
            // shape — one lock hold across mutation AND write — stays
            // the correct serialization exactly because no writer
            // thread exists and this lock is the only ordering point.
            let mut guard = self.lock();
            f(&mut guard);
            return state::commit(&self.sync_dir, &guard);
        };
        let gen = {
            let mut guard = self.lock();
            f(&mut guard);
            lane.bump()
        };
        lane.poke();
        if durable {
            lane.wait_durable(gen)
        } else {
            Ok(())
        }
    }

    /// Test introspection for the spec §4.6 counters — total physical
    /// writes this lane has performed.
    #[cfg(test)]
    pub(crate) fn lane_write_count(&self) -> u64 {
        self.lane.as_ref().map(|l| l.write_count()).unwrap_or(0)
    }

    /// Test seam: the spec §4.6 "fake writer" hook — delay/fault
    /// injection per physical write.
    #[cfg(test)]
    pub(crate) fn set_lane_write_hook(
        &self,
        hook: Option<Arc<dyn Fn() -> Result<()> + Send + Sync>>,
    ) {
        if let Some(lane) = &self.lane {
            lane.set_write_hook(hook);
        }
    }
}

impl Drop for Shared {
    /// Wake the writer thread so it drains outstanding work and exits —
    /// otherwise every test's `Shared` would leak one parked OS thread.
    /// There is no graceful-shutdown protocol beyond this (spec §4.2:
    /// "进程退出即消亡"); outstanding byte work IS finished first so a
    /// dying owner's last batch is not abandoned mid-flight.
    fn drop(&mut self) {
        if let Some(lane) = &self.lane {
            lane.shutdown();
        }
    }
}

/// The `Shared` construction `open`/`open_readonly` share — the only
/// difference between them is upstream (dir creation, which `load` runs)
/// and whether a write lane is minted (`writable`).
fn shared_of(
    mirror: &Path,
    sync_dir: PathBuf,
    out: state::LoadOutcome,
    writable: bool,
) -> Result<(Arc<Shared>, Option<Corruption>)> {
    let state_arc = Arc::new(Mutex::new(out.state));
    let lane = writable.then(|| {
        let lane = CommitLane::spawn(&sync_dir, Arc::clone(&state_arc));
        state::register_lane(&sync_dir, &lane);
        lane
    });
    let shared = Arc::new(Shared {
        mirror: mirror.to_path_buf(),
        journal_dir: sync_dir.join("journal"),
        sync_dir,
        state: state_arc,
        lane,
        notify: Notify::default(),
        suspend: Mutex::new(Suspended::default()),
        fail_closed: std::sync::atomic::AtomicBool::new(out.corruption.is_some()),
    });
    Ok((shared, out.corruption))
}

/// Everything shared by executors: the disk triple (`Shared`), the wire
/// endpoint (`None` in tests that never reach the network), and the B
/// probe reconcile uses.
#[derive(Clone)]
pub struct ExecCtx {
    pub shared: Arc<Shared>,
    pub base: Option<Base>,
    pub token: Option<String>,
    /// `file present on B?` — production wires `store_exists`; tests inject a
    /// static answer. `Unknown` must be returned on probe failure.
    pub b_probe: Arc<dyn Fn(&str) -> BProbe + Send + Sync>,
}

impl ExecCtx {
    /// Test convenience — production builds `ExecCtx` through
    /// `ExecParts::into_ctx` (daemon startup); unit tests need a
    /// network-free context three fields shorter.
    #[cfg(test)]
    pub fn new(shared: Arc<Shared>) -> Self {
        Self {
            shared,
            base: None,
            token: None,
            b_probe: Arc::new(|_| BProbe::Unknown),
        }
    }
}

/// `Arc<Shared>` IS the `SessionStore` the reconcile seam asks for: it
/// clones records out under the lock and commits one record at a time —
/// the pass never holds the lock across file IO. Commits funnel through
/// `commit_state` (spec §4.5: "`SessionStore::mutate_commit` 经同一漏斗"),
/// so the per-record durable semantics the reconcile verdicts rely on are
/// exactly `commit_state`'s ack — the writer thread performs the IO.
impl SessionStore for Arc<Shared> {
    fn sids(&self) -> BTreeSet<String> {
        self.lock().sessions.keys().cloned().collect()
    }
    fn get(&self, sid: &str) -> Option<SessionRecord> {
        self.lock().sessions.get(sid).cloned()
    }
    fn mutate_commit(
        &mut self,
        _sync_dir: &Path,
        sid: &str,
        f: &mut dyn FnMut(&mut SessionRecord),
    ) -> Result<()> {
        self.commit_state(|st| f(st.sessions.entry(sid.to_string()).or_default()))
    }
}

// ----------------------------------------------------------------- tests ---

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sync::persist::fs::testkit::{setup_shared, tmp};
    use std::sync::atomic::{AtomicBool, Ordering as AOrd};
    use std::sync::{Barrier, Mutex as StdMutex};

    // cd-state-cpu spec §4.6 — commit funnel, ack semantics: a durable
    // commit blocks until the writer thread's write is on disk. The
    // assertion is the strongest form: the moment `commit_state` returns,
    // the mutation is readable from `state.json` itself, no sleeps.
    #[test]
    fn durable_commit_acks_only_after_disk_write() {
        let (mirror, shared) = setup_shared("shared", "durable-ack");
        shared
            .commit_state(|st| {
                st.sessions.insert("s1".into(), SessionRecord::default());
            })
            .unwrap();
        let disk = state::load(&shared.sync_dir).unwrap();
        assert!(
            disk.state.sessions.contains_key("s1"),
            "durable commit returned before the write landed"
        );
        assert!(shared.lane_write_count() >= 1);
        std::fs::remove_dir_all(&mirror).unwrap();
    }

    // Ack ordering across generations: a lazy commit followed by a
    // durable one must not lose the lazy one's content — durable ≥ gen
    // covers every earlier generation by construction.
    #[test]
    fn durable_ack_covers_earlier_lazy_generations() {
        let (mirror, shared) = setup_shared("shared", "ack-covers");
        shared
            .commit_state_lazy(|st| {
                st.sessions.insert("lazy".into(), SessionRecord::default());
            })
            .unwrap();
        shared
            .commit_state(|st| {
                st.sessions.insert("durable".into(), SessionRecord::default());
            })
            .unwrap();
        let disk = state::load(&shared.sync_dir).unwrap();
        assert!(disk.state.sessions.contains_key("lazy"));
        assert!(disk.state.sessions.contains_key("durable"));
        std::fs::remove_dir_all(&mirror).unwrap();
    }

    // Coalescing (spec §4.6's fake-writer test): the writer parked
    // inside its first write cannot serialize per-commit — the writes
    // that eventually happen must be FEWER than the commits issued.
    // Deterministic by construction: the write hook parks the first
    // write until every candidate commit has landed, so the candidates
    // can only ever ride one serialization.
    #[test]
    fn writer_coalesces_commits_into_fewer_writes() {
        let (mirror, shared) = setup_shared("shared", "coalesce");
        let writes_before = shared.lane_write_count();

        let gate = Arc::new(Barrier::new(2));
        let entered = Arc::new(AtomicBool::new(false));
        let once = Arc::new(StdMutex::new(false));
        {
            let gate_hook = Arc::clone(&gate);
            let entered_hook = Arc::clone(&entered);
            let once_hook = Arc::clone(&once);
            shared.set_lane_write_hook(Some(Arc::new(move || {
                let mut fired = once_hook.lock().unwrap();
                if !*fired {
                    *fired = true;
                    drop(fired);
                    entered_hook.store(true, AOrd::SeqCst);
                    gate_hook.wait();
                }
                Ok(())
            })));
        }
        // Drive the writer into the parked write with one commit.
        let probe_shared = Arc::clone(&shared);
        let probe = std::thread::spawn(move || {
            probe_shared
                .commit_state(|st| {
                    st.sessions.insert("probe".into(), SessionRecord::default());
                })
                .unwrap();
        });
        for _ in 0..400 {
            if entered.load(AOrd::SeqCst) {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(entered.load(AOrd::SeqCst), "writer must reach the write");
        // 30 lazy commits land while the writer is parked — none of
        // them can be inside the in-flight write's serialization.
        for i in 0..30 {
            shared
                .commit_state_lazy(|st| {
                    st.sessions.insert(format!("cand-{i}"), SessionRecord::default());
                })
                .unwrap();
        }
        // Release: everything coalesces into the write already in
        // flight plus at most ONE serialization for all 30 candidates.
        gate.wait();
        probe.join().unwrap();
        shared
            .commit_state(|st| {
                st.sessions.insert("flush".into(), SessionRecord::default());
            })
            .unwrap();
        let writes = shared.lane_write_count() - writes_before;
        assert!(
            writes <= 3,
            "32 commits must not cost 32 physical writes, got {writes}"
        );
        let disk = state::load(&shared.sync_dir).unwrap();
        assert!(disk.state.sessions.contains_key("cand-29"));
        assert!(disk.state.sessions.contains_key("probe"));
        std::fs::remove_dir_all(&mirror).unwrap();
    }

    // Failure propagation (spec §4.4): a write IO error surfaces to the
    // durable waiter as Err — caller semantics unchanged — and the next
    // commit retries (a write is tried again only because a NEWER
    // generation exists; the lane never hot-loops on a dead write).
    #[test]
    fn durable_waiter_fails_fast_on_write_error_then_recovers() {
        let (mirror, shared) = setup_shared("shared", "write-err");
        // `state.json` as a DIRECTORY: the rename step of write_atomic
        // fails deterministically (ENOTDIR/EISDIR) on any platform.
        std::fs::remove_file(state::state_path(&shared.sync_dir)).ok();
        std::fs::create_dir(state::state_path(&shared.sync_dir)).unwrap();
        let err = shared
            .commit_state(|st| {
                st.sessions.insert("s1".into(), SessionRecord::default());
            })
            .unwrap_err();
        assert!(
            err.to_string().contains("state write failed"),
            "write failure must propagate through the ack: {err:#}"
        );
        // Repair the path: the same mutation (still in memory) plus the
        // new one both land — a failed write loses nothing.
        std::fs::remove_dir(state::state_path(&shared.sync_dir)).unwrap();
        shared
            .commit_state(|st| {
                st.sessions.insert("s2".into(), SessionRecord::default());
            })
            .unwrap();
        let disk = state::load(&shared.sync_dir).unwrap();
        assert!(disk.state.sessions.contains_key("s1"),
            "the failed generation's content rides the next write");
        assert!(disk.state.sessions.contains_key("s2"));
        std::fs::remove_dir_all(&mirror).unwrap();
    }

    // Read-only open spawns no writer and registers no lane: committing
    // through it falls back to a direct write — race-free exactly
    // because nothing else in this process writes the file.
    #[test]
    fn readonly_open_has_no_writer_lane() {
        let mirror = tmp("shared", "readonly");
        let (shared, corruption) = Shared::open_readonly(&mirror).unwrap();
        assert!(corruption.is_none());
        assert!(shared.lane.is_none(), "readonly must not spawn a writer");
        assert!(
            state::lookup_lane(&shared.sync_dir).is_none(),
            "readonly must not register a lane"
        );
        shared
            .commit_state(|st| {
                st.sessions.insert("s1".into(), SessionRecord::default());
            })
            .unwrap();
        let disk = state::load(&shared.sync_dir).unwrap();
        assert!(disk.state.sessions.contains_key("s1"));
        std::fs::remove_dir_all(&mirror).unwrap();
    }

    // Lock-hold shape (spec §4.4): the fsync window is not covered by
    // the state lock — while the writer thread is parked inside a write
    // (the hook stands in for fsync latency), a reader takes the lock
    // freely and sees the in-memory mutation.
    #[test]
    fn readers_are_not_blocked_behind_write_io() {
        let (mirror, shared) = setup_shared("shared", "reader-free");
        let release = Arc::new(Barrier::new(2));
        let entered = Arc::new(AtomicBool::new(false));
        let once = Arc::new(StdMutex::new(false));
        {
            let release_hook = Arc::clone(&release);
            let entered_hook = Arc::clone(&entered);
            let once_hook = Arc::clone(&once);
            shared.set_lane_write_hook(Some(Arc::new(move || {
                let mut fired = once_hook.lock().unwrap();
                if !*fired {
                    *fired = true;
                    drop(fired);
                    entered_hook.store(true, AOrd::SeqCst);
                    release_hook.wait();
                }
                Ok(())
            })));
        }
        let bg = Arc::clone(&shared);
        let writer_side = std::thread::spawn(move || {
            bg.commit_state(|st| {
                st.sessions.insert("slow".into(), SessionRecord::default());
            })
            .unwrap();
        });
        for _ in 0..400 {
            if entered.load(AOrd::SeqCst) {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(entered.load(AOrd::SeqCst), "writer must reach the write");
        // The write is now parked mid-IO: the lock is provably not held
        // across it, or this read would block for the barrier's release.
        let seen = shared.with_state(|st| st.sessions.len());
        assert_eq!(seen, 1, "in-memory mutation is visible pre-durability");
        release.wait();
        writer_side.join().unwrap();
        std::fs::remove_dir_all(&mirror).unwrap();
    }
}
