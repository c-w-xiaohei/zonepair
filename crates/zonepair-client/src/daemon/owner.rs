//! Writer ownership for `zonepaird` (RFC-0003-zonepaird §3): the lock IS the
//! authority. `<mirror>/.sync/owner.lock` is `flock(LOCK_EX|LOCK_NB)` —
//! who asks "who is the writer" asks this acquire's result, never the
//! file's existence: `.sync/` may be deleted wholesale and any residue
//! file proves nothing.
//!
//! Three pieces live here:
//!
//! - `Owner` — the held lock. The holder's `flock` is bound to the inode
//!   it opened; a wholesale `.sync/` delete leaves it locked on an
//!   unlinked inode while a fresh spawn wins the race on a rebuilt
//!   `owner.lock` — two writers at once (`flock-vs-unlink`, §3.3).
//!   `stale()` is the one-line countermeasure: every main-loop round
//!   re-stats the path and compares inodes against the held fd.
//! - `spawn_daemon` — the detached spawn every "writer missing" detector
//!   shares (§3.2): `current_exe` re-invoked as `zonepaird`, stdin/stdout
//!   to `/dev/null` and stderr append-logged to `<mirror>/.sync/zonepaird.log`
//!   (one-generation rotation at 4MB, `/dev/null` fallback — the daemon's
//!   bring-up never depends on its log opening), detached from the
//!   spawning process so the stub's `process::exit` cannot reach it. N
//!   concurrent spawns self-converge on the lock — the losers exit at
//!   startup step 1, no election, no residue.
//! - `ensure_owner` — the §3.4 probe-and-spawn: lock free → release and
//!   spawn; lock held → `AlreadyOwned`, never spawn. `journal::append`
//!   does not depend on the writer being alive, so a failed
//!   `ensure_owner` never invalidates a written intent — the waiter's
//!   own revive loop (§7.1) retries later.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

use super::super::sync::state;

/// The lock file name inside `<mirror>/.sync/`.
pub const LOCK_FILE: &str = "owner.lock";
/// The live-writer pid, next to the lock — `zonepaird stop`'s SIGTERM
/// target. Not authoritative: only the flock answers "who owns".
pub const PID_FILE: &str = "owner.pid";
/// The daemon's stderr capture inside `<mirror>/.sync/` — the only
/// forensic channel a self-reborn writer has.
pub const STDERR_LOG_FILE: &str = "zonepaird.log";
/// Rotate `zonepaird.log` past this size at spawn time: one generation
/// (`.log.1`) is enough for forensics — no logging framework.
pub const STDERR_LOG_MAX_BYTES: u64 = 4 * 1024 * 1024;

/// A held owner lock: the file handle (the lock itself — dropped = lost)
/// plus the inode identity the self-check compares against.
pub struct Owner {
    file: std::fs::File,
    path: PathBuf,
    /// `(ino, dev)` of the fd at acquire time (unix); `None` elsewhere —
    /// the non-unix approximation falls back to "path exists".
    inode: Option<(u64, u64)>,
}

impl Owner {
    /// `stat(owner.lock)` vs the held fd's identity. Mismatch — or the
    /// path gone — means `.sync/` was rebuilt under us: another spawn
    /// can win the new file's lock, so we must yield at once.
    pub fn stale(&self) -> bool {
        let Ok(meta) = std::fs::metadata(&self.path) else {
            return true;
        };
        match self.inode {
            Some((ino, dev)) => {
                #[cfg(unix)]
                {
                    use std::os::unix::fs::MetadataExt;
                    meta.ino() != ino || meta.dev() != dev
                }
                #[cfg(not(unix))]
                {
                    let _ = (ino, dev);
                    false
                }
            }
            // Non-unix fallback: the path still exists — as close to
            // "same file instance" as a non-inode stat gets.
            None => false,
        }
    }

    /// Liveness heartbeat for external observers (`doctor`, tests): the
    /// lock file's mtime rides every main-loop round. Cheap — no
    /// allocation, no second file — and it cannot lie about being held:
    /// a dead process stops updating it.
    pub fn beat(&self) {
        let _ = self.file.set_modified(std::time::SystemTime::now());
    }

    /// Write `owner.pid` — `zonepaird stop`'s SIGTERM target.
    pub fn write_pid(&self, sync_dir: &Path) -> Result<()> {
        let tmp = sync_dir.join(format!(".{PID_FILE}.tmp"));
        let body = std::process::id().to_string();
        state::write_atomic(&tmp, &sync_dir.join(PID_FILE), body.as_bytes())
    }

    /// Release the pid file on graceful exit (the lock itself releases
    /// when `self` drops / the process dies — no explicit unlock needed).
    pub fn clear_pid(&self, sync_dir: &Path) {
        let _ = std::fs::remove_file(sync_dir.join(PID_FILE));
    }
}

/// Open `owner.lock` (creating `.sync/` 0700 + the file 0600) and try
/// `flock(LOCK_EX|LOCK_NB)`. `Ok(None)` means a live writer holds it —
/// the caller exits immediately (§3.2 self-convergence) or reports
/// "already owned" (`ensure_owner`).
pub fn acquire(sync_dir: &Path) -> Result<Option<Owner>> {
    state::ensure_dir(sync_dir, 0o700)?;
    let path = sync_dir.join(LOCK_FILE);
    let file = open_lock(&path).with_context(|| format!("opening {}", path.display()))?;
    match file.try_lock() {
        Ok(()) => {}
        Err(std::fs::TryLockError::WouldBlock) => return Ok(None),
        Err(std::fs::TryLockError::Error(e)) => {
            return Err(e).with_context(|| format!("locking {}", path.display()))
        }
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = file.set_permissions(std::fs::Permissions::from_mode(0o600));
    }
    Ok(Some(Owner {
        inode: inode_of(&file),
        file,
        path,
    }))
}

#[cfg(unix)]
fn open_lock(path: &Path) -> std::io::Result<std::fs::File> {
    use std::os::unix::fs::OpenOptionsExt;
    std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .mode(0o600)
        .open(path)
}

#[cfg(not(unix))]
fn open_lock(path: &Path) -> std::io::Result<std::fs::File> {
    std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open(path)
}

#[cfg(unix)]
fn inode_of(file: &std::fs::File) -> Option<(u64, u64)> {
    use std::os::unix::fs::MetadataExt;
    file.metadata().ok().map(|m| (m.ino(), m.dev()))
}

#[cfg(not(unix))]
fn inode_of(_file: &std::fs::File) -> Option<(u64, u64)> {
    None
}

/// Best-effort stderr capture for a freshly spawned daemon: append-open
/// `<mirror>/.sync/zonepaird.log` (creating `.sync/` and the file at
/// 0600 if absent), rotating first when the existing log passed
/// `STDERR_LOG_MAX_BYTES` — the old log renames over `zonepaird.log.1`,
/// one generation only. `None` on any failure: spawn falls back to
/// `/dev/null` because bring-up must never hinge on the log opening.
fn open_stderr_log(mirror: &Path) -> Option<std::fs::File> {
    let sync = state::sync_dir(mirror);
    state::ensure_dir(&sync, 0o700).ok()?;
    let log = sync.join(STDERR_LOG_FILE);
    let oversized = std::fs::metadata(&log)
        .map(|m| m.len() > STDERR_LOG_MAX_BYTES)
        .unwrap_or(false);
    if oversized {
        let _ = std::fs::rename(&log, sync.join(format!("{STDERR_LOG_FILE}.1")));
    }
    open_log_append(&log).ok()
}

#[cfg(unix)]
fn open_log_append(path: &Path) -> std::io::Result<std::fs::File> {
    use std::os::unix::fs::OpenOptionsExt;
    std::fs::OpenOptions::new()
        .append(true)
        .create(true)
        .mode(0o600)
        .open(path)
}

#[cfg(not(unix))]
fn open_log_append(path: &Path) -> std::io::Result<std::fs::File> {
    std::fs::OpenOptions::new()
        .append(true)
        .create(true)
        .open(path)
}

/// Try the lock without keeping it — `ensure_owner`'s probe (§3.2:
/// "trying the lock is the cheapest probe and is race-free by nature").
pub fn lock_free(sync_dir: &Path) -> bool {
    match acquire(sync_dir) {
        Ok(Some(owner)) => {
            drop(owner); // probe only — never claim it
            true
        }
        _ => false,
    }
}

// ------------------------------------------------------------------ spawn --

/// What `ensure_owner` decided — the caller distinguishes "spawn failed"
/// from "another writer exists" (§3.4: the intent append stands either
/// way; the waiter times out loudly instead).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EnsureOwner {
    /// A fresh `zonepaird` was spawned (or the spawner reported success).
    Spawned,
    /// The lock is held — a live writer exists; spawning is wrong.
    AlreadyOwned,
}

/// Detached spawn: re-invoke this binary as `stub-client zonepaird
/// [--mirror <dir>]` — same argv the CLI dispatch understands, stdin and
/// stdout to `/dev/null`, stderr append-logged to
/// `<mirror>/.sync/zonepaird.log` when that opens (rotating one
/// generation past 4MB), `/dev/null` when it does not (§3.2). Separated
/// from the parent's process group so stub's `process::exit` cannot
/// take it down. The child is deliberately un-reaped: it outlives us;
/// the lock file, not `waitpid`, is its liveness report.
pub fn spawn_daemon(mirror: &Path) -> Result<()> {
    let exe = std::env::current_exe().context("resolving current_exe")?;
    let mut cmd = std::process::Command::new(exe);
    let stderr = open_stderr_log(mirror)
        .map(std::process::Stdio::from)
        .unwrap_or_else(std::process::Stdio::null);
    cmd.arg("zonepaird")
        .arg("--mirror")
        .arg(mirror)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(stderr);
    #[cfg(unix)]
    {
        // `process_group(0)` detaches into a new process group — the
        // posix_spawnp setsid-equivalent std exposes without libc.
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }
    cmd.spawn()
        .map(|_| ())
        .with_context(|| format!("spawning zonepaird for {}", mirror.display()))
}

/// The §3.2/§3.4 entry every intent writer runs first: probe the lock —
/// held → `AlreadyOwned` (do NOT spawn); free → spawn one detached
/// `zonepaird`. N callers racing here spawn N daemons that converge on
/// the lock themselves.
///
/// `spawn` is the injection seam: production `spawn_daemon`, tests a
/// counter — so this function is assertable without ever starting a real
/// process.
pub fn ensure_owner(
    sync_dir: &Path,
    mirror: &Path,
    spawn: impl FnOnce(&Path) -> Result<()>,
) -> Result<EnsureOwner> {
    if !lock_free(sync_dir) {
        return Ok(EnsureOwner::AlreadyOwned);
    }
    spawn(mirror)?;
    Ok(EnsureOwner::Spawned)
}

/// `ensure_owner` with the production spawner bound — the signature CLI
/// and transport callers use.
pub fn ensure_owner_daemon(sync_dir: &Path, mirror: &Path) -> Result<EnsureOwner> {
    ensure_owner(sync_dir, mirror, spawn_daemon)
}

// ----------------------------------------------------------------- tests ---

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    fn tmp(name: &str) -> PathBuf {
        crate::sync::persist::fs::testkit::tmp("owner", name)
    }

    // T-630 halves: `.sync/` created 0700, `owner.lock` 0600; a second
    // acquire while held answers `None` (the losing spawn's exit path).
    #[test]
    fn t630_lock_modes_and_held_answers_none() {
        let mirror = tmp("t630");
        let sync = state::ensure_sync_dir(&mirror).unwrap();
        let owner = acquire(&sync).unwrap().expect("first acquire wins");
        assert_eq!(
            sync.metadata().unwrap().permissions().mode() & 0o777,
            0o700
        );
        assert_eq!(
            sync.join(LOCK_FILE)
                .metadata()
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o600
        );
        assert!(
            acquire(&sync).unwrap().is_none(),
            "held lock must not re-acquire — the self-convergence rule"
        );
        assert!(!lock_free(&sync));
        drop(owner);
        // Freed lock re-acquires (fd dropped = released).
        assert!(lock_free(&sync));
        std::fs::remove_dir_all(&mirror).unwrap();
    }

    // T-640: the held fd's inode vs the path's inode — delete `.sync/`
    // wholesale and the self-check must read stale (two-writer window).
    #[test]
    fn t640_inode_self_check_catches_sync_delete() {
        let mirror = tmp("t640");
        let sync = state::ensure_sync_dir(&mirror).unwrap();
        let owner = acquire(&sync).unwrap().unwrap();
        assert!(!owner.stale(), "same inode — not stale");
        // Recreate `.sync/` with a FRESH owner.lock while the held fd
        // still locks the unlinked inode — the flock-vs-unlink window.
        std::fs::remove_dir_all(&sync).unwrap();
        let rebuilt = state::ensure_sync_dir(&mirror).unwrap();
        open_lock(&rebuilt.join(LOCK_FILE)).unwrap();
        assert!(owner.stale(), "rebuilt owner.lock must read stale");
        std::fs::remove_dir_all(&mirror).unwrap();
    }

    // The path vanishing entirely also reads stale.
    #[test]
    fn inode_check_stales_on_missing_path() {
        let mirror = tmp("t640b");
        let sync = state::ensure_sync_dir(&mirror).unwrap();
        let owner = acquire(&sync).unwrap().unwrap();
        std::fs::remove_file(sync.join(LOCK_FILE)).unwrap();
        assert!(owner.stale());
        std::fs::remove_dir_all(&mirror).unwrap();
    }

    // §3.2/§3.4: lock held → AlreadyOwned, no spawn; lock free → the
    // injected spawner runs exactly once.
    #[test]
    fn ensure_owner_never_spawns_over_a_held_lock() {
        let mirror = tmp("ensure");
        let sync = state::ensure_sync_dir(&mirror).unwrap();
        let _owner = acquire(&sync).unwrap().unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let c = calls.clone();
        let out = ensure_owner(&sync, &mirror, move |_| {
            c.fetch_add(1, Ordering::SeqCst);
            Ok(())
        })
        .unwrap();
        assert_eq!(out, EnsureOwner::AlreadyOwned);
        assert_eq!(calls.load(Ordering::SeqCst), 0, "held lock → no spawn");
        drop(_owner);
        let out = ensure_owner(&sync, &mirror, |m| {
            assert_eq!(m, mirror.as_path());
            Ok(())
        })
        .unwrap();
        assert_eq!(out, EnsureOwner::Spawned);
        std::fs::remove_dir_all(&mirror).unwrap();
    }

    // A failing spawner propagates the error (loud) — the intent append
    // still stands; the caller reports instead of retrying silently.
    #[test]
    fn ensure_owner_spawn_failure_is_loud() {
        let mirror = tmp("ensure-fail");
        let sync = state::ensure_sync_dir(&mirror).unwrap();
        let err = ensure_owner(&sync, &mirror, |_| {
            Err(anyhow::anyhow!("spawn denied"))
        })
        .unwrap_err();
        assert!(format!("{err:#}").contains("spawn denied"));
        std::fs::remove_dir_all(&mirror).unwrap();
    }

    // stderr capture: an oversized existing log rotates to
    // `zonepaird.log.1` before the new file opens; anything smaller
    // appends in place — one generation is the whole retention policy.
    #[test]
    fn stderr_log_rotates_past_the_cap_and_appends_below_it() {
        let mirror = tmp("stderr-rotate");
        let sync = state::ensure_sync_dir(&mirror).unwrap();
        let log = sync.join(STDERR_LOG_FILE);
        let rotated = sync.join(format!("{STDERR_LOG_FILE}.1"));

        // Oversized: rename first, then append-open a fresh file.
        std::fs::write(&log, vec![b'x'; (STDERR_LOG_MAX_BYTES + 1) as usize]).unwrap();
        let f = open_stderr_log(&mirror).expect("oversized log still opens");
        assert_eq!(
            std::fs::read(&rotated).unwrap().len() as u64,
            STDERR_LOG_MAX_BYTES + 1,
            "the oversized generation survives as .log.1"
        );
        assert_eq!(log.metadata().unwrap().len(), 0, "new file starts empty");
        drop(f);

        // Below the cap: no rotation, and the open is append — existing
        // content survives what new writers add.
        std::fs::write(&log, b"kept").unwrap();
        std::fs::remove_file(&rotated).unwrap();
        let mut f = open_stderr_log(&mirror).unwrap();
        assert!(
            !rotated.exists(),
            "sub-cap log must not rotate away its evidence"
        );
        use std::io::Write;
        f.write_all(b"more").unwrap();
        drop(f);
        assert_eq!(std::fs::read(&log).unwrap(), b"keptmore");
        std::fs::remove_dir_all(&mirror).unwrap();
    }

    // Rotation replaces the previous `.log.1` — only the two newest
    // generations ever exist, no unbounded `.log.N` accretion.
    #[test]
    fn stderr_log_rotation_overwrites_the_older_generation() {
        let mirror = tmp("stderr-rotate-overwrite");
        let sync = state::ensure_sync_dir(&mirror).unwrap();
        let log = sync.join(STDERR_LOG_FILE);
        let rotated = sync.join(format!("{STDERR_LOG_FILE}.1"));
        std::fs::write(&rotated, b"ancient generation").unwrap();
        std::fs::write(&log, vec![b'y'; (STDERR_LOG_MAX_BYTES + 1) as usize]).unwrap();
        open_stderr_log(&mirror).unwrap();
        assert_eq!(
            std::fs::read(&rotated).unwrap().len() as u64,
            STDERR_LOG_MAX_BYTES + 1
        );
        std::fs::remove_dir_all(&mirror).unwrap();
    }

    // Best-effort by contract: `.sync/` uncreatable (a file sits where the
    // dir must go) or the log path unopenable both answer `None` — the
    // spawn side falls back to `/dev/null` instead of failing the daemon.
    #[test]
    fn stderr_log_failure_is_silent_and_never_fatal() {
        let mirror = tmp("stderr-blocked-dir");
        std::fs::create_dir_all(&mirror).unwrap();
        std::fs::write(mirror.join(".sync"), b"not a dir").unwrap();
        assert!(open_stderr_log(&mirror).is_none());
        std::fs::remove_dir_all(&mirror).unwrap();

        let mirror = tmp("stderr-blocked-log");
        let sync = state::ensure_sync_dir(&mirror).unwrap();
        std::fs::create_dir(sync.join(STDERR_LOG_FILE)).unwrap();
        assert!(open_stderr_log(&mirror).is_none());
        std::fs::remove_dir_all(&mirror).unwrap();
    }

    // A failed rotation (`.log.1` occupied by a directory) must not lose
    // the capture: the existing log still opens for append.
    #[test]
    fn stderr_log_survives_a_failed_rotation() {
        let mirror = tmp("stderr-rotation-fails");
        let sync = state::ensure_sync_dir(&mirror).unwrap();
        let log = sync.join(STDERR_LOG_FILE);
        std::fs::write(&log, vec![b'z'; (STDERR_LOG_MAX_BYTES + 1) as usize]).unwrap();
        std::fs::create_dir(sync.join(format!("{STDERR_LOG_FILE}.1"))).unwrap();
        assert!(open_stderr_log(&mirror).is_some());
        assert!(
            log.metadata().unwrap().len() > STDERR_LOG_MAX_BYTES,
            "rotation failing leaves the file in place, still appendable"
        );
        std::fs::remove_dir_all(&mirror).unwrap();
    }

    // The capture file is private: 0600 like everything else in `.sync/`.
    #[cfg(unix)]
    #[test]
    fn stderr_log_created_with_owner_only_mode() {
        let mirror = tmp("stderr-mode");
        let sync = state::ensure_sync_dir(&mirror).unwrap();
        open_stderr_log(&mirror).unwrap();
        assert_eq!(
            sync.join(STDERR_LOG_FILE)
                .metadata()
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o600
        );
        std::fs::remove_dir_all(&mirror).unwrap();
    }
}
