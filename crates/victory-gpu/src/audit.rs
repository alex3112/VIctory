//! Opt-in telemetry. No GPU commands or changes to submission policy.
use std::fs::File;
use std::io::{self, Write};
use std::path::Path;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::mpsc::{SyncSender, sync_channel};
use std::time::Instant;

struct Session {
    sender: SyncSender<String>,
    started: Instant,
}
static SESSION: OnceLock<Session> = OnceLock::new();
static LOST: AtomicU64 = AtomicU64::new(0);

// Only the serialized host submission writes this snapshot. The audit
// reader uses try_lock, never a host lock, and can report a stuck fence.
struct Active {
    submission: u64,
    lists: usize,
    phase: &'static str,
    list: usize,
    fence: u64,
    since: Instant,
    pipelines: String,
}
static ACTIVE: std::sync::Mutex<Option<Active>> = std::sync::Mutex::new(None);

pub(crate) struct SubmissionTrace(bool);
impl SubmissionTrace {
    pub(crate) fn begin(submission: u64, lists: usize) -> Self {
        let on = enabled();
        if on {
            *ACTIVE.lock().unwrap_or_else(|e| e.into_inner()) = Some(Active {
                submission, lists, phase: "drain_previous", list: 0, fence: 0,
                since: Instant::now(), pipelines: String::new(),
            });
        }
        Self(on)
    }
    pub(crate) fn phase(&self, phase: &'static str, list: usize, fence: u64) {
        if self.0 {
            if let Some(a) = ACTIVE.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
                a.phase = phase;
                a.list = list;
                a.fence = fence;
                a.since = Instant::now();
            }
        }
    }
    pub(crate) fn pipelines(&self, make: impl FnOnce() -> String) {
        if self.0 {
            let names = make();
            if let Some(a) = ACTIVE.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
                a.pipelines = names;
            }
        }
    }
}
impl Drop for SubmissionTrace {
    fn drop(&mut self) {
        if self.0 {
            *ACTIVE.lock().unwrap_or_else(|e| e.into_inner()) = None;
        }
    }
}
pub(crate) fn submission_state() -> String {
    match ACTIVE.try_lock() {
        Ok(a) => match a.as_ref() {
            Some(a) => format!("active_submission={} active_lists={} active_phase={} active_list={} active_fence={} active_phase_ms={} active_pipelines={:?}",
                a.submission, a.lists, a.phase, a.list, a.fence, a.since.elapsed().as_millis(), a.pipelines),
            None => "active_submission=0 active_phase=idle".into(),
        },
        Err(_) => "active_phase=transition".into(),
    }
}

pub(crate) fn enabled() -> bool {
    SESSION.get().is_some()
}

/// Start a fresh report. I/O failure is non-fatal to the caller.
pub fn start(data_dir: &Path) -> io::Result<()> {
    if std::env::var_os("VICTORY_AUDIT").is_none_or(|v| v != "1") || enabled() {
        return Ok(());
    }
    let path = data_dir.join("performance_audit.log");
    let mut file = File::create(&path)?;
    writeln!(file, "VIctory performance audit v4 | pid={} | diagnostics={}",
        std::process::id(), cfg!(feature = "diagnostics"))?;
    writeln!(file, "CPU timings can overlap; GPU submit+wait is wall time, not GPU execution.")?;
    for key in ["VICTORY_ASYNC_SUBMIT", "VICTORY_GPU_TIMING", "VICTORY_D3D12_DRED",
        "VICTORY_D3D12_DEBUG", "VICTORY_D3D12_GBV", "VICTORY_D3D12_WARP",
        "VICTORY_TEXTURE_MAX", "VICTORY_TEXTURES", "VICTORY_CARD_BUFFERS",
        "VICTORY_HOIST_CLAMPS", "VICTORY_SHADER_LOOP_LIMIT", "VICTORY_FPS",
        "VICTORY_SWEEP_TILES", "VICTORY_KEEP_RETIRED", "VICTORY_STAGING_BUDGET_MB",
        "VICTORY_QUEUE_WORKER", "VICTORY_FORCE_WAITS"] {
        writeln!(file, "{key}={}", std::env::var(key).unwrap_or_else(|_| "<default>".into()))?;
    }
    file.flush()?;
    let (sender, receiver) = sync_channel::<String>(32);
    let _ = std::thread::Builder::new().name("victory-audit-writer".into()).spawn(move || {
        use std::sync::mpsc::RecvTimeoutError;
        use std::time::Duration;
        let started = Instant::now();
        let interval = Duration::from_secs(1);
        let mut next = started;
        loop {
            // Independent of frame production, submission locks and GPU waits.
            // Only query OS counters and atomics here: a stalled GPU must not
            // stop memory reporting too.
            let now = Instant::now();
            if now >= next {
                let line = format!("[+{:.3}s] memory_sample presented={} {} {} {} {} {} {} {}",
                    started.elapsed().as_secs_f64(), crate::queue::presented(),
                    process_memory(), system_memory(), crate::memstat::atomic_report(),
                    crate::timeline::audit_state(), crate::texpool::audit_state(), crate::resource::placed_audit(), submission_state());
                if writeln!(file, "{line}").and_then(|()| file.flush()).is_err() {
                    tracing::warn!("performance audit memory write failed");
                    break;
                }
                next = now + interval;
            }
            match receiver.recv_timeout(next.saturating_duration_since(Instant::now())) {
                Ok(line) => {
                    if writeln!(file, "{line}").and_then(|()| file.flush()).is_err() {
                        tracing::warn!("performance audit write failed");
                        break;
                    }
                }
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => break,
            }
        }
    })?;
    let _ = SESSION.set(Session { sender, started: Instant::now() });
    tracing::info!(path = %path.display(), "performance audit enabled");
    Ok(())
}

pub(crate) fn emit(text: String) {
    if let Some(s) = SESSION.get() {
        let lost = LOST.swap(0, Relaxed);
        let line = format!("[+{:.3}s; dropped report records={lost}] {text}",
            s.started.elapsed().as_secs_f64());
        if s.sender.try_send(line).is_err() {
            LOST.fetch_add(lost.saturating_add(1), Relaxed);
        }
    }
}

/// A failure from the HLE or GPU allocation path. No allocation policy change.
pub fn record_memory_failure(context: &str, bytes: u64, error: &str) {
    if enabled() {
        emit(format!("allocation_failure context={context} bytes={bytes} error={error}"));
    }
}

fn system_memory() -> String {
    use windows_sys::Win32::System::ProcessStatus::{K32GetPerformanceInfo, PERFORMANCE_INFORMATION};
    let mut info = PERFORMANCE_INFORMATION {
        cb: size_of::<PERFORMANCE_INFORMATION>() as u32,
        ..Default::default()
    };
    // SAFETY: the buffer is a whole writable PERFORMANCE_INFORMATION.
    if unsafe { K32GetPerformanceInfo(&mut info, info.cb) } == 0 {
        return "system_memory=unavailable".into();
    }
    let mib = |pages: usize| pages.saturating_mul(info.PageSize) >> 20;
    format!("system_commit_mib={} system_commit_limit_mib={} system_ram_available_mib={} system_ram_total_mib={}",
        mib(info.CommitTotal), mib(info.CommitLimit), mib(info.PhysicalAvailable), mib(info.PhysicalTotal))
}

#[derive(Clone, Copy)]
pub(crate) enum C {
    SubmitLockUs, FenceWaitUs, PipelineWaitUs, PipelineBuildUs,
    PipelineBuilds, PipelineFailed, PresentUs, FramePaceUs, ClampCalls, ClampMaxRecords,
}
const COUNT: usize = 10;
static COUNTERS: [AtomicU64; COUNT] = [const { AtomicU64::new(0) }; COUNT];

pub(crate) fn add(c: C, n: u64) {
    if enabled() {
        COUNTERS[c as usize].fetch_add(n, Relaxed);
    }
}

pub(crate) fn timer() -> Option<Instant> {
    enabled().then(Instant::now)
}

pub(crate) fn elapsed(c: C, started: Option<Instant>) {
    if let Some(started) = started {
        add(c, started.elapsed().as_micros() as u64);
    }
}

pub(crate) fn counters(frames: f64) -> String {
    let a: [u64; COUNT] = std::array::from_fn(|i| COUNTERS[i].swap(0, Relaxed));
    let value = |c: C| a[c as usize] as f64 / frames;
    let ms = |c: C| value(c) / 1000.0;
    format!(
        "submit_lock_ms/frame={:.3} fence_wait_ms/frame={:.3} pipeline_wait_thread_ms/frame={:.3} \
         pipeline_build_worker_ms/frame={:.3} pipeline_builds={} pipeline_failures={} \
         present_call_ms/frame={:.3} frame_pace_thread_ms/frame={:.3} \
         clamp_calls/frame={:.1} clamp_max_records/frame={:.1}",
        ms(C::SubmitLockUs), ms(C::FenceWaitUs), ms(C::PipelineWaitUs),
        ms(C::PipelineBuildUs), a[C::PipelineBuilds as usize], a[C::PipelineFailed as usize],
        ms(C::PresentUs), ms(C::FramePaceUs), value(C::ClampCalls), value(C::ClampMaxRecords),
    )
}

/// Actual intervals between presented frames, nearest-rank percentiles.
pub(crate) fn frame_stats(samples: &mut [f64]) -> String {
    if samples.is_empty() {
        return "frame_intervals=0".into();
    }
    samples.sort_unstable_by(f64::total_cmp);
    let n = samples.len();
    let q = |percent: usize| samples[(n * percent).div_ceil(100).saturating_sub(1)];
    let mean = samples.iter().sum::<f64>() / n as f64;
    format!("frame_intervals={n} mean_ms={mean:.3} min_ms={:.3} p50_ms={:.3} p95_ms={:.3} p99_ms={:.3} max_ms={:.3}",
        samples[0], q(50), q(95), q(99), samples[n - 1])
}

pub(crate) fn process_memory() -> String {
    use windows_sys::Win32::System::ProcessStatus::{
        K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS_EX,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut c = PROCESS_MEMORY_COUNTERS_EX {
        cb: size_of::<PROCESS_MEMORY_COUNTERS_EX>() as u32,
        ..Default::default()
    };
    let cb = c.cb;
    // SAFETY: our own process and a sized extended counter structure.
    let ok = unsafe {
        K32GetProcessMemoryInfo(GetCurrentProcess(), (&mut c as *mut PROCESS_MEMORY_COUNTERS_EX).cast(), cb)
    };
    if ok == 0 {
        return "process_memory=unavailable".into();
    }
    format!("process_working_set_mib={} process_private_commit_mib={} page_faults_total={}",
        c.WorkingSetSize >> 20, c.PrivateUsage >> 20, c.PageFaultCount)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn frame_quantiles_and_empty_window() {
        assert_eq!(frame_stats(&mut []), "frame_intervals=0");
        let mut samples = [40.0, 10.0, 30.0, 20.0];
        assert_eq!(frame_stats(&mut samples),
            "frame_intervals=4 mean_ms=25.000 min_ms=10.000 p50_ms=20.000 p95_ms=40.000 p99_ms=40.000 max_ms=40.000");
    }
}
