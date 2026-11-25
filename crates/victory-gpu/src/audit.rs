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
    writeln!(file, "VIctory performance audit v1 | pid={} | diagnostics={}",
        std::process::id(), cfg!(feature = "diagnostics"))?;
    writeln!(file, "CPU timings can overlap; GPU submit+wait is wall time, not GPU execution.")?;
    for key in ["VICTORY_ASYNC_SUBMIT", "VICTORY_GPU_TIMING", "VICTORY_D3D12_DRED",
        "VICTORY_D3D12_DEBUG", "VICTORY_D3D12_GBV", "VICTORY_D3D12_WARP",
        "VICTORY_TEXTURE_MAX", "VICTORY_TEXTURES", "VICTORY_CARD_BUFFERS",
        "VICTORY_HOIST_CLAMPS", "VICTORY_SHADER_LOOP_LIMIT", "VICTORY_FPS"] {
        writeln!(file, "{key}={}", std::env::var(key).unwrap_or_else(|_| "<default>".into()))?;
    }
    file.flush()?;
    let (sender, receiver) = sync_channel::<String>(32);
    let _ = std::thread::Builder::new().name("victory-audit-writer".into()).spawn(move || {
        while let Ok(line) = receiver.recv() {
            if writeln!(file, "{line}").and_then(|()| file.flush()).is_err() {
                tracing::warn!("performance audit write failed");
                break;
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
