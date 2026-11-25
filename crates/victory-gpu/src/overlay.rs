//! Read-only HUD on a separate, click-through window and thread. No GPU
//! readback, synchronous messages to the title, or diagnostic modes required.
use std::collections::VecDeque;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};
use windows_sys::Win32::Foundation::*;
use windows_sys::Win32::Graphics::Gdi::*;
use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;
use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
use windows_sys::Win32::System::ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS_EX};
use windows_sys::Win32::System::Threading::*;
use windows_sys::Win32::UI::Input::KeyboardAndMouse::GetAsyncKeyState;
use windows_sys::Win32::UI::WindowsAndMessaging::*;

#[derive(Default)]
pub(crate) struct History(VecDeque<f64>);
impl History {
    pub(crate) fn push(&mut self, ms: f64) {
        if !ms.is_finite() || ms <= 0.0 { return; }
        if self.0.len() == 240 { self.0.pop_front(); }
        self.0.push_back(ms);
    }

    fn summary(&self) -> String {
        if self.0.is_empty() { return "Frame history: collecting...".into(); }
        let mut times: Vec<_> = self.0.iter().copied().collect();
        times.sort_by(f64::total_cmp);
        let percentile = |p: f64| times[((times.len() as f64 * p).ceil() as usize).saturating_sub(1)];
        let slow = times.len().div_ceil(100);
        let low = 1000.0 / (times[times.len() - slow..].iter().sum::<f64>() / slow as f64);
        format!("History {} frames | min {:.1}  P50 {:.1}  P95 {:.1}  P99 {:.1} ms\nMax {:.1} ms | 1% low {:.2} fps | >33ms {}  >50ms {}  >100ms {}",
            times.len(), times[0], percentile(0.5), percentile(0.95), percentile(0.99),
            times[times.len()-1], low, times.iter().filter(|&&t| t > 33.34).count(),
            times.iter().filter(|&&t| t > 50.0).count(), times.iter().filter(|&&t| t > 100.0).count())
    }
}

struct State {
    target: usize,
    text: String,
    history: Vec<f64>,
    updated: Instant,
    paint: String,
}
static STATE: OnceLock<Mutex<State>> = OnceLock::new();

pub(crate) fn publish(target: usize, text: String, history: &History) {
    let state = STATE.get_or_init(|| {
        let _ = std::thread::Builder::new().name("victory-overlay".into()).spawn(run);
        Mutex::new(State { target, text: String::new(), history: Vec::new(), updated: Instant::now(), paint: String::new() })
    });
    update(state, target, text, history);
}

fn update(state: &Mutex<State>, target: usize, text: String, history: &History) -> bool {
    // A HUD must never delay presentation behind window painting or UI work.
    let mut s = match state.try_lock() {
        Ok(s) => s,
        Err(std::sync::TryLockError::WouldBlock) => return false,
        Err(std::sync::TryLockError::Poisoned(e)) => e.into_inner(),
    };
    s.target = target;
    s.text = format!("{text}\n{}", history.summary());
    s.history = history.0.iter().copied().collect();
    s.updated = Instant::now();
    true
}

fn wide(s: &str) -> Vec<u16> { s.encode_utf16().chain([0]).collect() }

unsafe extern "system" fn proc(hwnd: HWND, msg: u32, wp: WPARAM, lp: LPARAM) -> LRESULT {
    // SAFETY: the UI thread owns the window and all GDI objects below.
    unsafe {
        if msg == WM_NCHITTEST { return HTTRANSPARENT as LRESULT; }
        if msg == WM_PAINT {
            let mut ps = PAINTSTRUCT::default();
            let dc = BeginPaint(hwnd, &mut ps);
            let mut rect = RECT::default();
            GetClientRect(hwnd, &mut rect);
            FillRect(dc, &rect, GetStockObject(BLACK_BRUSH) as HBRUSH);
            let font = GetWindowLongPtrW(hwnd, GWLP_USERDATA) as HFONT;
            let old = SelectObject(dc, font as _);
            SetBkMode(dc, TRANSPARENT as i32);
            SetTextColor(dc, 0x00d8f0d8);
            if let Some(state) = STATE.get() {
                let (paint, history) = {
                    let s = state.lock().unwrap_or_else(|e| e.into_inner());
                    (s.paint.clone(), s.history.clone())
                };
                let text = wide(&paint);
                let mut area = RECT { left: 12, top: 10, right: rect.right - 10, bottom: rect.bottom - 48 };
                let drawn = DrawTextW(dc, text.as_ptr(), (text.len()-1) as i32, &mut area, DT_LEFT | DT_NOPREFIX);
                if drawn == 0 && text.len() > 1 {
                    static WARNED: std::sync::Once = std::sync::Once::new();
                    WARNED.call_once(|| tracing::warn!("statistics overlay text could not be painted"));
                }
                // Frame-time trend, independently scaled; no GPU queries.
                if history.len() > 1 {
                    let max = history.iter().copied().fold(33.34, f64::max);
                    let pen = CreatePen(PS_SOLID, 1, 0x00d0d040);
                    let old_pen = SelectObject(dc, pen as _);
                    for (i, &ms) in history.iter().enumerate() {
                        let x = 12 + i as i32 * (rect.right - 24) / (history.len()-1) as i32;
                        let y = rect.bottom - 8 - (ms / max * 32.0) as i32;
                        if i == 0 { MoveToEx(dc, x, y, std::ptr::null_mut()); } else { LineTo(dc, x, y); }
                    }
                    SelectObject(dc, old_pen);
                    DeleteObject(pen as _);
                }
            }
            SelectObject(dc, old);
            EndPaint(hwnd, &ps);
            return 0;
        }
        DefWindowProcW(hwnd, msg, wp, lp)
    }
}

#[derive(Default)]
struct ProcessSample { cpu: u64, read: u64, written: u64, faults: u32, since: Option<Instant> }
impl ProcessSample {
    fn read(&mut self) -> String {
        // SAFETY: sized output structures for the current process/system.
        unsafe {
            let process = GetCurrentProcess();
            let mut mem = PROCESS_MEMORY_COUNTERS_EX::default();
            mem.cb = size_of_val(&mem) as u32;
            let memory_ok = K32GetProcessMemoryInfo(process, (&raw mut mem).cast(), mem.cb) != 0;
            let mut system = MEMORYSTATUSEX::default();
            system.dwLength = size_of_val(&system) as u32;
            GlobalMemoryStatusEx(&mut system);
            let (mut created, mut exited, mut kernel, mut user) = (FILETIME::default(), FILETIME::default(), FILETIME::default(), FILETIME::default());
            let times_ok = GetProcessTimes(process, &mut created, &mut exited, &mut kernel, &mut user) != 0;
            let to_u64 = |t: FILETIME| (u64::from(t.dwHighDateTime) << 32) | u64::from(t.dwLowDateTime);
            let cpu = to_u64(kernel) + to_u64(user);
            let mut io = IO_COUNTERS::default();
            GetProcessIoCounters(process, &mut io);
            let elapsed = self.since.map(|t| t.elapsed().as_secs_f64()).unwrap_or(0.0);
            let mib = |n: u64| n as f64 / 1048576.0;
            let rate = |n: u64, old: u64| if elapsed > 0.0 { mib(n.saturating_sub(old)) / elapsed } else { 0.0 };
            let percent = if elapsed > 0.0 && times_ok { cpu.saturating_sub(self.cpu) as f64 / 1e7 / elapsed * 100.0 / std::thread::available_parallelism().map_or(1, |n| n.get()) as f64 } else { 0.0 };
            let text = if memory_ok {
                format!("CPU process {:.1}% | RAM resident {:.0} MiB | private commit {:.0} MiB\nRAM peak {:.0} MiB | system free {:.0}/{:.0} MiB ({}% used)\nProcess I/O read {:.1} / write {:.1} MiB/s | page faults +{}",
                    percent, mib(mem.WorkingSetSize as u64), mib(mem.PrivateUsage as u64), mib(mem.PeakWorkingSetSize as u64),
                    mib(system.ullAvailPhys), mib(system.ullTotalPhys), system.dwMemoryLoad,
                    rate(io.ReadTransferCount, self.read), rate(io.WriteTransferCount, self.written), mem.PageFaultCount.saturating_sub(self.faults))
            } else { "Process memory unavailable".into() };
            let text = format!("{text}\nSystem commit {:.0}/{:.0} MiB", mib(system.ullTotalPageFile.saturating_sub(system.ullAvailPageFile)), mib(system.ullTotalPageFile));
            self.cpu = cpu; self.read = io.ReadTransferCount; self.written = io.WriteTransferCount;
            self.faults = mem.PageFaultCount; self.since = Some(Instant::now());
            text
        }
    }
}

fn run() {
    // SAFETY: this dedicated thread creates, paints and destroys its own window.
    unsafe {
        let class = wide("VIctoryMetricsOverlay");
        let instance = GetModuleHandleW(std::ptr::null());
        let wc = WNDCLASSW { lpfnWndProc: Some(proc), hInstance: instance, lpszClassName: class.as_ptr(), ..Default::default() };
        RegisterClassW(&wc);
        let hwnd = CreateWindowExW(WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW,
            class.as_ptr(), wide("VIctory statistics").as_ptr(), WS_POPUP, 0, 0, 720, 440,
            std::ptr::null_mut(), std::ptr::null_mut(), instance, std::ptr::null());
        if hwnd.is_null() { tracing::warn!("could not create statistics overlay"); return; }
        let font = CreateFontW(16, 0, 0, 0, 400, 0, 0, 0, 1, 0, 0, 0, 1, wide("Consolas").as_ptr());
        SetWindowLongPtrW(hwnd, GWLP_USERDATA, font as isize);
        SetLayeredWindowAttributes(hwnd, 0, 225, LWA_ALPHA);
        let mut visible = true;
        let mut memory_page = false;
        let mut keys = [false; 2];
        let mut sample = ProcessSample::default();
        let mut sampled = Instant::now() - Duration::from_secs(2);
        let mut process_text = String::new();
        let mut memory_text = String::new();
        let mut vram_text = String::new();
        let mut redrawn = Instant::now() - Duration::from_secs(1);
        let mut owner = std::ptr::null_mut();
        let mut placement = None;
        loop {
            let mut msg = MSG::default();
            while PeekMessageW(&mut msg, std::ptr::null_mut(), 0, 0, PM_REMOVE) != 0 {
                TranslateMessage(&msg); DispatchMessageW(&msg);
            }
            let Some(shared) = STATE.get() else { std::thread::sleep(Duration::from_millis(50)); continue; };
            let target = shared.lock().unwrap_or_else(|e| e.into_inner()).target as HWND;
            if IsWindow(target) == 0 { break; }
            if owner != target {
                SetWindowLongPtrW(hwnd, GWLP_HWNDPARENT, target as isize);
                owner = target;
            }
            let focused = GetForegroundWindow() == target;
            for (i, key) in [0x72, 0x73].into_iter().enumerate() {
                let down = GetAsyncKeyState(key) < 0;
                if focused && down && !keys[i] {
                    if i == 0 { visible = !visible; } else { memory_page = !memory_page; }
                    tracing::info!(visible, memory_page, "statistics overlay toggled");
                }
                keys[i] = down;
            }
            if !visible || !focused || IsIconic(target) != 0 {
                if placement.take().is_some() { ShowWindow(hwnd, SW_HIDE); }
            }
            else {
                if redrawn.elapsed() < Duration::from_millis(200) {
                    std::thread::sleep(Duration::from_millis(50));
                    continue;
                }
                redrawn = Instant::now();
                if sampled.elapsed() >= Duration::from_secs(1) {
                    process_text = sample.read();
                    memory_text = crate::memstat::report().replace(", ", "\n");
                    vram_text = crate::host::video_memory().map_or_else(|| "VRAM budget unavailable".into(), |[v, s]| format!(
                        "Driver VRAM {:.0}/{:.0} MiB | shared {:.0}/{:.0} MiB",
                        v.0 as f64 / 1048576.0, v.1 as f64 / 1048576.0, s.0 as f64 / 1048576.0, s.1 as f64 / 1048576.0));
                    sampled = Instant::now();
                }
                let mut s = shared.lock().unwrap_or_else(|e| e.into_inner());
                let age = s.updated.elapsed().as_secs_f64();
                s.paint = format!("VIctory | F3 hide/show | F4 performance/memory | age {age:.1}s\n{}\n\n{process_text}\n{vram_text}\n{}",
                    if memory_page { &memory_text } else { &s.text },
                    if memory_page { "Host allocations are not the driver's residency." } else { "GPU wait includes sync; CPU sub-costs may overlap. I/O is process-wide." });
                let lines = s.paint.lines().count() as i32;
                drop(s);
                let mut rect = RECT::default(); GetClientRect(target, &mut rect);
                let mut point = POINT { x: 12, y: 12 }; ClientToScreen(target, &mut point);
                let bounds = (point.x, point.y, 720.min((rect.right - 24).max(240)), (lines * 18 + 70).min((rect.bottom - 24).max(120)));
                if placement != Some(bounds) {
                    SetWindowPos(hwnd, HWND_TOPMOST, bounds.0, bounds.1, bounds.2, bounds.3, SWP_NOACTIVATE | SWP_SHOWWINDOW);
                    placement = Some(bounds);
                }
                InvalidateRect(hwnd, std::ptr::null(), 0);
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        DestroyWindow(hwnd);
        DeleteObject(font as _);
        UnregisterClassW(class.as_ptr(), instance);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn busy_ui_drops_a_snapshot_instead_of_blocking_the_game() {
        let state = Mutex::new(State {
            target: 1, text: "old".into(), history: Vec::new(),
            updated: Instant::now(), paint: String::new(),
        });
        let guard = state.lock().unwrap();
        assert!(!update(&state, 2, "new".into(), &History::default()));
        assert_eq!(guard.text, "old");
        drop(guard);
        assert!(update(&state, 2, "new".into(), &History::default()));
        assert!(state.lock().unwrap().text.starts_with("new"));
    }
    #[test]
    fn history_is_bounded_and_rejects_invalid_samples() {
        let mut h = History::default();
        for t in [0.0, -1.0, f64::NAN, f64::INFINITY] { h.push(t); }
        assert!(h.0.is_empty());
        for i in 1..=300 { h.push(i as f64); }
        assert_eq!(h.0.len(), 240);
        assert_eq!(h.0[0], 61.0);
    }
    #[test]
    fn slow_frame_and_percentile_statistics_are_defined() {
        let mut h = History::default();
        for _ in 0..99 { h.push(10.0); } h.push(100.0);
        let s = h.summary();
        assert!(s.contains("P50 10.0"));
        assert!(s.contains("P99 10.0"));
        assert!(s.contains("1% low 10.00"));
    }
}
