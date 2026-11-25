//! Which title memory the title wrote since a host copy took it (decision
//! 0007, "Copies of title memory"). The runtime reserves title GPU memory
//! with write watching (`victory-hle` xmem), so the kernel keeps a dirty bit
//! per 4 KiB page; asking for a range's written pages also clears them. Host
//! copies of the same memory can overlap (nested buffers, textures in
//! buffers), so what one asks is kept here for the others: a journal of
//! each page's last write, numbered by the asking.
//!
//! Physical-page views (xmem's sections) cannot be write-watched. There a
//! copy leaves each 64 KiB block it took read-only, and our vectored handler
//! takes the first write to it: it notes the block as written and gives the
//! write access back. The runtime tells us when it maps or unmaps views
//! ([`title_memory_remapped`]), which resets their protection.
//!
//! shortcut: writes into physical pages through another view (xmem's
//! `write_physical`, DirectStorage's page destinations) are not seen at
//! the guarded addresses; they go to the texture pool, whose textures are
//! hashed, not guarded. A system call writing into a guarded block fails
//! (none is known to).

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

/// Per-64-MiB-region epoch table. Incremented for only the region(s) touched
/// whenever a guarded block is opened (VEH guard fault or `writing`),
/// remapped (`title_memory_remapped`), or rewritten (`title_memory_written`).
/// While a span's region epoch stays unchanged, any range already guarded and
/// copied at that epoch cannot have been modified by the title — and
/// `copy_back` on one buffer no longer invalidates clean spans in other regions.
const REGION_SHIFT: usize = 26; // 64 MiB per bucket
const REGION_BUCKETS: usize = 1024;
static REGION_EPOCHS: [AtomicU64; REGION_BUCKETS] =
    [const { AtomicU64::new(1) }; REGION_BUCKETS];

#[inline]
fn bump_region_epochs(start: usize, end: usize) {
    if start >= end {
        return;
    }
    let first = start >> REGION_SHIFT;
    let last = (end - 1) >> REGION_SHIFT;
    let count = last - first + 1;
    if count >= REGION_BUCKETS {
        for slot in &REGION_EPOCHS {
            slot.fetch_add(1, Ordering::Release);
        }
    } else {
        for r in first..=last {
            REGION_EPOCHS[r & (REGION_BUCKETS - 1)].fetch_add(1, Ordering::Release);
        }
    }
}

#[inline]
pub(crate) fn region_epoch(start: usize, end: usize) -> u64 {
    if start >= end {
        return 0;
    }
    let first = start >> REGION_SHIFT;
    let last = (end - 1) >> REGION_SHIFT;
    let count = last - first + 1;
    let mut hash = 0u64;
    if count >= REGION_BUCKETS {
        for slot in &REGION_EPOCHS {
            hash = hash
                .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                .wrapping_add(slot.load(Ordering::Acquire));
        }
    } else {
        for r in first..=last {
            hash = hash
                .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                .wrapping_add(REGION_EPOCHS[r & (REGION_BUCKETS - 1)].load(Ordering::Acquire));
        }
    }
    hash
}

/// The unit tracked and copied: 64 KiB, the console's page (the kernel's
/// 4 KiB pages fold into it). Finer units made the checks of the world's
/// big buffers cost more than the copies they saved.
pub(crate) const PAGE: usize = 64 << 10;
/// The kernel's page, the unit of its answers.
const KERNEL_PAGE: usize = 4096;

struct Journal {
    /// Numbers the askings; 0 is "never".
    asked: u64,
    /// Page number → the asking that last found it written.
    written: BTreeMap<usize, u64>,
    /// Room for the kernel's answer.
    found: Vec<usize>,
}

static JOURNAL: Mutex<Journal> = Mutex::new(Journal {
    asked: 0,
    written: BTreeMap::new(),
    found: Vec::new(),
});

/// The pages of `[start, end)` (whole pages) as last written, for pages
/// written since tracking began: `(page number, asking)` in page order, and
/// this asking's number, which is higher than any in the list but those the
/// kernel just reported. `None` if the memory is not write-watched (nor,
/// with `guarded`, view memory, which is then guarded).
pub(crate) fn writes(start: usize, end: usize, guarded: bool) -> Option<(u64, Vec<(usize, u64)>)> {
    use windows_sys::Win32::System::Memory::GetWriteWatch;
    const WRITE_WATCH_FLAG_RESET: u32 = 1;
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    let j = &mut *j;
    let mut count = (last - first) * PAGE / KERNEL_PAGE;
    j.found.resize(count.max(j.found.len()), 0);
    let mut granularity = 0;
    #[cfg(feature = "diagnostics")]
    let started = std::time::Instant::now();
    // SAFETY: the answer buffer holds `count` entries.
    let failed = unsafe {
        GetWriteWatch(
            WRITE_WATCH_FLAG_RESET,
            (first * PAGE) as *const _,
            (last - first) * PAGE,
            j.found.as_mut_ptr().cast(),
            &mut count,
            &mut granularity,
        )
    } != 0;
    let guarded = if failed {
        count = 0;
        if !guarded {
            return None;
        }
        guard(first * PAGE, last * PAGE)?
    } else {
        Vec::new()
    };
    #[cfg(feature = "diagnostics")]
    {
        use crate::copystat::{C, add};
        add(C::Watches, 1);
        add(C::WatchUs, started.elapsed().as_micros() as u64);
    }
    j.asked += 1;
    let now = j.asked;
    let pages = j.found[..count].iter().map(|&at| at / PAGE).chain(guarded);
    for page in pages.collect::<Vec<_>>() {
        j.written.insert(page, now);
    }
    Some((
        now,
        j.written
            .range(first..last)
            .map(|(&p, &w)| (p, w))
            .collect(),
    ))
}

/// Same as [`writes`] for memory already known to be guarded `MEM_MAPPED`
/// view memory: skips the `GetWriteWatch` syscall (which always fails on
/// mapped views).
pub(crate) fn writes_guarded(start: usize, end: usize) -> Option<(u64, Vec<(usize, u64)>)> {
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    #[cfg(feature = "diagnostics")]
    let started = std::time::Instant::now();
    let guarded = guard(first * PAGE, last * PAGE)?;
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    let j = &mut *j;
    #[cfg(feature = "diagnostics")]
    {
        use crate::copystat::{C, add};
        add(C::Watches, 1);
        add(C::WatchUs, started.elapsed().as_micros() as u64);
    }
    j.asked += 1;
    let now = j.asked;
    for page in guarded {
        j.written.insert(page, now);
    }
    Some((
        now,
        j.written
            .range(first..last)
            .map(|(&p, &w)| (p, w))
            .collect(),
    ))
}

/// Guarded blocks of view memory.
const BLOCK: usize = PAGE;

struct Guards {
    aliases: BTreeMap<usize, Vec<usize>>,
    /// Blocks we made read-only.
    protected: BTreeSet<usize>,
    /// Of those, the ones not written since a copy took them.
    clean: BTreeSet<usize>,
    /// Unmapped blocks (`VirtualProtect` failed) within guarded ranges;
    /// cleared by [`title_memory_remapped`] when the runtime maps them.
    holes: BTreeSet<usize>,
}

static GUARDS: Mutex<Guards> = Mutex::new(Guards {
    aliases: BTreeMap::new(),
    protected: BTreeSet::new(),
    clean: BTreeSet::new(),
    holes: BTreeSet::new(),
});

fn guards() -> std::sync::MutexGuard<'static, Guards> {
    GUARDS.lock().unwrap_or_else(|e| e.into_inner())
}

/// For view memory: the pages of `[start, end)`'s blocks written since
/// they were last guarded (all of a block's pages), guarding them again
/// before the caller copies them. `None` if the memory is not a view.
fn guard(start: usize, end: usize) -> Option<Vec<usize>> {
    let (first, last) = (start / BLOCK, end.div_ceil(BLOCK));
    // Fast path: if we already guard or recorded a hole at `first`, we already
    // know this is a mapped view and can skip `QueryVirtualMemoryInformation`.
    {
        let mut g = guards();
        if g.protected.contains(&first) || g.holes.contains(&first) {
            let mut written = Vec::new();
            let mut run: Option<usize> = None;
            for block in first..=last {
                let open = block < last && !g.clean.contains(&block) && !g.holes.contains(&block);
                if open {
                    if g.protected.contains(&block) {
                        g.clean.insert(block);
                        written.extend(block * BLOCK / PAGE..(block + 1) * BLOCK / PAGE);
                    } else {
                        run.get_or_insert(block);
                        continue;
                    }
                }
                if let Some(from) = run.take() {
                    protect(&mut g, from, block, &mut written);
                }
            }
            return Some(written);
        }
    }
    // Slow path (first encounter): `start` is itself a mapped view, or if
    // `start` is an unmapped hole (common in 2 GiB sparse shadows), jump past
    // leading holes via `VirtualQuery` to find the first committed page.
    let probe = if mapped(start) {
        start
    } else {
        first_mapped(start, end)?
    };
    let check_each = probe != start;
    install();
    let mut g = guards();
    let mut written = Vec::new();
    let mut run: Option<usize> = None;
    for block in first..=last {
        let open = block < last
            && !g.clean.contains(&block)
            && !g.holes.contains(&block)
            && (!check_each || mapped(block * BLOCK));
        if open {
            if g.protected.contains(&block) {
                // Written through the runtime, still read-only.
                g.clean.insert(block);
                written.extend(block * BLOCK / PAGE..(block + 1) * BLOCK / PAGE);
            } else {
                run.get_or_insert(block);
                continue;
            }
        }
        if let Some(from) = run.take() {
            protect(&mut g, from, block, &mut written);
        }
    }
    Some(written)
}

/// Scans `[start, end)` jumping over unmapped holes; returns the first
/// committed address if it is a mapped view, or `None` if `[start, end)` has
/// no committed mapped view (or starts with committed non-view memory).
fn first_mapped(start: usize, end: usize) -> Option<usize> {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_MAPPED, MEMORY_BASIC_INFORMATION, VirtualQuery,
    };
    let mut at = start;
    while at < end {
        let mut info = MEMORY_BASIC_INFORMATION::default();
        // SAFETY: queries our own address space without dereferencing `at`.
        let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
        if n == 0 || info.RegionSize == 0 {
            return None;
        }
        if info.State == MEM_COMMIT {
            return (info.Type == MEM_MAPPED).then_some(at);
        }
        let next = (info.BaseAddress as usize).saturating_add(info.RegionSize);
        if next <= at {
            return None;
        }
        at = next;
    }
    None
}

/// Whether the current allocation at `at` is a mapped view. Asking only for
/// the allocation type avoids enumerating pages with identical protection,
/// which can span an entire large view after its blocks were guarded.
pub(crate) fn private_range(at: usize, len: usize) -> bool {
    use windows_sys::Win32::System::Memory::{MEM_PRIVATE, MEMORY_BASIC_INFORMATION, VirtualQuery};
    let Some(end) = at.checked_add(len) else { return false; };
    let mut info = MEMORY_BASIC_INFORMATION::default();
    // SAFETY: query only; no memory at `at` is dereferenced.
    unsafe {
        VirtualQuery(at as *const _, &mut info, size_of_val(&info)) != 0
            && info.Type == MEM_PRIVATE
            && (info.BaseAddress as usize).saturating_add(info.RegionSize) >= end
    }
}

fn mapped(at: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_MAPPED, MEMORY_BASIC_INFORMATION, MemoryRegionInfo, QueryVirtualMemoryInformation,
        VirtualQuery, WIN32_MEMORY_REGION_INFORMATION,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;

    let mut region = WIN32_MEMORY_REGION_INFORMATION::default();
    // SAFETY: both queries describe an address without dereferencing it. The
    // output buffers match their information classes.
    let queried = unsafe {
        QueryVirtualMemoryInformation(
            GetCurrentProcess(),
            at as *const _,
            MemoryRegionInfo,
            (&raw mut region).cast(),
            size_of_val(&region),
            std::ptr::null_mut(),
        )
    };
    if queried != 0 {
        // These are the Windows SDK's named bitfields in declaration order.
        // windows-sys exposes only the union's raw bits.
        const PRIVATE: u32 = 1 << 0;
        const MAPPED_DATA_FILE: u32 = 1 << 1;
        const MAPPED_IMAGE: u32 = 1 << 2;
        const MAPPED_PAGE_FILE: u32 = 1 << 3;
        const MAPPED_PHYSICAL: u32 = 1 << 4;
        // SAFETY: this union was filled by the successful query above.
        match unsafe { region.Anonymous.Flags } {
            MAPPED_DATA_FILE | MAPPED_PAGE_FILE | MAPPED_PHYSICAL => return true,
            PRIVATE | MAPPED_IMAGE => return false,
            _ => {}
        }
    }
    // Unknown flag combinations and failed allocation queries keep the
    // VirtualQuery type check, including for a free address.
    let mut info = MEMORY_BASIC_INFORMATION::default();
    // SAFETY: the output buffer has the required layout and size.
    let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    n != 0 && info.Type == MEM_MAPPED
}

/// Makes blocks `[from, to)` read-only and notes them guarded and clean;
/// block by block if they span views, recording unmapped blocks in `g.holes`
/// so subsequent checks neither retry `VirtualProtect` nor fault copying them.
fn protect(g: &mut Guards, from: usize, to: usize, written: &mut Vec<usize>) {
    let others: BTreeSet<usize> = g.aliases.range(from..to).flat_map(|(_, pages)| pages.iter().copied()).collect();
    protect_here(g, from, to, written);
    for page in others {
        if !g.protected.contains(&page) { protect_here(g, page, page + 1, written); }
    }
}

fn protect_here(g: &mut Guards, from: usize, to: usize, written: &mut Vec<usize>) {
    use windows_sys::Win32::System::Memory::{PAGE_READONLY, VirtualProtect};
    let set = |from: usize, to: usize| {
        let mut old = 0;
        // SAFETY: title view memory; only its protection changes.
        unsafe {
            VirtualProtect(
                (from * BLOCK) as *const _,
                (to - from) * BLOCK,
                PAGE_READONLY,
                &mut old,
            ) != 0
        }
    };
    let note = |g: &mut Guards, written: &mut Vec<usize>, from: usize, to: usize| {
        g.protected.extend(from..to);
        g.clean.extend(from..to);
        written.extend(from * BLOCK / PAGE..to * BLOCK / PAGE);
    };
    if set(from, to) {
        note(g, written, from, to);
        return;
    }
    for block in from..to {
        if set(block, block + 1) {
            note(g, written, block, block + 1);
        } else {
            g.holes.insert(block);
        }
    }
}

/// Gives write access back to the guarded blocks of `[start, end)` and
/// notes them written; true if any was guarded.
fn open(g: &mut Guards, start: usize, end: usize) -> bool {
    let others: BTreeSet<usize> = g.aliases.range(start / BLOCK..end.div_ceil(BLOCK)).flat_map(|(_, pages)| pages.iter().copied()).collect();
    let mut opened = open_here(g, start, end);
    for page in others { opened |= open_here(g, page * BLOCK, (page + 1) * BLOCK); }
    opened
}

fn open_here(g: &mut Guards, start: usize, end: usize) -> bool {
    use windows_sys::Win32::System::Memory::{PAGE_READWRITE, VirtualProtect};
    let blocks: Vec<usize> = g
        .protected
        .range(start / BLOCK..end.div_ceil(BLOCK))
        .copied()
        .collect();
    if blocks.is_empty() {
        return false;
    }
    for &block in &blocks {
        g.protected.remove(&block);
        g.clean.remove(&block);
    }
    // Coalesce contiguous blocks into a single VirtualProtect call to minimize
    // kernel transitions and multi-core TLB shootdowns.
    let mut run_start = blocks[0];
    let mut run_end = run_start + 1;
    for &block in &blocks[1..] {
        if block == run_end {
            run_end = block + 1;
        } else {
            let mut old = 0;
            // SAFETY: blocks we made read-only.
            unsafe {
                VirtualProtect(
                    (run_start * BLOCK) as *const _,
                    (run_end - run_start) * BLOCK,
                    PAGE_READWRITE,
                    &mut old,
                );
            }
            run_start = block;
            run_end = block + 1;
        }
    }
    let mut old = 0;
    // SAFETY: blocks we made read-only.
    unsafe {
        VirtualProtect(
            (run_start * BLOCK) as *const _,
            (run_end - run_start) * BLOCK,
            PAGE_READWRITE,
            &mut old,
        );
    }
    bump_region_epochs(start, end);
    true
}

/// Before our own write into title memory `[at, at + len)` through the
/// kernel (which does not fault into our handler): opens its guarded blocks.
/// The runtime calls it (as `kernel_will_write`) before the title's kernel
/// reads into its own memory.
pub fn writing(at: usize, len: usize) {
    open(&mut guards(), at, at + len);
}

/// The runtime mapped or unmapped title memory `[at, at + len)`: what it
/// shows is new and read-write, so it is no longer guarded, and any placed
/// textures backed by that range are evicted.
pub fn title_memory_remapped(at: usize, len: usize) {
    if len == 0 {
        return;
    }
    let end = at.saturating_add(len);
    {
        let mut g = guards();
        let stale: Vec<_> = g.aliases.range(at / BLOCK..end.div_ceil(BLOCK)).map(|(&p, _)| p).collect();
        for page in stale {
            for other in g.aliases.remove(&page).unwrap_or_default() {
                if let Some(links) = g.aliases.get_mut(&other) { links.retain(|&p| p != page); }
            }
        }
        let blocks: Vec<usize> = g
            .protected
            .range(at / BLOCK..end.div_ceil(BLOCK))
            .copied()
            .collect();
        let holes: Vec<usize> = g
            .holes
            .range(at / BLOCK..end.div_ceil(BLOCK))
            .copied()
            .collect();
        let changed = !holes.is_empty() || !blocks.is_empty();
        for block in holes {
            g.holes.remove(&block);
        }
        for block in blocks {
            g.protected.remove(&block);
            g.clean.remove(&block);
        }
        if changed {
            bump_region_epochs(at, end);
        }
    }
    crate::shadow::remapped(at as u64, end as u64);
    crate::resource::evict_placed_textures(at as u64, end as u64);
    crate::descriptor::host::bindless_memory_changed(at as u64, end as u64);
}

/// The title committed memory `[at, at + len)` it had reserved.
pub fn title_memory_committed(at: usize, len: usize) {
    if len != 0 {
        crate::shadow::filled(at as u64, (at + len) as u64);
    }
}

/// Every view of a physical page participates in write tracking.
pub fn title_memory_aliased(at: usize, others: &[usize]) {
    let mut g = guards();
    let page = at / BLOCK;
    for &other in others {
        for (a, b) in [(page, other / BLOCK), (other / BLOCK, page)] {
            let links = g.aliases.entry(a).or_default();
            if !links.contains(&b) { links.push(b); }
        }
    }
    if others.iter().any(|&p| g.protected.contains(&(p / BLOCK))) {
        protect(&mut g, page, page + 1, &mut Vec::new());
    }
}

/// The runtime wrote new contents into title memory `[at, at + len)` (a
/// streamed file arrived): textures placed there are uploaded again at their
/// next use, not when their turn to be looked at comes.
pub fn title_memory_written(at: usize, len: usize) {
    if len != 0 {
        bump_region_epochs(at, at.saturating_add(len));
        // The runtime writes through a view of its own, which neither the
        // kernel's write watch nor our guards see: the pages are noted
        // written here, or host copies of buffers made before the file
        // arrived kept what the memory held then for good (meshes read as
        // zeros: walls, floors and bodies missing, others each launch).
        {
            let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
            j.asked += 1;
            let now = j.asked;
            for page in at / PAGE..(at + len).div_ceil(PAGE) {
                j.written.insert(page, now);
            }
        }
        crate::resource::textures_rewritten(at as u64, (at + len) as u64);
    }
}

/// Opens a guarded block at `at` (or returns `true` if another thread already
/// made it read-write). Called both by our own VEH handler and by
/// `host::safe_copy`'s VEH handler on write faults.
pub(crate) fn try_open_guarded(at: usize) -> bool {
    let mut g = guards();
    if open(&mut g, at, at + 1) {
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::GuardFaults, 1);
        return true;
    }
    writable(at)
}

/// Registers the handler that takes writes to guarded blocks, once, ahead
/// of the crash reporter's.
fn install() {
    use windows_sys::Win32::Foundation::{EXCEPTION_ACCESS_VIOLATION, NTSTATUS};
    use windows_sys::Win32::System::Diagnostics::Debug::{
        AddVectoredExceptionHandler, EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_CONTINUE_SEARCH,
        EXCEPTION_POINTERS,
    };
    unsafe extern "system" fn on_write(info: *mut EXCEPTION_POINTERS) -> i32 {
        // SAFETY: the system passes the exception being dispatched.
        let r = unsafe { &*(*info).ExceptionRecord };
        const WRITE: usize = 1;
        if r.ExceptionCode != EXCEPTION_ACCESS_VIOLATION as NTSTATUS
            || r.NumberParameters < 2
            || r.ExceptionInformation[0] != WRITE
        {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        let at = r.ExceptionInformation[1];
        if try_open_guarded(at) {
            EXCEPTION_CONTINUE_EXECUTION
        } else {
            EXCEPTION_CONTINUE_SEARCH
        }
    }
    static ONCE: std::sync::Once = std::sync::Once::new();
    // SAFETY: the handler only touches our guard state and protections.
    ONCE.call_once(|| unsafe {
        AddVectoredExceptionHandler(1, Some(on_write));
    });
}

/// Whether `at` is committed read-write memory now.
fn writable(at: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_READWRITE, VirtualQuery,
    };
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    n != 0 && info.State == MEM_COMMIT && info.Protect == PAGE_READWRITE
}

#[cfg(test)]
mod tests {
    #[test]
    fn aliased_view_write_invalidates_the_other_views_guard() {
        use super::*;
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::Memory::{CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, PAGE_READWRITE, UnmapViewOfFile};
        // SAFETY: a pagefile section, mapped twice into this process.
        unsafe {
            let section = CreateFileMappingW(-1isize as _, std::ptr::null(), PAGE_READWRITE, 0, PAGE as u32, std::ptr::null());
            assert!(!section.is_null());
            let a = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, PAGE);
            let b = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, PAGE);
            assert!(!a.Value.is_null() && !b.Value.is_null());
            let (pa, pb) = (a.Value as usize, b.Value as usize);
            title_memory_aliased(pb, &[pa]);
            let (before, _) = writes_guarded(pb, pb + PAGE).expect("mapped view");
            (a.Value as *mut u8).write_volatile(7);
            assert_eq!((b.Value as *const u8).read_volatile(), 7);
            let (_, writes) = writes_guarded(pb, pb + PAGE).expect("mapped view");
            assert!(writes.iter().any(|&(page, when)| page == pb / PAGE && when > before));
            title_memory_remapped(pa, PAGE);
            title_memory_remapped(pb, PAGE);
            UnmapViewOfFile(a);
            UnmapViewOfFile(b);
            CloseHandle(section);
        }
    }
    use super::*;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
    };
    const MEM_WRITE_WATCH: u32 = 0x20_0000;

    #[test]
    fn allocation_query_recognizes_file_views_and_rejects_nonviews() {
        use std::os::windows::io::AsRawHandle;
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_READ, MapViewOfFile, PAGE_READONLY, UnmapViewOfFile,
        };
        let file = tempfile::tempfile().unwrap();
        file.set_len(PAGE as u64).unwrap();
        // SAFETY: a valid, open file and a read-only mapping of its contents.
        let section = unsafe {
            CreateFileMappingW(
                file.as_raw_handle(),
                std::ptr::null(),
                PAGE_READONLY,
                0,
                0,
                std::ptr::null(),
            )
        };
        assert!(!section.is_null());
        // SAFETY: an open section containing at least one page.
        let view = unsafe { MapViewOfFile(section, FILE_MAP_READ, 0, 0, PAGE) };
        assert!(!view.Value.is_null());
        assert!(mapped(view.Value as usize + 13));
        // SAFETY: our successful mapping and its section handle.
        unsafe {
            UnmapViewOfFile(view);
            CloseHandle(section);
        }
        assert!(!mapped((&raw const file) as usize)); // private stack allocation
        assert!(!mapped(mapped as *const () as usize)); // executable image
        assert!(!mapped(0)); // no allocation to query
    }

    #[test]
    fn written_pages_are_reported_to_every_asker() {
        // SAFETY: a fresh write-watched allocation, freed at the end.
        let p = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                4 * PAGE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(p, 0);
        let (start, end) = (p, p + 4 * PAGE);
        // Windows may reuse an address previously written by another test.
        // This test owns a new allocation, not that allocation's journal.
        JOURNAL.lock().unwrap().written.retain(|&page, _| !(start / PAGE..end / PAGE).contains(&page));
        let (_, before) = writes(start, end, false).expect("watched");
        assert!(before.is_empty());
        // SAFETY: committed pages of ours.
        unsafe { ((p + PAGE + 8) as *mut u8).write(1) };
        let (now, after) = writes(start, end, false).expect("watched");
        assert_eq!(after, [(p / PAGE + 1, now)]);
        // Another asker after the kernel's bit was cleared still sees it.
        let (later, again) = writes(p + PAGE, p + 2 * PAGE, false).expect("watched");
        assert!(later > now);
        assert_eq!(again, [(p / PAGE + 1, now)]);
        let mut plain = vec![0u8; PAGE];
        assert!(
            writes(
                plain.as_mut_ptr() as usize,
                plain.as_ptr() as usize + PAGE,
                true
            )
            .is_none()
        );
        // SAFETY: our allocation.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
    }

    /// A kernel read into a guarded view block fails; opened first, it
    /// lands and the block counts as written.
    #[test]
    fn kernel_reads_into_guarded_blocks_after_opening() {
        use std::io::{Read, Seek, Write};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, UnmapViewOfFile,
        };
        // SAFETY: a pagefile section of two blocks, mapped read-write.
        let view = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 2 * BLOCK).Value as usize
        };
        assert_ne!(view, 0);
        assert!(guard(view, view + 2 * BLOCK).is_some()); // now read-only
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(b"from the disk").unwrap();
        // `ReadFile` inside: the kernel writes the view.
        let read = |file: &mut std::fs::File| {
            file.rewind().unwrap();
            // SAFETY: 13 bytes of our view, nothing else refers to them.
            let into = unsafe { std::slice::from_raw_parts_mut((view + 100) as *mut u8, 13) };
            file.read(into).is_ok_and(|n| n == 13)
        };
        assert!(!read(&mut file), "the kernel does not fault into our handler");
        writing(view + 100, 13);
        assert!(read(&mut file));
        // SAFETY: the bytes just read.
        assert_eq!(unsafe { std::slice::from_raw_parts((view + 100) as *const u8, 13) }, b"from the disk");
        assert_eq!(guard(view, view + 2 * BLOCK), Some((view / PAGE..view / PAGE + 1).collect()));
        // SAFETY: our view.
        unsafe { UnmapViewOfFile(windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS { Value: view as *mut _ }) };
        title_memory_remapped(view, 2 * BLOCK);
    }
}
