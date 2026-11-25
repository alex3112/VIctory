//! Memory for host textures only read by shaders: slots in shared heaps.
//! A texture of its own in a committed resource is a driver allocation of
//! 64 KiB at least, made and freed through the kernel; Grand Theft Auto VI
//! streams in tens of thousands, many of some bytes.

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use windows::Win32::Graphics::Direct3D12::*;

/// Textures larger than this have memory of their own.
const LARGEST: u64 = 8 << 20;

struct Slab {
    /// `None` once no slot was used.
    memory: Option<Memory>,
    free: Vec<u32>,
    slots: u32,
    bytes: u64,
}

#[derive(Clone)]
enum Memory {
    Heap(ID3D12Heap),
    /// An upload buffer and where the CPU writes it.
    Upload(ID3D12Resource, usize),
}

// SAFETY: D3D12 heaps and resources are free-threaded; the CPU pointer is
// only metadata for the owning slab and is never dereferenced on drop.
unsafe impl Send for Memory {}

/// Slot size, and whether for staging, to slabs of such slots.
type Slabs = BTreeMap<(u64, bool), Vec<Slab>>;
static SLABS: Mutex<Slabs> = Mutex::new(BTreeMap::new());
/// This is an idle-cache budget, never a cap on live textures or allocations.
const IDLE_BUDGET: u64 = 64 << 20;
static IDLE_BYTES: AtomicU64 = AtomicU64::new(0);
static SLABS_MADE: AtomicU64 = AtomicU64::new(0);
static SLABS_REUSED: AtomicU64 = AtomicU64::new(0);
static SLABS_FREED: AtomicU64 = AtomicU64::new(0);

pub(crate) fn audit_state() -> String {
    format!("pool_idle_mib={} pool_slabs_made={} pool_slabs_reused={} pool_slabs_freed={}",
        IDLE_BYTES.load(Relaxed) >> 20, SLABS_MADE.load(Relaxed),
        SLABS_REUSED.load(Relaxed), SLABS_FREED.load(Relaxed))
}

/// A texture's memory, free again when dropped.
pub(crate) struct Slot {
    size: (u64, bool),
    heap: usize,
    index: u32,
}

impl Drop for Slot {
    fn drop(&mut self) {
        let retired = {
            let mut slabs = SLABS.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(of) = slabs.get_mut(&self.size) {
                let another_idle = of.iter().enumerate().any(|(i,s)|
                    i != self.heap && s.memory.is_some() && s.free.len() == s.slots as usize);
                if let Some(s) = of.get_mut(self.heap) {
                    s.free.push(self.index);
                    if s.free.len() == s.slots as usize && s.memory.is_some() {
                        if !another_idle && IDLE_BYTES.load(Relaxed).saturating_add(s.bytes) <= IDLE_BUDGET {
                            IDLE_BYTES.fetch_add(s.bytes, Relaxed);
                            None
                        } else {
                            SLABS_FREED.fetch_add(1, Relaxed);
                            s.memory.take()
                        }
                    } else {
                        None
                    }
                } else { None }
            } else {
                None
            }
        };
        // A slot reaches Drop only after its resource owner is retired
        // (or creation failed). Empty slabs need no second grace period.
        drop(retired);
    }
}

/// The slot size for `bytes`: a multiple of 4 KiB, or of a quarter of its
/// power of two.
fn class(bytes: u64) -> u64 {
    if bytes <= 64 << 10 {
        return bytes.next_multiple_of(4 << 10);
    }
    let step = ((bytes.next_power_of_two() / 2) / 4).max(64 << 10);
    bytes.next_multiple_of(step)
}

/// A free slot and its slab's memory, which `make` makes more of.
fn slot(size: (u64, bool), make: impl FnOnce(u64) -> Option<Memory>) -> Option<(Slot, Memory)> {
    let mut slabs = SLABS.lock().unwrap_or_else(|e| e.into_inner());
    let of = slabs.entry(size).or_default();
    let used = of.iter().position(|s| s.memory.is_some() && !s.free.is_empty() && s.free.len() < s.slots as usize)
        .or_else(|| of.iter().position(|s| s.memory.is_some() && !s.free.is_empty()));
    let at = match used.or_else(|| of.iter().position(|s| s.memory.is_none())) {
        Some(at) => at,
        None => {
            // Start small for rare size classes. Grow only after existing
            // slabs of this class are full, reducing kernel heap creation.
            let (count, min, max) = if !size.1 && of.is_empty() {
                (2, 256 << 10, 16 << 20)
            } else if !size.1 && of.len() >= 2 {
                (32, 1 << 20, 32 << 20)
            } else {
                (8, 1 << 20, 16 << 20)
            };
            let bytes = (size.0 * count).clamp(min, max).next_multiple_of(64 << 10);
            let slots = (bytes / size.0) as u32;
            of.push(Slab {
                memory: None,
                free: (0..slots).rev().collect(),
                slots,
                bytes: (u64::from(slots) * size.0).next_multiple_of(64 << 10),
            });
            of.len() - 1
        }
    };
    let s = &mut of[at];
    if s.memory.is_none() {
        s.memory = Some(make(s.bytes)?);
        SLABS_MADE.fetch_add(1, Relaxed);
    } else if s.free.len() == s.slots as usize {
        IDLE_BYTES.fetch_sub(s.bytes, Relaxed);
        SLABS_REUSED.fetch_add(1, Relaxed);
    }
    let index = s.free.pop()?;
    debug_assert!(index < s.slots);
    Some((Slot { size, heap: at, index }, s.memory.clone()?))
}

/// Upload memory for `bytes`: a buffer, where the CPU writes it from its
/// start, the offset in it; `None` if it is to have a buffer of its own.
pub(crate) fn staging(
    host: &crate::host::Host,
    bytes: u64,
) -> Option<(ID3D12Resource, *mut u8, u64, Slot)> {
    if cfg!(test) || bytes > LARGEST {
        return None;
    }
    let size = class(bytes.max(1));
    let (slot, memory) = slot((size, true), |bytes| {
        let (buffer, data) = host.upload_buffer(bytes)?;
        Some(Memory::Upload(buffer, data as usize))
    })?;
    let Memory::Upload(buffer, data) = memory else {
        return None;
    };
    let offset = u64::from(slot.index) * size;
    Some((buffer, data as *mut u8, offset, slot))
}

/// A texture as `desc` says in `state`, in a slot; `None` if it is to have
/// memory of its own.
pub(crate) fn place(
    device: &ID3D12Device,
    desc: &D3D12_RESOURCE_DESC,
    state: D3D12_RESOURCE_STATES,
) -> Option<(ID3D12Resource, Slot)> {
    if cfg!(test) || desc.Flags.0 != 0 || desc.SampleDesc.Count > 1 {
        return None;
    }
    let mut desc = *desc;
    desc.Alignment = 0;
    // Query the legal default first. Only a single-slice 2D texture fitting
    // in one default allocation is a candidate for small placement. Large
    // streamed textures must never issue the rejected 4 KiB probe.
    // SAFETY: a whole description.
    let mut info = unsafe { device.GetResourceAllocationInfo(0, &[desc]) };
    if info.SizeInBytes <= 64 << 10
        && desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D
        && desc.DepthOrArraySize == 1
        && desc.Layout == D3D12_TEXTURE_LAYOUT_UNKNOWN
    {
        let mut small = desc;
        small.Alignment = 4 << 10;
        // SAFETY: a whole description; fall back to the valid default if refused.
        let candidate = unsafe { device.GetResourceAllocationInfo(0, &[small]) };
        if candidate.Alignment == 4 << 10 && candidate.SizeInBytes != u64::MAX {
            desc = small;
            info = candidate;
        }
    }
    if info.SizeInBytes == 0 || info.SizeInBytes > LARGEST || info.Alignment > 64 << 10 {
        return None;
    }
    let size = class(info.SizeInBytes).next_multiple_of(info.Alignment.max(1));
    let (slot, memory) = slot((size, false), |bytes| {
        let heap = D3D12_HEAP_DESC {
            SizeInBytes: bytes,
            Properties: D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                ..Default::default()
            },
            Alignment: 64 << 10,
            Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES,
        };
        let mut made: Option<ID3D12Heap> = None;
        // SAFETY: a whole description.
        unsafe { device.CreateHeap(&heap, &mut made) }.ok()?;
        let heap = made?;
        crate::memstat::track(&heap, 4, bytes);
        Some(Memory::Heap(heap))
    })?;
    let Memory::Heap(heap) = memory else {
        return None;
    };
    let mut texture: Option<ID3D12Resource> = None;
    // SAFETY: the slot is inside the heap and no other texture's.
    unsafe {
        device.CreatePlacedResource(
            &heap,
            u64::from(slot.index) * size,
            &desc,
            state,
            None,
            &mut texture,
        )
    }
    .ok()?;
    Some((texture?, slot))
}
