# Performance audit (v1)

Opt-in telemetry; no changes to rendering, resource limits, shader
translation or submission policy. Build diagnostics for drop reasons
and detailed copy counters.

~~~powershell
cargo build --profile diag -p victory --features diagnostics
$env:VICTORY_AUDIT = "1"
Remove-Item Env:VICTORY_GPU_TIMING -ErrorAction SilentlyContinue
Remove-Item Env:VICTORY_D3D12_DRED -ErrorAction SilentlyContinue
Remove-Item Env:VICTORY_D3D12_DEBUG -ErrorAction SilentlyContinue
Remove-Item Env:VICTORY_D3D12_GBV -ErrorAction SilentlyContinue
target\diag\victory.exe run <game executable> --execute --console series-x
~~~

Keep your usual game options and console profile. Warm the same build's
shader/pipeline cache first. Record 30 seconds in the menu and about two
minutes in the world, both static and moving.

The report is performance_audit.log in the configured data directory
(default %LOCALAPPDATA%\victory\data). Each run replaces the previous file:
rename/copy it before starting another run. A bounded background writer
flushes records; abrupt exit can lose queued records. Records dropped under
I/O pressure are counted on the next successful write.

Roughly one-second windows report presented-frame interval statistics,
existing CPU preparation/copy/texture/copy-back/bindless metrics, CPU fence
and submission-lock waits, pipeline build/caller waits, present/pacing
time, indirect clamp calls and maximum requested records, pipeline/cache/
backlog counters, process working set/private commit, tracked GPU allocations
and driver local/non-local memory usage. Diagnostics include detailed copy
summary and drop reasons with shader identifiers when available.

Interpretation:
- GPU submit+wait is wall time, not GPU execution or idle. Fence waits
  may overlap with other timings.
- Copy/texture/bindless plan times are part of preparation; worker and caller
  times may overlap. Do not add them into a frame budget.
- Clamp calls are recorded invocations, not the number of modified arguments.
- Draw/dispatch/indirect counters are existing recording counters; drop
  reasons can overlap and are not a complete execution census.
- RAM counters describe the whole process. Tracked GPU allocations are
  not additive with driver totals; non-local usage does not prove paging.
- P95/P99 are nearest-rank values for each window, not session percentiles.

## Separate detailed GPU run

With the same diagnostics build and audit enabled, set:

~~~powershell
$env:VICTORY_GPU_TIMING = "1"
$env:VICTORY_D3D12_DRED = "1"
~~~

Existing GPU timestamps are mirrored into the report. The detailed mode
adds timestamps/markers and waits for individual runs, so its FPS is not
directly comparable with standard audit FPS. Timestamped commands do not
cover the full GPU timeline.

Not yet measured: actual GPU idle, per-shader approximation usage, actual
clamp modifications, individual resource allocation/free rates, and full
title CPU profiling. Existing VICTORY_PROFILE=<file> can sample submission
thread stacks separately.

## Memory & Stability (audit v2)

Every second the audit writer now records `memory_sample`, including system RAM,
system commit/limit, process resident/private bytes, presented frame count,
tracked GPU allocation categories, retired object count and sparse root/alias
counts. These samples do not depend on presenting a frame or on a GPU wait.
Allocation failures are recorded by the HLE and key GPU allocation paths.
Counters may overlap; sparse alias counts are not physical memory usage.

Host texture creation is deduplicated per address shard; small placement probes
are restricted to small candidate descriptions. First texture slabs are smaller,
and read-only textures avoid a second retirement delay while keeping the final
GPU retirement fence/frame policy. Bindless preparation coalesces repeated work;
its existing time budget, urgent ordering and null-view safety remain in place.
The D3D12 pipeline library uses a new filename (v2), preserving old files and
compiler crash history. The first run rebuilds this library. No mip limit,
shader policy, GPU timeout or experimental sparse-tile sweep default is changed.

Windows validation and gameplay measurements are still required.

## Resource Lifetime & Queue Synchronization (audit v3)

Windows production builds use a dedicated title queue worker by default. Each
ready queue advances one operation per round; fence signals and page/present
work remain ordered after preceding submissions and copy-back. Host D3D12
submission remains serialized. `VICTORY_QUEUE_WORKER=0` restores inline
execution. The worker does not bypass unmet waits by default; the old timeout
fallback can be explicitly requested with `VICTORY_FORCE_WAITS=1` for diagnosis.
The audit reports pending operations, busy and held queues, completed steps
and forced waits independently of presentation. The queue can still be stalled
by expensive preparation of a single submission or a genuinely missing write.

Aligned memory waits use acquire loads, and already-published early writes
are no longer treated as hypothetical values. Bindless plans are refreshed
per heap on a maintenance epoch that advances during loading, not just when
an image is presented. Shader write-range deduplication and copy-back grouping
use hashed keys instead of repeated linear scans.

Completed, unreferenced closed lists are trimmed from the spare pool.
Upload-only staging (explicitly Arc-owned by lists and submissions) is retired
after observed host fence completion, independently of presentation. Generic
texture/descriptor retirement retains its existing safety grace period; no
unsafe eviction of live descriptors or sparse memory is enabled. Windows build
and title runtime validation are required. No game executable patch, watchdog
bypass, shader disablement or quality changes are introduced.

## GPU Resource Lifecycle (audit v4)

The v3 dedicated queue worker increased producer run-ahead during a stalled
host submission. Inline pumping is restored as the default, keeping the
round-robin scheduler and strict dependency handling. Ready work is executed
on the caller; unmet waits return so that other queues can provide the
signal. The dedicated worker is experimental (`VICTORY_QUEUE_WORKER=1`).
Unmet waits are not bypassed unless explicitly enabled for diagnostics.
This is natural backpressure, not a hard live-resource limit; independent
recording threads may still allocate while a host fence is stalled.

Split lists retain the original recording's dynamic descriptor tables and
uploads until every pending part is done. Completed recordings are eligible
for reuse on initial list creation as well as reset; recycle checks also
require deferred fence completion. Released dynamic slots drop their retained
owners instead of holding them until the slot is overwritten. Duplicate
texture ownership changes no longer retire identical resources again.
The first-upload queue coalesces entries per host texture.

Texture heaps grow per populated size class and reuse empty slabs. At most
64 MiB of wholly idle slabs are cached, with one per class. This is solely a
cache bound; live allocation is not capped. Empty heaps whose final slot
has already been retired no longer receive a second frame grace period.
Generic live descriptor and texture retirement protection remains intact.

The independent audit reports live list/heap counts, slab allocation/reuse,
and active submission stage, list index, fence, elapsed stage time and up
to 32 pipeline identifiers for the currently issued list. A fence at the end
of a batch also covers preceding lists; the last listed pipeline is not
necessarily the one that stalled the device. Windows testing is required.

Descriptor preparation also checks whether the slot generation changed while
work was done outside the tracking lock; stale results are rescheduled instead
of removing a new slot. Placed textures under construction revalidate their
backing memory when a remap intervenes. Creation/remap/rejection/eviction
counters are reported separately. No LRU eviction of still-mapped textures
is introduced: safe eviction requires a complete ownership audit of raw
resource pointers and bindless references. The audit records command-list
allocation failures and reports device removal rather than masking it as OOM.
