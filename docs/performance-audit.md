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
