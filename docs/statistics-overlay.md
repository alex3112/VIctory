# Statistics overlay

Available in release and diagnostics builds. Visible by default while the game
has keyboard focus. F3 hides/shows it; F4 switches performance/memory pages.
It ignores clicks and never takes keyboard focus. Minimizing or switching to
another application hides it. It is a separate Windows overlay, not pixels
inserted into the game's render targets; some window-capture software may
therefore omit it. Exclusive fullscreen capture is not guaranteed.

Performance page:

- FPS, frame time, draws/dispatches, and a frame-time trend graph.
- Last 240 presented frames: minimum, maximum, P50/P95/P99, 1% low (inverse
  mean of the slowest 1%, rounded up), and counts over 33.34/50/100 ms.
- CPU submission preparation, GPU submission/wait, buffer refresh, texture
  upload and copy-back times/volumes, and bindless preparation costs.
- Missing-pipeline draws, dropped indirect work, compilation backlog,
  pipelines built/loaded and shader-cache hit/translation session counts.
- Actual adapter name and diagnostic switches, so WARP/validation are visible.
- Diagnostics builds additionally show submissions, refresh requests,
  avoided checks, guard faults, snapshot cost and targeted invalidations.

Both pages show process CPU (normalized across available logical CPUs),
resident/peak RAM, private commit, system memory/commit availability, process
I/O rates, page-fault deltas and the adapter's local/shared memory budgets.
The memory page also lists the runtime's resource categories and allocation
counts, staging memory and pooled snapshots. Allocations are not equivalent
to GPU residency; GPU wait is not a pure GPU timestamp or utilization reading.
Process I/O includes cache and file operations, not only disk traffic.

Game counters are published once per second of presentation. The overlay
paints at most 5 times per second and polls system counters at most once per
second, only when visible. The sample age remains visible during stalls.
No GPU debug layer, validation, timing serialization, capture or readbacks
are enabled by the HUD. It reads existing counters without consuming logs.
