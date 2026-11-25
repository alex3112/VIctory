# Belobug AMD integration

This merge combines `007a338` (our upstream-based runtime and local memory
optimizations) with Belobug AMD `4cbe9d7`. Both histories remain parents of
the merge, rather than replacing our branch with the fork's tree.

Conflict resolutions:

- Use Belobug's shader compiler and GPU submission, sparse-memory, VRAM,
  descriptor and indirect-clamp implementations as a coherent subsystem.
- Port our segmented snapshots to ordinary host buffers. Existing blocks
  are not copied again when a declared write range grows. Belobug's sparse
  tile snapshots and VRAM synchronization retain their own paths.
- Retain all refreshed intervals within a submission and refresh only holes.
  Unknown writes invalidate all coverage. Known copies retain unrelated
  coverage only when Windows verifies that the entire destination range is
  private memory; mapped/unknown destinations conservatively invalidate all.
- Keep our physical-page mapping implementation and alias notifications,
  connecting them to the fork's guards and streaming notifications.
- Keep the PC FidelityFX bridge and invalidate host binding state after a
  host library records commands.
- Retain `shader-info --effects` and `translate --cache --pipelines`, adapting
  the latter to Belobug's host-wave-specific shader cache and live pipeline
  library. Pipeline warm-up now revisits records and reuses the driver's
  library instead of skipping records through the former prebuilt-key index.
- Split `emit/lower.rs` into two textual source units to meet the existing
  asset-policy size limit without raising it or changing shader logic.

The local `victory.toml` still selects `E:\GTA6` and the existing external
cache/data directories. The shader compiler changed, so earlier translations
are not assumed compatible. Release and diagnostics executables use distinct
shader-cache build keys in this fork.

`perf/before-belobug` preserves the version from before this integration.
Switching source branches requires recompiling before using the launchers.
No game-frame performance improvement is established by compilation or unit
tests; compare the same scene after warming the combined build's cache.

Validation: two AMD shader execution tests fail both in this integration and
in a separate checkout of unmodified Belobug `4cbe9d7`:

- `emit::valu_tests::scalar_counted_in_a_divergent_loop_read_after_it`
- `emit::valu_tests::scalar_counted_in_nested_divergent_loops`

The rest of the workspace is validated with those two names excluded from
the test invocation; the tests themselves remain enabled in the source.


## 2026-11-25: synchronization to Belobug AMD 01aadd1

Source: https://github.com/Belobug/VIctory-Belobug-AMD/compare/4cbe9d7...01aadd1
(20 upstream commits, GPL-3.0-or-later; original authors and commit history
are recorded at the source link). The source commit is not present as a Git
object in this independent repository, so this is a consolidated integration,
not a two-parent merge. Existing local history is retained.

- Import integer render-target blend handling, ambient tint/depth shading,
  descriptor free-slot search changes and timeline/queue async coordination.
- Integrate all upstream changes to sparse snapshots, tile copy-back and
  experimental sweep; retain local segmented snapshots for ordinary buffers.
- Use the upstream view guard registry, including kernel-write pins and
  robust per-view unprotection, from our physical mapping implementation.
  The local physical-page registry remains for HLE writes and the local
  private-range check remains for refresh-coverage retention.
- Preserve local FidelityFX bridge, overlay, prebuild tools and audit hooks
  while applying upstream submission lifecycle and deferred completion.
- Async submission and experimental tile sweep retain upstream opt-in defaults.
- Update README reference status from Belobug; those results are not local
  benchmarks.

Validation: source-level integration and Git diff review only; Windows
compilation, tests and same-scene benchmark are still required.
