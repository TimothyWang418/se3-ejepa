# step83 — R²(N) crossover thickening: seeds 3 -> 10 per (N, arch) cell (frozen before any new seed runs)

Date: 2026-06-11 (night). Statistics thickening of the paper's FIRST pillar (E2 spectrum faithfulness):
same protocol byte-identical (NS=12,20,28,40; archs conv,mlp,gru; same training recipe/epochs), seeds 3-9
added to the published 0-2. No gate, no tuning. The crossover claim (conv holds R²≈0.98 at N=40 while
mlp/gru collapse R²<0, transition N≈28→40) is folded HONESTLY whatever the 10-seed medians/ranges say —
including if any conv seed degrades or any dense seed survives. Runs on the backup MacBook (CPU lane,
Intel; REUSE_N40=0 since step74 artifacts are not on that machine). Artifact merged Mac-side with the
canonical 3-seed JSON (lists append; medians recomputed); n=3 canonical preserved.
