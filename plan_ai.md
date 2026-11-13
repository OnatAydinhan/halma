# check_halma — Part 2 (AI) Plan (agent handoff)

A Rust AI engine plays check_halma, driven from the **existing** Python Tkinter GUI
through a proven subprocess line-protocol. Self-play for weight tuning lives in Rust;
the weight *fit* lives in Python.

## HANDOFF (read this first)

- **What this is:** Part 2 of check_halma. Part 1 (engine + 2-local-player GUI + clocks)
  is DONE: `halma8.py` (engine, tkinter-free), `halma8_gui.py` (UI), `halma8_test.py`
  (36 tests), `dist/check_halma.exe`. This plan adds an AI under `ai/` (Rust crate
  `halma8_ai`) plus Python glue.
- **Where we are (updated as tasks land):**
  - **GUI (cross-plan, `plan_gui.md`) ✅ FINISHED 2026-11-05 (T-G5 closed).**
    The GUI now has a Players tab (1–2 humans / 1–2 AIs per seat, incl. A/A),
    per-seat model loading (`best_weights.json` / `c1_incumbent.json` / any
    `cand_w*.json` — wired via T-G0's `setoption` path) and a Kibitzer
    (best line + eval). Model loading is therefore **NOT open** — the T-G5
    matrix (`_t_g5_matrix.log`) proved loaded weights drive play (extreme
    progress-only model diverges from both shipped models).
    `dist/check_halma.exe` is rebuilt with `ai.exe` + both default models
    bundled.
  - **T-0 … T-10.5 ✅** (all done, all gates green). In one line each: T-0 scaffold ·
    T-1 board/types round-trip · T-2 `legal_steps`/`legal_jumps` · T-3 `apply_move`/
    `end_turn`/chain/win/`IllegalMove` · T-4 search frame · T-5 6-term weighted eval ·
    T-6 move ordering + refined TT + bench · T-6.5 perf pass (no pruning) · T-7
    line-protocol server + smoke · T-8 `halma_ai.py` driver + GUI toggle · T-9
    `selfplay.rs` self-play CSV · T-10 `train_weights.py` + `weights.json` · T-10.5
    depth 4→6 + `progress` training label.
  - **T-12 ✅ DONE & verified 2026-11-01.** `selfplay --workers K` (thread-scoped, per-
    game deterministic seed, K-independent byte-identical CSV) + `tools_tune.py --workers`
    passthrough + `ai/tests/parallel_determinism.rs`. Re-verified green today: `halma8_test`
    36 OK; `cargo test` green (incl. parallel_determinism); 3 cross-checks PASS;
    `tools_ai_smoke` PASS. `selfplay.exe` rebuilt after the source change.
  - **T-11 — FULL VERIFIED RUN COMPLETE, but the fit is DEGENERATE (awaiting user decision).**
    The full loop now **runs to completion cleanly** (exit 0) and produces `elo_curve.svg`
    + a 3-row `tune_log.txt` (depth 6, 100 games, 12 workers). **BUT every iteration lands
    at R=988.41 (S=0.483, acc 0.7101):** the linear logistic fit collapses to **`defense`-only**
    weights each iteration, so the Elo curve is a flat line 12 below the 1000 anchor. This
    FAILS T-11's own done-when "at least one Rᵢ differs from 1000 by > 50" (988.41 is only
    11.6 off). Root cause: the loop regenerates training data at a **fixed `--seed 0`** with
    deterministic self-play, so every "iteration" is byte-identical → no iteration-to-iteration
    improvement is possible by construction. This is a **research/fit outcome, not a coding
    failure**. Per the plan's "report and stop if degenerate" clause → **STALLED on a design
    decision** (see T-11 ⚠ STATUS + "Next action"). Objective gates 1–6 are green: 36 tests,
    cargo test, build, 3-row log + valid SVG, determinism (all Rᵢ identical to 2dp).
  - **T-13 ✅ DONE 2026-11-01 — verdict (A) int-round.** Read-only `diag_fit.py`
    on `selfplay.csv`: the float fit has real multi-term shape (defense 1.10,
    base_presence 0.48, threat -0.32, progress 0.11, spread -0.15) but
    `int(round)` flattens it to `{defense:1}` (round-to-0 = 5/6); held-out acc
    0.71 with balanced classes → the label is fine, **not (B) weak label**.
    **T-14 (f64 weights) is the confirmed fix; T-14b not required.** See T-13
    RESULT (table + verdict). Gates green: `halma8_test` 36 OK, no `ai/` change.
  - **T-14 ✅ DONE & verified 2026-11-01.** `Weights` fields `i32 → f64`;
    `load` parses `f64`; `side_score → f64`; `evaluate`/`evaluate_with`/
    `evaluate_pair` cast back to `i32` (bit-identical for any integer weight
    set). `train_weights.py::to_raw_weights` now emits `round(w_std/std, 4)`,
    drops the hard all-zero `collapsed` fallback (kept the `at_chance` guard).
    All five gates green (see T-14 RESULT). **The int-round→defense-only
    collapse that flattened T-11's curve is removed** — a fresh `train_weights`
    fit now writes real-valued (non-zero-across-terms) weights.
  - **T-15 ✅ DONE & verified 2026-11-01.** Decoupled-seed loop: `tools_tune.py`
    now trains at `seed_base+i` and measures the h2h at `seed_base+1000+i`, `--workers`
    default 12. `py -u tools_tune.py --iterations 3 --games 100 --depth 6 --elo-games 30
    --workers 12 --seed-base 1000 --out-dir _t15` (background) ran to **exit 0**:
    `tune_log.txt` header now has `train_seed,measure_seed`; every seed distinct (1001/
    1002/1003 train · 2001/2002/2003 measure); each `weights_v{i}` is a distinct real float
    set **≠ `weights_v0`** (T-14 collapse-to-defaults is gone). Curve:
    `1000 / 988.41 / 1000 / 1000` → `|Δ(R1−R2)|=11.59>2`, so the "same point N×" flat
    line is **broken**. Determinism re-run (same `--seed-base`) reproduces
    `weights_v1` byte-identical **and** `R=988.41`. **Caveat:** mean R still sits at the
    1000 anchor because the d6 h2h is *low-signal* — 29–30/30 games end `ongoing` (W+L≈0–1)
    so `S` is pinned near 0.5 and a single decisive game moves `S` by only ~0.017 (ΔR≈±10);
    the spread is h2h noise, not a stable strength signal, so T-11's ">50 off anchor"
    done-when is NOT yet met (data/measurement limit, not a pipeline bug). See T-15 RESULT.
  - **Route (user, 2026-11-01): fix-pipeline → T-16 budget-search → T-17 refined grid.**
    **T-13 ✅ → T-14 ✅ → T-15 ✅ → T-16 ✅ (production search ran to exit 0, ACCEPTED)
    → T-17 ✅ DONE (d4 proxy tier COMPLETE; T-17e d6-confirm CLOSED 2026-11-04 —
    no cell cleared the gate; **C1 = 448/360 recorded as WINNER → `best_weights.json`**).**
    T-16 = `tools_budget_search.py`: per-term multiplicative factors `f_i ∈ [0.25, 4]`
    on the Rust defaults → **proxy** (depth-4 / 15 games) scores all candidates →
    **confirm** top-3 at depth-6 / 50 games × 3 independent seeds; accept iff
    `mean−1000 > 50 AND std < 20 AND min > 1000`, else report best-found + stop.
    Every head-to-head is now **double-blind** (both move-orders combined) with
    **`--maxmove 480`** default, cancelling the first-mover P1 bias (measured
    2026-11-01: identical-weights cap-480 finished games ran **17 P1 / 7 P2**).
    Do not re-run the seed-0 tune loop (provably byte-identical → always R=988.41).
    **T-16 ✅ DONE 2026-11-02 — production search ran to exit 0 and is ACCEPTED;
    see T-16 RESULT below (best = rank-2 cand[3], mean_R@depth6 = 1285.94, +285.94;
    one selection bug found & fixed).**
    **T-17 ✅ DONE; T-17e d6-confirm CLOSED 2026-11-04 (8/9 cells complete — the
    background run died at cell 8, 448/720; user closed it on the 8-cell data): no
    cell clears the gate — best = 448/540 at 1020.89 (+20.89 over the C1 anchor,
    min 1013.90; below the mean>+50 gate) → **USER OVERRIDE 2026-11-04: WINNER =
    448/540 (progress 448 / base 540 / defense 6 / spread 4 / mobility 9 / threat
    130) → `best_weights.json`** (T-16's 1285.94 weights now reference-only;
    C1 self-match sanity floor = 965.11; anchor for the next search =
    `best_weights.json`).** Full d4 surface done (boundary-hunt + STEP 1c
    wide shape + STEP 1d 2D pin — all 2026-11-03, all exit 0). CONVERGED (locked in
    the T-17e grid, consistent across passes): threat 130 · defense 6 · spread 4 ·
    mobility 9. D4 RESULTS:
    STEP 1c (`_t17c/`), progress axis, d4/25 — INTERIOR PEAK @ **progress 448**
    (R 1798.25) then COLLAPSE past 896 (896→993, 1792→1007, 3584→1014); base
    axis, d4/25, at LOW progress (24) — plateau @ **base 720–1440** (R 1219.87).
    STEP 1d 2D pin (`_t17d/`), true 2D grid progress×base, d4/60 double-blind,
    4 terms locked, incumbent anchor (1285.94, R=1000 scale): R table —
    **448/360 = 1668.84 (top)** · 224/360 = 1636.43 · 224/720 = 1636.43 ·
    112/720 = 1584.96 · 112/360 = 1563.75 · 448/720 = 1544.69 · 672/360 = 1076.49 ·
    672/720 = 1064.42. FINDINGS: (1) **progress dominates**; (2) **672
    collapses** (W=72/L=46 vs incumbent at d4) — rule it out; (3) **base ×
    progress INTERACTION**: at high progress, base 360 beats 720 (448/360 vs
    448/720 = +124); at low progress, base 720–1440 wins (STEP 1c) — the
    incumbent (progress 14, base 91) sits in a different region of this surface.
    CAVEAT: top 6 cells are all 0-loss, separated by ≤5 draw games out of 120
    total → within d4 noise; d6 is the arbiter (T-16 already showed d4 #1 ≠ d6 #1).
    **NEW INCUMBENT (user, 2026-11-03) = C1 = progress 448 / base 360 / defense 6 /
    spread 4 / mobility 9 / threat 130** (d4 #1, R 1668.84 vs the 1285.94 incumbent
    at the time) — promoted to the anchor: file `c1_incumbent.json` (**1000 = C1**).
    d4 #2 C2 = progress 224 / base 360 (R 1636.43, robust: 224/720 ties). Prior
    1285.94 incumbent = reference only now.
    **T-17e (✅ CLOSED 2026-11-04 — C1 stands, recorded as winner):** d6-confirm the FULL 3×3 grid
    **progress {224, 336, 448} × base {360, 540, 720}**, 4 terms locked
    (defense 6 / spread 4 / mobility 9 / threat 130), **anchored on C1
    (`c1_incumbent.json`)** — the 448/360 cell is C1's self-match (sanity ≈1000).
    **25 games/seed** (was 50 → cross-seed variance higher, `std<20` gate rougher),
    d6 ×3 seeds, double-blind, **`--skip-proxy`** (a pure d6 pass; the d4 screen is
    pure overhead when confirming ALL cells). Work: 9 × 25 × 3 × 2 ≈ **1 350 d6
    games**, ~10 s/game → ~3.5–4 h, 12 workers, run in background. Accept (on the
    C1 scale): `mean−1000>50 AND std<20 AND min>1000`; **if a winner lands ~+50, add
    a 2nd-seed tiebreak** before trusting it. See the T-17e block in the T-17 task
    section. `tools_budget_search.py` gained **`--skip-proxy`** this session (additive;
    default proxy-then-confirm unchanged). Surfaces: `_t17c/progress`,
    `_t17c/base_presence`, `_t17d/`.**
  - `halma8.py` / `halma8_test.py` are the **canonical oracle — never modified in this
    plan.** The Rust engine must *agree* with them.
- **First command in any new session:** `py halma8_test.py` → 36 tests, OK, exit 0.
- **Always-green gates (verify before stopping a task):**
  1. `py halma8_test.py` → 36 OK (engine untouched).
  2. `cargo test` (in `ai/`) → all green, including the Python cross-checks.
  3. `py ai/crosscheck_moves.py`, `py ai/crosscheck_rules.py`, `py ai/crosscheck_eval.py`
     → PASS, zero divergence.
  4. `py ai/tools_ai_smoke.py` → `PASS (0 illegal moves; protocol OK)`.
- **Hard rules (every session):**
  1. Do EXACTLY one task at a time; never "improve" outside the current task.
  2. **Separation:** `engine.rs` (rules) has no search; `search.rs` consumes engine.
     Python `halma8.py` stays the sole legality authority — the AI only *proposes*, the
     GUI feeds the move through the existing `apply_move`.
  3. **Rust: std-only, zero cargo deps** (match `kego_engine_v2`).
  4. Single dir `check_halma/`; Rust under `ai/` (crate `halma8_ai`). Don't touch
     `halma8.py`/`halma8_test.py`; touch `halma8_gui.py` only where a task says.
  5. Use **`py`**, not `python` (`python` is a broken MS Store alias on this box).
  6. Shell timeouts **≤ 60 000 ms**; longer `cargo` builds/tests → run in background and
     poll a status file.
  7. Each task ends only when its gate is green and pasted.
  8. **Keep this md current on completion:** on a green gate, tick the task heading ✅
     and refresh "Where we are" in the same pass. Never let the md lag the real state.
  9. **Long selfplay/train runs → background + unbuffered.** Run `py -u tools_tune.py`
     with `is_background: true`, then poll the on-disk `selfplay.csv` / `weights_vN.json`
     / `elo_curve.svg`. **NEVER** a foreground/≤60 s redirect to a `.log`: the loop dies
     mid-batch and Python's file-buffered stdout never flushes, leaving the log empty and
     all diagnostic output lost — exactly how T-11's two prior runs died (see T-11 ⚠ STATUS).
- **Rollback:** plain files, no git assumptions. `ai/` is fully additive and
  self-contained — delete it and everything reverts to the Part 1 state.

---

## The load-bearing invariant: two engines, one rule book

The whole risk of a second engine is **divergence** from the Python rules. We eliminate
it with a standing test, not discipline:

- `halma8_test.py` (Python) is the **canonical oracle.**
- Rust tests **shell out to `py`** and, over shared fixture positions, assert that both
  engines produce identical `legal_steps`/`legal_jumps`, and that replaying the same
  move list yields identical `grid`, `turn`, `turn_no`, `winner`, and the same
  `Ok(true)`/`Ok(false)`/`Err(Ill)` at each move.
- This must stay green at the end of **any** task that touches `engine.rs`. It is what
  stops the AI from proposing an "illegal" move Python rejects, or missing a legal one.

**The two subtle rules that most often get mis-ported (test them explicitly in Rust):**
1. **Chain-revisit ban:** during an open chain, a jump whose destination is the origin or
   an already-landed square is `IllegalMove` — *if the chain is still open*. But if *no*
   non-banned continuation exists, the chain simply **ends** (`Ok(false)`), not an error.
2. **Win = "full + invaded":** a player wins only when the *opponent's* 3×3 base is
   **completely occupied AND contains at least one of that player's own pieces.** (The
   opponent's own 9 starting pieces filling their base do *not* trigger a win.)

---

## Wire formats (single source of truth — do not change)

All coordinates 0-based `(r,c)`, `r` = row (0 = top), `c` = col (0 = left).

- **Board string** (`board`): 64 chars, row-major, index `i = r*8 + c`.
  `.` = empty, `1` = P1 (red), `2` = P2 (blue).
  Start: rows 0–2 have `2` in cols 5–7; rows 5–7 have `1` in cols 0–2; else `.`.
- **Position string** = `<board> <turn> <chain>`:
  - `turn`: `1` or `2`.
  - `chain`: `-` when no chain; else `C:o=<r><c>;b=<r1><c1>,<r2><c2>,...` with `o` =
    origin and `b` = banned squares in **sorted** order (determinism).
  - Python builds this from `grid`, `turn`, `chain_active`, `chain_banned`,
    `chain_starter`.
- **Move string** (AI output / what both apply):
  - step: `step <r>,<c> <nr>,<nc>`
  - jump: `jump <r>,<c> <mr>,<mc> <nr>,<nc>`
  - stop (AI ends a live chain, i.e. call `end_turn`): `stop`

---

## FFI (subprocess text protocol, kego-proven)

No ctypes / `.dll`. The Rust AI is a **binary that reads a line protocol on stdin and
writes to stdout** (a port of `kego_engine_v2/src/uci.rs` + its `tools_smoke.py::UciEngine`
driver). One call per turn; the whole search runs natively in the child.

Verbs: `uci` → `id name check_halma_ai` / `uciok`; `isready` → `readyok`;
`position <position_string>`; `go depth <N>` or `go movetime <ms>`; `stop` (must still
emit a legal `bestmove`); `setoption name Weights path=<file.json>`; `quit`. Output per
search: optional `info depth <D> score cp <X>`, then `bestmove <move_string>`.

---

## Borrow from kego / do NOT port

- **Borrow (framework — transfers directly):** repo shape; negamax/alpha-beta +
  iterative deepening; transposition table (EXACT/UPPER/LOWER, ply-adjusted mate);
  time management + `stop`; `splitmix64`-seeded Zobrist key init; make/unmake; the UCI
  protocol; the `tools_smoke.py` harness (with `halma8.py` as the oracle instead of
  chess).
- **Do NOT port (dead for halma):** check/mate, castling, en passant, promotions,
  quiescence, SEE/MVV-LVA, null-move, LMR, futility, RAZOR, PVS, killers, history,
  50-move, threefold, insufficient-material. Halma has **no captures, no check, no
  draws** — it is a positional/race search.
- **No pruning of any kind** (user-mandated 2026-11-01). The tree stays fully explored;
  only move *ordering* + sound alpha-beta bounds are kept. Move **ordering ≠ pruning** —
  it only reorders which branch is examined first.
- **The genuinely hard part:** the evaluation function (T-5). Halma's win is a moving
  spatial front; no settled halma eval exists to copy, so it is built from principles.

---

## Design decisions that deviate from the literal per-task text (kept — do not "fix")

- **Search is a world-score min-max** (P1 = maximizer, P2 = minimizer), not raw negamax:
  a live halma chain lets the *same* side move again, so two consecutive same-side nodes
  must both be maximizers — negamax's `-negamax(...)` sign flip can't express that.
  `evaluate(b) = side_score(P1) − side_score(P2)`, **turn-independent**; start position =
  exactly 0; exact anti-symmetry under 180°-rotate + colour-swap.
- **Mobility** = `steps + 2·jumps`. A blocker can *raise* this term (a jump counts 2× and
  a blocker *opens* a jump target), so the test asserts the exact value, not
  monotonicity in open neighbours.
- **Weights** load as int `i32` (`evaluate.rs::Weights::load`). `train_weights.py` is
  **pure stdlib** (numpy/pandas/sklearn are not installed on this box), standardized
  L2-logistic fit of the mover's 6 terms → label, with a **fallback to Rust defaults**
  whenever a class is missing or the fit degenerates, so `weights.json` is always valid.
  T-10.5 label `progress` (a weight-independent mid-game base-race proxy) is the default
  because full-game win/loss is almost always `ongoing`.

---

## Repository layout (final)

```
check_halma/
├─ halma8.py / halma8_gui.py / halma8_test.py   (Part 1 — canonical oracle, unchanged)
├─ plan.md / plan_ai.md                          (this file)
├─ dist/check_halma.exe
├─ halma_ai.py                                   (subprocess driver — T-8)
├─ train_weights.py                              (weight fit — T-10/10.5)
├─ weights.json                                  (generated)
├─ tools_tune.py                                 (T-11 — TBD)
└─ ai/                                           (Rust crate `halma8_ai`, std-only)
   ├─ Cargo.toml                                 (bins: ai, selfplay, bench; example: evaluate_dump)
   ├─ src/{types,engine,evaluate,search,server,main}.rs
   ├─ src/bin/selfplay.rs                        (T-9; T-11 adds --weights-a/--weights-b)
   ├─ src/bin/bench.rs
   ├─ examples/evaluate_dump.rs
   ├─ tests/{engine,search,eval}.rs
   └─ crosscheck_moves.py / crosscheck_rules.py / crosscheck_eval.py / tools_ai_smoke.py
```

Notes: `ai/` has **zero** Python deps (its only external input is `py` for cross-checks).
`train_weights.py` is the one place Python may reach beyond stdlib in principle, but in
practice it is stdlib-only on this box.

---

## Task sequence

Every task ends on a green gate, paste-verified, then stops.

### T-0 … T-10.5 ✅ (done — summary)

- **T-0** crate scaffold. **T-1** board/types + coordinate-string round-trip. **T-2**
  `legal_steps`/`legal_jumps`. **T-3** `apply_move`/`end_turn`/chain rules/`_check_win`/
  `IllegalMove` — the two subtle rules, cross-checked zero-divergence vs Python.
- **T-4** search frame (world-score min-max, iterative deepening, TT, time/stop).
  **T-5** real 6-term weighted eval (progress / base_presence / defense / spread /
  mobility / threat) + `Weights` JSON load/save. **T-6** move ordering + refined TT +
  release bench. **T-6.5** perf pass: make/unmake, per-ply scratch buffers, stronger
  ordering, time-budget default — **no pruning** (result-invariant).
- **T-7** line-protocol `server.rs` + `main.rs` + `ai/tools_ai_smoke.py` (the
  load-bearing integration gate). Known fix in this pass: `Search.stop` is
  `Arc<AtomicBool>` cloned before the search-thread spawn so `stop` genuinely aborts a
  deep `go` (an earlier raw-pointer version left stale stack memory and hung).
- **T-8** `halma_ai.py` (subprocess driver: spawn, position/go/bestmove/stop/quit,
  auto-respawn on death, hard ~10 s per-move deadline) + the `halma8_gui.py` "AI plays
  for P1/P2" toggles + "AI depth" knob + the `_ai_take_turn` chain loop. Python engine
  stays sole legality authority.
- **T-9** `ai/src/bin/selfplay.rs` (deterministic self-play CSV; 35% explore / 65%
  search over a seeded SplitMix64 RNG; fresh stateless `Search` per move; fixed-depth
  default so same-seed runs are byte-identical).
- **T-10** `train_weights.py` (pure-stdlib standardized L2 logistic fit with
  Rust-default fallback) + `ai/examples/evaluate_dump.rs` + `ai/crosscheck_eval.py`
  (Python mirror of all six eval terms asserts the weight→score contract on 7 positions).
- **T-10.5** self-play default depth 4→6 + `train_weights.py --label {final,progress}`
  (default **progress**) + `--mark N`. `progress` is a weight-independent mid-game
  base-race proxy defined for every row, so the fit never degenerates to the fallback.

**All gates green last verified 2026-11-01:** `py halma8_test.py` 36 OK; `cargo test`
green; `cargo build --release` clean; `crosscheck_moves.py` / `crosscheck_rules.py` /
`crosscheck_eval.py` PASS (zero divergence); `tools_ai_smoke.py` PASS (0 illegal moves).
`halma8.py` / `halma8_test.py` byte-unchanged throughout Part 2.

### T-11 — Close the loop: selfplay → train → re-selfplay, Elo curve, baseline   ⚠ FULL RUN GREEN, but FIT DEGENERATE — **STALLED on a design decision**

> **⚠ STATUS (verified 2026-11-01) — the run now COMPLETES cleanly; the fit is the blocker.**
> - **T-12 (parallel) is done + verified** → depth‑6 self-play is fast; this is no longer a
>   "too slow to run" problem.
> - **A full verified run WAS produced** via
>   `py -u tools_tune.py --depth 6 --games 100 --elo-games 30 --iterations 3 --workers 12`:
>   exit 0, `elo_curve.svg` + 3‑row `tune_log.txt`, `weights_v1/v2/v3.json`, valid self‑contained
>   SVG (anchor line + 4 points + polyline). Objective gates 1–6 are GREEN (all Rᵢ reproduce to
>   2dp across runs — determinism confirmed).
> - **BUT the fit is degenerate (fails the "non-trivial trend" done-when):** every iteration is
>   **identical** — S=0.4833, **R=988.41**, held‑out acc 0.7101, and each `weights_v{i}.json`
>   collapses to **`{"defense":1}`** (all other terms 0). The curve is a flat line 11.6 below the
>   1000 anchor — the orchestrator flags it: *"All Rᵢ within 50 of anchor. Fit may still be
>   degenerate."*
> - **Why it is flat, by construction (NOT a scheduling/threading bug):** the loop regenerates
>   its training CSV at a **fixed `--seed 0`**, and self-play is deterministic, so every
>   "iteration" sees **byte-identical training rows** → the same `progress` label → the same
>   defense-only logistic solution → the same S. There is no iteration-to-iteration learning
>   possible with one fixed seed; the "curve" is the same point plotted 3×. This reproduces the
>   T‑10.5 observation that `progress`/linear fits collapse onto a single (defense) term.
> - **Conclusion / next action (user decision, 2026-11-01):** pick ONE of —
>   (a) **advance the seed per iteration** (`--seed = base + i`) so each iteration trains on
>   fresh, different data and the Elo curve can actually move (smallest change, reuses everything);
>   (b) **change the label / features** (`--label final`, or more games→rows) if `progress` is
>   the wrong target for this linear model; or
>   (c) **accept** T‑12's parallel speedup as the real win and treat the flat ≈988 curve as
>   "current AI ≈ baseline; no measurable lift yet" and stop tuning. **Do NOT** re-run the
>   seed‑0 loop — it is provably byte-identical and will always give R=988.41.
> - **Stale partial artifacts** from the earlier (killed) attempt are archived under
>   `_stale_t11_partial/` (selfplay.csv, weights_v0/v1, elo_v1.csv, tune_log.txt).

- **Goal:** `tools_tune.py` — orchestrates the *whole* pipeline, **tracks engine
  strength as an Elo rating against a fixed anchor**, and writes an **SVG chart** of
  Elo over iterations that opens in any browser.
- **Files touched (in scope):**
  - `ai/src/bin/selfplay.rs` — add **head-to-head mode**: `--weights-a <file>` and
    `--weights-b <file>` (both default to Rust defaults = current behavior). When both
    are set, P1 uses A and P2 uses B; each side loads its own weight set before its
    first move (fresh stateless `Search` per move, existing pattern). Determinism gate
    unchanged: same `--seed` + same A/B files → byte-identical CSV.
  - `tools_tune.py` — orchestrator. CLI: `--iterations N` (3), `--games N` (50),
    `--depth D` (6), `--seed S` (0), `--elo-games N` (30 head-to-head games per Elo
    measurement), `--out-dir DIR` (default `.`).
  - `elo_chart.py` (or inlined) — pure-stdlib SVG writer → `elo_curve.svg`. No
    matplotlib/numpy/pandas; hand-rolled `<svg>` primitives.
  - **NOT touched:** `halma8.py`, `halma8_test.py`, `evaluate.rs`, `evaluate_dump.rs`,
    `search.rs`, `server.rs`, `halma8_gui.py`, `crosscheck_eval.py`, `tools_ai_smoke.py`.

- **Elo design (implement exactly this):**
  - **Anchor (iteration 0):** Rust default weight set (`Weights::default()`), rating =
    **1000**, fixed reference. Iteration `i ≥ 1` is `weights_v{i}.json`.
  - **Elo is a one-shot rating vs the fixed anchor** (inverse Elo expected-score), not a
    running ladder — simpler, reproducible, order-independent.
  - **Measurement (per iteration i):**
    1. `selfplay --weights-a weights_v{i}.json --weights-b weights_v0.json --games
       N_elo --depth D --seed S_elo --out elo_v{i}.csv` (`weights_v0.json` = Rust
       defaults, written at session start).
    2. From **P1's (weights_v{i})** view: `W` = games P1 won, `L` = games P2 won, `D` =
       `ongoing`, `N_total = W+L+D`.
    3. `S = (W + 0.5·D) / N_total`, clipped to `[0.01, 0.99]`.
    4. `R_i = 1000 + 400 · log10( S / (1 - S) )`.

     Interpretation: R=1000 → equal to baseline; R=1200 → ~64% win-rate vs baseline;
     R=800 → ~36% (weaker).
  - **Steps per iteration (in `tools_tune.py`):**
    1. `py train_weights.py selfplay.csv --label progress -o weights_v{i}.json`.
    2. Elo measurement (protocol above) → `R_i`.
    3. Append a baseline row to `tune_log.txt` and print it:
       `iteration, seed, N, D, S, R_i (Elo), test-accuracy, median-move-ms`.
    4. Re-generate the training CSV: `selfplay --games N --depth D --seed S --out
       selfplay.csv` (overwrite) — the loop continues.
    5. After all iterations: write `elo_curve.svg`.
  - **SVG spec (`elo_curve.svg`, self-contained, no refs):** 720×360 px, white bg.
    X = iteration (0..N, with 0 = anchor); Y = Elo (auto-scale with ≥±100 padding
    around 1000). 5 light-grey horizontal grid lines + one vertical per iteration.
    Polyline through all `R_i` (incl. anchor); 4 px filled circles at each point;
    iteration label under each point, `R_i` value above; dashed horizontal anchor line at
    1000 labelled "Rust defaults (anchor = 1000)"; title
    "check_halma — AI Elo over self-play iterations". Pure inline attributes/style; no
    external fonts/CSS/JS.

- **Gate (must all be green to call T-11 done):**
  1. `py halma8_test.py` → 36 OK, exit 0.
  2. `cargo test` (in `ai/`) green. Head-to-head with **both** A and B set to the
     Rust-default file gives **S ≈ 0.5** (assert `0.35 ≤ S ≤ 0.65`; loose bound due to
     small sample + slow halma resolution, but a lopsided result means a bug in the
     head-to-head path).
  3. `cargo build --release --bin selfplay` clean.
  4. `py -u tools_tune.py --iterations 3 --games 100 --depth 6 --elo-games 30
     --workers 12` (run in the **background**) runs to completion and produces
     `tune_log.txt` (3 iteration rows) + `elo_curve.svg`. (`--games` must stay **≥100**
     per the user; `--workers` comes from T-12 and does not exist until T-12 lands.)
  5. `elo_curve.svg` is valid and self-contained — shows the anchor line, 4 points
     (i=0,1,2,3), and the polyline.
  6. **Determinism:** two same-seed runs give the same `R_i` values to 2 decimals.
  7. **Sanity:** `R_0 = 1000` exactly; `R_1 ≠ 1000` (if it is, the fit is learning
     nothing or the head-to-head is lopsided in the wrong direction).

- **Done-when:** `tools_tune.py` runs the full loop; `elo_curve.svg` opens cleanly and
  shows a non-trivial Elo trend (at least one `R_i` differs from 1000 by > 50);
  `tune_log.txt` is parseable and human-readable; the five untouched files above are
  byte-unchanged; **this T-11 heading is ticked ✅ and "Where we are" refreshed.**
  If **all** iterations land at exactly 1000, the fit is still degenerate — report the
  A/B data back to the user and stop.

- **Rollback:** delete `tools_tune.py`, `elo_chart.py` (or inline SVG code),
  `tune_log.txt`, `elo_curve.svg`, and the `--weights-a`/`--weights-b` flags in
  `selfplay.rs`. `train_weights.py` is unchanged by this task.

### T-12 — Parallel self-play   ✅ DONE & verified (2026-11-01)

**Status: DONE.** `selfplay.exe --workers K` (thread-scoped fan-out, per-game
`game_seed = splitmix_fold(master_seed, game_index)`, rows emitted in `game_index`
order → **byte-identical CSV for any K**) + `tools_tune.py --workers` passthrough on
every `selfplay` call + `ai/tests/parallel_determinism.rs` (asserts workers=2/5/1
identical). Verified green 2026-11-01: `halma8_test.py` 36 OK; `cargo test --release`
green incl. `parallel_determinism` (0.65s, `output_is_worker_count_independent` ok);
`crosscheck_moves/rules/eval` PASS (zero divergence); `tools_ai_smoke` PASS (0 illegal).
`--workers K` default 1 = serial. Depth-6 self-play is now practical (~4 min / 100 games
at K=12 on this 24-core box).

**Scope of this task (do first):**
1. `selfplay.rs` — add `--workers K` (default **1** = today's serial behaviour). Fan the
   games out across worker threads via `std::thread::scope`. **Give each game its own
   deterministic seed** `game_seed = splitmix_fold(master_seed, game_index)` (NOT the
   shared cross-game RNG stream), and **emit rows in `game_index` order** so the same
   `--seed` + `--games` yields **byte-identical CSV for any K** (determinism must not
   depend on thread scheduling). std-only, zero deps.
2. `tools_tune.py` — add a `--workers` CLI knob and pass it to **every** `selfplay.exe`
   call (initial CSV, each h2h, each regen). Default **1**; no other logic changes.
3. Re-run + **verify T-11** (see its ⚠ STATUS block): run `py -u tools_tune.py
   --depth 6 --games 100 --elo-games 30 --iterations 3 --workers 12` **in the
   background**, poll for `elo_curve.svg` + a 4-row `tune_log.txt`, then confirm every
   T-11 gate (S∈[0.35,0.65]; R_0=1000; ≥one R_i off 1000 by >50; two same-seed runs →
   same R_i to 2 dp; the 5 listed files byte-unchanged) and tick **T-11 ✅**.

- **Gate (parallel):** `cargo test` green — add a test asserting **K=4 output == K=1
  output** for the same `--seed`/`--games` (determinism is K-independent); a 24-game
  depth-4 sample at K=8 ≤ ~1.5× (soft; the old "256 games K=4 ≤ 1.25× serial" upper bar
  still applies). `py halma8_test.py` 36 OK; `crosscheck_moves.py` / `crosscheck_rules.py`
  / `crosscheck_eval.py` PASS; `tools_ai_smoke.py` PASS.
- **Done-when:** `--workers` is K-independent & byte-stable; T-11 has a green run
  (elo_curve.svg + tune_log.txt + all gates) at **depth 6, parallel, ≥100 games**, and
  **both** the T-12 and T-11 headings are ticked ✅ with "Where we are" refreshed.

### T-13 — Diagnose the degenerate fit (go / no-go)   ✅ DONE (2026-11-01)

> **Read first (first command this session):** `py halma8_test.py` → 36 OK.
> This task is **read-only** — it prints a table so we name the failing link
> before we touch `evaluate.rs`. Do not advance to T-14 until the verdict line
> is pasted and names (A) or (B) below.

> **RESULT (2026-11-01) — verdict = (A) int-round.** Throwaway `diag_fit.py`
> (reuses `train_weights`'s own functions, so the fit here is bit-identical to
> the loop's) on `selfplay.csv` (100 games, 9,867 labelled rows, 1,800 tied
> dropped, classes 5361 ahead / 4506 behind). Reproduced int set = `{defense:1}`
> — byte-equal to `weights_v1.json`. Held-out acc **0.7101**; **fit kept (no
> fallback) = Y**; round-to-0 = **5/6**.
>
> | term          | mean(train) | std(train) | w_std(O1) | float w (w_std/std) | int rnd |
> | ------------- | ----------: | ---------: | --------: | ------------------: | ------: |
> | progress      |     -45.437 |     13.255 |    1.4021 |            0.1058   |       0 |
> | base_presence |       0.872 |      1.060 |    0.5080 |            0.4792   |       0 |
> | defense       |      -0.820 |      1.015 |    1.1179 |            1.1016   |       1 |
> | spread        |      -4.312 |      2.277 |   -0.3305 |           -0.1451   |       0 |
> | mobility      |      34.976 |      4.940 |   -0.0818 |           -0.0166   |       0 |
> | threat        |       1.489 |      1.463 |   -0.4639 |           -0.3171   |       0 |
>
> **Reasoning:** acc 0.71 with balanced classes → the base-race label *is*
> carrying signal (rules out (B) weak label). The float fit has genuine
> multi-term shape (defense 1.10, base_presence 0.48, threat -0.32, progress
> 0.11, spread -0.15); `int(round)` flattens it to `{defense:1}` —
> `base_presence` 0.4792 is a live coefficient just-missed. **→ T-14
> (real-valued / f64 Weights) is the fix; T-14b signed label NOT required.**
> Gates: `py halma8_test.py` 36 OK; no `ai/` source changed.
> Re-run any time: `py diag_fit.py selfplay.csv`.

- **Two candidate causes to tell apart:**
  - **(A) int-round:** `train_weights.to_raw_weights` does `int(round(w_std/std))`.
    The six terms differ 10–30× in raw scale (e.g. `progress` ≈ −45…−135 vs
    `base_presence` 0…9), so a standardised O(1) coefficient maps to a raw value
    < 0.5 on the large-scaled terms → rounds to **0** → the learned vector looks
    "all-zero / near-zero" → the `collapsed`/`at_chance` guard in `main` fires
    → **fallback to `DEFAULT_WEIGHTS`** → `weights_v{i}.json == weights_v0.json`
    → the "h2h" is literally defaults-vs-defaults → S≈0.483 → R≈988. (Hypothesis
    **A**; the most likely one.)
  - **(B) weak label:** `load_rows_progress` stamps one binary base-race outcome
    ("who led at ply 50?") on every row of the game and drops whole games at a
    tied reference position. Coarse, temporally misaligned supervision → small
    coefficients even before rounding. If B dominates, float (T-14) alone won't
    fix it and we additionally switch to a **signed** regression target.
- **Build (a throwaway diagnostic — either `train_weights.py --diagnose <csv>`
  or a new tiny `diag_fit.py`):** on `selfplay.csv` (depth 6, ≥100 games),
  print for each of the 6 terms:
  | term            | raw mean | raw std | w_std (O(1)) | raw float weight (`w_std/std`) | `int(round)` after |
  | --------------- | -------: | ------: | -----------: | ----------------------------: | -----------------: |
  plus: # labelled rows, # dropped ties, # `win`/`loss`, held-out accuracy at
  the existing split, and a **verdict line**:
  `"round-to-0 count = N/6"` + `"fit kept (no fallback)? Y/N"`.
- **Gate (green = T-13 done):** table prints; `halma8_test.py` 36 OK; **no `ai/`
  source changed**; verdict line names (A) or (B). **All green 2026-11-01 — see
  RESULT above (table + (A) verdict).**
- **Done-when:** the table + verdict are pasted into the T-13 note. **This
  determines T-14's scope** (A → float only; B → float **and** signed label).
  **Resolved 2026-11-01: verdict (A) → T-14 = float only (T-14b not required).**

### T-14 — Real-valued weights (a *budget* across 6 features is only representable in float)  ✅ DONE & verified (2026-11-01)

> **RESULT (verified 2026-11-01) — all five gates green.**
> - `evaluate.rs`: `Weights` fields `i32 → f64`, `load` parses `f64`, `save`
>   emits floats, `side_score → f64`; `evaluate`/`evaluate_with`/
>   `evaluate_pair` cast back to `i32`. Defaults unchanged (`6.0, 30.0, 25.0,
>   4.0, 3.0, 55.0`). New `weights_fractional_roundtrip` test (55.5 exact).
> - `train_weights.py::to_raw_weights` → `round(w_std/std, 4)` (float); the
>   hard all-zero `collapsed` fallback was **dropped**, the `at_chance`
>   (held-out acc ≤ 0.5) guard **kept**.
> - `selfplay.rs` h2h log lines print the `f64` values via `{}` (verified:
>   `threat 55.5` appears in the head-to-head line; the other default ints print
>   as `6`, `30`, … — the `{}` Display for an integer-valued `f64`).
> - `crosscheck_eval.py` unchanged (generic: reads whatever `weights.json` holds);
>   with the current integer `weights.json` it reproduces **exact** scores (delta 0).
> - Gates: (1) `py halma8_test.py` → **36 OK**; (2) `cargo test --release` →
>   **all green** (anti_symmetry, evaluate_pair_*, weights_load_save_roundtrip,
>   weights_load_missing_file_gives_defaults, weights_fractional_roundtrip,
>   parallel_determinism); (3) `crosscheck_moves`/`crosscheck_rules`/
>   `crosscheck_eval` → **PASS, zero divergence**; (4) `tools_ai_smoke` →
>   **PASS (0 illegal moves)**; (5) fractional `_t14_frac.json` (`threat 55.5`)
>   loaded in Rust, round-tripped, and `_t14_frac.json` vs `weights.json` h2h ran
>   one game to completion printing `threat 55.5`.
> - **Effect on T-11's flat curve:** the root cause (int-round → defense-only
>   collapse) is gone. A fresh `train_weights.py selfplay.csv --label progress`
>   now writes a multi-term float weight set instead of `{defense:1}` — so the
>   T-15 loop no longer compares defaults-vs-defaults.
> - Throwaway verification files (`_t14_frac.json`/`.csv`) removed after use.

- **Goal:** `Weights` fields become `f64`. Because the 6 terms differ 10–30× in
  raw scale, any meaningful "distribute a budget" lives in **float** space —
  an integer weight can't express a fraction of a CP-unit that's small relative
  to the biggest term. World score stays `i32`, rounded, so every existing
  invariant (start=0, exact 180°+colour-swap anti-symmetry, alpha-beta integer
  comparison, all 36 Python tests, all `cargo test` invariants) is **bit-identical
  for any integer weight set** (small ints are exact in f64; products < 2⁵³ are
  exact; `Σ (f64 int)·(i32)` == the integer sum).
- **Files (in scope):**
  - `ai/src/evaluate.rs` — struct fields `i32 → f64`; `load` parses `f64`;
    `save` emits floats; `side_score → f64`; `evaluate` / `evaluate_with` /
    `evaluate_pair` return `f64 as i32` (rounds to the same integer as the old
    i32·i32 path for integer weights).
  - `ai/src/bin/selfplay.rs` — head-to-head log lines print floats (currently
    prints ints); no behaviour change.
  - `train_weights.py::to_raw_weights` — drop `int(round(...))`; emit
    `round(w_std/std, 4)`. **Keep the `at_chance` fallback; drop the hard
    all-zero `collapsed` fallback** (floats don't vanish by rounding).
  - `ai/crosscheck_eval.py` — existing 7-position integer-weight asserts **must
    produce identical integer scores** (regression guard that the f64 path is
    bit-identical to the old i32 path at integer weights). Optionally add **one
    fractional weight set** assert to lock float-in.
  - NOT touched: `halma8.py` / `halma8_test.py` (oracle).
- **Spec (implement exactly this):**
  1. `Weights { progress, base_presence, defense, spread, mobility, threat: f64 }`
     with the **same default values** `6.0, 30.0, 25.0, 4.0, 3.0, 55.0`.
  2. `load` parses `parse::<f64>()` (still tolerates missing field / bad file →
     default, one field at a time). `save` writes a flat JSON object with floats.
  3. `side_score` returns `f64`; `evaluate` / `evaluate_with` / `evaluate_pair`
     cast the result back to `i32` → `search.rs` and every existing integer
     weight-set test see **identical** integer scores as before.
  4. `train_weights.py::to_raw_weights` → `{TERMS[t]: round(w_std[t]/std[t], 4)
     for t in range(6)}`; keep the `WEIGHT_SANITY_BOUND` warn but drop the
     hard all-zero collapse fallback.
  5. `crosscheck_eval.py` — existing integer-weight asserts must reproduce the
     **same integer scores** as the pre-T-14 run (bit-identical). Optionally add
     a single fractional-weight-set assert (e.g. `threat: 55.5`) that passes
     both sides.
- **Gate (all green to call T-14 done):**
  1. `py halma8_test.py` → 36 OK.
  2. `cargo test` in `ai/` green — includes `anti_symmetry_*`, `evaluate_pair_*`,
     `weights_load_save_roundtrip`, `weights_load_missing_file_gives_defaults`.
  3. `py ai/crosscheck_moves.py` / `crosscheck_rules.py` / `crosscheck_eval.py`
     all PASS with **zero divergence** (integer weight sets unchanged).
  4. `py ai/tools_ai_smoke.py` PASS.
  5. **New assert:** a JSON file `{ "threat": 55.5, ... }` loads in Rust,
     round-trips via `save → load`, and `selfplay --weights-a <file>` runs one
     game to completion without error, printing the fractional weight in its
     head-to-head line.
- **Done-when:** all five gates green; `weights_v1.json` (fresh run of T-15's
  loop next task) no longer equals `weights_v0.json` because the fit is *no
  longer rounding to zero*; "Where we are" refreshed. **If T-13 verdict was
  (B), also queue T-14b:** re-fit with a signed regression target
  `(progress_1 − progress_2)` in place of the binary tied-dropped label — uses
  every game (no drops), far less noisy.

### T-15 — Decoupled-seed tuning loop (the curve must actually move)  ✅ DONE & verified (2026-11-01)

> **RESULT (verified 2026-11-01) — all gates green; flat-line-by-construction bug is GONE.**
> Run: `py -u tools_tune.py --iterations 3 --games 100 --depth 6 --elo-games 30 --workers 12
>      --seed-base 1000 --out-dir _t15` (background) → exit 0, `tune_log.txt` (header + i=1,2,3)
> + valid self-contained `elo_curve.svg`.
>
> - **Decoupled seeds (the whole point of T-15):** training seeds **1001, 1002, 1003**
>   vs measurement seeds **2001, 2002, 2003** — no family overlaps, every `train_seed` and
>   every `measure_seed` distinct. `tune_log.txt` header now carries both columns.
> - **Weights no longer collapse to defaults (T-14 fix holds under the loop):** each
>   `weights_v{i}.json` is a distinct real float set, clearly ≠ `weights_v0.json`:
>   - v1 `{progress 0.1022, base_presence 0.1849, defense 0.8772, spread -0.1159,
>     mobility -0.0308, threat -0.2789}` (acc 0.6795)
>   - v2 `{progress 0.0968, base_presence 0.149, defense 0.8598, spread -0.0949,
>     mobility -0.0298, threat -0.2346}` (acc 0.6790)
>   - v3 `{progress 0.1012, base_presence 0.2087, defense 0.9039, spread -0.1182,
>     mobility -0.0243, threat -0.2684}` (acc 0.6710)
> - **Elo curve (variance gate — the flat line is broken):**
>   `i=0:1000` · `i=1:988.41 (W0 L1 D29)` · `i=2:1000.00 (W0 L0 D30)` · `i=3:1000.00
>   (W0 L0 D30)`. `|R_1 − R_2| = 11.59 > 2` → **not** the same point plotted 3× (contrast
>   T-11, where all three were byte-identical 988.41 because every `weights_v{i}` collapsed
>   to `{defense:1}` = defaults).
> - **Determinism gate (re-run, throwaway `_t15d` then deleted):** re-ran
>   `--iterations 1 --seed-base 1000` → byte-identical `weights_v1.json` **and** same h2h
>   `W0 L1 D29 → S=0.48333 → R=988.41` (selfplay seeded per-game + pure-function logistic
>   fit ⇒ same seed ⇒ same R).
> - **Always-green 1–4 re-confirmed this pass:** `py halma8_test.py` **36 OK**; `cargo
>   test --release` **green** (63 lib + `parallel_determinism`); `crosscheck_moves/rules/
>   eval` **PASS** (zero divergence); `tools_ai_smoke` **PASS (0 illegal)**.
> - **HONEST CAVEAT / next:** the pipeline fix (real float weights + decoupled seeds) is in
>   place and the "same point N×" defect is gone. **But the mean R is still sitting at the
>   1000 anchor** because the *head-to-head measurement itself is low-signal*: in every h2h
>   batch 29–30/30 games end **ongoing** at the move cap (W+L ≈ 0–1), so `S` is pinned near
>   0.5 and a single decisive game swings `S` by only 0.5/30 ≈ 0.017 → ΔR ≈ ±10. The R
>   spread we see is therefore near-coincidental h2h noise, **not** a stable signal that the
>   fit is stronger/weaker. The original T-11 ">50 off anchor" done-when is **still not met**
>   — that is a data/measurement limitation (halma rarely finishes in 30 d6 games), not a
>   pipeline bug, and is exactly the "report and stop if degenerate" branch the plan
>   anticipated. **T-16's stricter 3-seed, std<20, min>1000 confirmation is the real test**
>   (and it may well "report best-found + stop" against this ongoing-game floor). Do NOT
>   treat the ≈1000 mean as evidence the AI has improved.
> - `halma8.py` / `halma8_test.py` / `evaluate.rs` / `search.rs` / `selfplay.rs` byte-unchanged.
> - **Post-land addendum (2026-11-01):** the h2h measurement in `tools_tune.py` was then
>   upgraded from single-order / cap-120 to **double-blind (both move-orders) + `--maxmove 480`**
>   default as part of T-16's h2h fix. The T-15 numbers above are the pre-fix state; the core
>   T-15 change (decoupled train/measure seeds) is unchanged and still verified. Re-run this
>   loop (any `--seed-base`) to see the de-biased, higher-finish-rate curve.

- **Goal:** kill "same point plotted N×". In `tools_tune.py`:
  - **Training seed** for iteration `i` = `seed_base + i` (`--seed` becomes
    `--seed-base`, default 0).
  - **Measurement (h2h) seed** for iteration `i` = `seed_base + 1000 + i`
    (kept distinct from the training seeds, so the game data is not the fitted
    data).
  - `--workers` default **12** on every `selfplay` call (keeps T-12).
- **Files (in scope):** `tools_tune.py` only. **Not touched:** `evaluate.rs`,
  `search.rs`, `selfplay.rs`, `crosscheck_*`, `halma8_test.py`, `halma8.py`.
- **Spec:**
  1. `train(csv_path, w_i_path, args.seed_base + i)` — fresh data per iter.
  2. `run_selfplay(args.elo_games, D, seed_base + 1000 + i, elo_csv,
     weights_a=w_i, weights_b=w0)` — h2h on a *different* seed.
  3. Regenerate the *next* training CSV with `seed_base + (i + 1)` (only if
     `i < iterations`). This is exactly the current step-4 logic, just re-pointed
     at the next training seed instead of re-using the same one.
  4. Log line gains `train_seed` and `measure_seed` columns.
- **Gate (green = T-15 done):**
  1. Always-green 1–4 (36 / `cargo test` / 3 crosschecks / smoke) still green.
  2. `py -u tools_tune.py --iterations 3 --games 100 --depth 6 --elo-games 30
     --workers 12 --seed-base 1000 --out-dir _t15` run in the **background**
     completes with exit 0 and writes `tune_log.txt` (4 rows: i=0 anchor + 3
     iterations) + a valid self-contained `elo_curve.svg`; every
     `train_seed`/`measure_seed` is distinct.
  3. **Variance:** ≥2 of the R_i (i≥1) agree with each other to within 2 Elo AND
     differ from each other in a *meaningful* way (|R_i − R_j| > 2 for at least
     one pair). **This proves the loop has real variance even if the mean is
     still near 1000** — the flat-line bug is gone.
  4. **Determinism:** two re-runs with the same `--seed-base` give identical
     R_i to 2 dp.
- **Done-when:** gates green; **`weights_v{i}.json` differs from
  `weights_v0.json`** (diff the files!) — confirming the float fit is no longer
  falling back to defaults — and the curve has real variance. Tick T-15 ✅ and
  refresh "Where we are".

### T-16 — Budget-constrained weight search (affordable form of the grid idea)  ✅ DONE & verified 2026-11-02 · **production search ACCEPTED**

> **STATUS (production run COMPLETE 2026-11-02) — ✅ ACCEPTED.** Random 40-candidate
> sweep screened + top-3 confirmed at d6/×3 seeds; **all 3 cleared the gate**, and
> the best float weight set (beat the Rust-default anchor by a stable **+285.94 Elo**)
> is in `best_weights.json`. **One selection bug found in the first run and fixed**
> (best-pick sorted by candidate index, not by score). Tooling + h2h-measurement
> fixes below are unchanged/re-verified.
> - **Built (`tools_budget_search.py`, pure stdlib):** `--sampler {random,ascent}`
>   (random default; per-term factors `f_i ~ LogUniform(0.25,4)`), `--candidates 40`,
>   **proxy** scores every candidate (d4/15), **confirm** top-3 (d6/50 × 3 seeds
>   `5000+j`) — this is all the T-16 default knobs. Verdict: **accept** iff `mean−1000 > 50
>   AND std < 20 AND min > 1000`, else report best-found + stop (a valid research outcome).
>   Outputs: `budget_search_log.csv` · `budget_search_verdict.txt` · `best_weights.json`.
> - **h2h measurement fixed — double-blind + cap (the real blocker of T-11/T-15's flat ≈1000):**
>   every head-to-head now plays **both move-orders** and combines `W=W_A+L_B, L=L_A+W_B,
>   D=D_A+D_B` so the first-mover P1 bias cancels, and `--maxmove` defaults to **480**
>   (cap 120 left ~95% of games "ongoing" = a noise floor; 480 is where most games resolve).
>   First-mover bias measured 2026-11-01, **identical weights both sides** (cap 480): finished
>   games ran **17 P1 / 7 P2** → the double-blind A/B is what removes it (`_dblind_analysis.py`).
>   This h2h fix landed in **both** `tools_budget_search.py` (T-16) **and** `tools_tune.py` (T-15).
> - **Gates (all green 2026-11-01, re-run against the on-disk code):** `py halma8_test.py`
>   **36 OK** · `cargo test --release` **green** (incl. `parallel_determinism`) ·
>   `crosscheck_moves/rules/eval` **PASS (zero divergence)** · `tools_ai_smoke`
>   **PASS (0 illegal)** · `tools_budget_search.py` proxy smoke → **`BUDGET_SMOKE_DONE`** ·
>   `tools_tune.py` smoke → **`TUNE_SMOKE_DONE`**.
> - **Ran 2026-11-02 — the deliverable (launched via `run_t16.bat` wrapper, exit 0):**
>   `py -u tools_budget_search.py --candidates 40 --seed-base 0 --proxy-games 15
>     --proxy-depth 4 --confirm-games 50 --confirm-depth 6 --maxmove 480 --workers 12
>     --max-confirm 3 --out-dir .`
>   **Launcher on disk (2026-11-01): `run_t16.bat`** at the project root wraps exactly this
>   command — fail-fast checks for `selfplay.exe` + `weights_v0.json`, then a minimized
>   background launch logging to `_t16_run.log`. **Not yet executed** (user runs it later).
>   Poll the on-disk `budget_search_log.csv` / `budget_search_verdict.txt`. Do **not** treat
>   the proxy smoke (cap 60, d3, 2 candidates, `--skip-confirm`) as a result — it only proves
>   the pipeline runs. **Do not** re-run seed-0 `tools_tune.py` (provably byte-identical R=988.41).

> **RESULT (production run 2026-11-02, `run_t16.bat` → exit 0):**
> - **All 3 confirmed candidates cleared the gate** (`mean>1050 & std<20 & min>1000`):
>   | rank | cand idx | confirm mean_R@depth6 | std | min |
>   |---|---|---|---|---|
>   | 1 | cand[0]  | 1265.49 | 7.44 | 1257.59 |
>   | **2** | **cand[3]** | **1285.94** ← best | 3.01 | 1281.68 |
>   | 3 | cand[32] | 1230.19 | 4.28 | 1224.97 |
> - **Verdict: ACCEPTED; best = rank-2 cand[3], mean_R 1285.94, delta +285.94** —
>   a *stable*, seed-consistent, noise-rejected real improvement over the defaults.
> - **Best weights (`best_weights.json`, = `cand_w3.json`):**
>   `progress 14.18, base_presence 91.49, defense 14.77, spread 7.56,
>   mobility 9.07, threat 91.60`.
> - **Selection bug found & fixed (this run only, tool is now correct):** original
>   `tools_budget_search.py` line ~366 sorted the accepted set by **candidate index**
>   (`key=lambda x: x[0]`) instead of by score, so it wrote the *weakest* of the 3
>   (cand[32]) to `best_weights.json`. Fixed to `key=lambda x: x[1]`, and this run's
>   `best_weights.json` + `budget_search_verdict.txt` were re-pointed to the true best
>   (cand[3]). **No re-run was needed** (all CSVs were on disk).
> - **Proxy-vs-confirm agreement (gate 2):** the top-2 by proxy (cand[0], cand[3])
>   *swapped* at depth 6 (proxy 1301>1241 → confirm 1265<1286). Depth-4 is a usable
>   **filter** but **not a precise top-ranker** (±~40 Elo), which is why T-17 confirms
>   5 rather than 3.
> - **Pattern analysis (top-10 vs bottom-10 by proxy_R, per-term factor averages):**
>   **progress is the dominant lever** (strong ~2.55× vs weak ~0.65× default; every
>   strong candidate sits at progress 14–17). **defense trends *down*** (all 3 winners
>   run it at 15–20, below the default 25 — likely over-weighted). **threat is bimodal
>   across winners** (2 at ~43, 1 at ~92, all still strong → threat likely *interacts*
>   with progress, not additive). base_presence mild-positive; **spread & mobility are
>   weak** (~neutral). This is the basis for T-17's structured box.
> - Always-green gates re-verified (36 tests / `cargo test` / 3 crosschecks / smoke)
>   unchanged from the 2026-11-01 STATUS above.
>
- **Goal:** find a float weight set that **beats the Rust-default anchor by a
  stable >50 Elo margin at depth 6**, scored by a **cheap proxy** during the
  sweep and by a **full h2h** only for the final few. This is the "distribute a
  budget across 6 features" the user proposed, made *affordable* by (i) floats
  (T-14) so the budget is representable, and (ii) cheap-proxy-gating
  (T-15's depth-6 h2h is only ever run on the top candidates, not a 6-dim grid).
- **Files (new, pure stdlib):** `tools_budget_search.py`. **NOT touched:** any
  `ai/` Rust file, `halma8.py`, `halma8_test.py`, `halma8_gui.py`.
- **Search space (budget-normalised multiplicative factors — the "budget" is
  each feature's share of CP relative to the default weight set):**
  - Parameterisation: for each of the 6 terms, a multiplicative factor
    `f_i ∈ [0.25, 4.0]` applied to the Rust default weight for that term, i.e.
    `w_i = default_i × f_i`. Keeping the *relative* shape near the defaults
    avoids the degenerate regions where one term is 100× another and the eval
    is dominated by a single signal.
  - Sampler (pick one, keep cheap — both ~40 evaluations):
    - **Random:** 40 uniform-in-log points (`f_i ~ LogUniform(0.25, 4)`),
      seeded for reproducibility (`--seed-base`, default 0).
    - **Coordinate ascent:** start at all `f_i = 1`.0`; for each `i` in turn,
      try `{0.5, 1, 2} × f_i` holding the others; keep the best by the cheap
      proxy; 2 passes (12 candidates/pass × 2 = 24 candidates).
- **Cheap scoring (≤ 30 s/candidate, depth 4, 15 games):**
  `selfplay --weights-a cand.json --weights-b weights_v0.json --games 15
    --depth 4 --workers 12 --seed S_i --out cand_s{i}.csv`
  → S, R_proxy. Rank by R_proxy. (Depth 4 is fast, and — we will verify in
  T-16 gate 3 — correlates with the depth-6 ranking for our small weight space;
  that's why depth 6 is reserved for confirmation, not search.)
- **Confirm the top-1..3 (depth 6, 50 games, *three independent seeds*):**
  `selfplay --weights-a BEST --weights-b weights_v0.json --games 50
    --depth 6 --workers 12 --seed (5000 + j) --out confirm_s{5000+j}.csv`
  for `j ∈ {0, 1, 2}`. Compute `S_j`, `R_j`. **Accept if** `mean(R) − 1000 > 50`
  **AND** `std(R) < 20` **AND** `min(R) > 1000` (all three must clear the
  anchor — kills the "one lucky seed" case). Otherwise reject and move to
  rank-2. If none accept, report the best-found set and stop (this is a
  research outcome, not a coding failure).
- **Files (outputs):** `tools_budget_search.py`; `budget_search_log.csv`
  (rank, `f`-vector, proxy R, confirm R@3 seeds, accept/reject reason);
  `best_weights.json` (accepted `w = default × f` vector, flat JSON floats);
  one-line verdict in the log.
- **Gate (all green to call T-16 done):**
  1. Always-green 1–4 (36 / `cargo test` / 3 crosschecks / smoke) green.
  2. **Correlation of proxies:** the top-3 candidates by proxy R also finish in
     the top-3 (or top-4, allowing for one rank change) by confirmed depth-6
     R — a loose "proxy is meaningful" sanity check. (If proxies and confirms
     don't agree at all, the proxy is the wrong signal → drop the cheap-proxy
     and confirm all 40 at depth 6; that's 20× slower but still feasible at
     T-15's parallelism.)
  3. **Acceptance:** ≥1 candidate clears the three-seed gate above and is
     written to `best_weights.json`.
  4. **Determinism:** two same-`--seed-base` runs of `tools_budget_search.py`
     produce identical `budget_search_log.csv`.
- **Done-when:** a single, noise-rejected (≥+50 Elo @ depth 6, std<20, 3/3
  confirm seeds) float weight vector is in `best_weights.json` and
  `best_weights.json`'s weights beat the Rust-default anchor in a head-to-head
  under the depth-6 protocol. This finally satisfies T-11's "non-trivial trend"
  done-when *with noise rejected*. Tick T-16 ✅ and refresh "Where we are".
  **If no candidate accepts, stop and report** the best-found set +
  proxy-vs-confirm correlation + verdict (this is the outcome class the
  T-11 "STALLED on a design decision" note anticipated).

### T-17 — Refined grid search around the T-16 best (structured, cheaper-per-candidate)  ✅ DONE (2026-11-04: d4 proxy tier COMPLETE + **T-17e d6-confirm CLOSED** — 8/9 cells, nothing clears the mean>+50 gate, **winner (user override 2026-11-04) = 448/540, the best d6 cell → `best_weights.json`**)

> **STATUS (2026-11-03, LATEST): T-17 d4 proxy tier COMPLETE — STEP 1c wide shape
> + STEP 1d 2D pin (both exit 0).** **progress = TRUE INTERIOR PEAK @ 448** (R 1798
> @448, then DECAYS: 896→993, 1792→1007, 3584→1014; also collapses at 672 — does
> NOT keep rising). **base: at LOW progress plateaus 720–1440; high progress wants
> LOWER base** (448/360 > 448/720, +124). STEP 1d 2D pin (`_t17d/`, progress×base,
> 4 terms locked, d4/60 double-blind) R table vs the 1285.94 incumbent —
> **448/360 = 1668.84 (top)** · 224/360 = 1636.43 · 224/720 = 1636.43 · 112/720 =
> 1584.96 · 112/360 = 1563.75 · 448/720 = 1544.69 · 672/360 = 1076.49 ·
> 672/720 = 1064.42. FINDINGS: (1) **progress dominates**; (2) **672 collapses**
> (W=72/L=46 vs incumbent at d4); (3) **base × progress INTERACTION** (high progress
> → base 360 > 720). Top 6 all 0-loss, ≤5-draw apart → within d4 noise (d6 is the
> arbiter; T-16 showed d4 #1 ≠ d6 #1).
> **NEW INCUMBENT (user, 2026-11-03) = C1 = progress 448 / base 360 / defense 6 /
> spread 4 / mobility 9 / threat 130** — promoted to the anchor (file
> `c1_incumbent.json`, 1000 = C1); prior 1285.94 incumbent now reference-only.
> **T-17e (✅ CLOSED 2026-11-04):** d6-confirm the full 3×3 grid
> **progress {224,336,448} × base {360,540,720}** around C1, 25/seed, `--skip-proxy`.
> See the **T-17e** block below.
>
> **✅ FINAL (2026-11-04): T-17e CLOSED — C1 = WINNER.** The d6 run completed cells
> 0–7, then the background process died at cell 8 (448/720) seed 5000 (A-side CSV
> only; tool never wrote its end-of-run files). Verdict reconstructed by hand from
> `_t17e_launch.log` + per-cell CSVs and written to `_t17e/budget_search_log.csv` +
> `budget_search_verdict.txt`. Ranked d6 means vs the C1 anchor (1000 = C1):
> **448/540 = 1020.89 (best, +20.89, std 9.88, min 1013.90)** · 224/540 = 997.68 ·
> 336/720 = 981.44 · **C1 self-match 448/360 = 965.11 (sanity floor, std 9.95)** ·
> 336/360 = 951.03 · 224/720 = 909.96 · 224/360 = 894.86 · 448/720 = INCOMPLETE.
> **No cell clears the gate (mean−1000>50 AND std<20 AND min>1000):** 448/540
> passes std (9.88) and min (1013.90) but its mean_delta is only +20.89 (< 50) →
> REJECTED; it sits +55.8 above the C1 self-match floor — the one cell worth a
> tiebreak seed if ever revisited. **USER DECISION 2026-11-04: C1 (progress
> 448 / base 360 / defense 6 / spread 4 / mobility 9 / threat 130) recorded as the
> winner → promoted to `best_weights.json`** (T-16's 1285.94 weights now
> reference-only). d6 shape (arbiter): **base 540 is the strongest column at every
> progress row** (224/540 997.68 · 336/540 974.47 · 448/540 1020.89); the d4 hint
> "high progress wants base ≤ 360" did NOT survive d6 (448/360 self = 965.11 <
> 448/540). C1 stands on the gate, not the ranking.
>
> **T-17e — d6 confirm around the new C1 incumbent (LAUNCHED 2026-11-04, running
> in background — PENDING results; user-defined 2026-11-03).** Output: `_t17e/`
> (`budget_search_log.csv` + `budget_search_verdict.txt`); launch log
> `_t17e_launch.log`.
>
> Full 3×3 d6 grid — confirm EVERY cell (not top-N), rank by d6 mean vs the C1 anchor:
>
> |  progress \ base | 360                 | 540       | 720      |
> |---|---|---|---|
> | **224** | C2              | (new)     | C3       |
> | **336** | (new)           | (new)     | (new)    |
> | **448** | C1 (sanity ≈1000) | (new)   | (new)    |
>
> Locked 4 terms (all 9 cells): defense **6** · spread **4** · mobility **9** ·
> threat **130**. Anchor: **`c1_incumbent.json` = C1** (progress 448 · base 360 ·
> defense 6 · spread 4 · mobility 9 · threat 130); ratings on the **C1 scale
> (1000 = C1)**. C1's own cell (448/360) = self-match sanity check → ≈1000.
>
> **Params:** `--grid --progress 224,336,448 --base 360,540,720 --defense 6
> --threat 130 --pin spread=4 --pin mobility=9` (→ 3×3 = 9 cells). **25 games/seed**
> (was 50), d6, double-blind per order, **`--skip-proxy`** (drop the pure-overhead d4
> screen; confirm all 9 cells directly at d6, rank by d6 mean).
>
> **`--skip-proxy` (added this session):** `tools_budget_search.py` always ran a d4
> proxy screen before confirm. Since T-17e confirms ALL cells, the proxy filters
> nothing → pure overhead. `--skip-proxy` makes a pure d6 pass first-class (additive;
> default proxy-then-confirm unchanged). With it: banner prints `proxy: SKIPPED`, all
> cells are confirmed, they're ranked by d6 mean, and the verdict reports the d6
> (not proxy) mean. `cand_w*.json` still written for every cell; proxy CSVs omitted.
>
> **Work estimate:** per cell 25 × 3 seeds × 2 orders = **150 d6 games** (was 300);
> 9 cells × 150 = **1 350 d6 games** (d4 proxy dropped via --skip-proxy). At
> ~10 s/game (T-16) → **~3.5–4 h**, 12 workers, **background** (`py -u`,
> `is_background: true`).
>
> **Accepted tradeoff (user, 2026-11-03):** 50→25/seed → each seed's sample rougher
> (~25 finished/ongoing) → cross-seed variance up → `std<20` gate easier to fail.
> Ranking/mean stay fine; a cell near the gate could flip on noise. OK for a
> "call-it-a-day" screen — **but if a winner lands ~right at +50, add a 2nd-seed
> tiebreak** first. Optional trim: drop the 448/360 self-match → 8 cells (~3.5 h).
>
> **Accept gate (on the C1 scale):** `mean−1000 > 50 AND std < 20 AND min > 1000`
> vs the `c1_incumbent.json` anchor. Report **absolute ≈ 1285.94 + (C1-scale mean −
> 1000)** when a winner emerges. If a cell clears → write it to `best_weights.json`
> (or promote to new incumbent per user) — else **C1 stays the incumbent**.
>
> **Launch (background, one line — maxmove uses the tool default 480):**
>
> ```
> py -u tools_budget_search.py --grid \
>   --progress 224,336,448 --base 360,540,720 --defense 6 --threat 130 \
>   --pin spread=4 --pin mobility=9 --anchor-file c1_incumbent.json \
>   --confirm-games 25 --confirm-depth 6 --workers 12 --skip-proxy --out-dir _t17e
> ```
>
> Poll `_t17e/budget_search_log.csv`.
>
> **STATUS (2026-11-04, IN FLIGHT — launched, not yet complete; user closing session
> with it still running).** Launched ~22:57 UTC as background job `bg_bdf55760`
> (pid 23920) → `_t17e/` per-cell `cand_confirm_*_s*.csv` + `_t17e_launch.log`.
> Last check: **39/54 order-CSVs (≈72%)** — cells 0–5 done, cell 6 (C1 self-match)
> on seed 2, cells 7–8 queued.
>
> **⚠ RECOVERY / CLOSE-SESSION NOTE (read this first):** `tools_budget_search.py`
> does **NOT** resume — a fresh launch **redoes ALL 9 cells** (overwrites `_t17e/`).
> So if it's still running, **let it finish before closing** if you can.
> **Done =** `_t17e/budget_search_log.csv` AND `_t17e/budget_search_verdict.txt`
> both exist (written only at the very end), OR `_t17e_launch.log` ends with a
> `VERDICT ->` line. If the background process was killed early: cells 0–5 per-cell
> CSVs are still on disk, but the final log/verdict/`best_weights.json` are **not**
> written → either relaunch the command above (clean full rerun), or aggregate
> cells 0–5 from the per-cell CSVs for a partial read.
>
> **Partial means (C1 scale, 1000 = C1), cells 0–5 complete:**
> 0 224/360 → **894.9** · 1 224/540 → **997.7** · 2 224/720 → **910.0** ·
> 3 336/360 → **951.0** · 4 336/540 → **974.5** · 5 336/720 → **981.4**.
> Best so far = **224/540 (997.7), still UNDER the anchor** → nothing clears the
> `mean−1000>50` gate yet → **C1 would stand** on current partial data. base=540
> column is the strongest within each progress row.
> **Watch:** C1 self-match (cell 6 = 448/360) seed 5000 = **951.04** — a true
> self-match should center ≈1000; confirm its 3-seed mean (within ~1 SE of 50 games
> = not alarming, but if it settles ~950 there's a small residual candidate-side
> bias; the *relative* ranking + gate still hold).
>
> **Also this session:** added a defensive UTF-8 stdout guard to `tools_budget_search.py`
> (`main()` reconfigures stdout/stderr to UTF-8) — the `--skip-proxy` NO-ACCEPT verdict
> prints `—`/`−` (non-ASCII) on this box's cp1252 console and would crash the run at the
> final print. Guard = 6 lines, **no logic change**, byte-compiled clean.
>
> **STATUS (2026-11-03, STEP 1 boundary-hunt — superseded by STEP 1c/1d above):**
> two d4 passes (both exit 0): Pass 1 = `t17_boundary_hunt.py` (`_t17b/`), Pass 2 =
> `_t17b_pass2.py`. CONVERGED local-max axes (consistent both passes): **threat 130 ·
> defense 6 · spread 4 · mobility 9**. UNCONVERGED then: **progress** (d4 argmax
> 42→80, "still rising") and **base** (bimodal 90/120/150) — this is what drove the
> wider STEP 1c/1d that found the true progress peak @448 and base 360. SANITY OK
> then: incumbent-exact weights → R 1011.59 ≈ 1000. Full curves: `_t17b/summary.txt`,
> `_t17b_pass2/summary_pass2.txt`.
>
> **STATUS (2026-11-02, history): T-17a, Pass A, and Pass B all DONE at d4 (proxy tier).**
> (T-17a 24-pt box: top cand p18/b90/d18/t90 → R 1082.59; Pass A: progress
> still rising @24, threat peak ~110, base peak 75–90, defense weakest; Pass B:
> grid1 progress24×threat130 → 1161.21, grid2 progress24×base105 → 1133.61 — all
> boundary corners.)
>
> **T-17a DONE (this session):** the 3×2×2×2 box was proxy-screened
> (d4/15, double-blind, cap 480, anchor = incumbent `best_weights.json`,
> `--skip-confirm`, exit 0). Full log `_t17_grid_proxy.log`; CSV `_t17_grid/budget_search_log.csv`.
> Ranking vs incumbent (1000 = the 1285.94 model; grid order = p/b/d/t, spread 4 /
> mobility 9 pinned): **1) cand[23] p18/b90/d18/t90 → R 1082.59 (8W-1L-21D);
> 2) cand[21] p18/b90/d12/t90 → 1070.44 (10-4-16); 3-way tie at 1011.59: cand[5]
> p12/b90/d12/t90, cand[8] p15/b60/d12/t45, cand[18] p18/b60/d18/t45; six points
> at 1000.00 incl. the incumbent-nearest point cand[15] p15/b90/d18/t90 (4-4-22) —
> sanity check passed exactly.** Surface reads: top moves progress/base/defense
> **UP from the incumbent**; threat=90 holds 4 of the top 6; low corners
> (p12/b60/d12) 880–950.
>
> **T-17 PASS A DONE (this session):** 4 single-axis main-effect d4 scans
> (7 values each, `--skip-confirm`, anchor = incumbent, spread 4 / mobility 9
> pinned), ~13 min total, all 4 runs exit 0. CSVs `_t17_sweepA..D/budget_search_log.csv`,
> logs `_t17_passA/{A_progress,B_defense,C_threat,D_base}.log`. Read:
> **progress** monotone rising, argmax at the *edge* of the tested range (24) — no
> interior peak found. **defense** flat low (4–12 best, ~1035), mild drop above
> 15 — weakest lever. **threat** clear peak at ~110 in this context (defense 15
> held), down on both sides. **base** peaked flat at 75–90, declining both sides
> (incumbent's own 91.49 sits right there).
>
> **T-17 PASS B DONE (this session):** 2 two-axis d4 interaction grids
> (progress×threat, progress×base; 12 points each; `--skip-confirm`, anchor =
> incumbent, spread 4 / mobility 9 pinned, defense 8 / base 90 [grid 1] and
> defense 8 / threat 110 [grid 2] held from Pass A's argmaxes), ~10.5 min, both
> runs exit 0. CSVs `_t17_pair1/budget_search_log.csv`,
> `_t17_pair2/budget_search_log.csv`, logs `_t17_passB/pair{1,2}_*.log`. Read:
> **grid 1 (progress×threat)**, defense 8 / base 90 held —
> best cell **progress 24 × threat 130 = R 1161.21**, the strongest single point
> found anywhere in T-17 so far, at the *outer corner* of the drawn box (both
> progress and threat still rising at their tested maxima — no interior maximum
> found for either in this regime). **grid 2 (progress×base)**, defense 8 /
> threat 110 held — best cell **progress 24 × base 105 = R 1133.61**, a close
> second, again at the top progress / top base corner. **spread and mobility
> have NOT been surface-mapped as free parameters anywhere in T-17 yet** — they
> have only ever been pinned (4 / 9) and carried over from T-16's "roughly
> neutral" read.
>
> **WHY THE ROUTE CHANGED (user, 2026-11-02, after seeing Pass A+B):** the
> current best (progress 24 × threat 130) is a **boundary corner** of a small,
> hand-drawn box — progress, threat, and base are ALL still rising at their
> tested maxima in exactly the regime that produced the top cell, and 2 of the
> 6 parameters (spread, mobility) were never actually searched at all in T-17.
> Confirming that corner at d6 now would likely just be a "best of what we
> bothered to draw," not the true apex. Cheaper and safer to **find real
> internal boundaries for every parameter first** (one cheap d4 line-scan per
> parameter, expanding outward until the curve turns down), then spend the d6
> budget around a point with that verification.
>
> **Route history:** T-16 random sweep showed **progress-dominant**; T-17a
> confirmed the incumbent sits below a grid corner. Proxy is a filter, not a
> top-ranker (T-16 top-2 swap) → mapping must precede d6.
>
> **Why not a blind random re-roll:** T-16's top-10 vs bottom-10 factor averages show
> progress carries the lift (strong ~2.55× vs weak ~0.65×), defense trends *down* from
> its default, threat is bimodal (2 winners at ~43, 1 at ~92), while spread & mobility
> are ~neutral. A grid that *pins* the flexible-weak terms and *lattices* the
> signal-bearing terms spends the same/less compute to actually explore the interesting
> region (and to resolve the threat-low vs threat-high split that random sampling can't).
>
> **Grid box 3 × 2 × 2 × 2 = 24 candidates** (`spread` pinned at 4, `mobility` pinned
> at 9 — the T-16 winners' near-values — both held fixed):
>
> | term (weight) | values | n |
> |---|---|---|
> | progress | 12 · **15** · 18 | 3 |
> | defense  | **12** · 18 | 2 |
> | threat   | **45** · 90 | 2 |
> | base     | 60 · **90** | 2 |
>
> (bolded = the T-16 best `cand[3]` values — `progress 15, defense 15, threat ~45,
> base ~90` — so the confirmed 1286 point sits inside the grid and the neighbours
> bracket it. Use the nearest box values; exact best is progress 14.18 / defense
> 14.77 / threat 91.60 / base 91.49.)
>
> **Anchor (DECIDED 2026-11-02 by user):** every T-17 head-to-head plays against the
> **incumbent** `best_weights.json` (= T-16 best cand[3], 1285.94), NOT `weights_v0.json`.
> Rationale: at +286 vs the 1000 anchor (~84% score), extra-strength candidates
> separate by only a few score-points (poor resolution near the S→1 region) and games
> blow out early; against the incumbent games sit near 50/50, last longer, and more
> finish at the cap-480 line — the condition where our double-blind h2h is most
> trustworthy. Elo bridge for reporting: **absolute ≈ 1285.94 + (incumbent-scale R − 1000)**;
> the grid point nearest the incumbent's weights (progress 15, defense 12, threat 90,
> base 90) is the built-in sanity check and should read ≈1000 at confirm.
>
> **Plan (next agent) — ROUTE CHANGED 2026-11-02 by user: extensive d4 passes → report →
> d6 confirm only after review.**
>
> **Code is DONE (this session, smoke-verified, T-17a run exit 0):**
> `tools_budget_search.py` has `--grid` (absolute-weight cartesian grid), `--pin TERM=VALUE`,
> `--anchor-file PATH` (default stays `weights_v0.json`). No new code needed for the passes.
> **Do NOT re-run the T-17a 24-point box** (results above are the record).
>
> Common switches for every pass below:
> `py -u tools_budget_search.py --grid <axis lists> --pin spread=4 --pin mobility=9
>   --proxy-games 15 --proxy-depth 4 --maxmove 480 --workers 12
>   --anchor-file best_weights.json --skip-confirm`
> (≈30 s/candidate, measured: 24 pts ≈ 25 min. Run each pass with
> `is_background: true`, log to file; `py -u` + cp1252 guard already in the script.)
>
> **PASS A — single-axis main-effect scans (~28 points, ~15 min).** Hold the other
> 3 axes at P* = (progress 15, defense 15, threat 90, base 90) and sweep each axis
> alone, 7 values each (4 separate runs, `--out-dir _t17_sweepA`…`_t17_sweepD` or one dir):
> - progress: `--progress 6,9,12,15,18,21,24   --defense 15 --threat 90 --base 90`
> - defense:  `--progress 15 --defense 4,8,12,15,18,22,26 --threat 90 --base 90`
> - threat:   `--progress 15 --defense 15 --threat 20,37,55,72,90,110,130 --base 90`
> - base:     `--progress 15 --defense 15 --threat 90 --base 30,45,60,75,90,105,120`
> Report the 4 R-curves (mono / peaked / bimodal, and the argmax of each).
>
> **PASS B — 2D interaction grids over the strongest axis pairs (~36 points, ~20 min).**
> Pick the two strongest axes from Pass A (a-priori suspects: progress & base, both
> monotone-up in T-17a; second pair per Pass A). Pin the other 2 axes at their Pass-A
> argmax, then e.g. `--progress 12,15,18,21 --base 60,90,120` and
> `--progress 12,15,18,21 --threat 60,90,120` (each as one run; `--out-dir _t17_pair1/2`).
> This is the pass that settles threat-low vs threat-high at the *raised* base/progress
> level — T-17a's bimodality was measured around the incumbent, not the top.
>
> **SURFACE REPORT (before any d6):** combined main-effect curves + interaction
> grids → identify the local-maximum region and the shape (peaked/monotone/bimodal)
> of each axis at the frontier. Present the full table to the user and **STOP**
> (staged rule: user reviews proxy evidence before deep sims; may reorder/add passes).
>
> **THEN d6 confirm — ONLY on user's shortlist (top ~5 incl. neighbours of the top
> for local structure):** same tool, flip to `--confirm-games 50 --confirm-depth 6
> --max-confirm 5` (drop `--skip-confirm`), same seeds scheme `5000+j`, same
> `--anchor-file best_weights.json`; ~30 min per confirmed candidate. Accept iff,
> **on the incumbent scale**, `mean−1000 > 50 AND std < 20 AND min > 1000` (beats
> the incumbent by a stable 50+); report **absolute ≈ 1285.94 + (mean − 1000)**;
> write the winner to `best_weights.json` (only by construction is it a strict beat).
>
> **Done-when:** a d6/3-seed confirmed winner (on the incumbent scale: mean−1000>50,
> std<20, min>1000 — i.e. strictly beats the 1285.94 incumbent by a stable 50+;
> report absolute ≈ 1285.94 + (mean−1000)) is written to `best_weights.json`, the
> incumbent sanity point reads ≈1000 at confirm, always-green gates still green,
> and "Where we are" refreshed to T-17 ✅ + the new best. **If nothing clears the
> gate (the incumbent may itself be the local optimum), STOP and report** the
> 24-point proxy surface + the confirmed top-5 table — that localizes the eval
> landscape and is a valid outcome (the incumbent then stays as best).

**Rollback (T-13..T-17):** T-13 (no files).
T-14: revert `evaluate.rs` (f64 → i32 in the struct, the parse, and the save
formatting, plus `side_score` return, and the `evaluate`/`evaluate_with`/
`evaluate_pair` casts); revert `train_weights.py::to_raw_weights` (re-add
`int(round(...))`; restore the `all(v==0)` collapsed check); revert
`ai/src/bin/selfplay.rs` head-to-head log to int formatting.
T-15: revert the seed plumbing in `tools_tune.py` (restore single `--seed`,
remove `--seed-base`).
T-16: delete `tools_budget_search.py`, `budget_search_log.csv`,
`best_weights.json` (keep `cand_w*.json`/`cand_*.csv` if they're the only
evidence of the ACCEPTED result — they back the RESULT table above).
T-17: revert the `--grid` / `--candidates-file` / `--pin` / `--anchor-file`
additions in `tools_budget_search.py`; delete the grid batch's `grid.json`,
`cand_*.json/csv`,
and any grid batch log/verdict.

**DEFERRED (after T-11 is verified, only if still wanted — the old T-12 extras):**
- **Release polish:** a small `build_release.bat` (mirror `kego_engine_v2/deploy.ps1`):
  `cargo build --release` in `ai/`, copy `target/release/{ai,selfplay}.exe` to
  `check_halma/`, re-run `ai/tools_ai_smoke.py`, then `py halma8_test.py`.
- **Distribution (optional):** bundle `dist/` with `check_halma.exe` + `ai.exe` +
  `selfplay.exe` + `weights.json` + a short `README_AI.txt`.
- **Gate:** `cargo test` green; smoke tests green; `py halma8_test.py` 36 OK.
- **Done-when:** one command (a) builds, (b) smoke-tests, (c) leaves a complete `dist/`.

---

## Decisions locked in (single source of truth)

| Question | Decision |
|---|---|
| FFI shape | **Subprocess text protocol** (kego-proven UCI-flavored). No `.dll`, no PyO3/CTypes. |
| Wire format | 64-char board + `turn` + `chain` token; `step R,C NR,NC` / `jump R,C MR,MC NR,NC` / `stop`. Do not change. |
| Search | World-score min-max (P1 max / P2 min), iterative deepening, TT, time/stop. Not raw negamax (see design notes). |
| Borrow from kego | alpha-beta + ID + TT + time/stop + `splitmix64` keys + make/unmake + UCI protocol + `tools_smoke.py`. |
| Do NOT port | No captures/check/draw machinery; **no pruning of any kind** (user-mandated). |
| The hard part | **The evaluation function** (T-5) — build from principles. |
| Python deps | Training side only, and in practice stdlib-only on this box. Engine stays std-only. |
| Never changes | `halma8.py`, `halma8_test.py` (canonical oracle). |
| Two subtle rules | **Chain-revisit ban** + **full-and-invaded win** — tested in Rust and cross-checked. |

## Open items (do NOT decide inside the tasks above)

- **Draw handling:** out of scope — halma has no draws by rule.
- **Neural eval:** T-5/T-10 is *linear* eval + logistic fit. A neural eval (NNUE-style)
  is a separate project (Part 3), needs a real training stack and a different loop.
- **Multi-threaded search:** overkill for a 2-player local game.
- **GUI "AI difficulty" slider:** trivial follow-up, not in scope.

## Rollback

- `ai/` is additive and self-contained; delete it and everything reverts.
- `halma_ai.py`, `train_weights.py`, `tools_tune.py`, `weights.json`, and (T-11)
  `tune_log.txt` / `elo_curve.svg` are new files — deleting them reverts.
- `halma8_gui.py` has one small addition (the AI toggle + `_ai_take_turn`). Revert
  from the Part 1 file if it breaks; every other behavior is identical.
- **No `halma8.py` or `halma8_test.py` changes at any point in this plan.**
