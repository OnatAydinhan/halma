# check_halma — Part 3 (Neural AI) Plan (agent handoff)

An AlphaZero-style neural network learns to play check_halma from raw board state —
no hand-coded eval, no hand-coded features. The network discovers what matters
through self-play.

## HANDOFF (read first)

- **What this is:** Part 3 of check_halma. A standalone PyTorch neural network under
  `azero/` that replaces the 6-term linear eval with a learned value + policy.
  Parts 1 (GUI) and 2 (Rust AI, `ai/`) are DONE and untouched by this plan.
- **Where we are:** See the "Where we are (updated as tasks land)" section below —
  all code tasks T-0 to T-14 are DONE (T-14 = endgame curriculum: near-win start
  states give the value head real +/-1 outcomes, plus a held-out finish-rate metric).
  Remaining work is launching/monitoring training (T-8). `halma8.py` (rule oracle)
  and the Rust engine (`ai/`) are the reference for all rule/legality questions.
- **First command to run:** `nvidia-smi` — note the CUDA driver version, then install
  the matching PyTorch build (see T-0).
- **Always-green gates (before AND after every task):**
  - `py halma8_test.py` → 36 OK
  - `cargo test` (in `ai/`) → green
- **Hard rules:**
  1. `halma8.py` is the rule oracle. The NN never re-implements legality or
     move application. All rule queries go through `Halma8` methods.
  2. The NN is standalone Python under `azero/`. It does NOT modify `ai/`,
     `halma8.py`, `halma8_gui.py`, or any existing file outside `azero/`.
  3. Every task has a test gate that must be green before the next task starts.
  4. No non-ASCII in any `print()` call (Windows cp1252 landmine).
  5. Checkpoints go in `azero/checkpoints/`. The training run must be resumable.
  6. Update this file as each task lands (tick the heading, refresh the
     "Where we are" line) — same rule as `plan_ai.md` / `plan_gui.md`.
- **Rollback path:** delete the `azero/` directory. Nothing else is affected.

## Where we are (updated as tasks land)

- T-0 done (2026-11-08): PyTorch 2.14.1+cu126 on 4090, numpy 2.5.4, azero/ created.
- T-1 done (2026-11-08): azero/board.py + tests (10/10 pass), halma8_test.py 36 OK.
- T-2 done (2026-11-08): azero/model.py (HalmaNet, ~1.23M params) + tests (8/8 pass), halma8_test.py 36 OK.
- T-3 done (2026-11-08): azero/mcts.py (MCTSNode + mcts_search, PUCT, terminal + stuck detection, dirichlet noise, deepcopy states) + tests (5/5 pass). azero suite 23/23, halma8_test.py 36 OK. Next: T-4 self-play.
- T-4 done (2026-11-08): azero/selfplay.py (play_one_game + self_play_batch; negamax value fill; discard no-winner games; optional start_state for testing) + tests (6/6 pass, incl. a deterministic forced-win value-fill check). azero suite 29/29, halma8_test.py 36 OK.
- T-5 done (2026-11-08): azero/train.py (train_step, train with cosine LR, save/load checkpoint) + tests (6/6 pass). azero suite 35/35, halma8_test.py 36 OK.
- T-6 done (2026-11-08): azero/elo.py (score_to_elo, evaluate_elo, K=64), azero/pipeline.py (run_pipeline with baseline, self-play, train, checkpoint, Elo eval, log), azero/run.py (CLI with --resume) + tests (7/7 pass). azero suite 41/41, halma8_test.py 36 OK. Next: T-7 evaluation vs Rust engine.
- T-7 done (2026-11-08): azero/evaluate.py (evaluate_nn_vs_rust + main CLI). NN vs Rust engine (best_weights.json, 448/540) via the halma_ai.HalmaAI subprocess driver; halma8 stays the legality oracle, per-game abort guard. NN moves use MCTS (add_noise=False). Seating is double-blind by default (NN=P1 in odd games, P2 in even -- cancels the ~+58 Elo first-move edge; `--no-alternate` for NN=P1 only); default `max_moves=480`. Gate `py azero/evaluate.py --games 2 --sims 10 --max-moves 16 --movetime 80 --device cpu` runs both seats clean and prints NN/Rust/ongoing. azero suite 41/41, halma8_test.py 36 OK. Next: T-8 launch (background) + monitoring.
- T-4 REVISED (2026-11-08, session 3): **self-play deadlock found and fixed.** The "attacker piece in a full enemy base" win is ~never reached by a weak model (measured: 800/800 random games -> 0 winners even at 400 moves; 200/200 games DID fill an enemy base to 9/9 but always with the DEFENDER's own pieces). Old `play_one_game` discarded ALL no-winner games -> ~0 samples -> the net trains nothing (the log's repeated `samples=0`). **FIX (user-approved):** `azero/selfplay.py` now KEEPS no-winner games, using the dense MCTS root-value (`root.value_sum/root.visits`, negamax) as the value target; true +/-1 outcome is still used when a winner exists. `mcts_search` is called with `return_root=True`. Test gate updated in `test_selfplay.py` (`test_tuple_shapes` value relaxed to [-1,1]; NEW `test_no_winner_yields_dense_data`). GATES GREEN: azero 42 passed, halma8 36 OK. Smoke: samples 0 -> 500.
- T-3 BUGFIX (2026-11-08, session 3): `azero/mcts.py` `compute_priors` built the illegal-action mask on CPU (`torch.full` w/o device) -> crashed on CUDA (`cuda:0` vs `cpu`). Fixed: mask now built with `device=logits.device`. CPU test gates never caught it; surfaced by the first CUDA run.
- T-8 IN PROGRESS (2026-11-08, session 4): **NOT yet launched — handed to the user to run in isolation** (they close the ollama/assistant session first so the 4090 is free; VRAM is a non-issue — the net is ~1.23M params). Prep done this session: gates re-verified green (halma8 36 OK, azero 42 pass); throughput measured at **~1.6 ms per (move x sim)** (`_probe_small.py`, under contention — the run is Python-MCTS-bound, so an isolated free-GPU launch runs at about this rate). Chosen launch: `py -u azero/run.py --rounds 10 --games 120 --sims 120 --epochs 5 --max-moves 100 --device cuda` (~50 min/round incl. the ~10 min Elo eval, ~8h for 10 rounds). Monitor: `azero/training_log.txt` (watch `elo=` climb off 1000) + `nvidia-smi`. Resume after a crash: `py -u azero/run.py --resume azero/checkpoints/round_N.pt --rounds <remaining> --games 120 --sims 120 --epochs 5 --max-moves 100 --device cuda`.
- T-9 done (2026-11-08, session 6): **parallel self-play workers implemented.** `azero/selfplay.py` gained a module-level `selfplay_worker(args)` (spawn-safe: rebuilds `HalmaNet` from a CPU state dict, plays its chunk, returns CPU tuples); `azero/pipeline.py` gained `_collect_selfplay(...)` (splits `games_per_round` into N contiguous chunks, runs one `selfplay_worker` per chunk under `mp.get_context("spawn").Pool`, concatenates) plus a `workers` param on `run_pipeline`; `azero/run.py` gained `--workers` (default 1 = unchanged sequential behavior). GATES GREEN: azero **44 pass** (was 42), halma8 **36 OK**; end-to-end `--workers 2` and `--workers 1` both log `samples=80` for `--games 8` (parity). Throughput gate #4 (CUDA `--workers 1` vs `4`) left as a user-launched isolated measurement (report-only). Next: T-10 batched MCTS leaf evals.
- T-10/T-11 PLANNED (2026-11-08, session 5): **two self-play speedups spec'd as handoff tasks — no code yet, implement in a later session, each with its own test gate.** The run is measured **Python-MCTS-bound** (~1.6 ms/move·sim) so the 4090 sits mostly idle. T-10 = batch B leaf evals per search into one `model(batch)` call in `mcts.py` (amortizes batch-1 launch overhead, finally loads the GPU). T-11 = a cheap `halma_clone(state)` helper to replace `copy.deepcopy` per child (removes the most-allocated object in the hot loop). See their sections near the end of this file; they compose per-process, on top of T-9's cross-core parallelism.
- **T-10 DONE (2026-11-08, session 7):** batched MCTS leaf evals implemented in `azero/mcts.py`. `mcts_search(..., batch_size=1)` is a new param; **`batch_size=1` is the default and bit-exactly reproduces the pre-T-10 search** (proven by a new reference-comparison test). Batched path uses **virtual losses** (Leela-Zero/KataGo style) so the B descents in a step see each other and don't pile onto one node; one `model(B,7,8,8)` forward pass per iteration. Threaded through: `selfplay.py` (`mcts_batch` on `play_one_game`/`self_play_batch`/`selfplay_worker`), `pipeline.py` (`run_pipeline` + `_collect_selfplay`, both sequential and the worker-args path), `run.py` (`--mcts-batch`, default 1). **GATES GREEN: azero 45 pass (was 44, +1), halma8 36 OK.** Key finding: with an *untrained (near-constant)* net the `best_action` legitimately differs between `batch_size=1` and `>1` (PUCT tie-breaking / visit-order is all the signal there is) — so the test gate asserts `batch_size=1 == original` **exactly** (vs a reference copy of the old loop) plus `batch_size>1` sound/deterministic/legal, rather than cross-batch-size equality (see T-10 section). **TIMING GATE DONE (2026-11-08):** `py _t10_timing.py` on the 4090 → `batch=1`: 7200 fwd_calls / 2.653 ms per(move·sim); `batch=32`: 240 fwd_calls / 0.179 ms → **30× fewer forward calls, ~14.8× lower wall-clock**. T-10 fully DONE (code + equiv gate + timing gate all green).
- **T-11 DONE (2026-11-08, session 8):** `halma_clone(state)` added to `azero/board.py` (uses `Halma8.__new__(Halma8)` + 8-field copy: grid row-wise, turn, winner, turn_no, chain_active, chain_banned set-copy, chain_starter, chain_current — all shared refs where immutable). `azero/mcts.py` swaps both `copy.deepcopy` call sites (root init + per-child expansion) to `halma_clone`; drops `import copy`. New tests: `test_board.py` — `test_halma_clone_fields_match` (all fields equal on a mid-chain state) + `test_halma_clone_independence` (mutating clone's `grid`/`chain_banned` does not affect source, and vice versa); `test_mcts.py` — `test_mcts_halma_clone_equiv` (same `best_action` as a `copy.deepcopy`-based reference MCTS, 2 positions x 3 seed/noise combos). **GATES GREEN: azero 48 pass (was 45, +3), halma8 36 OK.** All planned code tasks T-0 to T-11 are complete; remaining work: launch the training run (T-8 handoff command, user-launched in isolated free-GPU session).
- **T-8 PROGRESS (2026-11-09):** rounds 1-2 launched/done, **both Elo 1000.0** (round 1 score 0.500 vs baseline; round 2 confirmed all-draw in standalone `elo.py` — the net still can't decide games within `max_moves=100`, so more eval games add no signal — the lever is **more rounds**). Eval now runs vs the **champion** (best-trained net, resolves to latest `round_*.pt`), not the frozen baseline. NEXT: lean round-3 launch (command in the T-8 section) — user-launched isolated, free GPU.
- **T-12 DONE (2026-11-09):** **teacher value target added — the value head now gets a real training signal.** Root cause of the flat ~1000: within `max_moves` every game is a draw, so the value target is the near-constant dense MCTS estimate (~0 for a weak net) and the head has no informative gradient. `azero/value_target.py` (new) distills the 6-term eval (`best_weights.json`, 448/540) into a mover-perspective value in [-1,1] (`eval_signed` negamax + `tanh(eval/scale)`, `scale` auto-calibrated to p85 |eval| over random-walk positions). `selfplay.py` gained `value_teacher`/`teacher_weights`/`teacher_scale` on `play_one_game`/`self_play_batch`/`selfplay_worker` (W=0 → byte-identical original target; W=1 → value head distills the eval); `pipeline.py` resolves the weights path (default `best_weights.json`) + auto-calibrates scale and threads it through both `_collect_selfplay` paths; `run.py` gained `--value-teacher`/`--teacher-weights`/`--teacher-scale`. **Smoke:** W=0 mean|v|=0.019 (dead) → W=1.0 mean|v|=0.509, span [-0.963,0.962]. **GATES GREEN: azero 55 pass (was 48, +7), halma8 36 OK.** Use for the round-3 launch: add `--value-teacher 0.5` (jump-start with `1.0`, then fade to 0 for pure self-play). See the T-12 section.
- **T-13 DONE (2026-11-09):** **distance-to-enemy-base (centroid) shaping target added — a focused "getting closer to the win" value signal on top of the T-12 eval teacher.** `azero/value_target.py` gained `piece_centroid`/`centroid_distance`/`distance_advantage` (tanh of the mover-vs-defender centroid-distance difference to the base each advances toward; symmetric, negamax-consistent) + `calibrate_distance_scale` (p85 |adv| over random walks). `selfplay.py`/`pipeline.py`/`run.py` gained `dist_teacher`/`dist_scale` (default 0 = off) that convex-blend up to TWO shaping terms — `W_eval*eval + W_dist*dist + W_base*base`, residual on the base — so the two teachers compose without double-weighting. **Magnitude is a dial:** the pipeline auto-calibrates `dist_scale` to 1.0, giving mean|v| ~0.11 at 20 moves (the 0.027 figure was at scale 4.0); `--dist-scale 0.5` doubles it to ~0.21 with zero saturation. The payload is the directional sign — the "push forward" gradient the all-draw run lacks. **GATES GREEN: azero 65 pass (was 62, +3 new selfplay blend/off/convex gates; the 7 T-13 value_target tests counted in the prior 62), halma8 36 OK.** Use in the launch: add `--dist-teacher 0.5` alongside `--value-teacher 0.5`; fade both to 0 once the net wins on its own. See the T-13 section.
- **T-14 DONE (2026-11-10):** **endgame curriculum — teach the net to FINISH, not just approach.** T-12/T-13 taught approach (closer + more invaders) but the net caps at *1 invader in a 6/9 base*; within `max_moves` every fresh-board game is a draw, so the value head never sees a true terminal ±1. T-14 self-plays **from positions already 1–3 moves from a win** (a base at 8/9 full with ≥1 invader, to-move = the attacker), so those games' value target is the real ±1 outcome, plus a held-out **finish-rate** metric the full-game Elo is blind to. **Implemented + gated:** `azero/endgame.py` (`make_endgame_positions` + `serialize`/`deserialize`), `azero/selfplay.py` (`start_states` batch + spawn-safe worker), `azero/pipeline.py` (`endgame_n`/`endgame_full` supplement on top of `games_per_round`, both sequential + worker paths), `azero/run.py` (`--endgame`/`--endgame-full`), `azero/finish_rate.py` (held-out convert metric), `finish_rate=` logged per round; `tests/test_endgame.py` + `tests/test_finish_rate.py`. **GATES GREEN: azero 76 pass (was 65, +11), halma8 36 OK; CLI `--help` exposes all flags.** Launch recipe (teachers OFF) in the T-14 section below.
- **T-15 DONE (2026-11-12):** **T-14 deadlock fix — the endgame near-win games were NOT actually producing the true ±1 target the curriculum depends on.** Diagnosis (probes `azero/_probe_conv.py` / `_probe_evalue.py` / `_probe_evalue_fix.py`, round_6.pt): the net's softmax prior on the winning move is near-uniform (**0.002–0.108**, mostly ~0.01) and its value head reads ~0 (dead), so the net does not "know" the win; at the fresh-board **low `n_sims` + Dirichlet-noise** default the low-prior winning action gets starved to ~0 visits and the game never converts → the value target stays ~0 (chicken-and-egg deadlock stays alive even on the curriculum). Measured conversion of a win-in-1 position (noise off): **30 sims → 3/10, 60 → 6/10, 200 → 9/10**. **FIX:** the endgame path now runs **noise-OFF at a dedicated (higher) sim budget**, distinct from the fresh-board `--sims`/noise (fresh games keep noise on for diversity). Wired: `azero/selfplay.py` `selfplay_worker` (reads `endgame_n_sims`/`endgame_add_noise` from args), `azero/pipeline.py` `_collect_selfplay` (both the sequential path and the worker-args path thread the two knobs), `azero/run.py` (`--endgame-sims` default 200, `--endgame-noise` off by default). **Verified empirically (4090):** `_probe_conv.py` 9/10 convert at 200 sims; `_probe_evalue_fix.py` end-to-end through `_collect_selfplay` → **sequential path 20/68 samples are genuine ±1 (18×+1, 2×−1), worker path 8/8 all +1** (was ~0 before the fix). **GATES GREEN: azero 78 pass (was 76, +2 new T-15 gates in `test_pipeline.py::TestEndgameCurriculum`: `test_endgame_uses_dedicated_budget_and_noise_off`, `test_endgame_defaults_are_noise_off`), halma8 36 OK.** See the T-15 section below.
- **DIAGNOSIS (2026-11-12, NO code change — check only) — MCTS `select()` sign bug; the likely real reason endgame conversion caps below 1.0.** Confirmed by reading `azero/mcts.py` against the value convention (not just the probes). The code stores node values in **negamax** (each node from its own to-move perspective): `backprop` uses `sign = +1 if node.turn == leaf_turn else -1`; and `model.py` ("value convention is negamax: +1 = player to move wins"), `value_target.py`, and `selfplay.py` (dense target = `root.value_sum/root.visits`) all commit to it. But `select()` computes `q = child.value_sum / max(child.visits, 1)` with a **positive** sign — i.e. the *child's* to-move perspective = the **opponent's** value — so PUCT is selecting the move best for the opponent (worst for the side to move). Negamax PUCT must negate: `q = -(child.value_sum / max(child.visits, 1))`. **Why it stayed hidden:** with a dead value head (~0 on near-wins) `q≈0` for every child, so the PUCT *exploration* term dominates and conversion still rises with sims (30→3/10, 60→6/10, 200→9/10) — that rising-with-sims curve is exactly the T-15 signature, so it was (reasonably) attributed to sims/noise and that fix genuinely helped; the sign only fully bites once the value head carries real signal, where it steers selection *away* from the winning child. **Why fixes kept "failing":** the correct fix is ONE sign flip in `select()`; a flip in `backprop` instead would corrupt the negamax leaf/terminal values and the selfplay dense target (both correct today) and regress. **Next agent (NOT applied here):** flip that one line in `azero/mcts.py::select`; then re-run `azero/_probe_mcts.py` + `azero/_probe_conv.py` (winning child should get the most visits; conversion → 10/10) and add a test that a forced 1-move win is selected.

---

## Architecture overview

### Input encoding (7 channels, each 8x8)

| channel | meaning | values |
|---------|---------|--------|
| ch0 | P1 (red) pieces | 1 where grid==1, else 0 |
| ch1 | P2 (blue) pieces | 1 where grid==2, else 0 |
| ch2 | whose turn | all 1s if turn==P1, all 0s if turn==P2 |
| ch3 | chain active | all 1s if chain_active, else all 0s |
| ch4 | chain banned squares | 1 where chain_banned, else 0 |
| ch5 | chain current position | one-hot at chain_current, else 0 |
| ch6 | chain origin | one-hot at chain_starter, else 0 |

Tensor shape: `(batch, 7, 8, 8)` float32.

### Action space (fixed, 417 actions)

A CNN needs a fixed output size, so the action space is enumerated once and
reused for every position. Illegal actions are masked to `-inf` before softmax.

**Steps (224):** for each of the 64 squares `from=(r,c)`, for each of 4 orthogonal
directions, `to = (r+dr, c+dc)`. If `to` is in-bounds, it is a valid step action.
Count: 4 corners x 2 + 24 edges x 3 + 36 interior x 4 = 8 + 72 + 144 = 224.

**Jumps (192):** for each `from=(r,c)`, for each of 4 orthogonal directions,
`to = (r+2*dr, c+2*dc)`. If `to` is in-bounds, it is a valid jump action.
(The middle square is implied: `mid = (from + to) / 2`.)
Count: 96 vertical + 96 horizontal = 192.

**Stop (1):** end the chain. Only legal when `chain_active` is true.

**Total: 224 + 192 + 1 = 417.**

The policy head always outputs 417 floats. At inference/training time, the
indices of legal actions at the current position are used to extract the
relevant logits, softmax over that subset, and renormalize.

### Network

```
Input:  (B, 7, 8, 8) float32

Trunk (shared):
  Conv2d(7, 64, 3x3, padding=1)  ->  BatchNorm2d  ->  ReLU
  Conv2d(64, 64, 3x3, padding=1) ->  BatchNorm2d  ->  ReLU
  Conv2d(64, 64, 3x3, padding=1) ->  BatchNorm2d  ->  ReLU
  Flatten  ->  (B, 4096)
  Linear(4096, 256)  ->  ReLU

Policy head:
  Linear(256, 417)   ->  raw logits  (B, 417)

Value head:
  Linear(256, 1)  ->  tanh  ->  (B, 1) in [-1, +1]

Total parameters: ~1.5M (print and verify in T-2)
```

Value convention: `+1` = certain win for the player to move, `-1` = certain loss.
This is the negamax convention. It works correctly with chains because
`board.turn` at each node is always well-defined.

### Search: MCTS with PUCT (AlphaZero-style)

- No random rollouts. Leaf values come from the value head.
- PUCT: `score(a) = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))`
  where `Q` = average value, `P` = policy prior (masked softmax over legal actions),
  `N` = visit count.
- Dirichlet noise at the root (self-play only): `prior = (1-eps)*policy + eps*Dir(alpha)`
  with `alpha=0.3`, `eps=0.25`.
- Default: 200 simulations per position (tunable).

### Training

- Loss: `CE(policy_logits, mcts_policy) + MSE(value_pred, game_outcome)`
  - Policy loss uses **soft targets** (MCTS visit distribution), not one-hot:
    `loss_policy = -(target * log_softmax(logits)).sum(dim=-1).mean()`
  - Value loss: standard MSE.
- Optimizer: Adam, lr=0.002, gradient clipping max_norm=5.0.
- Schedule: cosine annealing over the training epochs per round.

---

## T-0 — Environment setup [DONE]

**Goal:** PyTorch with CUDA working; `azero/` directory created.

**Steps:**

1. Run `nvidia-smi` and note the "CUDA Version" line (e.g. 12.4).
2. Install the matching PyTorch. Use `py -m pip` (not `python`):
   ```
   py -m pip install torch --index-url https://download.pytorch.org/whl/cu118
   ```
   (substitute `cu118` with the correct build for the driver version found in step 1;
   common values: `cu118`, `cu121`, `cu124`)
3. Verify: write `azero/_check_gpu.py`:
   ```python
   import torch
   print("cuda available:", torch.cuda.is_available())
   print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
   ```
   Run: `py azero/_check_gpu.py`
4. Create the directory structure:
   ```
   azero/
     __init__.py          (empty)
     tests/
       __init__.py        (empty)
     checkpoints/         (empty, add a .gitkeep)
   ```

**Done when:** `_check_gpu.py` prints `cuda available: True` and the 4090 name.
`azero/__init__.py` and `azero/tests/__init__.py` exist.

**Risks:** CUDA build mismatch. If `torch.cuda.is_available()` returns False,
check `nvidia-smi` again and try a different `--index-url`.

---

## T-1 — Board encoding + action list [DONE]

**Goal:** The two foundational APIs the rest of the project depends on:
a function that converts a `Halma8` state into a 7-channel tensor, and the
fixed 417-action list with index lookup.

**Files to create:**
- `azero/board.py`
- `azero/tests/test_board.py`

### `azero/board.py` — required API

```
encode_board(state: Halma8) -> torch.Tensor          # shape (1, 7, 8, 8) float32
ACTION_LIST: list[tuple]                             # 417 entries, fixed order
action_index(action: tuple) -> int                   # dict lookup, O(1)
decode_action(index: int) -> halma8_move             # ACTION_LIST entry -> Halma8 move tuple
legal_action_indices(state: Halma8) -> list[int]     # indices into ACTION_LIST that are legal
```

**`ACTION_LIST` construction order (deterministic):**

1. For `from_sq` in 0..63 (row-major):
   For each of 4 directions `DIRS = [(-1,0),(1,0),(0,-1),(0,1)]`:
   - Step: `to = (r+dr, c+dc)`. If in-bounds → append
     `("step", r, c, to_r, to_c)`.
   - Jump: `to = (r+2*dr, c+2*dc)`. If in-bounds → append
     `("jump", r, c, to_r, to_c)`.
2. Append `("stop",)` at the very end.

Total: 224 steps + 192 jumps + 1 stop = 417. Build an `index_map` dict
from action tuple to index for O(1) lookup.

**`encode_board(state)`:**

Read `state.grid` (8x8 list of 0/1/2), `state.turn`, `state.chain_active`,
`state.chain_banned` (set of (r,c) tuples), `state.chain_starter`,
`state.chain_current`. Build a 7x8x8 float32 array, wrap in a tensor with
a leading batch dim: shape `(1, 7, 8, 8)`.

Channel details:
- ch0: 1.0 where `state.grid[r][c] == 1`, else 0.0
- ch1: 1.0 where `state.grid[r][c] == 2`, else 0.0
- ch2: 1.0 everywhere if `state.turn == 1`, else 0.0 everywhere
- ch3: 1.0 everywhere if `state.chain_active`, else 0.0 everywhere
- ch4: 1.0 at each square in `state.chain_banned`, else 0.0
- ch5: 1.0 at `state.chain_current` (if not None), else 0.0 everywhere
- ch6: 1.0 at `state.chain_starter` (if not None), else 0.0 everywhere

**`decode_action(index)`:**

Convert an `ACTION_LIST` entry to the move tuple that `Halma8.apply_move` accepts:
- `("step", r, c, nr, nc)` → `("step", (r,c), (nr,nc))`
- `("jump", r, c, nr, nc)` → `("jump", (r,c), (mid_r,mid_c), (nr,nc))`
  where `mid = ((r+nr)//2, (c+nc)//2)`
- `("stop",)` → this is not an `apply_move` call; it is `end_turn()`.
  Return a sentinel like `None` and let the caller handle it.

**`legal_action_indices(state)`:**

For each index in 0..417, check whether the action is legal at `state`:
- Step: `state.legal_steps(r, c)` contains `(nr, nc)`, AND the square `(r,c)`
  is occupied by `state.turn`, AND chain is not active (steps are not allowed
  during a chain).
- Jump: `state.legal_jumps(r, c)` contains `((mid_r,mid_c), (nr,nc))`, AND
  the square `(r,c)` is occupied by `state.turn`, AND (if chain is active)
  the destination is not in `state.chain_banned`, AND (if chain is active)
  `(r,c) == state.chain_current`.
- Stop: legal iff `state.chain_active` is True.

Return a sorted list of legal action indices.

**`azero/tests/test_board.py` — required tests:**

1. `len(ACTION_LIST) == 417`.
2. Every entry is one of the three valid forms (step/jump/stop tuple).
3. `encode_board(Halma8())` (start position) has shape `(1, 7, 8, 8)`.
   Spot-check: ch0[0][5][0] == 1.0 (P1 at (5,0)), ch1[0][0][5] == 1.0
   (P2 at (0,5)), ch2 all 1s (P1 to move), ch3 all 0s (no chain).
4. `encode_board` on a mid-chain state: set up a chain, verify ch3/ch4/ch5/ch6.
5. `legal_action_indices(Halma8())` (start position) returns a non-empty list,
   and every returned index corresponds to a legal step or jump (no Stop,
   since no chain is active at start).
6. `action_index` round-trip: for every action in `ACTION_LIST`,
   `ACTION_LIST[action_index(a)] == a`.
7. `decode_action` for a step action produces a tuple that
   `Halma8.apply_move` accepts (apply it, verify no exception).

**Gate:** `py -m pytest azero/tests/test_board.py -v` → all pass.
`py halma8_test.py` → 36 OK.

---

## T-2 — Neural network [DONE]

**Goal:** A PyTorch `nn.Module` with the architecture described above,
forward + backward working on CPU and GPU.

**Files to create:**
- `azero/model.py`
- `azero/tests/test_model.py`

### `azero/model.py` — required API

```
class HalmaNet(nn.Module):
    def __init__(self): ...
    def forward(self, x):           # x: (B, 7, 8, 8) -> (logits, value)
        # returns (policy_logits: (B, 417), value: (B, 1))
    def policy_and_value(self, x):  # same as forward, alias for clarity

def policy_loss(logits, target):
    # soft-target cross-entropy
    # logits: (B, 417), target: (B, 417) probability distribution
    # returns scalar tensor

def value_loss(pred, target):
    # MSE
    # pred: (B, 1), target: (B, 1) in [-1, 1]
    # returns scalar tensor

def total_loss(logits, policy_target, value_pred, value_target):
    # policy_loss + value_loss
```

**Implementation notes:**

- Use `nn.Conv2d`, `nn.BatchNorm2d`, `nn.ReLU`, `nn.Flatten`, `nn.Linear`.
- `forward` returns raw logits (no softmax) and tanh value.
- `policy_loss`: use `F.log_softmax(logits, dim=-1)` then
  `-(target * log_probs).sum(dim=-1).mean()`.
  Do NOT use `F.cross_entropy` (it expects hard class indices).
- `value_loss`: `F.mse_loss(pred.squeeze(-1), target.squeeze(-1))`.
- Provide a `count_parameters()` helper that returns the total int.

**`azero/tests/test_model.py` — required tests:**

1. `HalmaNet()` instantiates without error.
2. Forward on random input `(2, 7, 8, 8)` (CPU): outputs shapes `(2, 417)`
   and `(2, 1)`. Values are finite (no NaN/Inf).
3. Forward on GPU (skip gracefully if no GPU): same shapes.
4. `count_parameters()` is between 1,000,000 and 3,000,000.
5. Backward: create random targets, compute `total_loss`, call `.backward()`.
   All parameters have non-None `.grad`.
6. `policy_loss` on a one-hot target vs. uniform logits is finite and positive.
7. Save/load: `state_dict()` → `load_state_dict()` → outputs identical.

**Gate:** `py -m pytest azero/tests/test_model.py -v` → all pass.
`py halma8_test.py` → 36 OK.

---

## T-3 — MCTS [DONE]

**Goal:** AlphaZero-style MCTS that uses the network's policy + value to
search a position and return the best action + the root policy distribution.

**Files to create:**
- `azero/mcts.py`
- `azero/tests/test_mcts.py`

### `azero/mcts.py` — required API

```
class MCTSNode:
    state: Halma8              # deep-copied board state at this node
    parent: MCTSNode | None
    action: tuple | None       # the ACTION_LIST entry that led here
    children: dict             # action_tuple -> MCTSNode
    visits: int
    value_sum: float
    prior: float               # set at creation from parent's policy
    expanded: bool

def mcts_search(
    state: Halma8,
    model: HalmaNet,
    n_sims: int = 200,
    c_puct: float = 1.5,
    dirichlet_alpha: float = 0.3,
    dirichlet_eps: float = 0.25,
    device: str = "cpu",
    add_noise: bool = True,    # False for evaluation
) -> tuple[tuple, np.ndarray]:
    """
    Returns:
      best_action: the ACTION_LIST entry with the most visits at the root
      root_policy: np.ndarray of shape (417,), the normalized visit
                   distribution over ALL actions (0.0 for illegal ones)
    """
```

**Implementation details (critical for correctness):**

1. **Root creation:** Create the root `MCTSNode` with a deep copy of `state`.
   Run `model(encode_board(state))` once to get the root's policy + value.
   If `add_noise`, apply Dirichlet noise to the policy before using it as priors.
   Set `root.prior` for each legal child from the (noised) policy.

2. **Legal actions at a node:** Call `legal_action_indices(node.state)` from
   `azero/board.py`. This gives the list of legal `ACTION_LIST` indices.

3. **PUCT selection:** Among the node's children that correspond to legal
   actions, pick the one maximizing:
   ```
   Q = child.value_sum / max(child.visits, 1)
   U = c_puct * child.prior * sqrt(node.visits) / (1 + child.visits)
   score = Q + U
   ```
   If the node has no children yet (not expanded), expand it first.

4. **Expansion:** Run `model(encode_board(node.state))` to get policy + value.
   For each legal action index, create a child node. Set the child's `prior`
   from the policy (softmax over legal actions, extract the component for
   this action). Apply the action to a copy of the state to get the child's
   state. Mark the node as expanded.

5. **Terminal detection:** Before expanding, check if `node.state.winner is not None`.
   If so, the node is terminal:
   - Value = +1.0 if the winner is the player to move at this node, else -1.0.
   - Do not expand. Use this value for backpropagation.
   Also check if there are no legal actions (stuck):
   - Value = -1.0 (the player to move loses).

6. **Leaf evaluation (non-terminal):** Use the value head output.
   `v = value.item()` (scalar in [-1, +1], from the perspective of the player
   to move at this node).

7. **Backpropagation:** Walk from the leaf up to the root. For each node on
   the path, determine the sign:
   - If `node.state.turn == leaf.state.turn`: contribution = `+v`
   - Else: contribution = `-v`
   Update: `node.visits += 1`, `node.value_sum += contribution`.
   
   This is equivalent to: `contribution = v * (-1) ** (parity_distance)`
   where `parity_distance` is the number of turn-flips between the node and
   the leaf. In practice, just compare `node.state.turn` with
   `leaf.state.turn` at each step.

8. **Deep copy of state:** Use `copy.deepcopy(state)` when creating child nodes.
   This is safe and correct; the game state is small (a few hundred bytes).
   Do NOT try to use make/unmake here — it adds complexity for no benefit
   at this scale.

9. **`model.eval()` during search:** Set `model.eval()` before the MCTS loop
   (disables dropout, uses running BN stats). Set `model.train()` before
   training steps.

10. **`torch.no_grad()` during search:** Wrap the entire MCTS loop in
    `torch.no_grad()` to avoid building a computation graph.

**`azero/tests/test_mcts.py` — required tests:**

1. MCTS on the start position with a random-init model, `n_sims=20`,
   `add_noise=False`: returns a legal action (verify it is in
   `legal_action_indices(start_state)`).
2. `root_policy` has shape `(417,)`, sums to ~1.0 (within 1e-5), and all
   entries are >= 0.
3. Forced win: set up a position where P1 can win in 1 move (8 P2 pieces
   in P2's base + 1 P1 piece + 1 empty base square, P1 has a piece adjacent
   to that empty square). MCTS with `n_sims=50` finds the winning move.
4. After `n_sims` simulations, `root.visits == n_sims` (within tolerance —
   some sims may hit terminal nodes at the root, which is fine).
5. No NaN/Inf in any node's `value_sum` or `visits` after search.

**Gate:** `py -m pytest azero/tests/test_mcts.py -v` → all pass.
`py halma8_test.py` → 36 OK.

---

## T-4 — Self-play [DONE]

**Goal:** Play full games using MCTS + the network, collecting
(position, policy, value) training tuples.

**Files to create:**
- `azero/selfplay.py`
- `azero/tests/test_selfplay.py`

### `azero/selfplay.py` — required API

```
def play_one_game(
    model: HalmaNet,
    device: str = "cpu",
    n_sims: int = 200,
    max_moves: int = 100,
    add_noise: bool = True,
) -> list[tuple[torch.Tensor, np.ndarray, float]]:
    """
    Play one full game from the start position.
    Returns a list of training tuples:
      (encoded_state: (7,8,8) tensor,
       policy: (417,) numpy array,
       value: float in {-1.0, +1.0})
    Empty list if the game hits max_moves with no winner.
    """

def self_play_batch(
    model: HalmaNet,
    n_games: int = 500,
    device: str = "cpu",
    n_sims: int = 200,
    max_moves: int = 100,
) -> list:
    """
    Play n_games games, return a flat list of all training tuples.
    Games are played sequentially (each game is independent).
    """
```

**Implementation details:**

1. **Game loop:** Create `Halma8()`. For each move (up to `max_moves`):
   a. Record the current state: `enc = encode_board(state).squeeze(0)` (7,8,8).
   b. Run `mcts_search(state, model, n_sims, device=device, add_noise=True)`.
   c. Get `best_action` and `root_policy`.
   d. Apply the action:
      - If `best_action` is `("stop",)`: call `state.end_turn()`.
      - Otherwise: `mv = decode_action(action_index(best_action))`;
        call `state.apply_move(mv)`.
   e. Store `(enc, root_policy)` in a local list (value filled later).
   f. If `state.winner is not None`: break.

2. **Fill in value targets after the game ends:**
   - If `winner is not None`: for each stored `(enc, policy)` at move index `i`,
     the player to move was `state.turn` at that point (we need to track this —
     store `mover` alongside). Value = `+1.0` if `mover == winner`, else `-1.0`.
   - If the game hit `max_moves` with no winner: return an empty list
     (discard the game — no clean value signal).

3. **Track the mover:** At each step, before applying the move, record
   `mover = state.turn`. Store it alongside the encoding.

4. **No parallelism in T-4:** Play games sequentially. This keeps the code
   simple and correct. Parallelism can be added later if self-play is too slow.
   Each game is ~50-100 moves, each move is one MCTS search (~200 sims).
   On a 4090 this should be ~1-5 seconds per game.

5. **Memory:** The model stays on `device` throughout. The board states are
   small Python objects. No GPU memory pressure from self-play itself.

**`azero/tests/test_selfplay.py` — required tests:**

1. `play_one_game` with a random-init model, `n_sims=10`, `max_moves=30`,
   `device="cpu"`: completes without exception. Returns a list (may be empty
   if the game hits max_moves, but with a random model it should usually finish).
2. If the list is non-empty, every tuple has the right shapes:
   enc `(7,8,8)`, policy `(417,)`, value in {-1.0, +1.0}.
3. `self_play_batch` with `n_games=3`, `n_sims=5`, `max_moves=20`:
   completes, returns a list (possibly empty).
4. All policy arrays sum to ~1.0 (within 1e-4).
5. No NaN/Inf in any tensor.

**Gate:** `py -m pytest azero/tests/test_selfplay.py -v` → all pass.
`py halma8_test.py` → 36 OK.

**Note:** With a random-init model, most games will hit `max_moves` and
return empty lists. That is expected. The self-play code is correct; the
model just doesn't play well yet. T-5/T-6 will fix that.

---

## T-5 — Training loop [DONE]

**Goal:** Train the network on self-play data, with save/load checkpoints.

**Files to create:**
- `azero/train.py`
- `azero/tests/test_train.py`

### `azero/train.py` — required API

```
def train_step(model, optimizer, batch, device) -> float:
    """
    One training step.
    batch: list of (enc_tensor, policy_np, value_float)
    Returns the scalar loss value (float).
    """

def train(
    model: HalmaNet,
    data: list,
    n_epochs: int = 10,
    batch_size: int = 256,
    lr: float = 0.002,
    device: str = "cpu",
    optimizer: torch.optim.Optimizer = None,  # create if None
) -> torch.optim.Optimizer:
    """
    Train for n_epochs on the data list.
    Returns the (updated) optimizer so the caller can save it.
    """

def save_checkpoint(model, optimizer, round_no, path):
    """Save model + optimizer state to path."""

def load_checkpoint(model, optimizer, path, device):
    """Load model + optimizer state from path. Returns the saved round_no."""
```

**Implementation details:**

1. **`train_step`:**
   - Stack the batch: `states = torch.stack([enc for enc,_,_ in batch])` → (B,7,8,8)
   - `policies = torch.tensor(np.stack([p for _,p,_ in batch]), dtype=torch.float32)` → (B,417)
   - `values = torch.tensor([v for _,_,v in batch], dtype=torch.float32).unsqueeze(1)` → (B,1)
   - Move to device.
   - `model.train()` (enable BN running stats update, dropout if any).
   - Forward: `logits, value_pred = model(states)`.
   - Loss: `loss = policy_loss(logits, policies) + value_loss(value_pred, values)`.
   - `optimizer.zero_grad()`; `loss.backward()`.
   - `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)`.
   - `optimizer.step()`.
   - Return `loss.item()`.

2. **`train`:**
   - Create optimizer if None: `torch.optim.Adam(model.parameters(), lr=lr)`.
   - Create scheduler: `torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)`.
   - For each epoch: shuffle the data, iterate in batches of `batch_size`,
     call `train_step`. Track and print the average loss per epoch.
   - `scheduler.step()` after each epoch.
   - Return the optimizer.

3. **`save_checkpoint`:**
   ```python
   torch.save({
       'model_state': model.state_dict(),
       'optimizer_state': optimizer.state_dict(),
       'round': round_no,
   }, path)
   ```

4. **`load_checkpoint`:**
   ```python
   ckpt = torch.load(path, map_location=device)
   model.load_state_dict(ckpt['model_state'])
   optimizer.load_state_dict(ckpt['optimizer_state'])
   return ckpt['round']
   ```

**`azero/tests/test_train.py` — required tests:**

1. Create a small synthetic dataset: 64 random (7,8,8) tensors, random
   policy distributions (417-dim, normalized), random values in {-1,+1}.
2. `train` for 2 epochs, `batch_size=16`, `device="cpu"`: completes,
   loss decreases from epoch 1 to epoch 2 (or at least is finite).
3. `save_checkpoint` → file exists. `load_checkpoint` → model weights
   are identical (compare a few param tensors with `torch.allclose`).
4. `train_step` returns a finite float (no NaN/Inf).
5. After training, `model.eval()` forward pass still works (no shape errors).

**Gate:** `py -m pytest azero/tests/test_train.py -v` → all pass.
`py halma8_test.py` → 36 OK.

---

## T-6 — Full pipeline + CLI [DONE]

**Goal:** The end-to-end self-play → train → repeat loop, runnable as a
single command. Checkpointing and logging included.

**Files to create:**
- `azero/pipeline.py`
- `azero/run.py`

### `azero/pipeline.py`

```
def run_pipeline(
    n_rounds: int = 5,
    games_per_round: int = 500,
    n_sims: int = 200,
    n_train_epochs: int = 10,
    batch_size: int = 256,
    lr: float = 0.002,
    device: str = "cuda",
    checkpoint_dir: str = "azero/checkpoints",
    start_round: int = 1,
    model: HalmaNet = None,       # None = create fresh
    optimizer = None,
) -> HalmaNet:
    """
    Run the full self-play -> train loop.
    Returns the final model.
    """
```

**Per-round Elo tracking (USER REQUIREMENT):**

The user wants a **clear, evolving Elo number per round** so they can watch
the network get stronger over training — not just a loss curve. Conventions:

- **Baseline = the model at the start of training** (the random-init model
  before round 1's training). It is **pinned at 1000 Elo by definition**.
  Build it once before round 1 and hold a frozen copy for the whole run
  (it never trains). Save it to `azero/checkpoints/baseline.pt` so a resumed
  run uses the same baseline (do NOT re-baseline off a fresh random model).
- After **each round's** training, evaluate the **current model vs the
  baseline** in a **fixed, cheap** self-match batch and report the model's Elo.
  Use a separate, small sim count for this eval (e.g. `n_sims=100`) and a
  fixed game count (e.g. `n_games=40`, alternating who is P1/P2) so the eval
  stays fast and comparable across rounds.
- **Elo formula (baseline-relative):** baseline is fixed at 1000, so the
  model's expected score against it is `E = 0.5`. Compute the model's
  observed score `S = (wins + 0.5*draws) / n_games`, then
  `elo = 1000 + K * (S - E)` = `1000 + K * (S - 0.5)` with **K = 64**.
  - S = 1.0 → 1032 (always beats baseline)
  - S = 0.5 → 1000 (no better than random init)
  - S = 0.0 → 968 (worse than random init)
  Draws: a check_halma game ends by `winner is not None` or `max_moves`;
  count `max_moves`-ended games as 0.5. This is a **baseline-relative Elo
  proxy** — its whole purpose is a monotonic, readable trend, so keep the
  method and K constant across every round (do not change K mid-run).
- Add a step to the per-round flow (below) that runs this eval and records it,
  so the trend is visible in `training_log.txt` and on stdout.

**Per-round flow:**
1. `data = self_play_batch(model, games_per_round, device, n_sims)`
   Print: `round {r}: self-play done, {len(data)} samples collected`
2. `optimizer = train(model, data, n_train_epochs, batch_size, lr, device, optimizer)`
   Print: `round {r}: training done`
3. `save_checkpoint(model, optimizer, r, os.path.join(checkpoint_dir, f"round_{r}.pt"))`
   Print: `round {r}: saved checkpoint`
4. **Elo eval (see requirement above):** `score, elo = evaluate_elo(
      model, baseline_model, n_games=40, n_sims=100, device=device)`
   Print: `round {r}: elo {elo:.1f} (score {score:.3f} vs baseline)`
5. Append one line to `azero/training_log.txt`:
   `round={r}  samples={N}  avg_loss={L:.4f}  elo={elo:.1f}  score={score:.3f}  time_s={T:.1f}`

**After all rounds:** save a final checkpoint as `azero/checkpoints/final.pt`.
Print a summary, including a **round-by-round Elo table**:

```
round   samples   avg_loss   elo    score
1        ...       ...       1002.4  0.562
2        ...       ...       1009.1  0.675
...
```

`evaluate_elo` lives in `azero/elo.py` (a small, standalone self-match
referee that reuses `play_one_game`-style NN-vs-NN play: one side is
`model`, the other is the frozen `baseline_model`, alternating P1/P2;
returns `(score, elo)` per the formula above). It does NOT need MCTS
search quality to be high — it only needs a consistent, fair comparison.

**`azero/elo.py` — required API:**
```
K = 64
def score_to_elo(score) -> float:
    # 1000 + K * (score - 0.5)

def evaluate_elo(model, baseline_model, n_games=40, n_sims=100,
                 device="cpu", max_moves=100) -> tuple[float, float]:
    # Play n_games NN-vs-NN (model vs baseline_model), alternating who is P1/P2.
    # score = (wins + 0.5*draws) / n_games, from `model`'s perspective.
    # Returns (score, score_to_elo(score)).
    # A game is a "win" if `model`'s side is the winner, a "draw" if it
    # hits max_moves, a "loss" if baseline_model's side wins.
```

### `azero/run.py` — CLI entry point

```
Usage:
  py azero/run.py --rounds 5 --games 500 --sims 200 --epochs 10 --device cuda
  py azero/run.py --rounds 1 --games 5 --sims 10 --epochs 2 --device cpu  (smoke)
  py azero/run.py --resume azero/checkpoints/round_3.pt --rounds 5 ...
```

Parse args with `argparse`. If `--resume` is given, load the checkpoint
(model + optimizer + round number), set `start_round = saved_round + 1`,
and continue.

**`azero/tests/test_pipeline.py` — smoke test:**

1. Run `run_pipeline(n_rounds=1, games_per_round=3, n_sims=5,
   n_train_epochs=1, batch_size=4, device="cpu")`.
2. Completes without exception.
3. A checkpoint file exists in `azero/checkpoints/`.
4. `azero/training_log.txt` exists and has at least 1 line, and that line
   contains an `elo=` and a `score=` field.
5. The checkpoint loads correctly (model weights match).
6. The Elo **formula** is exact: `score_to_elo(0.5) == 1000`,
   `score_to_elo(1.0) == 1032`, `score_to_elo(0.0) == 968`.
7. `evaluate_elo(model, model, n_games=4, n_sims=3, device="cpu")`
   (a self-match sanity check) returns a score in `[0, 1]` and a finite
   elo in `[968, 1032]` — verifies the referee runs without error and the
   number is bounded. (Do NOT assert it is exactly 0.5 / 1000: a self-match
   has a first-mover bias and stochastic play, so the score is near — not
   equal to — 0.5.)

**Gate:** `py -m pytest azero/tests/test_pipeline.py -v` → all pass.
`py halma8_test.py` → 36 OK.

---

## T-7 — Evaluation (NN vs current best) [DONE]

**Goal:** Measure how the trained NN plays against the current best engine
(the 448/540-weight Rust engine, `best_weights.json`).

**Files to create:**
- `azero/evaluate.py`

### `azero/evaluate.py`

```
def evaluate_nn_vs_rust(
    model: HalmaNet,
    n_games: int = 50,
    n_sims: int = 200,
    device: str = "cuda",
) -> tuple[int, int, int]:
    """
    Play n_games games: NN as P1, Rust engine as P2.
    The Rust engine is driven via the existing halma_ai.py subprocess
    (the same path the GUI uses).
    Returns (nn_wins, rust_wins, ongoing).
    """

def main():
    # Load model from azero/checkpoints/final.pt
    # Run evaluation
    # Print results
```

**Implementation approach:**

The Rust engine speaks a line protocol (the `server.rs` UCI-like protocol:
`uci` / `position` / `go` / `stop`). The existing `halma_ai.py` driver
already handles spawning the Rust engine subprocess, sending positions,
and receiving moves. Reuse that driver.

The referee loop:
1. Create a `Halma8` state.
2. P1 (NN): encode → MCTS → best_action → decode → `apply_move`.
3. P2 (Rust): send the current position string to the Rust engine subprocess,
   receive a move string, parse it, `apply_move`.
4. Repeat until `state.winner is not None` or max moves.
5. Count the result.

**Note:** This task requires the Rust engine to be built
(`cargo build --release` in `ai/`). If it is not built, build it first.

**Gate:** `py azero/evaluate.py --games 5 --sims 20 --device cpu` runs
without crashing and prints win/loss/ongoing counts.

---

## T-8 — Launch (background) + monitoring [IN PROGRESS]

**Goal:** Launch a long training run in the background. Close the session.
Check progress later.

**CURRENT LAUNCH (2026-11-09) — lean round-3, user-launched isolated (free GPU):**
Rounds 1-2 are DONE, **both Elo 1000.0** (round 1 score 0.500; round 2 all-draw in
standalone `elo.py` — the net can't yet decide games within `max_moves=100`, so a bigger
eval adds no signal; the lever is **more rounds**). Eval now runs vs the **champion**
(best-trained net; resolves to latest `round_*.pt`), not the frozen baseline. This
**supersedes** the `--rounds 10` command further down:

    py -u azero/run.py --rounds 1 --device cuda --resume azero/checkpoints/round_2.pt ^
        --games 120 --sims 100 --epochs 10 --max-moves 100 ^
        --elo-games 120 --elo-sims 100 --workers 8 --mcts-batch 32

- `--resume round_2.pt` -> `start_round=3` (`load_checkpoint` returns round 2); trains and
  saves `round_3.pt`, then evaluates it vs the champion (round_2.pt).
- Est. ~40-70 min total: self-play 120 games @ 100 sims (~30-40 min) + 10-epoch train +
  120-game eval (~20-30 min). GPU-bound; 8 workers time-slice the 4090.
- Monitor: `type azero\training_log.txt` (watch `elo=` move off 1000.0) + `nvidia-smi`.
- Round 4+ (once the pace is confirmed, or bump `--rounds 3` for the round-3-5 trend):
    py -u azero/run.py --rounds 3 --device cuda --resume azero/checkpoints/round_3.pt ^
        --games 120 --sims 100 --epochs 10 --max-moves 100 ^
        --elo-games 120 --elo-sims 100 --workers 8 --mcts-batch 32

**STATUS (2026-11-08, session 3):** NOT yet launched. CUDA confirmed working
(4090, torch 2.14.1+cu126). **Blocker:** Python MCTS self-play is slow even on
CUDA — 8 games × 40 sims timed out past 60 s; the per-round Elo eval (40 games
× 100 sims) is also slow. The 10–20 h estimate below is likely a **lower bound**.
All other tasks (T-0 to T-7) are done; gates green (azero 42 pass, halma8 36 OK).

**STATUS (2026-11-08, session 4):** Gates re-verified green (halma8 36 OK,
azero 42 pass). Throughput measured ~**1.6 ms per (move × sim)** via `_probe_small.py`
(under contention). The bottleneck is Python MCTS, not the GPU, so an isolated
free-GPU launch runs at about this rate. Launch is now a **user-launched isolated
job** — the user closes the ollama/assistant session first so the 4090 is free.
Recommended command (below): `py -u azero/run.py --rounds 10 --games 120 --sims 120
--epochs 5 --max-moves 100 --device cuda` → ~50 min/round (incl. the ~10 min Elo
eval), ~8 h for 10 rounds. `training_log.txt` was cleared of stale smoke lines so
the real run's log starts clean.

**Session-4 additions:** (a) added `--elo-games` / `--elo-sims` CLI flags to
`azero/run.py` (default 40/100 = unchanged big-run behavior) so the per-round Elo
eval size is tunable for quick validation runs; (b) **resume path smoke-verified
end-to-end** on CPU: fresh round 1 → `final.pt` → `--resume final.pt` resumed at
round 2 with round numbering continuing and `load_checkpoint` restoring weights
exactly (also covered by `test_pipeline.py`, azero 42 pass). To validate the
continuing setup before the big run: `py -u azero/run.py --rounds 3 --games 60
--sims 40 --epochs 5 --elo-games 30 --elo-sims 40 --max-moves 100 --device cuda`
(fresh, ~30 min) then `py -u azero/run.py --resume azero/checkpoints/final.pt
--rounds 3 --games 60 --sims 40 --epochs 5 --elo-games 30 --elo-sims 40
--max-moves 100 --device cuda` (continues at round 4, ~30 min).

**Steps (next agent — in order):**

1. **Measure throughput** (before choosing launch params):
   ```
   py _probe_timing.py
   ```
   This runs a small CUDA self-play batch (8 games × 40 sims) and prints
   the per-(game × sim) rate plus extrapolated wall-clock for several
   `--games/--sims` combos. Use the numbers to pick a realistic config.
2. **Pick realistic launch params.** Balance wall-clock vs. data volume.
   Smaller `--sims` (e.g. 50–100) with more `--games` is usually better
   for throughput than the 300-sims default.
3. **Verify the T-6 smoke test** still passes on the chosen device:
   `py azero/run.py --rounds 1 --games 5 --sims 10 --epochs 2 --device cpu`
4. **Launch the full run** (with the params from step 2):
   ```
   py azero/run.py --rounds N --games G --sims S --epochs E --device cuda
   ```
   Run as a **background process** so the session can be closed.
   Use the shell tool with `is_background: true`.
5. The run writes:
   - `azero/checkpoints/baseline.pt` (frozen pre-training model, = 1000 Elo)
   - `azero/checkpoints/round_N.pt` after each round
   - `azero/checkpoints/final.pt` at the end
   - `azero/training_log.txt` (one line per round, incl. `elo=` and `score=`)
6. To check progress later (new session):
   ```
   type azero\training_log.txt
   dir azero\checkpoints\
   ```
7. To resume after a crash:
   ```
   py azero/run.py --resume azero/checkpoints/round_5.pt --rounds N --games G --sims S --epochs E --device cuda
   ```

**Monitoring (in a separate terminal):**
- `nvidia-smi` — GPU utilization during self-play + training
- `azero/training_log.txt` — one line per round (samples, loss, **elo**, time).
  This is the headline trend to watch: `elo` should climb away from 1000
  (the random-init baseline) over the rounds.

**Expected rough timeline (4090 — estimates, verify with `_probe_timing.py`):**
- Self-play: games × moves × sims × forward-pass cost. Python MCTS overhead
  adds 2–5× on top of pure GPU time. The `_probe_timing.py` output will give
  a concrete number per (game × sim).
- Training: ~samples × epochs / batch — typically ~1 min per round.
- Total per round: dominated by self-play. See `_probe_timing.py` output.

**Gate:** The background process is running. `training_log.txt` is being
appended. `nvidia-smi` shows GPU activity.

---

## T-9 — Parallel self-play workers (N OS processes) [DONE]

**Goal:** Cut self-play wall-clock ~N× by running N OS processes that each do
self-play and share the 4090. Least risky of the three speedups (no MCTS
rewrite, no state-copy change) — the recommended first win.

**Why it wins:** The `_probe_timing.py` / `_probe_small.py` measurements show
the run is **Python-MCTS-bound** (~1.6 ms per move·sim), **not** GPU-bound. A
single process therefore leaves the 4090 mostly idle and burns only one CPU
core on the MCTS loop. N OS processes → N CPU cores running the MCTS loop in
parallel; the GPU is time-sliced across N CUDA contexts (a ~1.23M-param net is
a drop in the 24 GB VRAM). Net: near-linear wall-clock reduction in the
self-play phase, which dominates each round. Data volume per round is
**unchanged** — only wall-clock drops.

**Files to touch (all inside `azero/`; nothing else):**
- `azero/run.py` — add a `--workers` flag (default `1` = unchanged sequential behavior).
- `azero/pipeline.py` — add a `workers` param to `run_pipeline(...)`; fan-out + merge.
- `azero/selfplay.py` — add a **module-level** `selfplay_worker(args)` (see below).
- `azero/tests/test_selfplay.py` — new worker test.

**Do NOT touch:** `azero/mcts.py`, `azero/train.py`, `azero/elo.py`, `halma8.py`.

**Design:**
- When `workers > 1`, split `games_per_round` into `workers` contiguous chunks
  (a non-divisible remainder is assigned to one chunk, so total games played
  is exactly `games_per_round`). For each chunk run `self_play_batch` in a
  worker process, then concatenate the returned tuples into one `data` list.
  The train / checkpoint / Elo / log steps that follow are unchanged.
- **Windows uses the `spawn` start method**, so the worker target must be a
  **module-level function** (not a closure/lambda) and the model must be passed
  as **data**, not a live `torch` module (a live CUDA module is not picklable).
  Concretely, a module-level function in `azero/selfplay.py`:
  ```
  def selfplay_worker(args):
      # args (picklable): model_state_dict, n_games, n_sims, max_moves,
      #                  device, seed, add_noise
      # -> build HalmaNet(), load_state_dict(model_state_dict), .eval(),
      #    run self_play_batch on `n_games` games, return the flat tuple list
  ```
  The parent loads `model.state_dict()` once per round and sends the **same**
  dict to every worker so all N see identical weights.
- All N workers may target the same physical GPU (same `CUDA_VISIBLE_DEVICES`).
  If VRAM ever shows pressure, set a distinct `CUDA_VISIBLE_DEVICES` per worker
  to spread across GPUs (not needed on a single 4090 with this net size).

**Test gate:**
1. New `test_selfplay_worker` (`azero/tests/test_selfplay.py`): call
   `selfplay_worker` directly with a tiny chunk (`n_games=2, n_sims=5,
   device="cpu"`); assert it returns a non-empty list of correctly-shaped
   tuples `(state(7,8,8), policy(417,), value)`.
2. End-to-end: `py azero/run.py --rounds 1 --games 8 --sims 10 --epochs 2
   --workers 2 --device cpu` runs clean and logs a `samples=` count consistent
   with `games_per_round=8` (same sample count as a `--workers 1` run, modulo RNG).
3. All existing gates still green: `py halma8_test.py` → 36 OK; the azero test
   suite green.
4. **Throughput check (report, do not assert):** a small CUDA self-play batch
   (e.g. `--games 40 --sims 20`) on `--device cuda` with `--workers 1` vs
   `--workers 4`; record the wall-clock delta on the self-play phase (expect
   roughly 2–4× speedup, close to linear until the CPU cores saturate).

**Risks / notes:**
- `spawn` re-imports torch in each worker (~a few seconds each). Negligible
  over a whole round; only matters for tiny test runs.
- If `--workers` is very large (8+), the self-play phase can flip from
  CPU-bound to GPU-context-bound; start at 2–4 and read the 4090 util.

---

## T-10 — Batch the network evals in MCTS [DONE 2026-11-08, session 7]

**DONE (2026-11-08, session 7).** Shipped as specified above, with two adjustments
the implementation forced (both documented, both gate-verified green — azero 45 pass,
halma8 36 OK):

1. **Default stays `batch_size=1`** (per the Risk note), and it is **bit-exactly
   equivalent to the pre-T-10 search** — proven by `test_mcts_batch_equiv`, which
   runs a **verbatim reference copy of the old loop** and asserts `batch_size=1`
   returns the same `best_action` and a bit-identical `root_policy` (no tolerance)
   across positions/seeds/`add_noise`.
2. **Virtual losses** added to the batched descent (Leela-Zero/KataGo canonical
   trick). Without them all B descents in a step see identical stale counts and
   pile onto one node (degenerate search); with them each descent sees the others'
   virtual loss and spreads out, and `batch_size=1` (a lone leaf) cancels the VL and
   is identical to the original.
3. **Test-gate relaxation (deviation from the spec text):** the spec asked to assert
   `best_action(batch_size=1) == best_action(batch_size=16)`. That does NOT hold and
   should not be asserted: with an *untrained* (near-constant) net — value ≈ -0.03 and
   a near-uniform policy for *every* position, verified empirically — the best move is
   decided purely by PUCT tie-breaking / visit order, which legitimately differs
   between batched and sequential descent. This is exactly the "order-sensitive" case
   the Risk note anticipated. The meaningful model-independent anchor (item 1) is kept
   as the hard gate; `batch_size>1` is asserted sound/deterministic/legal instead.
   With a *trained* net (real self-play) the two should converge much closer.

**TIMING GATE DONE (2026-11-08):** `py _t10_timing.py` run on the 4090 (240 samples,
`games=6/sims=30/max_moves=40`). Result — batch=32 is a large, unambiguous win:

| batch | fwd_calls | per(move·sim) |
|-------|-----------|---------------|
| 1     | 7200      | 2.653 ms      |
| 32    | 240       | 0.179 ms      |

30× fewer forward calls and **~14.8× lower** wall-clock per (move·sim). The batched
path loads the GPU as intended; `batch=1` stays the exact bit-identical reference.
T-10 is now fully DONE (code + equiv gate + report-only timing gate all green).
Next: T-11 (`halma_clone` to replace `deepcopy` per child).

**Goal:** Within a single `mcts_search`, evaluate B leaf states in **one**
`model(batch)` forward pass instead of one per leaf. Amortizes the fixed
launch overhead of a batch-1 forward call and finally gives the 4090 a real job.

**Why it wins:** Today `expand()` in `azero/mcts.py` calls `net(node.state)` →
`model(x)` with `x` of shape `(1,7,8,8)` — one forward pass per leaf. A batch-1
forward on a CNN has meaningful fixed launch overhead relative to the math.
Stacking B leaves into one `(B,7,8,8)` call cuts per-leaf GPU overhead by ~B×
and saturates the GPU better. Composes with T-9 (T-9 spreads work across cores;
T-10 loads the GPU within each process).

**Files to touch:**
- `azero/mcts.py` — restructure the sim loop + factor `net()` out of `expand()`.
- Optionally `azero/selfplay.py` (pass `batch_size` through to `mcts_search`)
  and `azero/run.py` (a `--mcts-batch` knob if you want it tunable from the CLI).
- `azero/tests/test_mcts.py` — new equivalence test.

**Do NOT touch:** `azero/train.py`, `azero/elo.py`, `halma8.py`.

**Design:**
- Add a `batch_size` param to `mcts_search(...)` (default e.g. 32;
  `batch_size=1` must reproduce the current behavior **exactly**).
- Change the sim loop from "1 leaf per iteration" to "B leaves per iteration":
  1. Select B leaf nodes via B independent root-to-leaf descents (each descent
     is the existing `select()` loop).
  2. Encode all B states → one `(B,7,8,8)` tensor (per-state `encode_board`,
     then `torch.stack`).
  3. **One** `model(x)` call → `(B,417)` logits and `(B,1)` values.
  4. For each of the B leaves, expand using its own `logits[i]` / `value[i]`
     (existing per-node expansion, but fed the precomputed net output instead of
     calling `net()`).
  5. Backprop each of the B leaves with its own value (existing `backprop`, unchanged).
- Factor `net()` out so `expand()` can accept precomputed `(logits, value)`.
- `n_sims` is still the **total** leaf evaluations, done in
  `ceil(n_sims / batch_size)` outer iterations of B leaves (last iteration takes
  the remainder if `n_sims` is not a multiple of `batch_size`).
- **Dirichlet-noise caveat:** noise is applied only at the root. In batched mode
  the first iteration (root unexpanded) has all B leaves be the root — apply
  noise to that single root expansion **once**, not B times; later iterations
  apply no noise. This preserves the "noise at root, once" semantics.

**Test gate:**
1. New `test_mcts_batch_equiv` (`azero/tests/test_mcts.py`): for a fixed seed and
   a couple of start positions, assert `mcts_search(batch_size=1)` and
   `mcts_search(batch_size=16)` return the **same** `best_action` (and the same
   `root_policy` within 1e-6). This is the behavior-preserving anchor.
2. Existing `test_mcts.py` (5 tests) stays green unchanged (they use the default).
3. Full azero suite green; `py halma8_test.py` → 36 OK.
4. **Throughput check (report, do not assert):** a `_probe_timing.py`-style run
   comparing per-(move·sim) wall-clock and forward-call count at
   `batch_size=1` vs `batch_size=32` on `--device cuda`; expect fewer forward
   calls and a wall-clock reduction on the self-play phase.

**Risks / notes:**
- This is the **riskiest** of the three (it rewrites the inner search loop).
  Keep `batch_size=1` as the default and gate on the equivalence test before
  enabling `batch_size>1` in the hot run.
- If the equivalence test is order-sensitive (RNG), pin the seed and compare
  the deterministic `best_action` rather than the full `root_policy`.

---

## T-11 — Cheaper state copy (replace `copy.deepcopy`) [DONE]

**Goal:** Replace `copy.deepcopy(state)` in `azero/mcts.py` with a lightweight
field-by-field clone of `Halma8`. The state is tiny; `deepcopy`'s generic
protocol (memo dict, per-type dispatch, `__deepcopy__` handling) costs more
than the data it copies.

**Why it wins:** `mcts.py` deep-copies the state **once per search** (root) and
**once per child per node expansion** — i.e. `n_sims × (legal actions)` copies
per move, the single most-allocated object in the hot loop. A hand-rolled copy
of the 9 mutable `Halma8` fields skips the deepcopy machinery entirely.

**Files to touch:**
- `azero/board.py` (or `azero/mcts.py`) — add `halma_clone(state)`.
- `azero/mcts.py` — swap both `copy.deepcopy(state)` call sites (root init +
  per-child expansion) to `halma_clone(state)`.
- `azero/tests/test_board.py` (or a new `azero/tests/test_halma_clone.py`) — tests.

**Do NOT touch:** `halma8.py` (hard rule #2 — no edits outside `azero/`).

**The 9 `Halma8` fields to copy** (verified against `halma8.py` `__init__`):
- `grid` — `list[list[int]]` (8×8). Copy as `[row[:] for row in state.grid]`
  (fresh row list; ints are immutable so sharing them is safe).
- `turn` — `P1`/`P2` (scalar, share).
- `winner` — `None`/`P1`/`P2` (scalar, share).
- `turn_no` — `int` (scalar, share).
- `chain_active` — `bool` (scalar, share).
- `chain_banned` — `set[(r,c)]`. Copy as `set(state.chain_banned)` (**new** set;
  tuples inside are immutable).
- `chain_starter` — `None` or `(r,c)` (immutable, share).
- `chain_current` — `None` or `(r,c)` (immutable, share).

**Design:** a module-level helper (this is the only code in the spec — a
signature, not an implementation):
```
def halma_clone(state) -> Halma8:
    # build an empty Halma8 without calling __init__, then assign the 9 fields:
    #   grid (row-wise copy), turn, winner, turn_no, chain_active (scalars),
    #   chain_banned (set copy), chain_starter, chain_current (shared refs)
    # -> returns an independent Halma8
```
The clone must be **independent**: mutating the clone's `grid` / `chain_banned`
must not affect the source, and vice versa.

**Test gate:**
1. New `test_halma_clone` (`azero/tests/test_board.py`): build a `Halma8` in a
   known mid-chain state (a few jumps so `chain_active=True`, `chain_banned`
   non-empty, `chain_starter`/`chain_current` set); `c = halma_clone(state)`;
   assert all 9 fields match (grid equality, turn, winner, turn_no, chain_active,
   chain_banned, chain_starter, chain_current).
   Independence: `c.grid[0][0] = 99; assert state.grid[0][0] != 99`;
   `c.chain_banned.add((7,7)); assert (7,7) not in state.chain_banned`.
2. Behavior equivalence (`azero/tests/test_mcts.py`): for a fixed seed,
   `mcts_search` using `halma_clone` returns the **same** `best_action` as the
   current `copy.deepcopy` version on a small set of positions.
3. Full azero suite green; `py halma8_test.py` → 36 OK.
4. **Throughput check (report, do not assert):** a `_probe_timing.py` run at
   `batch_size=1` (T-10 off) comparing per-(move·sim) wall-clock for
   `copy.deepcopy` vs `halma_clone`; expect a measurable drop (the copy is the
   most-allocated object per sim).

**Risks / notes:**
- If `Halma8` ever gains a new mutable field, `halma_clone` will silently miss
  it. The field-equality gate catches that immediately.
- `chain_starter` / `chain_current` are tuples (immutable) — sharing the
  reference is safe. `chain_banned` is a set — **must** be copied, not shared.
- Composes with T-10: T-11 removes the per-child allocation cost, T-10 removes
  the per-leaf forward-call overhead — together they target the two biggest
  per-sim costs.

---

## T-12 — Teacher value target (give the value head a real signal) [DONE 2026-11-09]

**Goal:** Break the value head out of its ~0 attractor. Within `max_moves` the
net never wins, so **every** game is a draw and the value target is the dense
MCTS root value — near-constant (~0) for a weak net. The value head therefore
gets no informative gradient, and the run sits at Elo ~1000. Bootstrap it by
blending a **teacher** value target derived from the hand-crafted 6-term eval
(`best_weights.json`, 448/540) into the self-play value target, then fade to 0
for pure self-play. The teacher never plays and is not part of the trained net.

**Why this, not more eval games:** the eval (448/540) already knows which side
is winning; it is a dense, everywhere-nonzero signal, unlike true game outcomes
(zero within the move cap). Distilling it gives the value head a discriminative
target that MCTS/PUCT can then use to steer self-play toward actual wins.

**Files:**
- `azero/value_target.py` (new) — the teacher. `load_weights(path)`;
  `eval_signed(grid, turn, weights)` = `ai/crosscheck_eval.py::py_evaluate`
  negated for P2 (mover-perspective negamax); `normalized_value(grid, turn,
  weights, scale)` = `tanh(eval_signed / scale)` in [-1, 1]; `calibrate_scale(
  weights)` = p85 |eval| over ~40 random-walk positions (sub-second, CPU) so
  mid-range positions land near tanh(1)≈0.76 and only extremes saturate.
- `azero/selfplay.py` — `value_teacher`/`teacher_weights`/`teacher_scale` on
  `play_one_game` / `self_play_batch` / `selfplay_worker`. In `play_one_game`,
  each record also snapshots the grid (pre-move) and, when the teacher is on,
  sets `value = W*normalized_value + (1-W)*base` (`base` = the original
  true-outcome/dense target). **W=0 is byte-identical to the original
  behavior** (guarded by `value_teacher > 0 and teacher_weights is not None`).
- `azero/pipeline.py` — `run_pipeline`/`_collect_selfplay` accept the three
  params; when `value_teacher>0` it resolves the weights path (default
  `best_weights.json`), auto-calibrates `scale` (unless `--teacher-scale > 0`),
  and forwards to both the sequential and worker paths.
- `azero/run.py` — `--value-teacher` (default 0.0), `--teacher-weights`,
  `--teacher-scale` (default 0 = auto-calibrate).

**Design:**
- `W` in [0,1] is the bootstrap weight. `W=1.0` → the value head distills the
  eval (strongest jump-start); `W=0.5` → half distillation / half self-play;
  `W=0.0` → exactly the original pure self-play target.
- `scale` is the tanh divisor; auto-calibrated from live sample positions so
  the eval's magnitude maps cleanly into [-1, 1] (the eval itself spans ~±200).
- The teacher is **CPU-only, no extra model params** — it only shapes training
  targets. Removing it (`W=0`) leaves the net and the rest of the pipeline
  unchanged.

**Test gate (all green):**
- `azero/tests/test_value_target.py` (new): `eval_signed` P2 is exactly the
  negation of P1 (mover-perspective); `normalized_value` bounded in [-1,1] and
  sign-flips with turn; `calibrate_scale` finite and ≥1; `load_weights` loads.
- `azero/tests/test_selfplay.py`:
  - `test_value_teacher_blends_informative_target` — W=1.0 target has
    `mean|v| > 0.1` and ≥ the W=0 baseline (the ~0 dense MCTS target).
  - `test_value_teacher_off_is_original_behavior` — W=0 reproduces the exact
    pre-teacher value target (no hidden behavior change).
- `py -m pytest azero/tests -q` → **55 pass** (was 48, +7); `py halma8_test.py` → **36 OK**.

**Smoke (2-game self-play, W=0 vs W=1, fixed model seed):**
- W=0.0 (original): mean|v|=0.0185, std=0.0225, range [-0.042, 0.068] — the dead ~0 target.
- W=1.0 (teacher):  mean|v|=0.5093, std=0.6054, range [-0.963, 0.962] — a fully informative, well-spread value target.

**Recommended use (round-3 launch):** add `--value-teacher 0.5` to the T-8
command (or `1.0` for a hard jump-start). Once `elo=` climbs off 1000 and the
net starts producing wins, fade `W` toward 0 so the self-play signal takes over.

**Diagnostic kept:** `azero/_probe_stuck.py` — loads a checkpoint, plays games,
and reports value-target / model-value / policy / loss-split stats + how far the
net pushes into the enemy base. The acceptance check for "is the value head
alive now?".

---

## T-13 — Distance-to-enemy-base (centroid) shaping target [DONE 2026-11-09]

**Goal:** Add a second, *focused* value-target bootstrap on top of the T-12
eval teacher. The 6-term eval is a "who is better right now" snapshot in which
advancement toward the win is one diluted term; the win condition is an
*invasion* (a piece of the attacker's in a full enemy base). T-13 adds an
explicit "am I getting closer to the win" gradient: each side's 9-piece
**centroid** distance to the enemy base center it advances toward, as a
mover-perspective signed advantage in [-1, 1]. Same fade-to-0 pattern as the
eval teacher — it never plays and is not part of the trained net.

**Why:** measured fact (T-4 revised + T-8 rounds): the weak net's pieces barely
leave the home bases within `max_moves` (800/800 random games -> 0 winners even
at 400 moves). The eval teacher says "who is better" but both sides are
symmetric and similar; the distance term breaks the symmetry with a
directional signal — whichever side's mass is closer to the enemy base gets the
positive value — pushing MCTS-prioritized play toward advancement.

**Files:**
- `azero/value_target.py` — the new target. `enemy_base_center(mover)` maps
  P1 -> P2 base center `(1.0, 6.0)` and vice versa (base centers are the
  exact 3x3 home-square averages: P1 `(6.0, 1.0)`, P2 `(1.0, 6.0)`);
  `piece_centroid(grid, player)` = mean (r, c) of that player's pieces (None
  if it has none); `centroid_distance(grid, player, target)` = Euclidean
  centroid-to-target distance; `distance_advantage(grid, mover, scale)` =
  `tanh((d_their - d_mine) / scale)` in [-1, 1] — **symmetric and
  negamax-consistent** (`advantage(g, P1) == -advantage(g, P2)` exactly);
  `calibrate_distance_scale()` = p85 of raw |advantage| over ~40 random-walk
  positions (CPU, sub-second), mirroring `calibrate_scale`.
- `azero/selfplay.py` — `dist_teacher`/`dist_scale` on `play_one_game` /
  `self_play_batch` / `selfplay_worker`. The value blend generalized from one
  teacher to **two** shaping terms with the residual staying on the base:
  `value = (W_eval*eval + W_dist*dist + W_base*base) / (W_eval + W_dist +
  W_base)` where `W_base = max(0, 1 - W_eval - W_dist)`. Both 0 -> `value ==
  base` exactly (byte-identical pre-teacher behavior, preserved by the
  `w_eval == 0 and w_dist == 0` short-circuit).
- `azero/pipeline.py` — `run_pipeline`/`_collect_selfplay` accept
  `dist_teacher`/`dist_scale`; when `dist_teacher > 0` it auto-calibrates the
  scale (unless `--dist-scale > 0`) and threads it through both the
  sequential and the worker-args paths.
- `azero/run.py` — `--dist-teacher` (default 0.0) and `--dist-scale`
  (default 0 = auto-calibrate).

**Design:**
- The two teachers **compose, they don't stack**: with `--value-teacher 0.5
  --dist-teacher 0.5` the target is `0.5*eval + 0.5*dist` (W_base = 0); with
  `--value-teacher 0.5` alone it is `0.5*eval + 0.5*base` as before. This is
  convex in all three terms, so the target always stays in [-1, 1].
- **Magnitude is a dial (`--dist-scale`), not a bug.** Probe (seeded,
  n_sims=10, max_moves=20, W=1.0): dist_scale 4.0 / 2.0 / 1.0 / 0.5 ->
  mean|v| 0.027 / 0.054 / 0.108 / 0.207, max|v| 0.077 / 0.153 / 0.299 / 0.549,
  with ZERO saturated positions at any scale (tanh keeps gradation, so the
  "closer vs much closer" ordering survives). The weak net's pieces barely
  advance within the cap, so the raw |advantage| is small — but the pipeline
  AUTO-CALIBRATES the scale to 1.0 (random-walk p85 clamps at the floor), so
  the actual launch signal is mean|v| ~0.11 at 20 moves (baseline dense-MCTS
  target: 0.013). Bigger signal wanted: `--dist-scale 0.5` (~0.21). The
  payload is the *direction* (correct sign, non-constant) — that is what
  breaks the all-draw symmetry; magnitude only sets learning speed (MSE
  gradient is proportional to |target|).
- Zero at the symmetric start position (both sides equally advanced) — no
  spurious preference before any moves.

**Test gate (all green):**
- `azero/tests/test_value_target.py` (7 new T-13 tests): base-center mapping;
  start-position centroids exactly `(6.0, 1.0)` / `(1.0, 6.0)`; centroid
  distance finite; `distance_advantage` == 0 at the symmetric start;
  exact negation under mover swap (`a1 == -a2`); P1-advanced position gives
  P1 advantage > 0.5 and stays in [-1, 1]; `calibrate_distance_scale` >= 1
  and finite.
- `azero/tests/test_selfplay.py` (3 new gates):
  - `test_dist_teacher_blends_informative_target` — W=1.0 target spreads
    materially (> 1.5x) beyond the near-constant baseline and is non-zero in
    absolute terms (the threshold reflects the measured small magnitude, not
    an aspirational 0.1).
  - `test_dist_teacher_off_is_original_behavior` — `dist_teacher=0.0`
    reproduces the exact pre-T-13 value target element-wise (no hidden
    behavior change).
  - `test_both_teachers_blend_is_convex_and_informative` — with both teachers
    on, per-position |v| stays within the convex hull of the two individual
    teacher targets and the blend remains informative.
- `py -m pytest azero/tests -q` → **65 pass** (was 62, +3); `py halma8_test.py` → **36 OK**.

**Recommended use (round-3+ launch):** add `--dist-teacher 0.5` alongside the
existing `--value-teacher 0.5` (total shaping weight 1.0, W_base = 0 — the
value head distills 50/50 of the two teachers); add `--dist-scale 0.5` if a
stronger dist signal is wanted (2x magnitude, zero saturation). Once `elo=`
climbs off 1000 and the net produces wins, fade BOTH weights toward 0 for
pure self-play.

---

## T-14 — Endgame curriculum: teach the net to FINISH [DONE 2026-11-10]

**RESULT (2026-11-10):** all five implementation steps landed and gated.
`azero/endgame.py` (`make_endgame_positions` + `serialize`/`deserialize`),
`azero/selfplay.py` (`start_states` on `self_play_batch` + spawn-safe
`selfplay_worker`), `azero/pipeline.py` (`endgame_n`/`endgame_full` supplement
on top of `games_per_round`, both sequential + worker paths), `azero/run.py`
(`--endgame`/`--endgame-full`/`--finish-rate`), `azero/finish_rate.py` (held-out
convert metric), `finish_rate=` logged each round; tests `test_endgame.py` +
`test_finish_rate.py`. **GATES GREEN: azero 76 pass (was 65, +11), halma8 36 OK;
CLI `--help` exposes all flags.** The text below is the original spec (now
implemented); the live next step is the **launch recipe**.

**Goal:** un-stick the ~1000 Elo plateau by feeding the net positions that are
*already 1–3 moves from a win*, so the value target for those games is a true
terminal ±1 (the signal the all-draw start-from-fresh run never produces).
T-12/T-13 taught **approach** (measured working: closer to the enemy base,
~2.6× more invaders) but the net caps at **1 invader in a 6/9 base**. T-14
teaches **finish**: from a base at 8/9 full with an invader already in, the win
is a few moves away — and `play_one_game`'s value target for those games is the
real outcome (`winner is not None` → `base = ±1`), not the dense-MCTS ~0.

**Why this, not a longer horizon:** measured (T-4 revised) — 800/800 random
games → **0 winners even at 400 moves**; the weak net's pieces never reach the
enemy base in force, so raising `--max-moves` cannot help. The lever is not
"allow more moves," it is **"start closer to a win."** That is a curriculum,
not a longer search budget.

**Win rule to target (`halma8.py:155–167`):** `_full_and_invaded(base,
attacker)` = base all-9 non-empty **AND** contains ≥1 attacker piece. P1 wins
on `P2_BASE` (rows 0–2, cols 5–7); P2 wins on `P1_BASE` (rows 5–7, cols 0–2).
A near-win start = some base **8/9 full with ≥1 invader** (one more push fills
it; the invader is already in). Keep it at **8/9, not 9/9**, so `_check_win()`
has not already fired (a 9/9 + invader state is already terminal).

**Implementation order (each step has a green gate before the next):**

1. **`azero/endgame.py` (new) — position synthesizer.**
   - `make_endgame_positions(n, full=8, attacker_min=1, seed=None) -> list[Halma8]`
     returns `n` **independent** `Halma8` states, each **non-terminal** and
     **winnable-in-few**, each with exactly one base at `full`/9 occupied and
     ≥`attacker_min` of the *attacker's* pieces in it, and `state.turn` set to
     the attacker (so the +1 lands on the correct mover). Uses `base_cells(player)`
     (`halma8.py:14`) for the target base. **Primary strategy — template
     (reliable core):** place `full - attacker_min` defender pieces +
     `attacker_min` attacker pieces into the 9 base cells (rest of that base
     empty), scatter each side's remaining pieces in their own half (off both
     bases) to avoid an accidental double-win and unreachable clutter; set
     `turn` to the attacker; ensure `winner is None` and ≥1 legal move (else
     discard + regenerate). Deterministic, always finishable, no long walk.
     **Secondary — random-walk top-up (generalization, optional):** bounded
     `Halma8` walks with the chain-stop guard (see the replay-gotcha memory)
     keeping any non-terminal position where a base reaches the target fullness
     with an invader; supplement the templates with on-distribution positions
     (if < ~50% of `n` is reached this way, top up with templates).
   - `serialize(state) -> dict` / `deserialize(d) -> Halma8` — plain-Python
     round-trip of `grid`, `turn`, `winner`, `turn_no`, `chain_active`,
     `chain_banned`, `chain_starter`, `chain_current` (the same 8 fields
     `halma_clone` copies, T-11) so states cross the `spawn` worker boundary
     (T-9) without torch.
   - **Gate:** `test_endgame.py` — (a) `make_endgame_positions(20)` returns 20
     states, each `winner is None` with ≥1 legal move; (b) each has exactly one
     base at `full`/9 with ≥1 invader and the attacker's other pieces outside it;
     (c) `state.turn` **is** the attacker (hard gate — a wrong turn inverts the
     value signal); (d) one generated state is winnable within a handful of moves
     by a depth-limited finish (or the Rust engine via `halma_ai`); (e)
     `deserialize(serialize(s))` reproduces all fields.

2. **`azero/selfplay.py` — start-from-state self-play.**
   - `self_play_batch(..., start_states=None)`: when `start_states` is a list,
     play `len(start_states)` games, game g from `start_states[g]`, forwarding to
     the **existing** `play_one_game(..., start_state=...)` (already there since
     T-4). `start_states=None` is byte-identical to today's behavior.
   - `selfplay_worker(args)` threads `start_states` as a list of `serialize()`
     dicts in the worker args → rebuild with `deserialize()` before playing.
   - **Gate:** `test_selfplay.py` — `self_play_batch(start_states=[s])` returns a
     non-empty tuple list; from a forced-win start state the value targets hit
     ±1 for the winner's positions (true-outcome path, not dense-MCTS);
     `start_states=None` still reproduces the original batch element-wise.

3. **`azero/pipeline.py` — mix endgame games into a round.**
   - `run_pipeline(..., endgame_n=0, endgame_full=8)`: when `endgame_n > 0`,
     the round additionally plays `endgame_n` endgame games **on top of** the
     normal `games_per_round` fresh-board games — a *supplement* to, not a
     replacement of, the approach signal (the policy head still trains on both;
     the value head gets real ±1 from the endgame games). Thread through
     `_collect_selfplay` for **both** the sequential and the worker-args paths
     (endgame states serialized per chunk).
   - **Gate:** `test_pipeline.py` — a 2-game round with `endgame_n=2` logs the
     extra samples and does not crash; `endgame_n=0` is unchanged.

4. **`azero/run.py` — CLI.** `--endgame` (int, endgame games per round, default
   0 = off) and `--endgame-full` (int, target base fullness, default 8).

5. **`azero/finish_rate.py` (new) — the metric that can SEE the finish** (the
   full-game Elo is blind to it — the exact measurement trap from the T-4/T-8
   diagnosis). `finish_rate(model, n_positions=120, full=8, max_moves=40,
   n_sims=50, device, seed) -> float`: generate a **fresh, held-out** set of
   endgame positions (seed **different** from training), play each with
   `add_noise=False`, return the fraction the net converts to a true win within
   `max_moves`. Report it in `training_log.txt` each round alongside `elo=`.
   This is not gold-plating: without it we are back to "Elo 1000, did it work?"
   - **Gate:** `test_finish_rate.py` — returns a float in [0,1]; on a trivially
     winnable state set it is > 0 (a greedy finish wins at least some).

**Test gate (all green, after every step):**
- `py -m pytest azero/tests -q` → **65 pass + the new endgame/finish_rate gates**;
- `py halma8_test.py` → **36 OK**.

**Risks / mitigations:**
- **Off-distribution overfit** (net memorizes synthetic configs, not real
  games). → Endgame games are a supplement (default `--endgame 40–80` on top of
  `--games 120`), never a full replacement; keep the random-walk top-up for
  on-distribution coverage; trust the held-out `finish_rate` (separate seed) —
  if it climbs, generalization is real; if it stalls while in-distribution wins
  look fine, dial `--endgame` back.
- **Turn attribution inverts the value signal.** → `state.turn` = attacker is a
  hard test gate; the +1 must go to the side holding the invader.
- **`spawn` worker can't pickle a live `Halma8`.** → states travel as
  `serialize()` dicts (model still travels as a CPU state dict, unchanged from
  T-9).
- **Synthetic state accidentally terminal** (base 9/9 + invader). → generate at
  `full=8`; `make_endgame_positions` discards any state with `winner is not None`.

**Launch recipe (next session, isolated free-GPU — the net is ~1.23M params, VRAM is a non-issue):**
```
py azero/run.py --rounds 2 --device cuda --resume azero/checkpoints/round_5.pt \
  --games 120 --sims 100 --epochs 10 --elo-games 120 --elo-sims 100 \
  --workers 8 --mcts-batch 32 --endgame 60 --endgame-full 8 \
  --finish-rate 120 --finish-rate-sims 30
```
**`--finish-rate 120` is required** (default 0 = off): without it the pipeline
skips the metric and the round log has no `finish_rate=` column — that column is
the leading indicator T-14 exists to produce. `--finish-rate-sims 30` is the
MCTS budget for the held-out near-win positions.
Teachers **off** here (default 0): the endgame games' true ±1 outcomes *are* the
value signal, and blending a teacher into them would dilute it; the fresh-board
games still train the policy head. Expect ~2× the lean-round ~45–50 min for the
endgame extra. The **`finish_rate` column climbing off 0.00** is the leading
indicator that the value head now knows how to finish, and will precede any full-game
Elo movement.

**Done-when:**
- `make_endgame_positions` produces valid, winnable, correctly-turned positions
  (test-gated); `finish_rate` is a working held-out metric (test-gated).
- A round with `--endgame 60` logs a **`finish_rate > 0`** on the held-out set and
  the endgame games' value targets include real ±1 (not all ~0).
- `finish_rate` climbs over 2 rounds; azero + halma8 gates green.
- This file updated (T-14 ticked ✅, "Where we are" line refreshed).

---

## T-15 — Endgame deadlock fix: make the near-win games actually produce a true ±1 [DONE 2026-11-12]

**Problem (found while validating T-14):** the endgame curriculum only helps if the
near-win games actually decide in the attacker's favour, so the value head gets a
real terminal ±1 target. They were *not*. Measured on `round_6.pt` (probes
`azero/_probe_conv.py`, `_probe_evalue.py`, `_probe_evalue_fix.py`):

- The net's softmax prior on the winning move is near-uniform (**0.002–0.108**,
  mostly ~0.01) and the value head reads ~0 on these positions (dead). The net
  does **not** "know" the win — it can only find it if MCTS explores it.
- At the fresh-board default (**low `n_sims` + Dirichlet root noise**), that
  low-prior winning action gets starved to ~0 visits, the game never converts,
  and the value target stays ~0. The chicken-and-egg deadlock stays alive *even
  on the curriculum*.
- Conversion of a win-in-1 position vs sim budget (noise off): **30 sims → 3/10,
  60 → 6/10, 200 → 9/10.** The lever is the sim budget (+ noise off), not a
  bigger fresh-board search.

**Fix:** the endgame near-win games run **noise-OFF at a dedicated (higher) sim
budget**, separate from the fresh-board `--sims`/noise (fresh games keep noise on
for diversity). The two knobs thread the whole path:

- `azero/selfplay.py` — `selfplay_worker` reads `endgame_n_sims` /
  `endgame_add_noise` from its `args` dict and applies them to the endgame chunk
  (fresh-board chunk keeps `n_sims`/`add_noise`).
- `azero/pipeline.py` — `_collect_selfplay` accepts `endgame_n_sims` (default 200)
  and `endgame_add_noise` (default **False**) and threads them through **both**
  the sequential path (explicit `add_noise`/`n_sims` on the endgame
  `self_play_batch` call) and the worker-args path.
- `azero/run.py` — `--endgame-sims` (default 200) and `--endgame-noise`
  (`store_true`, off by default).

**Verified empirically (4090):**
- `_probe_conv.py`: 9/10 convert at 200 sims (noise off).
- `_probe_evalue_fix.py` end-to-end through `_collect_selfplay`:
  **sequential path → 20/68 samples are genuine ±1 (18×+1, 2×−1); worker path →
  8/8 all +1.** Before the fix these were ~0.

**Gates (azero 78 pass, was 76, +2; halma8 36 OK):**
- `test_pipeline.py::TestEndgameCurriculum::test_endgame_uses_dedicated_budget_and_noise_off`
  — the noise-off / dedicated-sims endgame path yields a true terminal +1.
- `test_pipeline.py::TestEndgameCurriculum::test_endgame_defaults_are_noise_off`
  — pins `endgame_add_noise` defaulting to `False` on `_collect_selfplay` and
  `run_pipeline` (regression guard against reverting to the noise-on default).

**Launch recipe (isolated free-GPU) — now with the fix's defaults applied:**
```
py azero/run.py --rounds 2 --device cuda --resume azero/checkpoints/round_5.pt \
  --games 120 --sims 100 --epochs 10 --elo-games 120 --elo-sims 100 \
  --workers 8 --mcts-batch 32 --endgame 60 --endgame-full 8 \
  --endgame-sims 200 --finish-rate 120 --finish-rate-sims 30
```
Keep `--endgame-noise` **off** (the default). `--endgame-sims` defaults to 200,
the budget at which ~9/10 win-in-1 positions convert. Watch the `finish_rate`
column climb off 0.00 — it is the leading indicator the value head has learned to
finish, and will precede any full-game Elo movement.

---

## File structure (final)

```
azero/
  __init__.py
  _check_gpu.py          # T-0: GPU verification (can delete after T-0)
  board.py               # T-1: encode_board, ACTION_LIST, action_index,
                         #      decode_action, legal_action_indices
  model.py               # T-2: HalmaNet, policy_loss, value_loss, total_loss
  mcts.py                # T-3: MCTSNode, mcts_search
  selfplay.py            # T-4: play_one_game, self_play_batch (+T-12 value-teacher, +T-13 dist-teacher, +T-14 start_states, +T-15 endgame noise-off/dedicated-sims)
  value_target.py        # T-12: teacher value target (eval->mover-perspective [-1,1]); T-13: centroid distance-to-enemy-base
  train.py               # T-5: train_step, train, save/load_checkpoint
  elo.py                 # T-6: evaluate_elo (NN vs baseline, baseline=1000)
  pipeline.py            # T-6: run_pipeline (+T-14 endgame curriculum, +T-15 endgame_n_sims/endgame_add_noise knobs)
  run.py                 # T-6: CLI entry point (argparse; +T-14 --endgame, +T-15 --endgame-sims/--endgame-noise)
  evaluate.py            # T-7: evaluate_nn_vs_rust
  endgame.py             # T-14: make_endgame_positions (near-win synthesizer), serialize/deserialize
  finish_rate.py         # T-14: finish_rate (held-out "does it convert near-wins to wins" metric)
  tests/
    __init__.py
    test_board.py        # T-1 gate
    test_model.py        # T-2 gate
    test_mcts.py         # T-3 gate
    test_selfplay.py     # T-4 gate (+T-12 value-teacher gates, +T-13 dist-teacher gates)
    test_value_target.py # T-12 gate (teacher eval) + T-13 gate (distance target)
    test_endgame.py      # T-14 gate (synthesizer + serialize round-trip + winnability + turn)
    test_finish_rate.py  # T-14 gate (finish_rate metric)
    test_train.py        # T-5 gate
    test_pipeline.py     # T-6 gate
  checkpoints/
    .gitkeep
  training_log.txt       # T-6/T-8: one line per round
```

---

## Design decisions (rationale)

1. **Python, not Rust, for the NN.** PyTorch is Python-native. The game is
   small enough that Python's overhead is negligible compared to the GPU
   forward passes. The Rust engine remains for the GUI; the NN is a separate
   concern under `azero/`.

2. **Fixed action list (417 actions), not variable.** A CNN needs a fixed
   output size. Masking illegal actions before softmax is the standard, clean
   solution. 417 is small enough for a single Linear layer.

3. **Value head outputs "outcome for the player to move" in [-1, +1].**
   Negamax convention. Works correctly with chains because `board.turn` at
   each node is always well-defined, even mid-chain.

4. **No random rollouts.** The value head replaces the rollout. This is the
   key AlphaZero improvement over vanilla MCTS and is much more
   sample-efficient.

5. **Dirichlet noise only at the root, only during self-play.** Standard
   exploration mechanism. Not applied during evaluation (T-7) so the NN
   plays its best move.

6. **Soft policy loss (CE with soft targets), not hard CE.** The MCTS visit
   distribution is a soft target. The network learns to match the full
   distribution, not just the argmax. This is critical for learning quality.

7. **`halma8.py` is the rule oracle.** The NN never re-implements legality.
   All rule queries go through `Halma8` methods. This guarantees the NN only
   plays legal moves and stays in sync with the GUI's rules.

8. **`copy.deepcopy` for MCTS node states, not make/unmake.** The game state
   is small (a few hundred bytes). Deep copy is simple, correct, and fast
   enough. Make/unmake adds complexity for no measurable benefit at this
   scale.

9. **Sequential self-play in T-4.** Keep it simple and correct first.
   Parallelism can be added later if self-play is the bottleneck.

---

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| CUDA build mismatch (torch vs driver) | Check `nvidia-smi` first in T-0; install matching build |
| Python MCTS too slow (self-play bottleneck) | Batch NN forward calls; reduce `n_sims`; if still too slow, move MCTS to Rust (future work) |
| Overfitting to self-play patterns | Dirichlet noise + fresh self-play data each round |
| Windows cp1252 crash on non-ASCII print | No non-ASCII in any print (hard rule #4) |
| Training diverges (NaN loss) | Gradient clipping (max_norm=5.0); monitor `training_log.txt` |
| Chain state encoding wrong | T-1 test gate verifies encoding on known mid-chain positions |
| Games hit max_moves (random model) | Expected in early rounds; discard those games (no value signal) |
| Rust engine not built for T-7 | Run `cargo build --release` in `ai/` before T-7 |

---

## Out of scope (future work)

- GUI integration (loading the NN model into the Tkinter GUI as an AI seat)
- Rust-based MCTS (for speed, if Python is too slow)
- Multi-GPU training (not needed for a 1.5M param network)
- Distribution/packaging of the NN as a standalone executable
- AlphaZero-style "perfect play" (unbounded MCTS) — overkill for this game

---

## Quick reference: the 6-term eval being replaced

For context, the NN replaces this (from `ai/src/evaluate.rs`):

| term | meaning | best weight (448/540) |
|------|---------|----------------------|
| progress | distance toward target base | 448 |
| base_presence | pieces in target base | 540 |
| defense | blocking opponent's path | 6 |
| spread | piece distribution | 4 |
| mobility | available moves | 9 |
| threat | immediate winning threats | 130 |

The NN learns all of this (and more) from raw board state. No human decides
what matters.
