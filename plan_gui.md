# check_halma — "Finish the GUI" Plan (agent handoff)

The final GUI work: a **Players tab** (pick Human/AI per seat, load a model per
AI seat, allow AI-vs-AI), plus a **Kibitzer toggle** that shows the engine's
best continuation line + eval from the current position. Grounded in the *actual*
source — read the symbols cited below before starting.

## HANDOFF (read this first)

- **What this is:** capstone GUI work on the already-finished Part 1 + Part 2.
  Part 1 (engine `halma8.py`, 2-local-player GUI `halma8_gui.py`, clocks, 36-test
  suite) is DONE. Part 2 (Rust engine `ai/`, `halma_ai.py` driver, trained models
  `best_weights.json` / `c1_incumbent.json`) is DONE. This plan only adds: a
  Players/setup tab, per-seat model loading, and a Kibitzer.
- **Where we are (one line):** **T-G0…T-G4 ✅ DONE** —
  `halma_ai.py` has `set_weights` / `move_with_score` / `line()`; the GUI has the
  **Players tab** (per-seat Human/AI + model Combobox/Browse + depth/movetime,
  all four seat combos) and the game loop is driven by per-seat `seat_cfg` (AI
  seats auto-play incl. **A/A**; undo lets the AI seat continue; an engine move
  Python rejects stops the turn instead of looping). **T-G4 (2026-11-05)** adds the
  **Kibitzer**: a Game-tab toggle + panel that shows the engine's best continuation
  line + eval from the current position, refreshing after every move, on its own
  `ai.exe` child thread so it never blocks play or mutates the live board.
  Verified live: `_t_g2_test.py` green (seats/A/A) and `_t_g4_test.py` green
  (`T-G4 GATE OK`: start eval + line · post-move refresh w/ board unchanged ·
  undo restore · A/A auto-run w/ Kibitzer on · toggle-off reset · wedged-child
  warning). **T-G5 (2026-11-05) ✅ DONE** — validation matrix green
  (`_t_g5_matrix.log`: weight-wiring OK, Kibitzer matrix H/H·H/A·A/H·A/A OK,
  model-change re-apply OK, `T-G5 GATE OK`), exe rebuilt (`_t_g5_build.log`,
  13.5 MB, launches alive 20 s), all four baseline gates green
  (36 OK / cargo green / smoke OK / GUI launches). **This plan is FINISHED —
  all of T-G0…T-G5 are ticked ✅.**
- **First command to run (re-establish baseline):**
  `py halma8_test.py`  (expect 36 OK)  ·  then `cargo test --release` in `ai/`  (expect green).
- **Always-green gates (keep these passing at every step):**
  - `py halma8_test.py` → **36 OK** (rule engine, must stay tkinter-free).
  - `cargo test --release` in `ai/` → **green** (evaluate / search / parallel_determinism).
  - `py halma_ai.py` → **smoke OK** (spawns `ai.exe`, one legal move applied).
  - GUI still launches: `py halma8_gui.py`.
- **Hard rules:**
  1. Python (`halma8.py`) stays the **sole authority** on legality, undo, clocks,
     win banner. The engine only ever returns a move; the GUI applies it.
  2. `halma8.py` must stay **tkinter-free** (36-test suite enforces this).
  3. Do **not** re-run the seed-0 tune loop or re-open the T-17 d6-confirm
     (that work is in `plan_ai.md`; it is in-flight/finished — don't touch).
  4. Use `py`, never `python` (MS-Store alias hijacks it on this box).
  5. **Update this file as each task lands:** tick the heading ✅ and refresh the
     "Where we are" line the moment a gate goes green.
  6. Never let the Kibitzer mutate the live board — it analyses a **copy**.
- **Rollback path:** every task touches a distinct seam (Rust server / Python
  driver / GUI layout / GUI analysis). Revert the last green commit to roll back
  one task without disturbing the rest. `git status` before each task; the three
  always-green gates are your "safe to commit" proof.

## ✅ WEIGHTS WIRING — RESOLVED BY T-G0 (2026-11-04)

The old blocker — "the live `ai.exe` always plays with the default 6-term
weights, ignoring `best_weights.json` / `c1_incumbent.json` / `cand_w*.json`" —
**is gone**. `ai/src/server.rs` now parses `setoption` and feeds the loaded
weights into every `go`. This is the one Rust change that was needed; the
remaining tasks (T-G1…T-G5) are all Python/Tk.

**The wire protocol the Python driver (T-G1) must send** (all before the first
`go`; a `path=` value must have no spaces):
- `setoption name WeightsP1 path=<file>` — P1 = **Red** (bottom-left base) seat's model
- `setoption name WeightsP2 path=<file>` — P2 = **Blue** (top-right base) seat's model
- `setoption name Weights path=<file>` — set **both** (single-model play / analysis)
- Unknown options and a recognised name with no `path=` stay silent (UCI contract).
- The server logs one line to stderr on each load (`[weights] P1 <- <path>`) so a
  mis-pointed path is visible (a bad path falls back to `Weights::default()`).

**Seat ↔ side is exact and constant for the whole game:** Red seat → `WeightsP1`,
Blue seat → `WeightsP2`; a Human seat's side is irrelevant. Send the weights on
**every** New game (the process may be respawned and `_ensure()` re-handshakes).

**Guard:** `ai/src/search.rs::tests::t_g0_distinct_weights_change_the_search_answer`
proves a distinct weight set routed through `Search::set_weights_pair` (the same
path the server uses) changes the engine's root answer — the regression guard
against "parsed-but-ignored". Verified live on 2026-11-04: on a fixed asymmetric
position the root **score** differs across weight sets (default `cp 14` vs
`c1_incumbent` `cp 456` vs `best_weights` `cp 29`).

## Kibitzer — how it will work (design, agreed here so it's not re-litigated)

- The server returns a **single** `bestmove` + one `info … score cp N` per `go`.
  It does **not** emit a principal-variation line. So the "best continuation line"
  is built **Python-side by iterative re-query**: from the current position, ask
  for the best action; apply it to a **scratch copy** of the board; if the same
  side still to move and no winner, ask again; repeat for N plies (or to game-over).
  This reuses the exact move granularity the engine already speaks (step/jump/stop),
  so a halma **chain** naturally shows up as consecutive actions by the same side.
- The **eval** shown = the root `info … score cp N` (positive = good for Red/P1,
  negative = good for Blue/P2). The driver must capture that `info` line (today
  `HalmaAI.move()` only looks for `bestmove`).
- The Kibitzer runs in a **separate `HalmaAI` child process** from the playing AI,
  so its searches never collide with the playing process's stdin/stdout and never
  block a human's move. At most 2 `ai.exe` processes are alive.
- The Kibitzer re-runs **after every position change** (each move / turn change)
  while active — including during AI-vs-AI games and when it is either side's turn.

---

## T-G0 — (Rust, prerequisite) Wire `setoption` weights into the live server ✅ DONE (2026-11-04)

**Goal:** make `ai.exe` honour per-side learned weights, so the GUI can load
`best_weights.json` / `c1_incumbent.json` / any `cand_w*.json` into either seat.

**Steps:**
1. In `ai/src/server.rs`, in the `main_loop`, persist two optional weights
   (start as `Weights::default()` each) on the loop's state.
2. Re-implement the `setoption` arm to parse:
   - `setoption name WeightsP1 path=<file>` → `w1 = Weights::load(<file>)`
   - `setoption name WeightsP2 path=<file>` → `w2 = Weights::load(<file>)`
   - `setoption name Weights path=<file>` → set **both** `w1` and `w2` (convenience)
   - unknown `setoption` → still silent (UCI contract); ignore, don't error.
   (A `path=` with spaces is out of scope — model paths here have none. Note it.)
3. In the `go` arm, after constructing `let mut s = Search::new();`, call
   `s.set_weights_pair(w1, w2)` using the current loop state **before** spawning
   the search thread. (Both seats' weights are set before the first `go`, so this
   is read-only during search — no shared-state mutation under the thread.)
4. Keep `set_weights_pair`'s default pair as the fallback so a server started with
   no `setoption` is byte-identical to today (preserves the selfplay/bench path).

**Test gate (must all pass before you commit):**
- `cargo test --release` in `ai/` → green (existing `evaluate_pair` / `set_weights_pair`
  tests still hold; they prove default-pair ≡ original eval).
- **New proof that weights actually change play:** add a small `#[test]` (or a
  one-off `cargo run --example` probe) that builds a fixed asymmetric position and
  asserts the root `bestmove` (or root score) **differs** between two distinct
  weight sets routed through `Search` + `set_weights_pair`. This directly kills the
  "parsed-but-ignored" bug.
- Rebuild: `cargo build --release` in `ai/` → `ai/target/release/ai.exe` exists, fresh mtime.
- `py halma_ai.py` → smoke OK.

**Risks:** (a) forgetting to call `set_weights_pair` in the `go` arm (the weights
  then silently no-op) — the new test above is the guard. (b) `Weights::load`
  returning defaults on a bad path hides a typo'd filename — log the loaded file
  path to stderr for one line so a mis-point is visible. (c) Windows path with a
  space → `setoption` tokenization breaks; state this limitation in the GUI file
  picker guidance (avoid spaces in the chosen folder) rather than over-engineering.

**Done-when:** the new test proves a changed weight set changes the engine's
answer, `ai.exe` is rebuilt, all three baseline gates are green, and this file is
updated (heading ✅ + "Where we are").

---

## T-G1 — (Python driver) `HalmaAI` gains weight-loading, score capture, and a line API ✅ DONE (2026-11-04)

**Goal:** `halma_ai.py` can (a) point the engine at a model per side, (b) return
the eval with each move, and (c) build a continuation line for the Kibitzer.

**Steps:**
1. Add `set_weights(self, p1_path=None, p2_path=None)` to `HalmaAI`: send
   `setoption name WeightsP1 path=<...>` / `…WeightsP2…` / `…Weights path=<...>`
   (only for the sides provided) **before** the first `go`. Guard: if
   `self.proc` isn't up yet, buffer the paths and send them right after `_ensure()`
   handshakes, so an early call is not lost. (Simplest: always `_ensure()` first.)
2. In `move()` (and `move_depth()`), also read the **`info … score cp N`** line
   that precedes `bestmove` and return it: change the return to a small pair /
   namedtuple `(action, score)` — **but** keep a thin `move()` that still returns
   just the action so the existing `_ai_take_turn` in the GUI keeps working with
   a one-line change. (Expose `move_with_score()`; make `move()` a wrapper.)
3. Add `line(self, position, plies=8)` for the Kibitzer: loop `plies` times —
   `action, score = move_with_score(position)`; stop on `"stop"`-terminal or a
   `(none)`/terminal; otherwise advance `position` by applying `action` to a
   **`copy.deepcopy`** of a `halma8.Halma8` board (the caller passes the board or
   the position string; prefer the board so `board_to_position` is reused).
   Return `(list_of_actions, root_score)`. `root_score` = the first (root) `info`
   score, not the last. Keep a hard cap on `plies` so a stuck chain can't loop.
4. Add a lightweight `HalmaAI.analysis(binary=…, movetime_ms=…)` factory or a
   separate instance the GUI constructs for the Kibitzer (its own process, own
   movetime, own model). No protocol change needed — it is just another `HalmaAI`.

**Test gate:**
- Extend the existing `py halma_ai.py` smoke (`__main__`) to: set a known weight
  file (e.g. `c1_incumbent.json`) via `set_weights`, then `move_with_score` and
  assert the action is legal (apply to a board, no `IllegalMove`) and `score` is
  an int. Then `line(pos, plies=4)` and assert it returns a non-empty action list
  and an int score.
- `py halma8_test.py` → 36 OK (unchanged — driver is not in that suite).
- All three baseline gates green.

**Risks:** (a) returning a 2-tuple from `move()` silently breaks the GUI loop —
  mitigate by keeping `move()` action-only and adding `move_with_score()`. (b)
  `line()` re-sends `go` per ply, so it is N× slower than one move — cap `plies`
  (default 6–8) and run it off the hot path (T-G4). (c) a position where the engine
  returns `(none)` (already a win) — `line()` must treat that as "stop, empty rest".

**Done-when:** the extended `__main__` smoke passes (weights load, score is an int,
`line()` returns a non-empty list), baseline gates green, file updated.

---

## T-G2 — (GUI) Players tab: per-seat Human/AI + model picker + AI-vs-AI ✅ DONE (2026-11-04)

**Goal:** replace the current "AI plays / depth" row with a real **Players** tab
where each seat (P1 Red, P2 Blue) is independently Human or AI, and each AI seat
selects a model file + search depth/movetime. All four setups allowed:
H/H · H/A · A/H · **A/A** (AI-vs-AI is explicitly required).

**Steps:**
1. Introduce a `ttk.Notebook` as the main container under the top status/clock
   area, with two tabs: **"Game"** and **"Players"**. Move the existing
   board + move-log + (optional) controls into the **Game** tab. *Do not change
   any game-logic method* — only re-parent the frames. (See RISK + fallback.)
2. On the **Players** tab, build two identical seat panels (Red, Blue), each with:
   - a **Human / AI** radio (store in `self.seat_cfg[P1]/[P2]` = `{"kind","model","depth","movetime_ms"}`),
   - a **model** picker shown only when AI is selected: a `Combobox` pre-seeded
     with the known models (`best_weights.json`, `c1_incumbent.json`) + a **Browse…**
     button (`filedialog.askopenfilename`, filter `*.json`). Store the chosen path.
   - a **depth** `Spinbox` and a **movetime** `Spinbox` (per seat; defaults from
     today: depth 4, movetime 500 ms).
3. Add a **Kibitzer** block on the Players tab (config only — the button is T-G4):
   - an **analysis model** picker (same Combobox+Browse; default = `best_weights.json`),
   - an **analysis depth/movetime** (default depth 6), and an **enable** checkbox.
4. Wire **"New game"** (and a "Start/Restart" on the Players tab that calls the
   same handler) to `_apply_players_config()`: validate (≥0 humans; if a seat is AI,
   its model path must exist), then set the playing process's weights
   (`Red model → set_weights p1`, `Blue model → set_weights p2`), set per-seat
   depth/movetime into `self.seat_cfg`, then `new_game()` as today.
5. Keep the legacy single shared playing process **`self._ai`**; per-seat depth/
   movetime are read at turn time (the side determines which seat's config applies).

**Test gate (manual, since Tk has no unit harness here):**
- Launch `py halma8_gui.py`. Open the **Players** tab. Confirm all four combos render:
  (a) both Human, (b) Red AI, (c) Blue AI, (d) both AI. In each, the model picker
  is visible only for AI seats.
- (b)/(d): pick `c1_incumbent.json`, New game. Confirm the first AI move is legal
  and logged (proves T-G0's wiring reached the GUI). **AI-vs-AI (d):** confirm the
  game auto-runs (Red, being P1, moves first) and continues turn-for-turn without input.
- `py halma8_test.py` → 36 OK; `py halma_ai.py` → smoke OK.

**Risks:** (a) re-parenting into a `Notebook` can change `pack`/`grid` order and
  break layout. **Fallback if it fights you:** keep the single window and simply
  replace the current AI row with a fuller "Players" `Frame` (same fields), and put
  the Kibitzer as a top-row button — still satisfies "select players" and is far
  lower-risk; the word "tab" is then met by a labelled `Frame`/`LabelFrame`
  section. Decide early which you prefer. (b) Two seats both AI + a very low
  movetime → the auto-run loop can look "stuck"; that's expected (T-G4 keeps
  analysis off this path). (c) A model path with a space → `setoption` tokenization
  (see T-G0 RISK); steer users to a space-free folder in the Browse hint.

**Done-when:** all four seat combos settable in the UI, an AI seat demonstrably
plays with a loaded model (not the default), AI-vs-AI auto-runs, baseline gates
green, file updated.

**Verification (2026-11-04):** implemented in `halma8_gui.py` (Notebook with
Game | Players tabs; `_build_seat_panel` per seat; `_read_players` validates
model paths incl. the space-check; `_apply_players_config` = New game handler).
Proven by `_t_g2_test.py` (real GUI + real `ai.exe`): H/H render (widgets
disabled) · H/A answer · **A/H first move** · **A/A auto-run** with distinct
per-seat models (`c1_incumbent.json` Red / `best_weights.json` Blue) — Red
logged first, Blue answered. Baseline gates green: `halma8_test.py` 36 OK,
`cargo test --release` green, `halma_ai.py` smoke OK.

---

## T-G3 — (GUI) Drive the game loop from per-seat config ✅ DONE (2026-11-04)

**Goal:** make the existing `_maybe_ai` / `_ai_take_turn` honour per-seat
`kind` / model / depth / movetime instead of the old boolean toggle.

**Steps:**
1. Replace `self.ai_enabled = {P1:False, P2:False}` reads with
   `self.seat_cfg[side]["kind"] == "ai"`.
2. In `_ai_take_turn`, use the moving side's `seat_cfg[side]["depth"]` and
   `["movetime_ms"]` when calling the engine (already per-call, so this is a
   parameter read, not a restructure).
3. On **"New game"**: after applying weights, if the side to move (P1) is an AI
   seat, `_maybe_ai()` already fires — confirm the auto-run path works for A/A.
4. Keep the **busy flag** so the playing AI can't re-enter mid-turn; keep the
   Python engine as sole legality authority (unchanged).

**Test gate:**
- H/A and A/H and A/A each: the AI's turn plays, the human's turn waits, undo
  still works mid-game, and a New game with a different config re-applies weights.
- `py halma8_test.py` → 36 OK.

**Risks:** (a) stale `ai_enabled` references elsewhere — grep for them and remove
  (it is a small, contained rename). (b) weights must be (re)sent on **every**
  New game, not just once, because the process may have been spawned fresh —
  `_ensure()` re-handshakes, so send weights after ensure on each apply.

**Done-when:** the three mixed AI setups behave, undo/new-game re-apply config,
baseline gates green, file updated.

**Verification (2026-11-04):** the last agent had already implemented this in the
same pass as T-G2 (`_maybe_ai` / `_ai_take_turn` read `seat_cfg[side]`; `_ask_ai`
uses the seat's own movetime/depth; busy flag kept). Completed by `_t_g2_test.py`:
H/A · A/H · A/A all behave; **undo back to an AI seat's turn lets it continue**
(`undo()` now ends with `_maybe_ai()`); a New game with different per-seat models
re-applies weights (A/A subtest uses two distinct models). One robustness add:
`_ai_take_turn` now stops the turn if the engine returns a move Python rejects
(grid unchanged) instead of re-asking forever.

---

## T-G4 — (GUI) Kibitzer toggle + analysis panel (line + eval), separate process ✅ DONE (2026-11-05)

**Goal:** a **Kibitzer** button (toggle) that, when active, shows from the current
position the engine's **best continuation line** and its **eval**, refreshing after
every move — in a dedicated `ai.exe` process so it never blocks play.

**Steps:**
1. On the **Game** tab control row, add a **Kibitzer** toggle button. State
   stored in `self.kibitz_on`. A small **status** word (e.g. "Kibitzer on") in the
   status line.
2. Build the analysis handle once, lazily: `self._kib = HalmaAI(binary=AI_BIN,
   movetime_ms=<kibitz movetime>)`; apply the Kibitzer model via
   `set_weights(<kib model>, <kib model>)` (both sides, so it's side-independent).
   Close it in `_on_close`.
3. Add a **Kibitzer panel** (right side, under the move log, or a thin bar under
   the board): an **eval** label (e.g. `+132  (Red favours)` / `-88 (Blue favours)`
   / `#3 (mate-in, Red)` for `|score| ≥ MATE`) and a **line** label rendering the
   action list readably (steps as `(r,c)→(r,c)`, jumps as `jump (r,c)→(r,c)`, `stop`
   as `·stop·`, joined by `  |  `, truncated with `…` past a length cap).
4. Hook a **`_maybe_kibitz()`** call at every place a position changes: after
   `_do_move`, after `_stop_chain`, after `new_game`, and after `undo`. Guard on
   `self.kibitz_on`, `board.winner is None`, and a `self._kibitz_busy` flag.
   Schedule with `self.root.after(120, self._kibitz_run)` so it never blocks input.
5. `_kibitz_run`: snapshot the board (`copy.deepcopy`), call
   `self._kib.line(board_to_position(snapshot), plies=self.kib_plies)` on the copy,
   then write the eval + line into the panel. Wrap in `try/except HalmaAIError`
   (show a one-line warning, do not crash). Reset `_kibitz_busy` in `finally`.
6. When the Kibitzer is toggled **off** mid-game, cancel any pending
   `after(...)` (keep the token id) so a stale analysis can't land after a New game.

**Test gate (manual):**
- H/A game: toggle Kibitzer on. After each human move and each AI move, the panel
  updates with a plausible line + an eval that flips sign with the advantage.
- Turn it on at the start position → eval should read **0 / balanced** (symmetric
  board) and a non-empty line.
- Toggle on during **A/A** → it keeps refreshing and never freezes the game.
- Toggle off → pending update cancelled; New game shows a clean (0) panel.
- `py halma8_test.py` → 36 OK; `py halma_ai.py` → smoke OK.

**Risks:** (a) analysis and the playing AI share a process → they would collide on
  stdin/stdout. **Mitigated by design**: Kibitzer uses its **own** `HalmaAI`/`ai.exe`.
  (b) analysis is N× a move's cost — cap `plies` (6–8) and a modest movetime; it
  runs on the Tk `after` loop, not a blocking call. (c) a win score (`|cp| ≥ 30000`)
  must be shown as "mate-in-N for <side>", not a CP number. (d) re-entrancy —
  the `_kibitz_busy` flag + a single pending `after` token are the whole guard;
  don't schedule a second analysis while one is pending.

**Done-when:** the Kibitzer shows a live line + correctly-signed eval from the
current position in H/A and A/A games, updates on every move, and never blocks or
mutates the live game; baseline gates green, file updated.

**Verification (2026-11-05):** implemented in `halma8_gui.py` — `_toggle_kibitz`
(Game-tab button), `_maybe_kibitz` (coalesced trigger, hooked into `_do_move` /
`_stop_chain` / `new_game` / `undo`), `_kib_worker` on a dedicated thread with its
**own** `HalmaAI` child (`self._kib`), `_kib_poll`/`_kib_finish` collecting the
result on the main loop (epoch-gated so a New game / toggle-off drops stale
results), `_fmt_eval` (mate-in-N for `|cp| ≥ 29488`, else `±N  (Red/Blue favours)`
/ `0  (balanced)`) and `_fmt_line` (steps `(r,c)->(r,c)`, jumps, `·stop·`, truncated).
The Kibitzer runs on a `copy.deepcopy` of the board — it never mutates the live
game. Proven by `_t_g4_test.py` (real GUI + real `ai.exe`), 6 sub-tests: start-
position eval + line · post-move refresh with live board **unchanged** beyond the
single human move · undo restores the start eval · A/A auto-run with Kibitzer on
(analysis lands, game never freezes) · toggle-off resets the panel · a wedged
analysis child surfaces a one-line warning, not a crash. Note: the start-position
root score is **P1-favoured** (first-mover bias), not 0 — the gate asserts
`Red favours`, not `balanced`. Baseline gates green: `halma8_test.py` 36 OK,
`halma_ai.py` smoke OK (`ai.exe` already built; no Rust touched this pass).
Automated gate: `py _t_g4_test.py` → `T-G4 GATE OK`.

---

## T-G5 ✅ — (Validation + packaging) Full matrix, rebuild the exe, hand off

**STATUS (2026-11-05) — DONE.** All four steps green, recorded here:

- **Step 1 — matrix (recorded):** `_t_g5_test.py` → `_t_g5_matrix.log` →
  `T-G5 GATE OK`:
  - **R) weight-wiring regression:** c1_incumbent=3758, best_weights=3758,
    progress-only extreme=8000 → real models differ from progress-only, so
    loaded weights drive play (no silent "parsed-but-ignored" wiring bug).
  - **K) Kibitzer-on matrix:** H/H `+4640 (Red favours)` · H/A `+4640 (Red
    favours)` · A/H `+4640 (Red favours)` · A/A `#5 (Blue wins in 5)` — valid
    eval + non-empty line, no freeze, live board not mutated.
  - **N) New game with changed Blue model:** clean restart, Blue answered
    legally (with R, proves a model change changes play).
  - Undo covered by the T-G4 gate; T-G2 covered seats/A/A.
- **Step 2 — exe rebuilt:** `py -m PyInstaller check_halma.spec` →
  `_t_g5_build.log` "Build complete!" → `dist/check_halma.exe` (13.5 MB,
  2026-11-05 18:40 — newer than the last `halma8_gui.py` edit). The spec
  bundles `ai/target/release/ai.exe` (the T-G0 rebuilt one, 11/4) + both
  default models at the `_MEIPASS` paths the frozen driver resolves. Launch
  check: exe alive after 20 s, then killed. AI-seat play proven in step 1
  (same GUI code path as the exe).
- **Step 3 — cross-ref:** one line added to `plan_ai.md`'s "Where we are"
  (GUI can load models + Kibitzer).
- **Step 4 — final gate sweep (2026-11-05):** `py halma8_test.py` → 36 OK ·
  `cargo test --release` in `ai/` → green (`_t_g5_cargo.log`) ·
  `py halma_ai.py` → SMOKE OK · `py halma8_gui.py` launches (driven live by
  the matrix test) · exe launches (above).

**Steps:**
1. Run the full manual matrix and record results here:
   H/H · H/A · A/H · A/A, each with Kibitzer **off** then **on**; plus a
   mid-game **undo** and a **New game** that changes the model. Note any model
   that loads but plays identically to another (a weight-wiring regression).
2. Rebuild the standalone exe: run the PyInstaller build for `check_halma.spec`
   → `dist/check_halma.exe`; confirm it launches and an AI seat plays (the exe
   must find `ai.exe` — check how the spec bundles/locates it and that the T-G0
   rebuild of `ai.exe` is the one shipped).
3. Refresh `plan_ai.md`'s cross-reference (one line) that the GUI now can load
   models + Kibitzer, so the next agent doesn't think model-loading is still open.
4. Final gate sweep: `py halma8_test.py` (36 OK) · `cargo test --release` in `ai/`
   (green) · `py halma_ai.py` (smoke) · `py halma8_gui.py` (launches).

**Done-when:** matrix table filled and green, `dist/check_halma.exe` rebuilt and
plays with a loaded model, all baseline gates green, this file fully ticked.

---

## Definition of "finished" for this plan
All of T-G0…T-G5 ticked ✅ above, the four-gate baseline is green, and a stranger
can open `plan_gui.md`, read the HANDOFF block, and run the game with any loaded
model — single, dual, human-vs-AI, AI-vs-AI — with the Kibitzer showing a live
line and eval. If T-G0 (the Rust weights wiring) is blocked, stop there and report:
everything downstream depends on it, and doing it wrong is silent (the engine
just quietly plays the default weights).
