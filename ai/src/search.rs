//! Search skeleton: min-max alpha-beta + iterative deepening + transposition
//! table + time/stop. A faithful halma cut of `kego_engine_v2/src/search.rs` —
//! the capture machinery (quiescence, SEE/MVV-LVA, killers, history, null-move,
//! LMR, futility) and the chess terminal/draw rules (check, mate by check,
//! 50-move, threefold, insufficient material) are all dropped, since halma has
//! no captures, no check, and no draws.
//!
//! **Why min-max instead of negamax:** a halma *chain* lets the same player
//! move again (a jump that still has an open continuation keeps the turn).
//! Negamax's `-negamax(...)` sign flip assumes the side alternates and cannot
//! express same-player-again nodes. We therefore search on a **world score**
//! axis — P1 (turn==P1) is the maximizer, P2 the minimizer — and pass
//! `(alpha, beta)` world-bounds *unchanged* to every child. Two consecutive P1
//! nodes (a live chain) are just two consecutive maximizers; correctness is
//! trivial and the TT bound bookkeeping stays uniform.
//!
//! Move set at a node is the full **engine-legal** action set (every own piece's
//! step, every own piece's jump, plus `Stop` when a chain is live). The engine
//! (Rust `apply_move`/`end_turn`) is the sole legality authority, so a move the
//! search proposes is, by construction, one the Python engine will accept too —
//! the invariant the T-7 protocol gate relies on.

use crate::engine::{Board, Move};
use crate::evaluate::{evaluate_pair, Weights};
use crate::types::{idx, row_col, DIRS, SIZE};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// Terminal (win) base score. `|score| > MATE` is impossible for a real eval.
pub const MATE: i32 = 30000;
const INF: i32 = 32000;
/// Scores with magnitude >= this are "mate in N" (need a ply adjustment in the TT).
const MATE_BOUND: i32 = MATE - 512;

#[inline]
fn clamp(v: i32) -> i32 {
    v.clamp(-MATE, MATE)
}

/// Cheap **move-ordering proxy** (T-6). Estimates how good a child of `b`
/// is *after* `a` is played, used only to decide which branch to descend into
/// first. Deliberately O(1) in `a` — it does **not** clone the board or apply
/// the move (that was the dominant per-node cost at depth 6, paid once per
/// action). The real leaf score is still the full `evaluate` at `depth == 0`;
/// this only orders.
///
/// Because every action at a node shares the same parent board, ranking by the
/// *change* caused by the move is equivalent to ranking by the child's absolute
/// static value (the shared parent term cancels). Signal, strongest first:
/// - landing in the target base (the win lever) — biggest weight;
/// - a jump (continues the chain / holds the turn);
/// - forward progress toward the target base.
#[inline]
fn action_order_score(b: &Board, a: &Play) -> i32 {
    use crate::types::{base_cells, P1, P2};
    let (from, to) = match a {
        Play::Step { from, to } => (*from, *to),
        Play::Jump { from, to, .. } => (*from, *to),
        Play::Stop => return -1000, // voluntarily end a live chain: lowest priority
    };
    let mover = b.turn;
    let target = if mover == P1 { P2 } else { P1 }; // P1 fills P2's base, and vice versa
    let base = base_cells(target);
    let in_base = |sq: (u8, u8)| base.iter().any(|&(r, c)| (r, c) == sq);
    let mut s = (in_base(to) as i32) - (in_base(from) as i32);
    s *= 200; // base entry/exit is the strongest ordering lever (×2 weight)
    if matches!(a, Play::Jump { .. }) {
        s += 40; // a jump is a chain-continuing power move (halma-specific)
        // T-6.5: a landing that itself still has an open geometric jump is
        // chain-fertile. Cheap grid test (at most 8 lookups), no allocation.
        let g = &b.grid;
        let (tri, tci) = (to.0 as i8, to.1 as i8);
        for &(dr, dc) in DIRS.iter() {
            let (jr, jc) = (tri + 2 * dr, tci + 2 * dc);
            if (0..SIZE as i8).contains(&jr)
                && (0..SIZE as i8).contains(&jc)
                && g[idx((tri + dr) as u8, (tci + dc) as u8)] != 0
                && g[idx(jr as u8, jc as u8)] == 0
            {
                s += 8;
                break;
            }
        }
    }
    // forward progress (weakest lever): P1 wants a smaller row, P2 a larger row
    let dr = to.0 as i32 - from.0 as i32;
    s += if mover == P1 { -dr } else { dr };
    s
}

// ---------------------------------------------------------------------------
// Transposition table
// ---------------------------------------------------------------------------
const TT_SIZE: usize = 1 << 20; // 1M entries (T-6)
const TT_MASK: usize = TT_SIZE - 1;
const TT_EXACT: u8 = 0;
const TT_UPPER: u8 = 1; // fail-low:  true score <= cached
const TT_LOWER: u8 = 2; // fail-high: true score >= cached

#[derive(Clone, Copy)]
struct TTEntry {
    key: u64,
    depth: u8,
    score: i32, // stored ply-adjusted (see tt_store_score / tt_load_score)
    flag: u8,
    best: Option<Play>,
}

impl Default for TTEntry {
    fn default() -> Self {
        TTEntry {
            key: 0,
            depth: 0,
            score: 0,
            flag: TT_UPPER, // never exact; guarantees `replace` for a real entry
            best: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Move actions
// ---------------------------------------------------------------------------

/// A search action: exactly one `apply_move` application, or `end_turn` (`Stop`).
/// `Move` (step/jump) is `Copy`, so `Play` is too.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Play {
    Step {
        from: (u8, u8),
        to: (u8, u8),
    },
    Jump {
        from: (u8, u8),
        mid: (u8, u8),
        to: (u8, u8),
    },
    Stop,
}

impl Play {
    #[allow(dead_code)] // consumed by tests and the T-7 protocol (Play -> Move); not in the hot loop
    fn as_move(&self) -> Move {
        match self {
            Play::Step { from, to } => Move::Step { from: *from, to: *to },
            Play::Jump { from, mid, to } => Move::Jump {
                from: *from,
                mid: *mid,
                to: *to,
            },
            Play::Stop => panic!("Stop is end_turn, not apply_move"),
        }
    }
}

/// All engine-legal actions at `b`, in a deterministic order (row-major piece
/// scan, `DIRS` order inside each piece, `Stop` last), written into the
/// caller-provided buffer `out` (cleared first). This list is what the search
/// recurses over and what it may return as the best move.
///
/// T-6.5: writing into a *reused* per-ply buffer (rather than allocating a
/// fresh `Vec` per node) removes the dominant per-node heap allocation. The
/// 4-direction scan is inlined here (not the per-piece `legal_steps` /
/// `legal_jumps` helpers, which would allocate ~36 small `Vec`s per node). The
/// rule logic is byte-for-byte the engine's: a step = empty orthogonal
/// neighbour in-bounds; a jump = middle occupied + destination empty +
/// destination in-bounds (destination in-bounds implies the middle is in-bounds).
fn gen_actions_into(b: &Board, out: &mut Vec<Play>) {
    out.clear();
    let g = &b.grid;
    let turn = b.turn;

    if b.chain_active {
        // During a chain, only the chain piece (at its current landing) may
        // jump again, or the turn may be stopped. No steps, no other pieces.
        if let Some((cr, cc)) = b.chain_current {
            let (ri, ci) = (cr as i8, cc as i8);
            for &(dr, dc) in DIRS.iter() {
                let (nr, nc) = (ri + dr, ci + dc);
                let (jr, jc) = (ri + 2 * dr, ci + 2 * dc);
                if (0..SIZE as i8).contains(&jr)
                    && (0..SIZE as i8).contains(&jc)
                    && g[idx(nr as u8, nc as u8)] != 0
                    && g[idx(jr as u8, jc as u8)] == 0
                    && !b.chain_banned[idx(jr as u8, jc as u8)]
                {
                    out.push(Play::Jump {
                        from: (cr, cc),
                        mid: (nr as u8, nc as u8),
                        to: (jr as u8, jc as u8),
                    });
                }
            }
        }
        out.push(Play::Stop);
        return;
    }

    // No chain active: full move set for the current player.
    for i in 0..64 {
        if g[i] != turn {
            continue;
        }
        let (r, c) = row_col(i);
        let (ri, ci) = (r as i8, c as i8);
        for &(dr, dc) in DIRS.iter() {
            let (nr, nc) = (ri + dr, ci + dc);
            if (0..SIZE as i8).contains(&nr)
                && (0..SIZE as i8).contains(&nc)
                && g[idx(nr as u8, nc as u8)] == 0
            {
                out.push(Play::Step {
                    from: (r, c),
                    to: (nr as u8, nc as u8),
                });
            }
            let (jr, jc) = (ri + 2 * dr, ci + 2 * dc);
            if (0..SIZE as i8).contains(&jr)
                && (0..SIZE as i8).contains(&jc)
                && g[idx(nr as u8, nc as u8)] != 0
                && g[idx(jr as u8, jc as u8)] == 0
            {
                out.push(Play::Jump {
                    from: (r, c),
                    mid: (nr as u8, nc as u8),
                    to: (jr as u8, jc as u8),
                });
            }
        }
    }
}

/// Allocating single-shot wrapper — used only for the `search` fallback path
/// (never in the recursion hot loop), where a throwaway `Vec` is fine.
fn gen_actions(b: &Board) -> Vec<Play> {
    let mut v = Vec::with_capacity(40);
    gen_actions_into(b, &mut v);
    v
}

/// Per-ply scratch buffers shared down the recursion (T-6.5). Each ply owns its
/// own `actions` and `order` buffers, so a parent never clobbers a child's list
/// while it is still being searched, and neither buffer is heap-allocated per
/// node. Indexed by `ply`; a node at ply `p` only ever touches `[p]`, its
/// children `[p+1]`, etc. — disjoint slots, no aliasing.
pub struct Scratch {
    actions: Vec<Vec<Play>>,
    order: Vec<Vec<(i32, usize)>>,
}

impl Scratch {
    fn new(plies: usize) -> Self {
        Scratch {
            actions: vec![Vec::with_capacity(40); plies.max(1)],
            order: vec![Vec::with_capacity(40); plies.max(1)],
        }
    }
}

// ---------------------------------------------------------------------------
// T-6.5: make/unmake — one working board mutated in place and restored by a
// plain-Copy snapshot, replacing the per-branch `b.clone()` + validated
// `apply_move` (which re-allocated a `Vec` or two for its own legality
// checks).
// ---------------------------------------------------------------------------

/// Full mutable state of a `Board`. Every field is `Copy`, so snapshot +
/// restore is a flat copy with zero heap traffic. Unmaking restores
/// *exactly* this — piece placement, `turn`, `winner`, `turn_no`, and the
/// chain `active` / `banned` / `starter` state.
#[derive(Clone, Copy)]
struct Snap {
    grid: [u8; 64],
    turn: u8,
    winner: Option<u8>,
    turn_no: u32,
    chain_active: bool,
    chain_banned: [bool; 64],
    chain_starter: Option<(u8, u8)>,
    chain_current: Option<(u8, u8)>,
}

#[inline]
fn snap(b: &Board) -> Snap {
    Snap {
        grid: b.grid,
        turn: b.turn,
        winner: b.winner,
        turn_no: b.turn_no,
        chain_active: b.chain_active,
        chain_banned: b.chain_banned,
        chain_starter: b.chain_starter,
        chain_current: b.chain_current,
    }
}

#[inline]
fn unmake(b: &mut Board, s: Snap) {
    b.grid = s.grid;
    b.turn = s.turn;
    b.winner = s.winner;
    b.turn_no = s.turn_no;
    b.chain_active = s.chain_active;
    b.chain_banned = s.chain_banned;
    b.chain_starter = s.chain_starter;
    b.chain_current = s.chain_current;
}

/// Non-allocating port of `Board::has_open_jump` (identical rule: a legal
/// geometric jump from `to` onto a non-banned destination).
#[inline]
fn has_open_jump_nb(b: &Board, to: (u8, u8)) -> bool {
    let g = &b.grid;
    let (ri, ci) = (to.0 as i8, to.1 as i8);
    for &(dr, dc) in DIRS.iter() {
        let (nr, nc) = (ri + dr, ci + dc);
        let (jr, jc) = (ri + 2 * dr, ci + 2 * dc);
        if (0..SIZE as i8).contains(&jr) && (0..SIZE as i8).contains(&jc) && g[idx(nr as u8, nc as u8)] != 0
        {
            let di = idx(jr as u8, jc as u8);
            if g[di] == 0 && !(b.chain_active && b.chain_banned[di]) {
                return true;
            }
        }
    }
    false
}

/// Apply a `Play` mutatively with **no validation** (every play `gen_actions`
/// emits is engine-legal, including the chain-ban filter) and **no helper
/// allocations**. Postconditions mirror `Board::apply_move` exactly:
/// step → turn ends (turn flip + `turn_no`, no-op after a win); jump → chain
/// stays live iff the landing keeps an open jump and no winner was set;
/// `Stop` → `end_turn` (no-op once a winner is set).
#[inline]
fn make_play(b: &mut Board, p: Play) {
    let flip = if b.turn == crate::types::P1 {
        crate::types::P2
    } else {
        crate::types::P1
    };
    match p {
        Play::Step { from, to } => {
            let fi = idx(from.0, from.1);
            let ti = idx(to.0, to.1);
            let v = b.grid[fi];
            b.grid[ti] = v;
            b.grid[fi] = 0;
            b.chain_active = false;
            b.chain_banned = [false; 64];
            b.chain_current = None;
            b.check_win();
            if b.winner.is_none() {
                b.turn = flip;
                b.turn_no += 1;
            }
        }
        Play::Jump { from, to, .. } => {
            let fi = idx(from.0, from.1);
            let ti = idx(to.0, to.1);
            let v = b.grid[fi];
            b.grid[ti] = v;
            b.grid[fi] = 0;
            if !b.chain_active {
                b.chain_active = true;
                b.chain_starter = Some(from);
                b.chain_banned = [false; 64];
                b.chain_banned[fi] = true; // origin revisit ban
            }
            b.chain_banned[ti] = true;
            b.chain_current = Some(to);
            b.check_win();
            if b.winner.is_some() {
                // a winning jump: chain cleared, turn NOT flipped (the
                // engine's `end_turn` is a no-op once a winner is set)
                b.chain_active = false;
                b.chain_banned = [false; 64];
                b.chain_current = None;
            } else if has_open_jump_nb(b, to) {
                return; // chain still live: turn unchanged
            } else {
                b.chain_active = false;
                b.chain_banned = [false; 64];
                b.chain_current = None;
                b.turn = flip;
                b.turn_no += 1;
            }
        }
        Play::Stop => {
            if b.winner.is_some() {
                return; // end_turn no-op
            }
            b.turn = flip;
            b.turn_no += 1;
            b.chain_active = false;
            b.chain_banned = [false; 64];
            b.chain_current = None;
        }
    }
}

// ---------------------------------------------------------------------------
// Position key (deterministic splitmix64-seeded; no std::once cell, no deps)
// ---------------------------------------------------------------------------
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58476D1CE4E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D049BB133111EB);
    x ^= x >> 31;
    x
}

struct KeySet {
    piece: [[u64; 2]; 64],
    turn: [u64; 3],
    chain_active: u64,
    chain_banned: [u64; 64],
    chain_starter: [u64; 64],
}

fn keyset() -> &'static KeySet {
    use std::sync::OnceLock;
    static ONCE: OnceLock<KeySet> = OnceLock::new();
    ONCE.get_or_init(|| {
        let mut seed: u64 = 0x9E3779B97F4A7C15;
        let mut next =
            || {
                seed = seed.wrapping_add(1);
                splitmix64(0x85A308D313198A2E ^ seed)
            };
        let mut piece = [[0u64; 2]; 64];
        for s in 0..64 {
            piece[s][0] = next();
            piece[s][1] = next();
        }
        let turn = [next(), next(), next()];
        let chain_active = next();
        let mut chain_banned = [0u64; 64];
        for s in 0..64 {
            chain_banned[s] = next();
        }
        let mut chain_starter = [0u64; 64];
        for s in 0..64 {
            chain_starter[s] = next();
        }
        KeySet {
            piece,
            turn,
            chain_active,
            chain_banned,
            chain_starter,
        }
    })
}

/// Full search-relevant state: 64 piece cells (18 real + empty), turn, and the
/// active chain (flag + banned set + origin). `winner` and `turn_no` are
/// deliberately out of the key — a win is caught before the TT is consulted,
/// and `turn_no` is a bookkeeping counter that neither legality nor eval reads.
fn board_key(b: &Board) -> u64 {
    let ks = keyset();
    let mut k: u64 = 0;
    for s in 0..64 {
        let v = b.grid[s];
        if v != 0 {
            k ^= ks.piece[s][(v - 1) as usize];
        }
    }
    k ^= ks.turn[b.turn as usize];
    if b.chain_active {
        k ^= ks.chain_active;
        for s in 0..64 {
            if b.chain_banned[s] {
                k ^= ks.chain_banned[s];
            }
        }
        if let Some((r, c)) = b.chain_starter {
            k ^= ks.chain_starter[idx(r, c)];
        }
    }
    k
}

// ---------------------------------------------------------------------------
// Score <-> TT ply adjustment (world convention; both signs handled)
// ---------------------------------------------------------------------------
#[inline]
fn tt_store_score(score: i32, ply: i32) -> i32 {
    if score >= MATE_BOUND {
        score + ply
    } else if score <= -MATE_BOUND {
        score - ply
    } else {
        score
    }
}

#[inline]
fn tt_load_score(score: i32, ply: i32) -> i32 {
    if score >= MATE_BOUND {
        score - ply
    } else if score <= -MATE_BOUND {
        score + ply
    } else {
        score
    }
}

// ---------------------------------------------------------------------------
// Search session
// ---------------------------------------------------------------------------

pub struct Search {
    pub nodes: u64,
    pub stop: Arc<AtomicBool>,
    movetime_ms: u64,
    t0: Instant,
    tt: Vec<TTEntry>,
    max_depth: i32,
    /// Deepest depth **fully completed** by the last `search` call (T-6.5
    /// time-budget contract: on a mid-depth interrupt this is the depth whose
    /// result the returned move/score belong to; 0 if none completed).
    pub deep_completed: i32,
    /// T-11: per-side evaluation weights. `w1` scores P1's six terms, `w2`
    /// scores P2's. Both default to `Weights::default()`, i.e. `evaluate_pair`
    /// on defaults is byte-identical to the original `evaluate` — the search
    /// is unchanged unless a caller sets a distinct per-side pair (head-to-head).
    w1: Weights,
    w2: Weights,
}

impl Search {
    pub fn new() -> Self {
        Search {
            nodes: 0,
            stop: Arc::new(AtomicBool::new(false)),
            movetime_ms: 0,
            t0: Instant::now(),
            tt: vec![TTEntry::default(); TT_SIZE],
            // T-6.5: time is the primary axis — the depth is now only a
            // generous safety cap so a `movetime`-driven search can run as
            // deep as the budget allows. `go depth N` callers pass an explicit
            // (smaller) cap and are unaffected.
            max_depth: 64,
            deep_completed: 0,
            w1: Weights::default(),
            w2: Weights::default(),
        }
    }

    pub fn set_movetime_ms(&mut self, ms: u64) {
        self.movetime_ms = ms;
    }

    pub fn set_max_depth(&mut self, d: i32) {
        self.max_depth = d;
    }

    /// T-11: set the per-side evaluation weights for the head-to-head Elo
    /// measurement. `w1` scores P1's terms (maximizer), `w2` scores P2's
    /// (minimizer). Leaving both at `Weights::default()` (the `Search::new`
    /// state) is byte-identical to the pre-T-11 search, so every caller that
    /// does not call this — `server.rs`, `selfplay.rs` self-play, `bench.rs`,
    /// and all tests — is unaffected.
    pub fn set_weights_pair(&mut self, w1: Weights, w2: Weights) {
        self.w1 = w1;
        self.w2 = w2;
    }

    /// External stop hook (protocol `stop` verb will drive this in T-7).
    pub fn request_stop(&self) {
        self.stop.store(true, Ordering::Relaxed);
    }

    #[inline]
    fn is_stop(&self) -> bool {
        self.stop.load(Ordering::Relaxed)
    }

    #[inline]
    fn maybe_stop(&self) {
        if self.movetime_ms > 0
            && self.t0.elapsed().as_millis() as u64 >= self.movetime_ms
        {
            self.stop.store(true, Ordering::Relaxed);
        }
    }

    #[inline]
    fn tt_get(&self, key: u64) -> Option<TTEntry> {
        let e = &self.tt[(key as usize) & TT_MASK];
        if e.key == key {
            Some(*e)
        } else {
            None
        }
    }

    #[inline]
    fn tt_store(&mut self, key: u64, score: i32, flag: u8, depth: i32, ply: i32, best: Option<Play>) {
        let i = (key as usize) & TT_MASK;
        let rank = |f: u8| match f {
            TT_EXACT => 2,
            TT_LOWER => 1,
            TT_UPPER => 0,
            _ => 0,
        };
        let e = &self.tt[i];
        let d = depth.max(0) as u8;
        if !(d > e.depth || (d == e.depth && rank(flag) >= rank(e.flag))) {
            return;
        }
        self.tt[i] = TTEntry {
            key,
            depth: d,
            score: tt_store_score(score, ply),
            flag,
            best,
        };
    }

    /// One recursive node. `alpha`/`beta` are world-score bounds, unchanged by
    /// any sign flip (P1 max, P2 min). `b` is the **single working board**:
    /// children are made on it and unmaked before the next sibling (T-6.5 —
    /// no per-branch clone); `sc` supplies this ply's scratch buffers so the
    /// recursion never allocates per node (T-6.5). `capture_best` is
    /// true only at the root, so the root returns the best action as well.
    ///
    /// T-6.5 note on borrows: this node's buffers (`sc.actions[pi]`,
    /// `sc.order[pi]`) are re-borrowed per statement inside the child loop so
    /// no `&mut` borrow survives into `self.node(..., sc)`; the child only ever
    /// touches the slots at `ply+1` and below, which are distinct inner `Vec`s.
    #[inline]
    fn node(
        &mut self,
        b: &mut Board,
        depth: i32,
        alpha: i32,
        beta: i32,
        ply: i32,
        capture_best: bool,
        sc: &mut Scratch,
    ) -> (i32, Option<Play>) {
        self.nodes += 1;
        if self.nodes % 256 == 0 {
            self.maybe_stop();
        }
        if self.is_stop() {
            return (alpha, None);
        }

        let key = board_key(b);

        // Terminal: a win is a win at whatever depth/ply we hit it. Score for the
        // winner, closer to the root (lower ply) the larger magnitude -> both
        // "winners prefer faster" holds: P1 max wants MATE - ply high, P2 min
        // wants -MATE + ply low.
        if let Some(w) = b.winner {
            let sc = if w == crate::types::P1 {
                MATE - ply
            } else {
                -MATE + ply
            };
            return (sc, None);
        }

        // Leaf: static eval. Store as an exact entry so siblings at >=0 depth
        // can use it (the depth stored here is 0, matching the effective bound).
        // T-11: per-side weights (head-to-head); defaults == original `evaluate`.
        if depth <= 0 {
            let sc = evaluate_pair(b, &self.w1, &self.w2);
            self.tt_store(key, sc, TT_EXACT, 0, ply, None);
            return (sc, None);
        }

        let pi = ply as usize;
        {
            let acts = &mut sc.actions[pi];
            gen_actions_into(b, acts);
        }
        if sc.actions[pi].is_empty() {
            // Stuck (can happen if every own step/jump is blocked and no live
            // chain to `Stop`). Score is "mover dead": large-negative from the
            // mover's own perspective. P1 dead = very negative (P1 is the
            // maximizer -> this is a loss). P2 dead = very positive.
            let sc = if b.turn == crate::types::P1 {
                -(MATE - ply)
            } else {
                MATE - ply
            };
            return (sc, None);
        }

        let orig_alpha = alpha;
        let orig_beta = beta;
        let mut alpha = alpha;
        let mut beta = beta;
        let is_max = b.turn == crate::types::P1;

        // -- TT probe: pull a stored bound or exact value; adjust alpha/beta;
        // an exact (or bound-that-beats-the-window) entry is an immediate exit.
        let mut tt_best: Option<Play> = None;
        if let Some(e) = self.tt_get(key) {
            tt_best = e.best;
            let score = tt_load_score(e.score, ply);
            if e.depth as i32 >= depth {
                match e.flag {
                    TT_EXACT => return (score, if capture_best { tt_best } else { None }),
                    TT_LOWER if score >= beta => {
                        return (score, if capture_best { tt_best } else { None })
                    }
                    TT_UPPER if score <= orig_alpha => {
                        return (score, if capture_best { tt_best } else { None })
                    }
                    _ => {}
                }
                if e.flag == TT_LOWER {
                    if score > alpha {
                        alpha = score;
                    }
                } else if e.flag == TT_UPPER {
                    if score < beta {
                        beta = score;
                    }
                }
            }
        }
        if alpha >= beta {
            if capture_best {
                return (alpha, tt_best);
            }
            return (alpha, None);
        }

        // -- Move ordering: a cheap, allocation-free estimate per action
        // (no board clone / `apply_move` here — that was the dominant per-node
        // cost at depth 6). Written into this ply's scratch buffer (T-6.5 —
        // no per-node heap allocation). Deterministic (stable, index
        // tie-break); the TT best action is floated to the front afterwards.
        {
            let acts = &sc.actions[pi];
            let scored = &mut sc.order[pi];
            scored.clear();
            scored.reserve(acts.len());
            for (i, a) in acts.iter().enumerate() {
                scored.push((action_order_score(b, a), i));
            }
            if is_max {
                scored.sort_by(|x, y| y.0.cmp(&x.0).then(x.1.cmp(&y.1)));
            } else {
                scored.sort_by(|x, y| x.0.cmp(&y.0).then(x.1.cmp(&y.1)));
            }
            if let Some(t) = tt_best {
                if let Some(p) = scored.iter().position(|(_, i)| acts[*i] == t) {
                    scored.swap(0, p);
                }
            }
        }

        let mut best = if is_max { -INF } else { INF };
        let mut best_play: Option<Play> = None;

        let n = sc.order[pi].len();
        for k in 0..n {
            if self.is_stop() {
                break;
            }
            let a = sc.actions[pi][sc.order[pi][k].1];
            // Make/unmake (T-6.5): mutate the one working board, recurse on it,
            // restore the exact pre-move state before the next sibling.
            let s = snap(b);
            make_play(b, a);
            let v = self.node(b, depth - 1, alpha, beta, ply + 1, false, sc).0;
            unmake(b, s);
            if is_max {
                if v > best {
                    best = v;
                    best_play = Some(a);
                }
                if best > alpha {
                    alpha = best;
                }
                if alpha >= beta {
                    break;
                }
            } else {
                if v < best {
                    best = v;
                    best_play = Some(a);
                }
                if best < beta {
                    beta = best;
                }
                if alpha >= beta {
                    break;
                }
            }
        }

        let flag = if best <= orig_alpha {
            TT_UPPER
        } else if best >= orig_beta {
            TT_LOWER
        } else {
            TT_EXACT
        };
        if !self.is_stop() {
            self.tt_store(key, best, flag, depth, ply, best_play);
        }

        let best_val = clamp(best);
        (best_val, if capture_best { best_play } else { None })
    }

    /// Iterative deepening from `root`. One working board and one set of
    /// scratch buffers are reused for the whole search (T-6.5), and one
    /// transposition table across depths, so each deeper iteration builds on
    /// the previous one's best lines.
    ///
    /// T-6.5 time-budget contract: with a `movetime` set, deepening continues
    /// until the budget and returns the best **fully completed** depth's
    /// (score, legal action); a mid-depth interrupt never discards the last
    /// completed result. If even that first depth is interrupted, a first
    /// legal action from `root` is returned so the caller always has something
    /// legal to play. `self.deep_completed` records the deepest completed depth.
    pub fn search(&mut self, root: &Board, max_depth: i32) -> (i32, Option<Play>) {
        let cap = (max_depth.min(self.max_depth)).max(1);
        let mut b = root.clone(); // one working board for the whole search
        let mut scratch = Scratch::new((cap as usize) + 2);
        let mut best_score = 0i32;
        let mut best_play: Option<Play> = None;
        self.deep_completed = 0;
        for depth in 1..=cap {
            self.maybe_stop();
            if self.is_stop() {
                break;
            }
            let (score, play) = self.node(&mut b, depth, -INF, INF, 0, true, &mut scratch);
            if self.is_stop() {
                // This depth was interrupted: keep the last *completed*
                // depth's result; a play is usable as the legality fallback.
                if best_play.is_none() {
                    best_play = play;
                }
                break;
            }
            if let Some(p) = play {
                best_play = Some(p);
                best_score = score;
            }
            self.deep_completed = depth;
        }
        if best_play.is_none() {
            // interrupted before any depth completed — fall back to a legal action.
            best_play = gen_actions(root).into_iter().next();
            best_score = evaluate_pair(root, &self.w1, &self.w2);
        }
        (clamp(best_score), best_play)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Board, Move};
    use crate::types::{idx, P1, P2};

    fn assert_legal(b: &Board, p: Play) {
        let mut nb = b.clone();
        match p {
            Play::Step { .. } | Play::Jump { .. } => {
                match nb.apply_move(&p.as_move()) {
                    Ok(_) => {}
                    Err(_) => panic!("search returned an illegal Move: {:?}", p),
                }
            }
            Play::Stop => {
                // Stop is only legal as an `end_turn` after a jump in a live chain;
                // we do not validate engine-legality here because `end_turn` is
                // not `apply_move`. Tests only construct non-chain positions for
                // legality assertions.
            }
        }
    }

    // ---- T-4 test 1 — a one-ply win must be found and returned at depth 1.
    #[test]
    fn t4_depth1_immediate_win() {
        let mut b = Board::new();
        // P2 base (rows 0..3, cols 5..8) filled with P2, leaving (0,5) empty.
        for r in 0..3u8 {
            for c in 5..8u8 {
                if (r, c) != (0, 5) {
                    b.grid[idx(r, c)] = P2;
                }
            }
        }
        // A P1 piece just LEFT of the empty cell (0,4) — one step right fills the
        // last base cell while leaving the other 8 base cells as P2 (full + invaded).
        b.grid[idx(0, 4)] = P1;
        b.turn = P1;
        b.winner = None;

        let mut s = Search::new();
        let (score, play) = s.search(&b, 1);
        assert_eq!(play, Some(Play::Step { from: (0, 4), to: (0, 5) }));
        assert!(score as i64 >= MATE as i64 - 8, "expected a mate-scale score, got {}", score);
        assert_legal(&b, play.unwrap());
    }

    // ---- T-4 test 2 — a one-ply win is also returned at depth 2 (within horizon).
    #[test]
    fn t4_depth2_still_wins_immediately() {
        let mut b = Board::new();
        for r in 0..3u8 {
            for c in 5..8u8 {
                if (r, c) != (0, 5) {
                    b.grid[idx(r, c)] = P2;
                }
            }
        }
        b.grid[idx(0, 4)] = P1;
        b.turn = P1;

        let mut s = Search::new();
        let (score, play) = s.search(&b, 2);
        assert_eq!(
            play,
            Some(Play::Step {
                from: (0, 4),
                to: (0, 5)
            })
        );
        assert!(score as i64 >= MATE as i64 - 16);
    }

    // ---- T-4 test 3 — time budget: movetime=100ms must finish quickly.
    #[test]
    fn t4_movetime_bounded_returns_legal_move() {
        let b = Board::start();
        let mut s = Search::new();
        s.set_movetime_ms(100);
        s.set_max_depth(40);
        let t = std::time::Instant::now();
        let (_, play) = s.search(&b, 40);
        let dt = t.elapsed().as_millis();
        assert!(dt < 500, "search exceeded 500 ms wall clock: {} ms", dt);
        assert!(play.is_some(), "search must always return some legal action");
        let p = play.unwrap();
        match p {
            Play::Step { .. } | Play::Jump { .. } => {
                let mut nb = b.clone();
                assert!(nb.apply_move(&p.as_move()).is_ok());
            }
            Play::Stop => panic!("Stop is not a legal first root action"),
        }
    }

    // ---- T-4 test 4 — iterative deepening respects the max_depth cap.
    #[test]
    fn t4_iter_terminates_at_max_depth() {
        let b = Board::start();
        let mut s = Search::new();
        s.set_max_depth(3);
        let (_, p) = s.search(&b, 10); // cap must clamp to 3
        assert!(p.is_some());
    }

    // ---- T-4 — determinism: two identical searches give identical (score, move).
    #[test]
    fn t4_deterministic_same_input_same_output() {
        let b = Board::start();
        let mut s1 = Search::new();
        let (sc1, p1) = s1.search(&b, 4);
        let mut s2 = Search::new();
        let (sc2, p2) = s2.search(&b, 4);
        assert_eq!(sc1, sc2);
        assert_eq!(p1, p2);
    }

    // ---- T-4 — the search never proposes an illegal move on a mid-game position
    // (this guards the T-7 protocol gate: the Python engine will accept everything
    // the Rust search returns, because the search is built on the engine's apply_move).
    #[test]
    fn t4_midgame_move_is_engine_legal() {
        let mut b = Board::start();
        // A few plies of real play (mix of step and jump).
        assert!(b
            .apply_move(&Move::Step {
                from: (5, 2),
                to: (4, 2)
            })
            .is_ok());
        assert!(b
            .apply_move(&Move::Step {
                from: (0, 5),
                to: (0, 4)
            })
            .is_ok());
        let mut s = Search::new();
        let (_, p) = s.search(&b, 4);
        let p = p.expect("some move");
        match p {
            Play::Step { .. } | Play::Jump { .. } => {
                let mut nb = b.clone();
                assert!(
                    nb.apply_move(&p.as_move()).is_ok(),
                    "illegal move: {:?}",
                    p
                );
            }
            Play::Stop => panic!("Start position has no active chain; Stop is illegal here"),
        }
    }

    // ---- T-4 — the chain-active path is exercised: a live chain must still
    // produce a legal action (a continuation `Jump` or a `Stop`).
    #[test]
    fn t4_live_chain_returns_legal_action() {
        let mut b = Board::new();
        // P1 just jumped and is live: piece at (4,3), banned {(4,1),(4,3)}, starter
        // (4,1). P2 at (4,4) so (4,3) can jump to (4,5) (dst empty, mid occupied).
        b.grid[idx(4, 1)] = 0; // vacate the origin cell
        b.grid[idx(4, 3)] = P1;
        b.grid[idx(4, 4)] = P2;
        // Make the jump-continuation legal.
        b.chain_active = true;
        b.chain_starter = Some((4, 1));
        for (r, c) in [(4, 1), (4, 3)] {
            b.chain_banned[idx(r, c)] = true;
        }
        b.turn = P1;
        b.winner = None;

        let mut s = Search::new();
        let (_, p) = s.search(&b, 4);
        let p = p.expect("some action");
        match p {
            Play::Step { to, .. } | Play::Jump { to, .. } => {
                // destination must not be a banned cell for the move to be engine-legal.
                assert!(!b.chain_banned[idx(to.0, to.1)], "search returned a banned destination");
            }
            Play::Stop => {} // legal: voluntarily end a live chain.
        }
    }

    // ---- T-4 — the search must NOT prefer a move that the engine would reject
    // (chain-revisit ban) when an identical engine-legal alternative exists.
    #[test]
    fn t4_never_returns_banned_jump_destination() {
        let mut b = Board::new();
        // Chain live: piece at (4,5); the *only* geometric jump is back to (4,3),
        // which is banned. The search must not propose that banned jump.
        b.grid[idx(4, 5)] = P1;
        b.grid[idx(4, 4)] = P2;
        b.grid[idx(4, 3)] = P1; // would-be destination occupied -> not even a legal jump
        b.grid[idx(4, 6)] = P1;
        b.chain_active = true;
        b.chain_starter = Some((4, 7));
        // banned set includes (4,3) and the piece's current square (4,5).
        for (r, c) in [(4, 3), (4, 5)] {
            b.chain_banned[idx(r, c)] = true;
        }
        b.turn = P1;
        b.winner = None;

        let mut s = Search::new();
        let (_, p) = s.search(&b, 3);
        let p = p.expect("some action");
        // The only geometric jump (4,5)->(4,7) over (4,6) would land on a banned
        // origin (4,7) is the starter — banned? We made (4,7) not banned, so that
        // jump may or may not be the one chosen. The key check: whatever is
        // returned, if it's a Jump its `to` is *not* a banned cell.
        match p {
            Play::Jump { to, mid, .. } => {
                assert!(!b.chain_banned[idx(to.0, to.1)], "Jump to a banned square");
                let _ = mid; // silence
            }
            _ => {}
        }
    }

    // ---- T-6.5 — make/unmake must be byte-identical to the engine's own
    // `apply_move` / `end_turn` side effects, and unmake a perfect restore.
    // (This is the soundness proof for dropping the validated `apply_move`
    // from the hot loop.)
    #[test]
    fn t65_make_unmake_matches_engine_exactly() {
        fn check(b0: Board, play: Play, engine: impl FnOnce(&mut Board)) {
            let mut expected = b0.clone();
            engine(&mut expected);
            let mut w = b0.clone();
            let s = snap(&w);
            make_play(&mut w, play);
            assert_eq!(w, expected, "make_play diverged from engine state");
            unmake(&mut w, s);
            assert_eq!(w, b0, "unmake failed to restore pre-move state");
        }

        // 1. plain step (turn ends)
        {
            let b0 = Board::start();
            check(
                b0.clone(),
                Play::Step {
                    from: (5, 2),
                    to: (4, 2),
                },
                |b| {
                    b.apply_move(&Move::Step {
                        from: (5, 2),
                        to: (4, 2),
                    })
                    .unwrap();
                },
            );
        }
        // 2. chain-opening jump (turn stays with the mover)
        {
            let mut b0 = Board::new();
            b0.grid[idx(4, 1)] = P1;
            b0.grid[idx(4, 2)] = P2;
            b0.grid[idx(4, 4)] = P2;
            let p = Play::Jump {
                from: (4, 1),
                mid: (4, 2),
                to: (4, 3),
            };
            check(
                b0.clone(),
                p,
                |b| {
                    b.apply_move(&Move::Jump {
                        from: (4, 1),
                        mid: (4, 2),
                        to: (4, 3),
                    })
                    .unwrap();
                },
            );
        }
        // 3. `Stop` on a live chain (= engine `end_turn`)
        {
            let mut b0 = Board::new();
            b0.grid[idx(4, 3)] = P1;
            b0.grid[idx(4, 4)] = P2;
            b0.chain_active = true;
            b0.chain_starter = Some((4, 1));
            b0.chain_banned[idx(4, 1)] = true;
            b0.chain_banned[idx(4, 3)] = true;
            check(b0.clone(), Play::Stop, |b| b.end_turn());
        }
        // 4. a WINNING step (winner set, turn not flipped)
        {
            let mut b0 = Board::new();
            for r in 0..3u8 {
                for c in 5..8u8 {
                    if (r, c) != (0, 5) {
                        b0.grid[idx(r, c)] = P2;
                    }
                }
            }
            b0.grid[idx(0, 4)] = P1;
            b0.turn = P1;
            check(
                b0.clone(),
                Play::Step {
                    from: (0, 4),
                    to: (0, 5),
                },
                |b| {
                    b.apply_move(&Move::Step {
                        from: (0, 4),
                        to: (0, 5),
                    })
                    .unwrap();
                },
            );
        }
        // 5. a WINNING jump (winner set mid-chain, chain cleared, turn untouched)
        {
            let mut b0 = Board::new();
            for r in 0..3u8 {
                for c in 5..8u8 {
                    if (r, c) != (0, 5) {
                        b0.grid[idx(r, c)] = P2;
                    }
                }
            }
            b0.grid[idx(0, 3)] = P1;
            b0.grid[idx(0, 4)] = P1;
            b0.turn = P1;
            check(
                b0.clone(),
                Play::Jump {
                    from: (0, 3),
                    mid: (0, 4),
                    to: (0, 5),
                },
                |b| {
                    b.apply_move(&Move::Jump {
                        from: (0, 3),
                        mid: (0, 4),
                        to: (0, 5),
                    })
                    .unwrap();
                },
            );
        }
    }

    // ---- T-6.5 — with a time budget, deepening must return a legal move,
    // finish on budget, and remember the last fully completed depth.
    #[test]
    fn t65_time_budget_contract() {
        let mut b = Board::start();
        b.apply_move(&Move::Step {
            from: (5, 2),
            to: (4, 2),
        })
        .unwrap();
        let mut s = Search::new();
        s.set_movetime_ms(120);
        let t = std::time::Instant::now();
        let (_, play) = s.search(&b, 64);
        let dt = t.elapsed().as_millis();
        assert!(dt < 1000, "took {} ms against a 120 ms budget", dt);
        let p = play.expect("a legal action is always returned");
        match p {
            Play::Step { .. } | Play::Jump { .. } => {
                let mut nb = b.clone();
                assert!(nb.apply_move(&p.as_move()).is_ok(), "illegal: {:?}", p);
            }
            Play::Stop => panic!("no live chain at this root; Stop is illegal"),
        }
        assert!(s.deep_completed >= 1, "at least depth 1 must complete in budget");
    }

    // ---- T-11 — head-to-head plumbing: a default per-side weight pair must be
    // byte-identical to the plain (pre-T-11) search, proving the
    // `evaluate_pair` leaf swap is a no-op at defaults and that
    // `set_weights_pair` is a safe extension rather than a behavior change.
    #[test]
    fn t11_default_weights_pair_matches_plain_search() {
        let mut b = Board::start();
        // A couple of real plies so the root is a genuine mid-position.
        assert!(b.apply_move(&Move::Step { from: (5, 2), to: (4, 2) }).is_ok());
        assert!(b.apply_move(&Move::Step { from: (0, 5), to: (0, 4) }).is_ok());

        let (sc_plain, mv_plain) = {
            let mut s = Search::new();
            s.set_max_depth(5);
            s.search(&b, 5)
        };
        let (sc_pair, mv_pair) = {
            let mut s = Search::new();
            let d = Weights::default();
            s.set_weights_pair(d, d);
            s.set_max_depth(5);
            s.search(&b, 5)
        };
        assert_eq!(sc_plain, sc_pair, "default per-side pair must not change the score");
        assert_eq!(mv_plain, mv_pair, "default per-side pair must not change the move");
    }

    // ---- T-G0 — the live server's weights wiring must actually change play.
    // Feeding a distinct weight set through `Search::set_weights_pair` — the
    // exact path `server.rs`'s `setoption`→`go` arm now uses — must produce a
    // different root answer than the default pair on a fixed asymmetric
    // position. This is the guard against the "parsed-but-ignored" bug: if the
    // two were ever identical, the weights would be silently no-ops.
    #[test]
    fn t_g0_distinct_weights_change_the_search_answer() {
        // A fixed, non-symmetric mid-position (a couple of real plies each side).
        let mut b = Board::start();
        assert!(b.apply_move(&Move::Step { from: (5, 2), to: (4, 2) }).is_ok());
        assert!(b.apply_move(&Move::Step { from: (0, 5), to: (0, 4) }).is_ok());

        let depth = 3;

        let (sc_default, mv_default) = {
            let mut s = Search::new();
            s.set_max_depth(depth);
            s.search(&b, depth)
        };

        // A deliberately different, single-term (progress-only) weight set.
        let extreme = Weights {
            progress: 1000.0,
            base_presence: 0.0,
            defense: 0.0,
            spread: 0.0,
            mobility: 0.0,
            threat: 0.0,
        };
        let (sc_extreme, mv_extreme) = {
            let mut s = Search::new();
            s.set_weights_pair(extreme, extreme);
            s.set_max_depth(depth);
            s.search(&b, depth)
        };

        assert!(
            sc_default != sc_extreme || mv_default != mv_extreme,
            "distinct weight sets routed through Search must change the root \
             (score, bestmove); got default ({}, {:?}) == extreme ({}, {:?}) \
             — the weights are being silently ignored",
            sc_default, mv_default, sc_extreme, mv_extreme
        );
    }
}
