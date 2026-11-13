//! Evaluation — the six-term weighted positional model (T-5).
//!
//! Each term is a **pure** function of `(&Board, side)` and returns an
//! *unsigned* magnitude (distance, count, pair-count, …) in the "more positive
//! = better for that side" direction. `side_score` applies the weights;
//! `evaluate` gives the **world score**:
//!
//! ```text
//! evaluate(b) = side_score(b, P1) - side_score(b, P2)
//! ```
//!
//! Positive = good for P1 (the maximizer), negative = good for P2 (the
//! minimizer). `evaluate` is turn-independent — it scores the piece
//! arrangement only. The search (see `search.rs`) handles terminal (win)
//! detection and all depth/ply bookkeeping before it ever calls `evaluate`.

use crate::engine::Board;
use crate::types::{base_cells, idx, row_col, DIRS, P1, P2, SIZE};

/// The six named eval terms; each is a `f64` multiplier (CP units).
///
/// T-14: the six terms differ ~10–30× in raw scale, so a meaningful
/// "distribute a budget across six features" only exists in **float** space —
/// an integer weight can't express a small fraction of a CP-unit relative to
/// the biggest term (and `int(round(...))` was zeroing the small-scaled ones).
/// Small integers are exact in `f64`, so every integer weight set scores
/// bit-identically to the pre-T-14 i32·i32 path (products ≪ 2⁵³, exact).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Weights {
    pub progress: f64,
    pub base_presence: f64,
    pub defense: f64,
    pub spread: f64,
    pub mobility: f64,
    pub threat: f64,
}

impl Default for Weights {
    fn default() -> Self {
        Weights {
            progress: 6.0,
            base_presence: 30.0,
            defense: 25.0,
            spread: 4.0,
            mobility: 3.0,
            threat: 55.0,
        }
    }
}

impl Weights {
    /// Load a flat JSON object `{ "progress": N, "base_presence": N, … }`.
    /// Missing or unparseable fields keep the default for that field.
    /// A missing / unreadable file falls back entirely to the defaults.
    pub fn load(path: &str) -> Weights {
        let Ok(text) = std::fs::read_to_string(path) else {
            return Weights::default();
        };
        let mut w = Weights::default();
        let (open, close) = match (text.find('{'), text.rfind('}')) {
            (Some(o), Some(c)) if o < c => (o, c),
            _ => return w,
        };
        for field in text[open + 1..close].split(',') {
            let field = field.trim();
            if let Some(c) = field.find(':') {
                let key = field[..c].trim().trim_matches('"');
                let Ok(val) = field[c + 1..].trim().parse::<f64>() else {
                    continue;
                };
                match key {
                    "progress" => w.progress = val,
                    "base_presence" => w.base_presence = val,
                    "defense" => w.defense = val,
                    "spread" => w.spread = val,
                    "mobility" => w.mobility = val,
                    "threat" => w.threat = val,
                    _ => {}
                }
            }
        }
        w
    }

    /// Write the six fields as a flat JSON object.
    pub fn save(&self, path: &str) {
        let s = format!(
            "{{\"progress\": {}, \"base_presence\": {}, \"defense\": {}, \"spread\": {}, \"mobility\": {}, \"threat\": {}}}",
            self.progress,
            self.base_presence,
            self.defense,
            self.spread,
            self.mobility,
            self.threat,
        );
        let _ = std::fs::write(path, s);
    }
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

#[inline]
fn other(side: u8) -> u8 {
    if side == P1 {
        P2
    } else {
        P1
    }
}

/// Top-left `(r0, c0)` of a player's *home* 3×3 base (rows `r0..=r0+2`, cols
/// `c0..=c0+2`). P1: (5, 0); P2: (0, 5).
#[inline]
fn base_origin(side: u8) -> (u8, u8) {
    match side {
        P1 => (5, 0),
        P2 => (0, 5),
        _ => unreachable!(),
    }
}

/// Manhattan distance from cell `(r, c)` to the 3×3 base square
/// `[r0..=r0+2] × [c0..=c0+2]`.
#[inline]
fn dist_to_base(r: usize, c: usize, r0: usize, c0: usize) -> i32 {
    let dr = (r as i32) - (r0 as i32);
    let dbot = (r as i32) - ((r0 + 2) as i32);
    let dc = (c as i32) - (c0 as i32);
    let dright = (c as i32) - ((c0 + 2) as i32);
    let row_part = if dr < 0 { -dr } else if dbot > 0 { dbot } else { 0 };
    let col_part = if dc < 0 { -dc } else if dright > 0 { dright } else { 0 };
    row_part + col_part
}

fn for_each_piece<F: FnMut((u8, u8))>(b: &Board, side: u8, mut f: F) {
    for i in 0..64 {
        if b.grid[i] == side {
            f(row_col(i));
        }
    }
}

// ---------------------------------------------------------------------------
// six term functions — each pure, each "more positive = better for that side"
// ---------------------------------------------------------------------------

/// Sum over `side`'s pieces of Manhattan distance to the target base,
/// **negated** so that being closer (smaller sum) → larger value → better.
/// A piece one square closer raises this term by exactly +1.
pub fn progress(b: &Board, side: u8) -> i32 {
    let (r0, c0) = base_origin(other(side));
    let (r0i, c0i) = (r0 as usize, c0 as usize);
    let mut total = 0i32;
    for_each_piece(b, side, |(r, c)| {
        total += dist_to_base(r as usize, c as usize, r0i, c0i);
    });
    -total
}

/// Number of `side`'s pieces already inside the target base. +1 when a new
/// piece lands in the base.
pub fn base_presence(b: &Board, side: u8) -> i32 {
    base_cells(other(side))
        .iter()
        .filter(|&&(r, c)| b.grid[idx(r, c)] == side)
        .count() as i32
}

/// Negated count of the *opponent's* pieces already in `side`'s own base.
/// A single opponent piece entering our base lowers this term by exactly -1.
pub fn defense(b: &Board, side: u8) -> i32 {
    let opp = other(side);
    -(base_cells(side)
        .iter()
        .filter(|&&(r, c)| b.grid[idx(r, c)] == opp)
        .count() as i32)
}

/// Negated number of **unordered orthogonal adjacent pairs** among `side`'s
/// pieces (each pair counted once). Two adjacent pieces → -1; two pieces 3
/// apart → 0. Encourages spreading.
pub fn spread(b: &Board, side: u8) -> i32 {
    let mut pairs = 0i32;
    for i in 0..64 {
        if b.grid[i] != side {
            continue;
        }
        let (r, c) = row_col(i);
        let (ri, ci) = (r as i32, c as i32);
        // count only right + down to avoid double-counting
        for &(dr, dc) in &[(0_i32, 1_i32), (1_i32, 0_i32)] {
            let nr = ri + dr;
            let nc = ci + dc;
            if (0..SIZE as i32).contains(&nr) && (0..SIZE as i32).contains(&nc)
                && b.grid[idx(nr as u8, nc as u8)] == side
            {
                pairs += 1;
            }
        }
    }
    -pairs
}

/// Total `|legal_steps| + 2·|legal_jumps|` across `side`'s pieces.
/// One more open square adjacent to a piece → +1 for that piece.
/// T-6.5: computed with the inlined, allocation-free scan; the counts are
/// byte-identical to the `legal_steps` / `legal_jumps` `Vec`-based versions.
pub fn mobility(b: &Board, side: u8) -> i32 {
    let g = &b.grid;
    let mut total = 0i32;
    for i in 0..64 {
        if g[i] != side {
            continue;
        }
        let (ri, ci) = ((i / 8) as i8, (i % 8) as i8);
        for &(dr, dc) in DIRS.iter() {
            let (nr, nc) = (ri + dr, ci + dc);
            if (0..SIZE as i8).contains(&nr)
                && (0..SIZE as i8).contains(&nc)
                && g[idx(nr as u8, nc as u8)] == 0
            {
                total += 1; // open step
            }
            let (jr, jc) = (ri + 2 * dr, ci + 2 * dc);
            if (0..SIZE as i8).contains(&jr)
                && (0..SIZE as i8).contains(&jc)
                && g[idx(nr as u8, nc as u8)] != 0
                && g[idx(jr as u8, jc as u8)] == 0
            {
                total += 2; // open jump
            }
        }
    }
    total
}

/// Number of **empty** cells in the *opponent's* base that any of `side`'s
/// pieces can land on in one step or one jump. A position one move from being
/// filled (and the mover can land the last piece) scores ≥ 1 → the largest
/// single-term win signal.
/// T-6.5: scans from `side`'s pieces to their reachable destination cells and
/// marks the ones inside the target base — the same set of base cells the
/// original (base-cell → attacker) scan found, computed allocation-free.
pub fn threat(b: &Board, side: u8) -> i32 {
    let base = base_cells(other(side));
    let in_target_base = |i: usize| -> bool {
        let (r, c) = ((i / 8) as u8, (i % 8) as u8);
        base.iter().any(|cell| *cell == (r, c))
    };
    let g = &b.grid;
    let mut marked = [false; 64];
    for i in 0..64 {
        if g[i] != side {
            continue;
        }
        let (ri, ci) = ((i / 8) as i8, (i % 8) as i8);
        for &(dr, dc) in DIRS.iter() {
            let (nr, nc) = (ri + dr, ci + dc);
            let (jr, jc) = (ri + 2 * dr, ci + 2 * dc);
            if (0..SIZE as i8).contains(&nr)
                && (0..SIZE as i8).contains(&nc)
                && g[idx(nr as u8, nc as u8)] == 0
                && in_target_base(idx(nr as u8, nc as u8))
            {
                marked[idx(nr as u8, nc as u8)] = true; // step-landable base cell
            }
            if (0..SIZE as i8).contains(&jr)
                && (0..SIZE as i8).contains(&jc)
                && g[idx(nr as u8, nc as u8)] != 0
                && g[idx(jr as u8, jc as u8)] == 0
                && in_target_base(idx(jr as u8, jc as u8))
            {
                marked[idx(jr as u8, jc as u8)] = true; // jump-landable base cell
            }
        }
    }
    marked.iter().filter(|&&m| m).count() as i32
}

// ---------------------------------------------------------------------------
// compose
// ---------------------------------------------------------------------------

/// `side`'s weighted score = Σ `w_i × term_i(side)`, in `f64` (T-14).
///
/// `evaluate` / `evaluate_with` / `evaluate_pair` cast the world score back to
/// `i32`. For any **integer** weight set the `f64` result is an exact integer
/// (each `f64 * i32` product is exact for |product| ≪ 2⁵³ and the sum is too),
/// so the truncation `as i32` yields **bit-identical** scores to the old
/// i32·i32 path. The `i32` world score keeps every search invariant (alpha-beta
/// integer comparison, anti-symmetry, start=0) unchanged.
fn side_score(b: &Board, side: u8, w: &Weights) -> f64 {
    w.progress * progress(b, side) as f64
        + w.base_presence * base_presence(b, side) as f64
        + w.defense * defense(b, side) as f64
        + w.spread * spread(b, side) as f64
        + w.mobility * mobility(b, side) as f64
        + w.threat * threat(b, side) as f64
}

/// World score: `P1 score − P2 score`. Turn-independent.
pub fn evaluate(b: &Board) -> i32 {
    let w = Weights::default();
    (side_score(b, P1, &w) - side_score(b, P2, &w)) as i32
}

/// World score using an explicit `Weights` (exposed for tests and later
/// `Search::set_weights`).
pub fn evaluate_with(b: &Board, w: &Weights) -> i32 {
    (side_score(b, P1, w) - side_score(b, P2, w)) as i32
}

/// World score with **per-side** weights: `side_score(P1, w1) - side_score(P2, w2)`.
/// This is the head-to-head primitive (T-11): each side scores its own six terms
/// with its own weight set, so P1 (the maximizer) can run one weight set while
/// P2 (the minimizer) runs another. Collapses exactly to the existing functions:
/// `evaluate_pair(b, w, w) == evaluate_with(b, w)`, and with both `Weights::default()`
/// it is byte-identical to `evaluate(b)` — so the default path and every existing
/// test are unchanged when the search is not told otherwise.
pub fn evaluate_pair(b: &Board, w1: &Weights, w2: &Weights) -> i32 {
    (side_score(b, P1, w1) - side_score(b, P2, w2)) as i32
}

// ---------------------------------------------------------------------------
// tests — each term, anti-symmetry, and the sign invariant
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::idx;

    fn board_with(pieces: &[(u8, u8, u8)]) -> Board {
        let mut b = Board::new();
        for &(r, c, p) in pieces {
            b.grid[idx(r, c)] = p;
        }
        b
    }

    // ---- sign test: start-position is exactly 0 (symmetric) -----

    #[test]
    fn start_position_is_zero() {
        assert_eq!(evaluate(&Board::start()), 0);
    }

    // ---- anti-symmetry: mirror (rotate 180° + swap colors) is exact ----

    fn mirror(b: &Board) -> Board {
        let mut m = Board::new();
        for i in 0..64 {
            let v = b.grid[i];
            if v == 0 {
                continue;
            }
            let (r, c) = row_col(i);
            m.grid[idx(7 - r, 7 - c)] = if v == P1 { P2 } else { P1 };
        }
        m
    }

    #[test]
    fn anti_symmetry_mid_board_exact() {
        let b = board_with(&[(1, 4, P1), (3, 3, P1), (0, 6, P2), (5, 5, P2)]);
        let bm = mirror(&b);
        assert_eq!(evaluate(&bm), -evaluate(&b));
    }

    #[test]
    fn anti_symmetry_heavier_board_exact() {
        let b = board_with(&[
            (2, 4, P1),
            (4, 2, P1),
            (5, 4, P1),
            (6, 6, P1),
            (3, 5, P2),
            (4, 7, P2),
            (6, 5, P2),
            (7, 3, P2),
        ]);
        let bm = mirror(&b);
        assert_eq!(evaluate(&bm), -evaluate(&b));
    }

    // ---- per-term differential tests (each isolates one feature) ----

    fn weighted_term(fn_term: fn(&Board, u8) -> i32, w: f64, b: &Board, side: u8) -> f64 {
        w * fn_term(b, side) as f64
    }

    #[test]
    fn progress_moves_by_one_when_piece_one_closer() {
        // P1 in base, one piece at (4,0) — one square right → (4,1) — both
        // outside the target base (P2 base r0=0..2, c0=5..7).
        //   dist(4,0, r0=0, c0=5) = 4+5 = 9
        //   dist(4,1, r0=0, c0=5) = 4+4 = 8
        // Δ progress = +1 → weighted Δ = +w.progress
        let b1 = board_with(&[(4, 0, P1)]);
        let b2 = board_with(&[(4, 1, P1)]);
        let w = Weights::default();
        assert_eq!(
            weighted_term(progress, w.progress, &b2, P1) - weighted_term(progress, w.progress, &b1, P1),
            w.progress,
            "progress must rise by exactly w.progress when a piece moves 1 closer"
        );
    }

    #[test]
    fn base_presence_jumps_by_one_when_piece_enters_target_base() {
        // P1 at (3,7) → (2,7): (2,7) IS in P2 base (rows 0..2, cols 5..7).
        // base_presence: 0 → 1 → Δ = +1 → weighted Δ = +w.base_presence.
        let b1 = board_with(&[(3, 7, P1)]);
        let b2 = board_with(&[(2, 7, P1)]);
        let w = Weights::default();
        assert_eq!(
            weighted_term(base_presence, w.base_presence, &b2, P1)
                - weighted_term(base_presence, w.base_presence, &b1, P1),
            w.base_presence
        );
    }

    #[test]
    fn defense_drops_by_one_when_opp_piece_enters_our_base() {
        // P1 anchored at (5,1) (own base). Move a P2 piece INTO P1's base:
        // (4,0) → (5,0). defense(P1): 0 → -1 → Δ = -1 → weighted Δ = -w.defense.
        let b1 = board_with(&[(5, 1, P1), (4, 0, P2)]); // P2 OUTSIDE P1 base
        let b2 = board_with(&[(5, 1, P1), (5, 0, P2)]); // P2 INSIDE P1 base
        let w = Weights::default();
        let d1 = weighted_term(defense, w.defense, &b1, P1);
        let d2 = weighted_term(defense, w.defense, &b2, P1);
        assert_eq!(
            d2 - d1,
            -w.defense,
            "defense must drop by exactly w.defense when one opp piece enters our base"
        );
    }

    #[test]
    fn spread_adjacent_scores_lower_than_apart() {
        let b_adj = board_with(&[(4, 4, P1), (4, 5, P1)]); // adjacent pair
        let b_far = board_with(&[(4, 4, P1), (4, 7, P1)]); // 3 apart, no pair
        let w = Weights::default();
        assert!(
            weighted_term(spread, w.spread, &b_adj, P1)
                < weighted_term(spread, w.spread, &b_far, P1),
            "adjacent pair must score lower than 3-apart"
        );
    }

    #[test]
    fn mobility_counts_steps_plus_twice_jumps() {
        // b1: single P1 at (4,4). 4 open orthogonal steps, no jump possible
        // (no neighbour is an occupied middle) → mobility = 4 + 2*0 = 4.
        let b1 = board_with(&[(4, 4, P1)]);
        assert_eq!(mobility(&b1, P1), 4);
        // b2: a P2 now sits on the up-adjacent (3,4). That step is lost
        // (3 steps left) but a jump over it opens (middle (3,4) occupied,
        // dest (2,4) empty), contributing 2*1 → mobility = 3 + 2 = 5.
        // A blocker can therefore RAISE the mobility term: it opens a jump
        // target, so the term is not monotone in "open neighbours" — it
        // rewards usable jump options, which is the intended signal.
        let b2 = board_with(&[(4, 4, P1), (3, 4, P2)]);
        assert_eq!(mobility(&b2, P1), 5);
    }

    #[test]
    fn threat_positive_when_attacker_can_complete_in_one() {
        // P1 has 8 in P2 base, last empty cell (2,7), piece (3,7) can step onto it.
        let b = board_with(&[
            (0, 5, P1),
            (0, 6, P1),
            (0, 7, P1),
            (1, 5, P1),
            (1, 6, P1),
            (1, 7, P1),
            (2, 5, P1),
            (2, 6, P1),
            (3, 7, P1),
        ]);
        let w = Weights::default();
        assert!(
            weighted_term(threat, w.threat, &b, P1) > 0.0,
            "attacker must get a positive threat bonus when one step fills the base"
        );
        // Defender (P2) should NOT get a positive threat in this position.
        assert_eq!(weighted_term(threat, w.threat, &b, P2), 0.0);
    }

    // ---- Weights::load / save round-trip ----

    #[test]
    fn weights_load_save_roundtrip() {
        let w = Weights {
            progress: 7.0,
            base_presence: 33.0,
            defense: 22.0,
            spread: 5.0,
            mobility: 4.0,
            threat: 60.0,
        };
        let path = std::env::temp_dir().join(format!(
            "halma8_ai_weights_test_{}.json",
            std::process::id()
        ));
        w.save(path.to_str().unwrap());
        let w2 = Weights::load(path.to_str().unwrap());
        assert_eq!(w, w2);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn weights_fractional_roundtrip() {
        // T-14 (gate 5): a fractional (non-integer) weight must survive
        // save -> load, so the "budget" is no longer forced to zero by the old
        // integer rounding. 55.5 is exactly representable, so round-trip is exact.
        let w = Weights {
            progress: 6.0,
            base_presence: 30.0,
            defense: 25.0,
            spread: 4.0,
            mobility: 3.0,
            threat: 55.5,
        };
        let path = std::env::temp_dir().join(format!(
            "halma8_ai_weights_frac_test_{}.json",
            std::process::id()
        ));
        w.save(path.to_str().unwrap());
        let w2 = Weights::load(path.to_str().unwrap());
        assert_eq!(w, w2, "55.5 must round-trip exactly through save/load");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn weights_load_missing_file_gives_defaults() {
        let w = Weights::load("/nonexistent/path/does_not_exist.json");
        assert_eq!(w, Weights::default());
    }

    // ---- T-11 — evaluate_pair is the per-side generalization of evaluate_with ----

    #[test]
    fn evaluate_pair_default_pair_equals_evaluate() {
        let b = board_with(&[(1, 4, P1), (3, 3, P1), (0, 6, P2), (5, 5, P2)]);
        let d = Weights::default();
        assert_eq!(evaluate_pair(&b, &d, &d), evaluate(&b));
    }

    #[test]
    fn evaluate_pair_same_weights_equals_evaluate_with() {
        let b = board_with(&[(2, 4, P1), (4, 2, P1), (0, 6, P2), (3, 5, P2)]);
        let w = Weights {
            progress: 7.0,
            base_presence: 33.0,
            defense: 22.0,
            spread: 5.0,
            mobility: 4.0,
            threat: 60.0,
        };
        assert_eq!(evaluate_pair(&b, &w, &w), evaluate_with(&b, &w));
    }

    #[test]
    fn evaluate_pair_honours_distinct_side_weights() {
        // A non-symmetric position where the two sides carry different piece
        // counts: swapping which weight set scores which side must change the
        // world score (proves w1 and w2 are each applied to the correct side).
        let b = board_with(&[(4, 4, P1), (4, 5, P1), (3, 3, P2)]);
        let hi = Weights {
            progress: 0.0,
            base_presence: 0.0,
            defense: 0.0,
            spread: 0.0,
            mobility: 0.0,
            threat: 7.0,
        };
        let lo = Weights {
            progress: 3.0,
            base_presence: 3.0,
            defense: 3.0,
            spread: 3.0,
            mobility: 3.0,
            threat: 3.0,
        };
        assert_ne!(
            evaluate_pair(&b, &hi, &lo),
            evaluate_pair(&b, &lo, &hi),
            "per-side weights must be side-specific, not a shared weight set"
        );
    }
}
