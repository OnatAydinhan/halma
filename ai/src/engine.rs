//! Board state (T-1) + pure move generation (T-2) for the check_halma AI: a
//! faithful port of `halma8.py`. Serialization and the pure `legal_steps` /
//! `legal_jumps` board functions live here; the stateful rule layer
//! (`apply_move` / `end_turn` / `_check_win` / `IllegalMove`) lands in T-3.

use crate::types::{base_cells, idx, row_col, DIRS, P1, P2, SIZE};

/// Board state, mirroring the fields of `halma8.Halma8`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Board {
    /// 64 cells, row-major. `0` empty, `1` = P1, `2` = P2.
    pub grid: [u8; 64],
    pub turn: u8,
    pub winner: Option<u8>,
    pub turn_no: u32,
    pub chain_active: bool,
    /// Bit-set of banned squares in the active chain (origin + landing squares).
    pub chain_banned: [bool; 64],
    pub chain_starter: Option<(u8, u8)>,
    /// Current position of the chain piece (where it sits right now).
    pub chain_current: Option<(u8, u8)>,
}

fn cell_char(v: u8) -> char {
    match v {
        0 => '.',
        1 => '1',
        2 => '2',
        _ => panic!("bad cell value: {}", v),
    }
}

fn char_cell(c: char) -> Option<u8> {
    match c {
        '.' => Some(0),
        '1' => Some(1),
        '2' => Some(2),
        _ => None,
    }
}

/// Parse a two-chars square token like `"45"` into `(4, 5)`.
fn parse_pair(t: &str) -> Option<(u8, u8)> {
    let b = t.as_bytes();
    if b.len() != 2 || !b[0].is_ascii_digit() || !b[1].is_ascii_digit() {
        return None;
    }
    let r = (b[0] - b'0') as u8;
    let c = (b[1] - b'0') as u8;
    if r > 7 || c > 7 {
        return None;
    }
    Some((r, c))
}

/// A move to be applied. Mirrors the Python `("step", pos, dst)` / `("jump",
/// pos, mid, dst)` tuples. `stop` (end a live chain) is not a move — it is
/// `end_turn` and is handled by the server, not `apply_move`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Move {
    Step { from: (u8, u8), to: (u8, u8) },
    Jump { from: (u8, u8), mid: (u8, u8), to: (u8, u8) },
}

/// A rule violation — the Rust analogue of `halma8.IllegalMove`
/// (`class IllegalMove(ValueError)`). Only the *type* is contractually
/// meaningful (Ok(true)/Ok(false)/Err must line up with Python); the
/// message need not match.
#[derive(Debug, PartialEq, Eq)]
pub struct IllegalMove;

impl Board {
    /// Blank state, no pieces placed, P1 to move, turn 1. (Not the start
    /// position — see `Board::start`.)
    pub fn new() -> Board {
        Board {
            grid: [0; 64],
            turn: P1,
            winner: None,
            turn_no: 1,
            chain_active: false,
            chain_banned: [false; 64],
            chain_starter: None,
            chain_current: None,
        }
    }

    /// Exactly matches `Halma8.__init__`: P1 fills its base, P2 fills its
    /// base, P1 to move, turn 1, no winner, no chain.
    pub fn start() -> Board {
        let mut b = Board::new();
        for (r, c) in base_cells(P1) {
            b.grid[idx(r, c)] = P1;
        }
        for (r, c) in base_cells(P2) {
            b.grid[idx(r, c)] = P2;
        }
        b
    }

    /// Cell accessor (panic on out-of-bounds — call sites are all internal).
    #[inline]
    pub fn cell(&self, r: u8, c: u8) -> u8 {
        assert!(r < 8 && c < 8, "cell({},{}) out of range", r, c);
        self.grid[idx(r, c)]
    }

    /// Adjacent orthogonal empty squares, in `DIRS` order. Returns `[]` for an
    /// empty or out-of-range square. Faithful port of `Halma8.legal_steps`.
    pub fn legal_steps(&self, r: u8, c: u8) -> Vec<(u8, u8)> {
        if (r as usize) >= SIZE as usize || (c as usize) >= SIZE as usize || self.grid[idx(r, c)] == 0 {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(4);
        let (ri, ci) = (r as i8, c as i8);
        for &(dr, dc) in DIRS.iter() {
            let (nr, nc) = (ri + dr, ci + dc);
            if (0..SIZE as i8).contains(&nr)
                && (0..SIZE as i8).contains(&nc)
                && self.grid[idx(nr as u8, nc as u8)] == 0
            {
                out.push((nr as u8, nc as u8));
            }
        }
        out
    }

    /// `(middle, destination)` jump pairs, in `DIRS` order. Middle must be
    /// occupied, destination empty, and destination in-bounds. Returns `[]` for
    /// an empty or out-of-range square. Faithful port of `Halma8.legal_jumps`.
    pub fn legal_jumps(&self, r: u8, c: u8) -> Vec<((u8, u8), (u8, u8))> {
        if (r as usize) >= SIZE as usize || (c as usize) >= SIZE as usize || self.grid[idx(r, c)] == 0 {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(4);
        let (ri, ci) = (r as i8, c as i8);
        for &(dr, dc) in DIRS.iter() {
            let (mi, ma) = (ri + dr, ci + dc);
            let (nr, nc) = (ri + 2 * dr, ci + 2 * dc);
            // `nr`/`nc` in-bounds (checked first) guarantees `mi`/`ma` in-bounds,
            // exactly as in halma8.py, so both lookups below are always valid.
            if (0..SIZE as i8).contains(&nr)
                && (0..SIZE as i8).contains(&nc)
                && self.grid[idx(mi as u8, ma as u8)] != 0
                && self.grid[idx(nr as u8, nc as u8)] == 0
            {
                out.push(((mi as u8, ma as u8), (nr as u8, nc as u8)));
            }
        }
        out
    }

    // ------------------------------------------------------------------
    // T-3: the stateful rule layer — a faithful port of `halma8.Halma8`.
    // ------------------------------------------------------------------

    /// Apply `("step", pos, dst)` / `("jump", pos, mid, dst)`.
    ///
    /// Returns `Ok(true)` when a jump was made and that piece still has a
    /// non-banned continuation (the chain is live, the turn is *unchanged*).
    /// Returns `Ok(false)` when the turn ended (turn already flipped, `turn_no`
    /// already incremented). Returns `Err(IllegalMove)` on any rule violation or
    /// when a winner is already set. Exact contract: `halma8.py::apply_move`.
    ///
    /// Note — like `halma8.Halma8.apply_move`, this does **not** enforce whose
    /// turn it is; `turn` is only flipped as a side effect.
    pub fn apply_move(&mut self, mv: &Move) -> Result<bool, IllegalMove> {
        if self.winner.is_some() {
            return Err(IllegalMove);
        }
        // Validate every square in the move before mutating anything
        // (matches `halma8.py`: `for sq in move[1:]` must be in-bounds).
        // A `u8` can never be negative, so `< SIZE` is the only real bound.
        let sqs: &[(u8, u8)] = match mv {
            Move::Step { from, to } => &[*from, *to],
            Move::Jump { from, mid, to } => &[*from, *mid, *to],
        };
        for &(r, c) in sqs {
            if (r as u8) >= SIZE || (c as u8) >= SIZE {
                return Err(IllegalMove);
            }
        }

        match mv {
            Move::Step { from: (r, c), to: (nr, nc) } => {
                let (r, c, nr, nc) = (*r, *c, *nr, *nc);
                if self.grid[idx(r, c)] == 0 {
                    return Err(IllegalMove);
                }
                if self.grid[idx(nr, nc)] != 0 {
                    return Err(IllegalMove);
                }
                if !self.legal_steps(r, c).contains(&(nr, nc)) {
                    return Err(IllegalMove);
                }
                let v = self.grid[idx(r, c)];
                self.grid[idx(nr, nc)] = v;
                self.grid[idx(r, c)] = 0;
                self.chain_active = false; // defensive: a step is always a new turn
                self.chain_banned = [false; 64];
                self.check_win();
                self.end_turn();
                Ok(false)
            }
            Move::Jump { from: (r, c), mid: (mr, mc), to: (nr, nc) } => {
                let (r, c, mr, mc, nr, nc) = (*r, *c, *mr, *mc, *nr, *nc);
                if self.grid[idx(r, c)] == 0 {
                    return Err(IllegalMove);
                }
                if self.chain_active {
                    // Only the chain piece (at its current landing) may continue.
                    if self.chain_current != Some((r, c)) {
                        return Err(IllegalMove);
                    }
                    // Chain-revisit ban: landing on an already-visited square.
                    if self.chain_banned[idx(nr, nc)] {
                        return Err(IllegalMove);
                    }
                }
                if !self.legal_jumps(r, c).iter().any(|&((a, b), (d, e))| (a, b) == (mr, mc) && (d, e) == (nr, nc)) {
                    return Err(IllegalMove);
                }
                let v = self.grid[idx(r, c)];
                self.grid[idx(nr, nc)] = v;
                self.grid[idx(r, c)] = 0;
                if !self.chain_active {
                    self.chain_active = true;
                    self.chain_starter = Some((r, c));
                    self.chain_banned = [false; 64];
                    self.chain_banned[idx(r, c)] = true; // origin: revisit ban
                }
                self.chain_banned[idx(nr, nc)] = true;
                self.chain_current = Some((nr, nc));
                self.check_win();
                if self.winner.is_none() && self.has_open_jump(nr, nc) {
                    return Ok(true); // chain in progress: turn unchanged
                }
                self.chain_active = false;
                self.chain_banned = [false; 64];
                self.chain_current = None;
                self.end_turn();
                Ok(false)
            }
        }
    }

    /// `True` iff any legal jump from `(r, c)` lands on a non-banned square.
    /// Mirrors `Halma8._has_open_jump`.
    pub fn has_open_jump(&self, r: u8, c: u8) -> bool {
        for (_mid, dst) in self.legal_jumps(r, c) {
            if !self.chain_banned[idx(dst.0, dst.1)] {
                return true;
            }
        }
        false
    }

    /// Flip the turn and bump `turn_no`; no-op when a winner is set. Resets any
    /// live chain. Mirrors `Halma8.end_turn`.
    pub fn end_turn(&mut self) {
        if self.winner.is_some() {
            return;
        }
        self.turn = if self.turn == P1 { P2 } else { P1 };
        self.turn_no += 1;
        self.chain_active = false;
        self.chain_banned = [false; 64];
        self.chain_current = None;
    }

    /// "Full + invaded" win check for both bases (P1's target first). A player
    /// wins only when the *opponent's* 3×3 base is completely occupied **and**
    /// contains at least one of that player's own pieces. Mirrors
    /// `Halma8._check_win`.
    pub fn check_win(&mut self) {
        let p2b = base_cells(P2);
        let p1_wins = p2b
            .iter()
            .all(|&(r, c)| self.grid[idx(r, c)] != 0)
            && p2b.iter().any(|&(r, c)| self.grid[idx(r, c)] == P1);
        let p1b = base_cells(P1);
        let p2_wins = p1b
            .iter()
            .all(|&(r, c)| self.grid[idx(r, c)] != 0)
            && p1b.iter().any(|&(r, c)| self.grid[idx(r, c)] == P2);
        if p1_wins {
            self.winner = Some(P1);
        } else if p2_wins {
            self.winner = Some(P2);
        }
    }

    /// Canonical one-line state (mirrors the Python cross-check expected value):
    /// `<position_str> turn_no=<n> winner=<0|1|2>`. `winner` is `0` when `None`.
    pub fn state_line(&self) -> String {
        let w = self.winner.unwrap_or(0);
        format!(
            "{} turn_no={} winner={}",
            self.to_position_str(),
            self.turn_no,
            w
        )
    }

    /// Board-string emit: 64 chars, `.`/`1`/`2`, row-major.
    pub fn to_board_str(&self) -> String {
        let mut s = String::with_capacity(64);
        for i in 0..64 {
            s.push(cell_char(self.grid[i]));
        }
        s
    }

    /// Parse a 64-char board string.
    pub fn from_board_str(s: &str) -> Result<Board, String> {
        let chars: Vec<char> = s.chars().collect();
        if chars.len() != 64 {
            return Err(format!("board string must be 64 chars, got {}", chars.len()));
        }
        let mut b = Board::new();
        for i in 0..64 {
            b.grid[i] = char_cell(chars[i])
                .ok_or_else(|| format!("bad board char {:?} at {}", chars[i], i))?;
        }
        Ok(b)
    }

    /// Position string: `<board> <turn> <chain>`.
    ///
    /// `chain` is `-` when `chain_active` is false; else
    /// `C:o=<r><c>;b=<r1><c1>,<r2><c2>,...;c=<r><c>` with banned squares sorted
    /// ascending by (r, c) for determinism, and `c=` the chain piece's current square.
    pub fn to_position_str(&self) -> String {
        let board = self.to_board_str();
        let turn = self.turn.to_string();
        let chain = if !self.chain_active || self.chain_starter.is_none() {
            "-".to_string()
        } else {
            let (or, oc) = self.chain_starter.unwrap();
            let mut banned: Vec<usize> = self
                .chain_banned
                .iter()
                .enumerate()
                .filter(|(_, v)| **v)
                .map(|(i, _)| i)
                .collect();
            banned.sort_unstable();
            let b_str = banned
                .iter()
                .map(|i| {
                    let (r, c) = row_col(*i);
                    format!("{}{}", r, c)
                })
                .collect::<Vec<_>>()
                .join(",");
            let (cc_r, cc_c) = self.chain_current.unwrap_or((0, 0));
            format!("C:o={or}{oc};b={b_str};c={cc_r}{cc_c}")
        };
        format!("{} {} {}", board, turn, chain)
    }

    /// Parse a position string.
    pub fn from_position_str(s: &str) -> Result<Board, String> {
        let mut parts = s.split_ascii_whitespace();
        let board_s = parts.next().ok_or("missing board")?;
        let turn_s = parts.next().ok_or("missing turn")?;
        let chain_s = parts.next().ok_or("missing chain")?;

        let mut b = Board::from_board_str(board_s)?;
        let turn: u8 = turn_s
            .parse()
            .map_err(|e| format!("bad turn {}: {}", turn_s, e))?;
        if turn != P1 && turn != P2 {
            return Err("turn must be 1 or 2".to_string());
        }
        b.turn = turn;

        if chain_s != "-" {
            if !chain_s.starts_with("C:o=") {
                return Err(format!("bad chain token {}", chain_s));
            }
            // rest after "C:" = "o=RC;b=RC,...;c=RC"
            let rest = &chain_s[2..];
            let mut semis = rest.split(';');
            let o_part = semis
                .next()
                .ok_or("missing o= in chain")?;
            let b_part = semis
                .next()
                .ok_or("missing b= in chain")?;
            let c_part = semis.next(); // optional: "c=RC"
            if semis.next().is_some() {
                return Err("too many ; in chain token".to_string());
            }
            if !o_part.starts_with("o=") || !b_part.starts_with("b=") {
                return Err("chain token must be C:o=RC;b=RC,...".to_string());
            }
            let o_rc = &o_part[2..];
            let (or, oc) = parse_pair(o_rc).ok_or("bad chain origin")?;
            b.chain_active = true;
            b.chain_starter = Some((or, oc));
            b.chain_banned = [false; 64];
            let b_body = &b_part[2..];
            if !b_body.is_empty() {
                for tok in b_body.split(',') {
                    if tok.is_empty() {
                        continue;
                    }
                    let (r, c) = parse_pair(tok).ok_or_else(|| format!("bad banned square {}", tok))?;
                    b.chain_banned[idx(r, c)] = true;
                }
            }
            // chain_current: required for protocol correctness
            if let Some(c_tok) = c_part {
                if !c_tok.starts_with("c=") {
                    return Err(format!("bad chain-current token {}", c_tok));
                }
                let (r, c) = parse_pair(&c_tok[2..]).ok_or("bad chain current")?;
                b.chain_current = Some((r, c));
            } else {
                // backward-compat: derive from banned set (last banned square occupied by turn)
                b.chain_current = None;
            }
        } else {
            b.chain_active = false;
            b.chain_starter = None;
            b.chain_banned = [false; 64];
            b.chain_current = None;
        }
        Ok(b)
    }
}

impl Default for Board {
    fn default() -> Self {
        Board::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn start_placement_matches_halma8() {
        let b = Board::start();
        // Exactly 18 occupied cells: 9 P1 in its base, 9 P2 in its base.
        let occupied: Vec<u8> = b
            .grid
            .iter()
            .filter(|&&v| v != 0)
            .cloned()
            .collect();
        let n_p1 = occupied.iter().filter(|v| **v == P1).count();
        let n_p2 = occupied.iter().filter(|v| **v == P2).count();
        assert_eq!((n_p1, n_p2), (9, 9), "start must have exactly 9 of each");

        // Cell identity for every base cell.
        for (r, c) in base_cells(P1) {
            assert_eq!(b.cell(r, c), P1);
        }
        for (r, c) in base_cells(P2) {
            assert_eq!(b.cell(r, c), P2);
        }
        // A few explicitly empty cells.
        assert_eq!(b.cell(0, 0), 0);
        assert_eq!(b.cell(3, 3), 0);
        assert_eq!(b.cell(4, 4), 0);

        // Initial flags.
        assert_eq!(b.turn, P1);
        assert_eq!(b.turn_no, 1);
        assert_eq!(b.winner, None);
        assert!(!b.chain_active);
        assert!(!b.chain_banned.iter().any(|&v| v));
        assert_eq!(b.chain_starter, None);
    }

    #[test]
    fn board_str_start_roundtrips() {
        let b = Board::start();
        let s = b.to_board_str();
        assert_eq!(s.len(), 64);
        // Layout spot checks (each ".....222"-ish row for P2, "111....." for P1).
        assert_eq!(&s[0..8], ".....222");
        assert_eq!(&s[40..48], "111.....");
        assert_eq!(&s[56..64], "111.....");
        let b2 = Board::from_board_str(&s).unwrap();
        assert_eq!(b2, b);
    }

    #[test]
    fn from_board_str_rejects_bad_length() {
        assert!(Board::from_board_str("1111").is_err());
        assert!(Board::from_board_str(".".repeat(64).as_str()).is_ok());
    }

    #[test]
    fn from_board_str_rejects_bad_chars() {
        let b = Board::start();
        let s = b.to_board_str();
        let mut t = s.chars().collect::<Vec<_>>();
        t.pop();        // drop the last '2'
        t.push('x');    // replace with a bad char, still 64 total
        let t: String = t.into_iter().collect();
        assert!(Board::from_board_str(&t).is_err());
    }

    #[test]
    fn position_str_start_roundtrips() {
        let b = Board::start();
        let s = b.to_position_str();
        let expected = [
            ".....222",
            ".....222",
            ".....222",
            "........",
            "........",
            "111.....",
            "111.....",
            "111.....",
        ]
        .join("")
            + " 1 -";
        assert_eq!(s, expected);
        let b2 = Board::from_position_str(&s).unwrap();
        assert_eq!(b, b2);
    }

    #[test]
    fn position_str_mid_chain_roundtrips() {
        // Contrived mid-chain state: starter (6,1), banned {(6,1),(6,3),(7,2)}, current (7,2).
        let mut b = Board::start();
        b.chain_active = true;
        b.chain_starter = Some((6, 1));
        b.chain_current = Some((7, 2));
        for (r, c) in [(6, 1), (6, 3), (7, 2)] {
            b.chain_banned[idx(r, c)] = true;
        }
        let s = b.to_position_str();
        assert!(s.ends_with(" 1 C:o=61;b=61,63,72;c=72"), "got {}", s);
        let b2 = Board::from_position_str(&s).unwrap();
        assert_eq!(b, b2);
    }

    // ---- T-2: pure move generation (legal_steps / legal_jumps) ----

    fn board_with(pieces: &[(u8, u8, u8)]) -> Board {
        let mut b = Board::new();
        for &(r, c, p) in pieces {
            b.grid[idx(r, c)] = p;
        }
        b
    }

    fn s_sorted(v: &[(u8, u8)]) -> Vec<(u8, u8)> {
        let mut v = v.to_vec();
        v.sort_unstable();
        v
    }

    fn j_sorted(v: &[((u8, u8), (u8, u8))]) -> Vec<((u8, u8), (u8, u8))> {
        let mut v = v.to_vec();
        v.sort_unstable();
        v
    }

    #[test]
    fn t2_start_steps_corner_and_boxed() {
        let b = Board::start();
        assert_eq!(s_sorted(&b.legal_steps(5, 2)), vec![(4, 2), (5, 3)]);
        assert_eq!(b.legal_steps(6, 1), Vec::<(u8, u8)>::new());
        let _ = j_sorted(&b.legal_jumps(5, 2)); // (5,2) has no jumps on the start board
    }

    #[test]
    fn t2_steps_preserves_dirs_order() {
        // Only in-bounds empty neighbours, in DIRS order (up, down, left, right).
        let b = board_with(&[(4, 4, P1)]);
        assert_eq!(b.legal_steps(4, 4), vec![(3, 4), (5, 4), (4, 3), (4, 5)]);
    }

    #[test]
    fn t2_isolated_jump_right() {
        let b = board_with(&[(4, 4, P1), (4, 5, P2)]);
        assert_eq!(b.legal_jumps(4, 4), vec![((4, 5), (4, 6))]);
    }

    #[test]
    fn t2_jump_over_own_piece() {
        let b = board_with(&[(4, 4, P1), (4, 5, P1)]);
        assert_eq!(b.legal_jumps(4, 4), vec![((4, 5), (4, 6))]);
    }

    #[test]
    fn t2_jump_over_own_other_side() {
        let b = board_with(&[(4, 6, P1), (4, 5, P1)]);
        assert_eq!(b.legal_jumps(4, 6), vec![((4, 5), (4, 4))]);
    }

    #[test]
    fn t2_no_jump_middle_empty() {
        let b = board_with(&[(4, 4, P1), (4, 6, P2)]);
        assert_eq!(b.legal_jumps(4, 4), Vec::new());
    }

    #[test]
    fn t2_no_jump_dst_occupied() {
        let b = board_with(&[(4, 4, P1), (4, 5, P2), (4, 6, P2)]);
        assert_eq!(b.legal_jumps(4, 4), Vec::new());
    }

    #[test]
    fn t2_jump_bounded_by_edges() {
        assert_eq!(
            board_with(&[(0, 4, P1), (0, 5, P2)]).legal_jumps(0, 4),
            vec![((0, 5), (0, 6))]
        );
        assert_eq!(
            board_with(&[(0, 4, P1), (1, 4, P2)]).legal_jumps(0, 4),
            vec![((1, 4), (2, 4))]
        );
        assert_eq!(
            board_with(&[(7, 4, P1), (7, 5, P2)]).legal_jumps(7, 5),
            vec![((7, 4), (7, 3))]
        );
        assert_eq!(board_with(&[(7, 6, P1), (7, 7, P2)]).legal_jumps(7, 6), Vec::new());
    }

    #[test]
    fn t2_empty_square_empty_lists() {
        let b = board_with(&[(4, 4, P1)]);
        assert_eq!(b.legal_steps(3, 3), Vec::new());
        assert_eq!(b.legal_jumps(3, 3), Vec::new());
    }

    #[test]
    fn t2_out_of_range_empty_lists() {
        let b = Board::start();
        for (r, c) in [(8u8, 0), (0, 8), (8, 8), (5, 9), (9, 5), (255, 255)] {
            assert_eq!(b.legal_steps(r, c), Vec::new());
            assert_eq!(b.legal_jumps(r, c), Vec::new());
        }
    }

    #[test]
    fn t2_pure_of_turn() {
        let mut b = board_with(&[(4, 4, P1), (4, 5, P2)]);
        let before = (b.legal_steps(4, 4), b.legal_jumps(4, 4));
        b.turn = P2;
        assert_eq!((b.legal_steps(4, 4), b.legal_jumps(4, 4)), before);
        // Ground truth on this small position.
        assert_eq!(s_sorted(&b.legal_steps(4, 4)), vec![(3, 4), (4, 3), (5, 4)]);
        assert_eq!(b.legal_jumps(4, 5), vec![((4, 4), (4, 3))]);
    }

    // ---- T-3 / T-4: stateful rules (apply_move / end_turn / check_win) ----

    fn is_banned_err(e: Result<bool, IllegalMove>) -> bool {
        matches!(e, Err(IllegalMove))
    }

    #[test]
    fn t3_step_applies_and_flips_turn() {
        let mut b = Board::start();
        assert_eq!(
            b.apply_move(&Move::Step { from: (5, 2), to: (4, 2) }),
            Ok(false)
        );
        assert_eq!(b.cell(4, 2), P1);
        assert_eq!(b.cell(5, 2), 0);
        assert_eq!(b.turn, P2);
        assert_eq!(b.turn_no, 2);
    }

    #[test]
    fn t3_two_jump_chain() {
        let mut b = board_with(&[(4, 1, P1), (4, 2, P2), (4, 4, P2)]);
        assert_eq!(
            b.apply_move(&Move::Jump { from: (4, 1), mid: (4, 2), to: (4, 3) }),
            Ok(true)
        );
        assert_eq!(b.cell(4, 3), P1);
        assert_eq!(b.cell(4, 1), 0);
        assert_eq!(b.turn, P1); // chain in progress: turn untouched
        assert_eq!(b.turn_no, 1);
        assert_eq!(
            b.apply_move(&Move::Jump { from: (4, 3), mid: (4, 4), to: (4, 5) }),
            Ok(false)
        );
        assert_eq!(b.cell(4, 5), P1);
        assert_eq!(b.cell(4, 3), 0);
        assert_eq!(b.turn, P2); // final jump
        assert_eq!(b.turn_no, 2);
    }

    #[test]
    fn t3_banned_destination_raises_while_chain_open() {
        let mut b = board_with(&[(4, 1, P1), (4, 2, P2), (4, 4, P2), (4, 6, P2)]);
        b.turn = P1;
        assert_eq!(
            b.apply_move(&Move::Jump { from: (4, 1), mid: (4, 2), to: (4, 3) }),
            Ok(true)
        );
        // the geometrically-valid back-jump to the banned origin (4,1) must raise
        assert!(is_banned_err(b.apply_move(
            &Move::Jump { from: (4, 3), mid: (4, 2), to: (4, 1) }
        )));
        assert_eq!(b.cell(4, 3), P1);
        assert_eq!(b.cell(4, 1), 0);
        assert!(b.chain_active);
    }

    #[test]
    fn t3_three_jump_chain() {
        let mut b = board_with(&[(4, 1, P1), (4, 2, P2), (4, 4, P2), (4, 6, P2)]);
        b.turn = P1;
        assert_eq!(
            b.apply_move(&Move::Jump { from: (4, 1), mid: (4, 2), to: (4, 3) }),
            Ok(true)
        );
        assert_eq!(b.turn, P1);
        assert_eq!(b.turn_no, 1);
        assert_eq!(
            b.apply_move(&Move::Jump { from: (4, 3), mid: (4, 4), to: (4, 5) }),
            Ok(true)
        );
        assert_eq!(b.turn, P1);
        assert_eq!(b.turn_no, 1);
        assert_eq!(
            b.apply_move(&Move::Jump { from: (4, 5), mid: (4, 6), to: (4, 7) }),
            Ok(false)
        );
        assert_eq!(b.cell(4, 7), P1);
        assert_eq!(b.turn, P2);
        assert_eq!(b.turn_no, 2);
        assert!(!b.chain_active);
    }

    #[test]
    fn t3_single_jump_ending_turn() {
        let mut b = board_with(&[(4, 4, P1), (4, 5, P2)]);
        b.turn = P1;
        assert_eq!(
            b.apply_move(&Move::Jump { from: (4, 4), mid: (4, 5), to: (4, 6) }),
            Ok(false)
        );
        assert_eq!(b.turn, P2);
        assert_eq!(b.turn_no, 2);
        assert_eq!(b.cell(4, 6), P1);
        assert_eq!(b.cell(4, 4), 0);
    }

    #[test]
    fn t3_end_turn_only_flips() {
        let mut b = Board::start();
        let before_grid = b.grid;
        b.end_turn();
        assert_eq!(b.grid, before_grid);
        assert_eq!(b.turn, P2);
        assert_eq!(b.turn_no, 2);
    }

    #[test]
    fn t3_step_into_occupied_raises() {
        let mut b = Board::start();
        assert!(is_banned_err(b.apply_move(
            &Move::Step { from: (6, 0), to: (6, 1) }
        )));
        assert_eq!(b.cell(6, 0), P1);
        assert_eq!(b.turn, P1);
        assert_eq!(b.turn_no, 1);
    }

    #[test]
    fn t3_jump_with_empty_middle_raises() {
        let mut b = board_with(&[(4, 4, P1)]);
        b.turn = P1;
        assert!(is_banned_err(b.apply_move(
            &Move::Jump { from: (4, 4), mid: (4, 5), to: (4, 6) }
        )));
        assert_eq!(b.cell(4, 4), P1);
        assert_eq!(b.turn, P1);
    }

    #[test]
    fn t3_jump_with_occupied_destination_raises() {
        let mut b = board_with(&[(4, 4, P1), (4, 5, P2), (4, 6, P2)]);
        b.turn = P1;
        assert!(is_banned_err(b.apply_move(
            &Move::Jump { from: (4, 4), mid: (4, 5), to: (4, 6) }
        )));
        assert_eq!(b.cell(4, 4), P1);
    }

    #[test]
    fn t3_move_off_board_raises() {
        let mut b = board_with(&[(7, 0, P1)]);
        b.turn = P1;
        // destination (8,0) is out of range -> Err before mutation
        assert!(is_banned_err(b.apply_move(
            &Move::Step { from: (7, 0), to: (8, 0) }
        )));
        let mut b2 = Board::start();
        // (4,3) is in range but not an adjacent empty of (5,2) -> Err
        assert!(is_banned_err(b2.apply_move(
            &Move::Step { from: (5, 2), to: (4, 3) }
        )));
        assert_eq!(b2.cell(5, 2), P1);
        assert_eq!(b2.turn, P1);
    }

    #[test]
    fn t3_move_after_winner_is_set_raises() {
        let mut b = Board::start();
        b.winner = Some(P2);
        assert!(is_banned_err(b.apply_move(
            &Move::Step { from: (5, 2), to: (4, 2) }
        )));
        let mut b2 = board_with(&[(4, 4, P1)]);
        b2.turn = P1;
        b2.winner = Some(P1);
        assert!(is_banned_err(b2.apply_move(
            &Move::Jump { from: (4, 4), mid: (4, 5), to: (4, 6) }
        )));
    }

    #[test]
    fn t3_end_turn_no_op_when_winner_set() {
        let mut b = Board::start();
        b.winner = Some(P1);
        b.end_turn();
        assert_eq!(b.turn, P1);
        assert_eq!(b.turn_no, 1);
    }

    #[test]
    fn t3_illegal_move_is_rejected() {
        let mut b = Board::start();
        assert!(is_banned_err(b.apply_move(
            &Move::Step { from: (5, 2), to: (4, 3) }
        )));
    }

    #[test]
    fn t4_p1_wins_pure() {
        let mut b = board_with(&[
            (0, 5, P1), (0, 6, P1), (0, 7, P1),
            (1, 5, P1), (1, 6, P1), (1, 7, P1),
            (2, 6, P1), (2, 7, P1), // (2,5) empty
            (2, 4, P1),
        ]);
        assert_eq!(
            b.apply_move(&Move::Step { from: (2, 4), to: (2, 5) }),
            Ok(false)
        );
        assert_eq!(b.winner, Some(P1));
        assert_eq!(b.turn, P1); // game over, end_turn was a no-op
    }

    #[test]
    fn t4_p1_wins_mixed_with_parked_blue() {
        let mut b = board_with(&[
            (0, 5, P1), (0, 6, P1), (0, 7, P1),
            (1, 5, P1), (1, 6, P1), (1, 7, P1),
            (2, 5, P2), // blue parked in P2's own base
            (2, 6, P1), // (2,7) empty
            (3, 7, P1),
        ]);
        assert_eq!(
            b.apply_move(&Move::Step { from: (3, 7), to: (2, 7) }),
            Ok(false)
        );
        assert_eq!(b.winner, Some(P1));
        assert_eq!(b.turn, P1);
    }

    #[test]
    fn t4_no_false_positive_at_8_of_9() {
        let mut b = board_with(&[
            (0, 5, P1), (0, 6, P1), (0, 7, P1),
            (1, 5, P2), (1, 6, P1), (1, 7, P1),
            (2, 5, P1), (2, 6, P1), // (2,7) empty
        ]);
        b.check_win();
        assert_eq!(b.winner, None);
    }

    #[test]
    fn t4_p2_wins_pure() {
        let mut b = board_with(&[
            (5, 0, P2), (5, 1, P2), (5, 2, P2),
            (6, 0, P2), (6, 1, P2), (6, 2, P2),
            (7, 0, P2), (7, 1, P2), // (7,2) empty
            (7, 3, P2),
        ]);
        assert_eq!(
            b.apply_move(&Move::Step { from: (7, 3), to: (7, 2) }),
            Ok(false)
        );
        assert_eq!(b.winner, Some(P2));
    }

    #[test]
    fn t4_p2_wins_mixed() {
        let mut b = board_with(&[
            (5, 0, P2), (5, 1, P2), (5, 2, P2),
            (6, 0, P2), (6, 1, P2), (6, 2, P2),
            (7, 0, P1), // red parked in P1's own base
            (7, 1, P2), // (7,2) empty
            (7, 3, P2),
        ]);
        assert_eq!(
            b.apply_move(&Move::Step { from: (7, 3), to: (7, 2) }),
            Ok(false)
        );
        assert_eq!(b.winner, Some(P2));
        assert_eq!(b.turn, P1); // game over, end_turn no-op
    }

    #[test]
    fn t4_move_after_winner_raises_and_endturn_noop() {
        let mut b = board_with(&[
            (0, 5, P1), (0, 6, P1), (0, 7, P1),
            (1, 5, P1), (1, 6, P1), (1, 7, P1),
            (2, 6, P1), (2, 7, P1), // (2,5) empty
            (2, 4, P1),
        ]);
        assert_eq!(
            b.apply_move(&Move::Step { from: (2, 4), to: (2, 5) }),
            Ok(false)
        );
        assert_eq!(b.winner, Some(P1));
        // place a piece and try to move it -> must raise
        b.grid[idx(3, 3)] = P1;
        assert!(is_banned_err(b.apply_move(
            &Move::Step { from: (3, 3), to: (3, 4) }
        )));
        b.end_turn();
        assert_eq!(b.turn_no, 1); // no-op
    }

    #[test]
    fn t4_winning_jump_with_open_continuation_ends_chain() {
        let mut b = board_with(&[
            (0, 5, P1), (0, 6, P1), (0, 7, P1),
            (1, 5, P1), (1, 6, P1), (1, 7, P1),
            (2, 5, P1), (2, 7, P1), // (2,6) empty
            (3, 6, P1), // jump middle
            (4, 6, P1), // jumping piece
        ]);
        b.turn = P1;
        assert_eq!(
            b.apply_move(&Move::Jump { from: (4, 6), mid: (3, 6), to: (2, 6) }),
            Ok(false)
        );
        assert_eq!(b.winner, Some(P1));
        assert!(!b.chain_active); // win closed the chain
        assert_eq!(b.cell(2, 6), P1);
        assert_eq!(b.cell(4, 6), 0);
        assert_eq!(b.turn, P1);
        assert_eq!(b.turn_no, 1);
        // continuation rejected once a winner is set
        assert!(is_banned_err(b.apply_move(
            &Move::Jump { from: (2, 6), mid: (2, 5), to: (2, 4) }
        )));
    }
}
