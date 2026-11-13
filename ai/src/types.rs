//! Constants, board coordinate helpers, and the base-cell sets.
//!
//! Pure-data part of the `halma8.py` port. Coordinates are 0-based `(r, c)`
//! with `r` = row (0 = top), `c` = col (0 = left), exactly as in Python.

pub const SIZE: u8 = 8;

/// P1 = red, starting in the bottom-left 3x3, moves first.
pub const P1: u8 = 1;
/// P2 = blue, starting in the top-right 3x3.
pub const P2: u8 = 2;

/// Orthogonal directions only (no diagonals), matching `halma8.py::DIRS`.
pub const DIRS: [(i8, i8); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];

/// Flat row-major index `i = r*8 + c`.
#[inline]
pub fn idx(r: u8, c: u8) -> usize {
    (r as usize) * 8 + (c as usize)
}

/// Inverse of `idx`.
#[inline]
pub fn row_col(i: usize) -> (u8, u8) {
    ((i / 8) as u8, (i % 8) as u8)
}

/// P1 base: rows 5-7, cols 0-2 (bottom-left 3x3). 9 cells, fixed order.
pub fn p1_base() -> [(u8, u8); 9] {
    let mut a = [(0u8, 0u8); 9];
    let mut k = 0;
    for r in 5..8 {
        for c in 0..3 {
            a[k] = (r, c);
            k += 1;
        }
    }
    a
}

/// P2 base: rows 0-2, cols 5-7 (top-right 3x3). 9 cells, fixed order.
pub fn p2_base() -> [(u8, u8); 9] {
    let mut a = [(0u8, 0u8); 9];
    let mut k = 0;
    for r in 0..3 {
        for c in 5..8 {
            a[k] = (r, c);
            k += 1;
        }
    }
    a
}

/// The 9 base cells for a player, or a debug error for an unknown player.
pub fn base_cells(player: u8) -> [(u8, u8); 9] {
    match player {
        P1 => p1_base(),
        P2 => p2_base(),
        _ => panic!("unknown player: {}", player),
    }
}
