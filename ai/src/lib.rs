//! check_halma AI engine (Part 2).
//!
//! A rules port of the canonical Python engine (`halma8.py`) plus a negamax /
//! iterative-deepening search and a JSON-weighted positional eval. `halma8.py`
//! remains the authority on what the GUI accepts; this crate only *proposes*
//! moves.
//!
//! std-only by design — see `plan_ai.md`.

pub mod types;
pub mod engine;
pub mod evaluate;
pub mod search;
pub mod server;
