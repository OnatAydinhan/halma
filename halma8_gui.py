"""check_halma GUI — Part 1, T6 (full playable loop).

Tkinter frontend over the halma8 engine: board rendering, click-to-move,
jump-chain UX (Done button / re-click to stop), win banner, New game, a
move log, and undo. This file only imports the engine; the engine never
imports tkinter.

Run with:  py halma8_gui.py
"""

import copy
import os
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter import scrolledtext

import halma8
import halma_ai
from halma8 import P1, P2, NAME

CELL = 64
SIZE = halma8.Halma8.SIZE
BOARD_PX = CELL * SIZE

LIGHT = "#f0d9b5"
DARK = "#b58863"
P1_COLOR = "#d9534f"  # red
P2_COLOR = "#4a90d9"  # blue
SELECTION_COLOR = "#ffd700"
TARGET_COLOR = "#2ecc71"
JUMP_MID_COLOR = "#9b59b6"
CLOCK_ACTIVE_BG = "#fff3cd"  # soft highlight for the clock that is running

DEFAULT_BASE_SEC = 600        # 10 minutes
DEFAULT_INCR_SEC = 0          # no increment

HERE = os.path.dirname(os.path.abspath(__file__))
KNOWN_MODELS = ["best_weights.json", "c1_incumbent.json"]  # Browse… covers the rest
AI_SEAT_DEFAULTS = {"kind": "human", "model": "", "depth": 4, "movetime_ms": 500}


class HalmaGUI:
    def __init__(self, board):
        self.board = board
        self.selected = None  # (r, c) or None
        self.undo_stack = []  # snapshot dicts, oldest first

        # clock settings (seconds) and running clocks
        self.settings = {P1: {"base": DEFAULT_BASE_SEC, "inc": DEFAULT_INCR_SEC},
                         P2: {"base": DEFAULT_BASE_SEC, "inc": DEFAULT_INCR_SEC}}
        self.remaining = {P1: float(DEFAULT_BASE_SEC), P2: float(DEFAULT_BASE_SEC)}
        self.running = None          # no clock runs until the first move is made
        self._last = time.monotonic()

        self.root = tk.Tk()
        self.root.title("check_halma — 8x8, no diagonals")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Players (T-G2): per-seat config; one shared playing child process.
        # seat_cfg[side] = {"kind": "human"|"ai", "model": path, "depth": int,
        #                   "movetime_ms": int}; AI seats play with their model.
        self.seat_cfg = {P1: dict(AI_SEAT_DEFAULTS), P2: dict(AI_SEAT_DEFAULTS)}
        self._ai_busy = False
        self._ai = halma_ai.HalmaAI(movetime_ms=AI_SEAT_DEFAULTS["movetime_ms"])

        # Kibitzer (T-G2 config block; the toggle/panel land in T-G4)
        # 10 s total cap per query: per-ply = 10000 // plies (8 → 1250 ms/ply)
        self.kib_cfg = {"on": False, "model": os.path.join(HERE, "best_weights.json"),
                        "depth": 6, "movetime_ms": 1250}

        # Kibitzer live state (T-G4): the analysis runs in its own ai.exe
        # child (self._kib) on a worker thread, so it never blocks input.
        self.kibitz_on = bool(self.kib_cfg["on"])
        self.kib_plies = 8          # line length cap (T-G4 risk b)
        self._kib = None            # lazy HalmaAI analysis child
        self._kib_model = None      # resolved model path applied to self._kib
        self._kib_thread = None
        self._kibitz_busy = False   # analysis worker running
        self._kibitz_pending = False  # triggered while busy -> rerun on finish
        self._kibitz_epoch = 0      # bumped on New game / off; drops stale results
        self._kib_result = None     # ("ok"|"err", epoch, ...) set by the worker

        # status line (win banner lives here too) — always visible
        self.status = tk.Label(self.root, text="", anchor="w", padx=8, pady=4)
        self.status.pack(side=tk.TOP, fill=tk.X)

        # clock row: P1 on the left, P2 on the right; the running clock is
        # highlighted. Always visible so clocks show on either tab.
        clockf = tk.Frame(self.root)
        clockf.pack(side=tk.TOP, fill=tk.X, padx=4, pady=(0, 2))
        self.p1_clock = tk.Label(clockf, text="", bg=CLOCK_ACTIVE_BG,
                                 padx=10, pady=2, anchor="w",
                                 font=("Consolas", 12, "bold"))
        self.p1_clock.pack(side=tk.LEFT)
        self.p2_clock = tk.Label(clockf, text="", bg="white",
                                 padx=10, pady=2, anchor="e",
                                 font=("Consolas", 12, "bold"))
        self.p2_clock.pack(side=tk.RIGHT)

        # tabs: Game (board + log + controls) | Players (seats + Kibitzer config)
        nb = ttk.Notebook(self.root)
        nb.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=2, pady=2)
        game_tab = tk.Frame(nb)
        nb.add(game_tab, text="Game")
        players_tab = tk.Frame(nb)
        nb.add(players_tab, text="Players")

        # -- Game tab ------------------------------------------------------
        ctrl = tk.Frame(game_tab)
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=4, pady=2)
        self.new_btn = tk.Button(ctrl, text="New game", command=self._apply_players_config)
        self.new_btn.pack(side=tk.LEFT, padx=2)
        self.undo_btn = tk.Button(ctrl, text="Undo", command=self.undo)
        self.undo_btn.pack(side=tk.LEFT, padx=2)
        self.done_btn = tk.Button(ctrl, text="Done", command=self._stop_chain,
                                  state=tk.DISABLED)
        self.done_btn.pack(side=tk.LEFT, padx=2)
        self.settings_btn = tk.Button(ctrl, text="Settings", command=self._open_settings)
        self.settings_btn.pack(side=tk.LEFT, padx=2)
        self.kib_btn = tk.Button(ctrl, text="Kibitzer: off", command=self._toggle_kibitz)
        self.kib_btn.pack(side=tk.LEFT, padx=2)

        # canvas on the left, move log on the right
        main = tk.Frame(game_tab)
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(main, width=BOARD_PX, height=BOARD_PX,
                                highlightthickness=0, bg="white")
        self.canvas.pack(side=tk.LEFT, padx=2, pady=2)
        self.canvas.bind("<Button-1>", self.on_click)

        logf = tk.Frame(main)
        logf.pack(side=tk.LEFT, padx=(6, 4), pady=4, fill=tk.BOTH, expand=True)
        tk.Label(logf, text="Move log", anchor="w").pack(fill=tk.X)
        self.log = scrolledtext.ScrolledText(logf, width=40, height=16,
                                             wrap=tk.NONE, state=tk.DISABLED)
        self.log.pack(fill=tk.BOTH, expand=True)

        # Kibitzer panel (T-G4): eval + best continuation line for the
        # current position; refreshed while the Kibitzer toggle is on.
        kibf = tk.LabelFrame(logf, text="Kibitzer")
        kibf.pack(fill=tk.X, pady=(6, 0))
        self.kib_eval = tk.Label(kibf, text="Kibitzer off", anchor="w",
                                 font=("Consolas", 11, "bold"))
        self.kib_eval.pack(fill=tk.X, padx=6, pady=2)
        self.kib_line = tk.Label(kibf, text="", anchor="w", justify=tk.LEFT,
                                 wraplength=250)
        self.kib_line.pack(fill=tk.X, padx=6, pady=(0, 4))

        # -- Players tab ---------------------------------------------------
        self._build_players_tab(players_tab)
        self._kib_refresh_button()

        self._reset_clocks()
        self.redraw()
        self._update_timers()
        self.root.after(100, self._tick)

    def run(self):
        self.root.mainloop()

    # -- geometry helpers ------------------------------------------------
    def _cell_box(self, r, c):
        return c * CELL, r * CELL, (c + 1) * CELL, (r + 1) * CELL

    def _center(self, r, c):
        return c * CELL + CELL // 2, r * CELL + CELL // 2

    def _selected_targets(self):
        """Return (steps, jumps) offered for the selected piece right now.

        During a jump chain only non-banned jump destinations are offered and
        steps are hidden (a step cannot follow a jump in the same turn)."""
        b = self.board
        if self.selected is None:
            return [], []
        sr, sc = self.selected
        if b.chain_active:
            jumps = [(mid, dst) for (mid, dst) in b.legal_jumps(sr, sc)
                     if dst not in b.chain_banned]
            return [], jumps
        return b.legal_steps(sr, sc), b.legal_jumps(sr, sc)

    # -- rendering -------------------------------------------------------
    def redraw(self):
        cv = self.canvas
        cv.delete("all")

        for r in range(SIZE):
            for c in range(SIZE):
                x0, y0, x1, y1 = self._cell_box(r, c)
                color = LIGHT if (r + c) % 2 == 0 else DARK
                cv.create_rectangle(x0, y0, x1, y1, fill=color, outline="")

        for r in range(SIZE):
            for c in range(SIZE):
                p = self.board.grid[r][c]
                if p == 0:
                    continue
                cx, cy = self._center(r, c)
                rad = int(CELL * 0.35)
                cv.create_oval(cx - rad, cy - rad, cx + rad, cy + rad,
                               fill=P1_COLOR if p == P1 else P2_COLOR,
                               outline="white", width=2)

        # game over: freeze the board — no selection box or target dots
        if self.selected is not None and self.board.winner is None:
            self._draw_selection_and_targets()

        self._update_status()
        self._refresh_controls()

    def _draw_selection_and_targets(self):
        cv = self.canvas
        sr, sc = self.selected

        x0, y0, x1, y1 = self._cell_box(sr, sc)
        cv.create_rectangle(x0 + 2, y0 + 2, x1 - 2, y1 - 2,
                            outline=SELECTION_COLOR, width=4, fill="")

        steps, jumps = self._selected_targets()
        for (nr, nc) in steps:
            self._draw_target_dot(nr, nc)
        for mid, dst in jumps:
            self._draw_target_dot(dst[0], dst[1])
            cx, cy = self._center(*mid)
            rad = int(CELL * 0.12)
            cv.create_oval(cx - rad, cy - rad, cx + rad, cy + rad,
                           fill=JUMP_MID_COLOR, outline="white", width=1)

    def _draw_target_dot(self, r, c):
        cx, cy = self._center(r, c)
        rad = int(CELL * 0.15)
        self.canvas.create_oval(cx - rad, cy - rad, cx + rad, cy + rad,
                                fill=TARGET_COLOR, outline="white", width=2)

    def _update_status(self):
        b = self.board
        if b.winner is not None:
            self.status.config(
                text=f"*** {NAME[b.winner]} wins! ***   (New game to replay)")
            return
        if b.chain_active:
            self.status.config(
                text=f"{NAME[b.turn]} to move — jump available: continue jumping, "
                     f"or press Done to end this piece's jumps")
            return
        if self.selected is not None:
            sr, sc = self.selected
            steps, jumps = self._selected_targets()
            n = len(steps) + len(jumps)
            self.status.config(
                text=f"{NAME[b.turn]} to move — selected ({sr},{sc}); {n} legal move(s)")
            return
        self.status.config(text=f"{NAME[b.turn]} to move — click your piece to select it")

    def _refresh_controls(self):
        self.done_btn.config(state=tk.NORMAL if self.board.chain_active else tk.DISABLED)
        self.undo_btn.config(state=tk.NORMAL if self.undo_stack else tk.DISABLED)

    # -- clocks ------------------------------------------------------------
    def _reset_clocks(self):
        """Reset both clocks to base time. No clock runs until the first move is
        made: the first player's opening move is free, and only then does the
        opponent's clock start (handled when that first move ends the turn)."""
        self.remaining = {P1: float(self.settings[P1]["base"]),
                          P2: float(self.settings[P2]["base"])}
        self.running = None
        self._last = time.monotonic()

    @staticmethod
    def _fmt_time(secs):
        secs = max(0, int(round(secs)))
        m, s = divmod(secs, 60)
        return f"{m:02d}:{s:02d}"

    def _update_timers(self):
        active = self.running
        self.p1_clock.config(
            text=f"P1 (Red)   {self._fmt_time(self.remaining[P1])}",
            bg=CLOCK_ACTIVE_BG if active == P1 else "white")
        self.p2_clock.config(
            text=f"{self._fmt_time(self.remaining[P2])}   P2 (Blue)",
            bg=CLOCK_ACTIVE_BG if active == P2 else "white")

    def _tick(self):
        b = self.board
        if self.running is not None and b.winner is None:
            now = time.monotonic()
            dt = now - self._last
            self._last = now
            self.remaining[self.running] -= dt
            if self.remaining[self.running] <= 0:
                self.remaining[self.running] = 0.0
                self._on_timeout(self.running)
        self._update_timers()
        self.root.after(100, self._tick)

    def _on_timeout(self, player):
        """A player's clock hit zero: that player loses on time. Reuses the
        engine's winner flag so the existing freeze/banner logic all applies."""
        other = P2 if player == P1 else P1
        self.board.winner = other
        self._append_log(f"!! {NAME[player]} ran out of time — {NAME[other]} wins on time")
        self.running = None
        self._last = time.monotonic()
        self.redraw()

    # -- settings ----------------------------------------------------------
    def _open_settings(self):
        top = tk.Toplevel(self.root)
        top.title("Clock Settings")
        top.transient(self.root)
        top.grab_set()
        top.resizable(False, False)

        b1 = tk.StringVar(value=self._fmt_base(self.settings[P1]["base"]))
        i1 = tk.StringVar(value=self.settings[P1]["inc"])
        b2 = tk.StringVar(value=self._fmt_base(self.settings[P2]["base"]))
        i2 = tk.StringVar(value=self.settings[P2]["inc"])

        def entry(parent, r, c, var, width):
            e = tk.Entry(parent, textvariable=var, width=width)
            e.grid(row=r, column=c, padx=4, pady=2, ipady=1)
            return e

        tk.Label(top, text="").grid(row=0, column=0)
        tk.Label(top, text="Base (min)", width=11, anchor="center").grid(row=0, column=1)
        tk.Label(top, text="Increment (sec)", width=17, anchor="center").grid(row=0, column=2)

        tk.Label(top, text=NAME[P1], width=9, anchor="w").grid(row=1, column=0, padx=(8, 2))
        tk.Label(top, text=NAME[P2], width=9, anchor="w").grid(row=2, column=0, padx=(8, 2))
        entry(top, 1, 1, b1, 6)
        entry(top, 1, 2, i1, 6)
        entry(top, 2, 1, b2, 6)
        entry(top, 2, 2, i2, 6)

        def apply():
            try:
                nb1, ni1 = float(b1.get()), float(i1.get())
                nb2, ni2 = float(b2.get()), float(i2.get())
                if min(nb1, nb2) < 0 or ni1 < 0 or ni2 < 0:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Clock Settings",
                                     "Base (min) and Increment (sec) must be non-negative numbers.",
                                     parent=top)
                return
            self.settings[P1] = {"base": int(round(nb1 * 60)), "inc": int(round(ni1))}
            self.settings[P2] = {"base": int(round(nb2 * 60)), "inc": int(round(ni2))}
            self._apply_settings_now()
            top.destroy()

        btns = tk.Frame(top)
        btns.grid(row=3, column=0, columnspan=3, pady=10)
        tk.Button(btns, text="Apply", width=9, command=apply).pack(side=tk.LEFT, padx=6)
        tk.Button(btns, text="Cancel", width=9, command=top.destroy).pack(side=tk.LEFT, padx=6)

    @staticmethod
    def _fmt_base(base_sec):
        """Render a whole-minute base for the settings entry; fall back to raw seconds."""
        m = base_sec / 60
        return f"{m:.0f}" if m >= 1 and m == int(m) else f"{base_sec}s"

    def _apply_settings_now(self):
        """Apply new settings to a fresh, untouched board immediately; otherwise they
        take effect on the next New game (increment still applies to every move)."""
        if self.undo_stack == [] and self.board.winner is None:
            self._reset_clocks()
        self._update_timers()

    # -- input -----------------------------------------------------------
    def on_click(self, event):
        b = self.board

        # 1) game over: ignore board clicks (New game / Undo still work)
        if b.winner is not None:
            return

        c = event.x // CELL
        r = event.y // CELL
        if not (0 <= r < SIZE and 0 <= c < SIZE):
            return
        pos = (r, c)

        # 2) chain in progress: chained piece must either continue or stop
        if b.chain_active:
            if self.selected is not None:
                sr, sc = self.selected
                for mid, dst in b.legal_jumps(sr, sc):
                    if dst == pos and dst not in b.chain_banned:
                        self._do_move(("jump", (sr, sc), mid, dst))
                        return
            # re-clicked the chained piece, or clicked elsewhere -> stop the chain
            self._stop_chain()
            return

        # 3) already have a selection: try to move it
        if self.selected is not None:
            sr, sc = self.selected
            if pos != (sr, sc):
                if pos in b.legal_steps(sr, sc):
                    self._do_move(("step", (sr, sc), pos))
                    return
                for mid, dst in b.legal_jumps(sr, sc):
                    if dst == pos:
                        self._do_move(("jump", (sr, sc), mid, dst))
                        return
            else:
                # re-clicking the selected piece deselects it
                self._deselect()
                return

        # 4) select a piece that belongs to the current player
        if b.grid[r][c] == b.turn:
            self.selected = (r, c)
            self.redraw()
            return

        # 5) clicked an empty or opponent square: clear the selection
        self._deselect()

    # -- actions ---------------------------------------------------------
    def _deselect(self):
        if self.selected is not None:
            self.selected = None
            self.redraw()

    def _do_move(self, move):
        b = self.board
        player = b.turn
        tno = b.turn_no

        # snapshot current state for Undo before mutating anything
        self.undo_stack.append(self._snapshot())
        try:
            moved = b.apply_move(move)
        except halma8.IllegalMove as e:
            self.undo_stack.pop()  # invalid moves are not undone
            print(f"IllegalMove: {e}")
            return

        self._append_log(self._describe(player, tno, move))

        if moved:
            # a jump that leaves the chain open: keep the chained piece selected,
            # the mover still has the move, and their clock keeps running.
            self.selected = move[-1]
            # running is already == player; just keep the dt baseline fresh
            self._last = time.monotonic()
        else:
            # turn ended: the mover banked their increment; the clock hands to the
            # player who now has the move (or the game is over).
            self.remaining[player] += self.settings[player]["inc"]
            self.selected = None
            if b.winner is not None:
                self.running = None
            else:
                self.running = b.turn
            self._last = time.monotonic()
        self.redraw()
        self._maybe_ai()
        self._maybe_kibitz()

    def _stop_chain(self):
        b = self.board
        if b.chain_active:
            mover = b.turn
            b.end_turn()
            self.remaining[mover] += self.settings[mover]["inc"]
            if b.winner is not None:
                self.running = None
            else:
                self.running = b.turn
            self._last = time.monotonic()
        self.selected = None
        self.redraw()
        self._maybe_ai()
        self._maybe_kibitz()

    def _on_close(self):
        # let a running Kibitzer analysis finish (bounded by the driver's
        # timeouts) so its worker never touches a half-closed child
        if self._kib_thread is not None and self._kib_thread.is_alive():
            self._kib_thread.join(timeout=15)
        if self._kib is not None:
            self._kib.close()
        self._ai.close()
        self.root.destroy()

    # -- Players tab (T-G2) -------------------------------------------------
    def _build_players_tab(self, tab):
        head = tk.Frame(tab)
        head.pack(fill=tk.X, padx=8, pady=(8, 2))
        tk.Label(head, anchor="w",
                 text="Pick who sits at each seat. AI seats load their model on "
                      "New game. Start applies the setup and begins a fresh game.").pack()

        self.seat_widgets = {}
        self.seat_kind_var, self.seat_model_var = {}, {}
        self.seat_depth_var, self.seat_mt_var = {}, {}
        self.seat_widgets[P1] = self._build_seat_panel(tab, NAME[P1], P1)
        self.seat_widgets[P2] = self._build_seat_panel(tab, NAME[P2], P2)

        # Kibitzer config (toggle + analysis panel land in T-G4)
        kif = tk.LabelFrame(tab, text="Kibitzer (analysis)")
        kif.pack(fill=tk.X, padx=8, pady=6)
        self.kib_on_var = tk.BooleanVar(value=False)
        tk.Checkbutton(kif, text="Enable Kibitzer (best line + eval after every move)",
                       variable=self.kib_on_var).grid(row=0, column=0, columnspan=5,
                                                      sticky="w", padx=6, pady=2)
        self.kib_model_var = tk.StringVar(value="best_weights.json")
        tk.Label(kif, text="Model:").grid(row=1, column=0, sticky="w", padx=6, pady=2)
        self.kib_model_cb = ttk.Combobox(kif, width=34, values=KNOWN_MODELS,
                                         state="readonly",
                                         textvariable=self.kib_model_var)
        self.kib_model_cb.grid(row=1, column=1, sticky="w")
        tk.Button(kif, text="Browse…", width=9,
                  command=self._browse_kib_model).grid(row=1, column=2, padx=4)
        tk.Label(kif, text="Depth:").grid(row=1, column=3, padx=(12, 2))
        self.kib_depth_var = tk.IntVar(value=self.kib_cfg["depth"])
        tk.Spinbox(kif, from_=2, to=12, width=3,
                   textvariable=self.kib_depth_var).grid(row=1, column=4)
        self._refresh_kib_widgets()

        tk.Button(tab, text="Start game with this setup", width=28,
                  command=self._apply_players_config).pack(pady=10)
        self._refresh_seat_widgets(P1)
        self._refresh_seat_widgets(P2)

    def _build_seat_panel(self, tab, title, side):
        panel = tk.LabelFrame(tab, text=f"Seat — {title}")
        panel.pack(fill=tk.X, padx=8, pady=6)

        self.seat_kind_var[side] = tk.StringVar(value=self.seat_cfg[side]["kind"])
        self.seat_model_var[side] = tk.StringVar(
            value=os.path.basename(self.seat_cfg[side]["model"])
            if self.seat_cfg[side]["model"] else "")
        self.seat_depth_var[side] = tk.IntVar(value=self.seat_cfg[side]["depth"])
        self.seat_mt_var[side] = tk.IntVar(value=self.seat_cfg[side]["movetime_ms"])

        tk.Radiobutton(panel, text="Human", value="human",
                       variable=self.seat_kind_var[side],
                       command=lambda: self._on_seat_kind(side)).grid(
            row=0, column=0, sticky="w", padx=6, pady=2)
        tk.Radiobutton(panel, text="AI", value="ai",
                       variable=self.seat_kind_var[side],
                       command=lambda: self._on_seat_kind(side)).grid(
            row=0, column=1, sticky="w", padx=6)

        tk.Label(panel, text="Model:").grid(row=1, column=0, sticky="w", padx=6, pady=2)
        cb = ttk.Combobox(panel, width=30, values=KNOWN_MODELS,
                          state="readonly",
                          textvariable=self.seat_model_var[side])
        cb.grid(row=1, column=1, columnspan=2, sticky="w")
        browse_btn = tk.Button(panel, text="Browse…", width=9,
                               command=lambda: self._browse_model(side))
        browse_btn.grid(row=1, column=3, padx=4)
        tk.Label(panel, text="Depth:").grid(row=1, column=4, padx=(12, 2))
        depth_spin = tk.Spinbox(panel, from_=2, to=12, width=3,
                                textvariable=self.seat_depth_var[side])
        depth_spin.grid(row=1, column=5)
        tk.Label(panel, text="Movetime (ms):").grid(row=1, column=6, padx=(12, 2))
        mt_spin = tk.Spinbox(panel, from_=50, to=60000, increment=50, width=6,
                             textvariable=self.seat_mt_var[side])
        mt_spin.grid(row=1, column=7)
        return {"cb": cb, "browse": browse_btn, "depth_spin": depth_spin,
                "mt_spin": mt_spin}

    def _refresh_seat_widgets(self, side):
        """Enable a seat's model/depth/movetime row only when that seat is AI."""
        is_ai = self.seat_kind_var[side].get() == "ai"
        state = tk.NORMAL if is_ai else tk.DISABLED
        w = self.seat_widgets[side]
        for widget in (w["cb"], w["browse"], w["depth_spin"], w["mt_spin"]):
            widget.config(state=state)
        w["cb"].config(state=("disabled" if not is_ai else "readonly"))

    def _browse_model(self, side):
        path = filedialog.askopenfilename(
            parent=self.root, initialdir=HERE, title="Pick an AI model file",
            filetypes=[("Weight JSON", "*.json"), ("All files", "*.*")])
        if path:
            self.seat_model_var[side].set(path)

    def _browse_kib_model(self):
        path = filedialog.askopenfilename(
            parent=self.root, initialdir=HERE, title="Pick the Kibitzer model file",
            filetypes=[("Weight JSON", "*.json"), ("All files", "*.*")])
        if path:
            self.kib_model_var.set(path)

    def _refresh_kib_widgets(self):
        # T-G4: the Kibitzer widgets stay enabled; live on/off is the Game-tab
        # toggle button (self.kib_btn), so there is nothing to gate here.
        pass

    def _seat_model_path(self, side):
        """Resolve a seat's model field to an existing file path, or None."""
        text = self.seat_model_var[side].get().strip()
        if not text:
            return None
        if os.path.isabs(text):
            return text if os.path.exists(text) else None
        return os.path.join(HERE, text) if os.path.exists(os.path.join(HERE, text)) \
            else None

    def _on_seat_kind(self, side):
        self.seat_cfg[side]["kind"] = self.seat_kind_var[side].get()
        self._refresh_seat_widgets(side)
        b = self.board
        # flipped to AI while it is that seat's turn: start thinking now
        if (self.seat_cfg[side]["kind"] == "ai" and b.turn == side
                and b.winner is None and not self._ai_busy):
            self._maybe_ai()

    def _read_players(self):
        """Read the Players tab widgets into seat_cfg / kib_cfg; True if valid."""
        for side in (P1, P2):
            self.seat_cfg[side]["kind"] = self.seat_kind_var[side].get()
            self.seat_cfg[side]["depth"] = max(2, int(self.seat_depth_var[side].get()))
            self.seat_cfg[side]["movetime_ms"] = max(
                50, int(self.seat_mt_var[side].get()))
            self.seat_cfg[side]["model"] = self._seat_model_path(side) or ""
        self.kib_cfg["on"] = bool(self.kib_on_var.get())
        self.kib_cfg["depth"] = max(2, int(self.kib_depth_var.get()))
        kib_model = self.kib_model_var.get().strip()
        self.kib_cfg["model"] = (kib_model if os.path.isabs(kib_model)
                                 else os.path.join(HERE, kib_model))

        ok = True
        for side in (P1, P2):
            cfg = self.seat_cfg[side]
            if cfg["kind"] != "ai":
                continue
            if not cfg["model"]:
                self._players_error(f"{NAME[side]} seat is AI but has no model selected.")
                ok = False
            elif not os.path.exists(cfg["model"]):
                self._players_error(f"{NAME[side]} model file not found:\n{cfg['model']}")
                ok = False
            elif " " in cfg["model"]:
                self._players_error(
                    f"{NAME[side]} model path contains a space (the AI protocol "
                    "can't send it). Move the file to a space-free folder.")
                ok = False
        if not ok:
            return False
        if self.kib_cfg["on"]:
            if not os.path.exists(self.kib_cfg["model"]):
                self._players_error(
                    f"Kibitzer model file not found (leaving Kibitzer off):\n"
                    f"{self.kib_cfg['model']}")
                self.kib_cfg["on"] = False
            elif " " in self.kib_cfg["model"]:
                self._players_error(
                    "Kibitzer model path contains a space (the AI protocol "
                    "can't send it). Move the file to a space-free folder.")
                self.kib_cfg["on"] = False
        return True

    def _players_error(self, msg):
        messagebox.showerror("Players", msg, parent=self.root)

    def _apply_players_config(self):
        """Validate the Players tab, load the AI seats' models into the shared
        playing engine, then start a fresh game (Red/P1 moves first)."""
        if not self._read_players():
            return
        # the Players-tab checkbox is the setup-time Kibitzer toggle
        self.kibitz_on = bool(self.kib_cfg["on"])
        self._kib_refresh_button()
        # seat ↔ side is constant: Red → WeightsP1, Blue → WeightsP2. Sent on
        # every New game (the child may have been respawned with default weights).
        # A human seat's side is irrelevant — send no weights for it.
        m1 = self.seat_cfg[P1]["model"] if self.seat_cfg[P1]["kind"] == "ai" else None
        m2 = self.seat_cfg[P2]["model"] if self.seat_cfg[P2]["kind"] == "ai" else None
        self._ai.set_weights(m1, m2)
        self.new_game()

    # -- AI turn (Part 2, T-8) ---------------------------------------------
    def _clock_movetime_ms(self, player):
        """Clock-aware movetime: 10% of the mover's remaining clock, capped
        at 5 s. Falls back to the configured default when no clock is running
        yet (late start: nothing ticks before the first move is made)."""
        if self.running is None:
            # No clock active — use the seat's fixed default (500 ms).
            return self.seat_cfg[player]["movetime_ms"]
        remaining_ms = int(self.remaining[player] * 1000)
        return min(max(remaining_ms // 10, 50), 5000)  # 10%, min 50 ms, cap 5 s

    def _ask_ai(self, cfg, pos, player=None):
        """One engine request under a seat's own search settings.

        Clock-aware: when a clock is running, movetime is 10% of the mover's
        remaining time (capped 5 s); otherwise the fixed configured default."""
        if player is not None:
            mt = self._clock_movetime_ms(player)
            # Bump the hard timeout so the engine has room (search + IPC overhead)
            self._ai.timeout = max(self._ai.timeout, mt / 1000.0 + 2.0)
            return self._ai.move(pos, movetime_ms=mt)
        if cfg["movetime_ms"] > 0:
            return self._ai.move(pos, movetime_ms=cfg["movetime_ms"])
        return self._ai.move_depth(pos, depth=cfg["depth"])

    def _maybe_ai(self, delay_ms=200):
        """If the current mover's seat is an AI, schedule the AI to act (after a
        short delay so the human's own move paints first)."""
        b = self.board
        if b.winner is not None or self._ai_busy:
            return
        if self.seat_cfg[b.turn]["kind"] != "ai":
            return
        self.status.config(text=f"{NAME[b.turn]}: AI is thinking…")
        self.root.after(delay_ms, self._ai_take_turn)

    def _ai_take_turn(self):
        """Play one full AI turn (possibly a multi-jump chain) by feeding each
        ``bestmove`` through the existing move path, so logs/undo/clocks/winner
        all behave exactly as if a human had clicked. Python stays the sole
        authority on legality."""
        b = self.board
        if self._ai_busy or b.winner is not None:
            return
        player = b.turn
        if self.seat_cfg[player]["kind"] != "ai":
            return
        cfg = self.seat_cfg[player]
        self._ai_busy = True
        self.selected = None
        failed = False
        try:
            while (b.turn == player and b.winner is None
                   and self.seat_cfg[player]["kind"] == "ai"):
                grid_before = [row[:] for row in b.grid]
                pos = halma_ai.board_to_position(b)
                action = self._ask_ai(cfg, pos, player=player)
                if action == "stop":
                    if b.chain_active:
                        self._stop_chain()
                    break
                self._do_move(action)
                if b.grid == grid_before:
                    # Python rejected the engine's move; stop the turn instead of
                    # re-asking forever (a no-op move changes nothing).
                    failed = True
                    self.status.config(
                        text="AI returned a move Python rejected - turn stopped. "
                             "Undo or New game to continue.")
                    break
        except halma_ai.HalmaAIError as e:
            failed = True
            self.status.config(
                text=f"⚠ AI: {e}  — turn not played. Toggle AI off + re-click, or New game.")
        finally:
            self._ai_busy = False
            self.redraw()
            if not failed:
                self._maybe_ai()

    # -- Kibitzer (T-G4) -----------------------------------------------------
    def _kib_refresh_button(self):
        self.kib_btn.config(text=f"Kibitzer: {'on' if self.kibitz_on else 'off'}")

    def _toggle_kibitz(self):
        self.kibitz_on = not self.kibitz_on
        self.kib_on_var.set(self.kibitz_on)
        self._kib_refresh_button()
        if not self.kibitz_on:
            # drop anything in flight or pending (the epoch bump makes a
            # running worker's result land as stale and get discarded)
            self._kibitz_pending = False
            self._kib_result = None
            self._kibitz_epoch += 1
            self.kib_eval.config(text="Kibitzer off")
            self.kib_line.config(text="")
            return
        if self.board.winner is None:
            self.kib_eval.config(text="analysing…")
        self._maybe_kibitz()

    def _maybe_kibitz(self):
        """If the Kibitzer is on and the game is live, analyse the current
        position on a worker thread (its own ai.exe child, off the main loop).
        Coalesced: at most one analysis in flight; a trigger while busy just
        queues one follow-up, so a burst of moves yields the latest position."""
        if not self.kibitz_on or self.board.winner is not None:
            return
        if self._kibitz_busy:
            self._kibitz_pending = True
            return
        self._kibitz_pending = False
        self._kib_result = None
        snapshot = copy.deepcopy(self.board)
        args = (snapshot, self._kibitz_epoch, self.kib_cfg["model"],
                self.kib_cfg["movetime_ms"])
        self._kibitz_busy = True
        self._kib_thread = threading.Thread(target=self._kib_worker, args=args,
                                            daemon=True)
        self._kib_thread.start()
        self.root.after(150, self._kib_poll)

    def _kib_worker(self, snapshot, epoch, model, movetime_ms):
        """Analysis worker (dedicated thread; self._kib is used only from
        here). Never mutates the live board — `snapshot` is a deepcopy.
        The result is picked up by the main loop via `_kib_poll`."""
        try:
            # Enforce the 10 s total cap: per-ply movetime ≤ 10000 // plies
            plies = max(1, self.kib_plies)
            cap_per_ply = 10000 // plies
            actual_mt = min(movetime_ms, cap_per_ply)
            if self._kib is None:
                self._kib = halma_ai.HalmaAI(movetime_ms=actual_mt, timeout=12.0)
            else:
                self._kib.movetime_ms = actual_mt
                self._kib.timeout = max(self._kib.timeout, actual_mt / 1000.0 + 2.0)
            if model and os.path.exists(model):
                if self._kib_model != model:
                    # both sides, so the analysis is side-independent
                    self._kib.set_weights(model, model)
                    self._kib_model = model
            actions, root = self._kib.line(snapshot, plies=plies)
            self._kib_result = ("ok", epoch, actions, root)
        except Exception as e:   # HalmaAIError or anything else: warn, don't crash
            self._kib_result = ("err", epoch, str(e))
        finally:
            self._kibitz_busy = False

    def _kib_poll(self):
        """Main-loop poller: collect the worker's result (Tk must not be
        touched from the worker thread) and keep polling until it lands."""
        res = self._kib_result
        if res is not None:
            self._kib_result = None
            self._kib_finish(res)
            return
        if self._kibitz_busy:
            self.root.after(150, self._kib_poll)
            return
        # worker finished and its result was handled or dropped: if something
        # was triggered while it ran, analyse the freshest position now
        if self._kibitz_pending and self.kibitz_on and self.board.winner is None:
            self._kibitz_pending = False
            self._maybe_kibitz()

    def _kib_finish(self, res):
        kind, epoch = res[0], res[1]
        if epoch != self._kibitz_epoch:
            return  # stale (New game / toggled off since) — drop silently
        if kind == "ok":
            self.kib_eval.config(text=self._fmt_eval(res[3]))
            self.kib_line.config(text=self._fmt_line(res[2]))
        else:
            self.kib_eval.config(text="⚠ " + res[2])
            self.kib_line.config(text="")

    @staticmethod
    def _fmt_eval(score):
        if score is None:
            return "eval: — (engine sent no score)"
        # MATE - ply convention (ai/src/search.rs: MATE = 30000)
        if abs(score) >= 29488:
            n = max(1, 30000 - abs(score))
            side = "Red" if score > 0 else "Blue"
            return f"#{n}  ({side} wins in {n})"
        if score == 0:
            return "0  (balanced)"
        side = "Red" if score > 0 else "Blue"
        return f"{score:+d}  ({side} favours)"

    @staticmethod
    def _fmt_line(actions):
        if not actions:
            return "(no line — position is terminal)"
        parts = []
        for a in actions:
            if a == "stop":
                parts.append("·stop·")
            elif a[0] == "step":
                parts.append(f"{a[1]}->{a[2]}")
            else:
                parts.append(f"jump {a[1]}->{a[3]}")
        text = "  |  ".join(parts)
        if len(text) > 220:
            text = text[:217] + "…"
        return text

    def _describe(self, player, tno, move):
        if move[0] == "step":
            _, src, dst = move
            return f"{tno} {NAME[player]}: {src} -> {dst} step"
        _, src, mid, dst = move
        return f"{tno} {NAME[player]}: {src} jumps over {mid} -> {dst} jump"

    # -- new game / undo -------------------------------------------------
    def new_game(self):
        self.board = halma8.Halma8()
        self.selected = None
        self.undo_stack = []
        self._set_log("")
        self._reset_clocks()
        self.redraw()
        self._update_timers()
        # Kibitzer: drop any in-flight analysis of the previous position and
        # (if on) start a fresh one from the clean board
        self._kibitz_pending = False
        self._kib_result = None
        self._kibitz_epoch += 1
        self.kib_eval.config(text="analysing…" if self.kibitz_on else "Kibitzer off")
        self.kib_line.config(text="")
        self._maybe_ai()
        self._maybe_kibitz()

    def undo(self):
        if not self.undo_stack:
            return
        snap = self.undo_stack.pop()
        b = self.board
        b.grid = snap["grid"]
        b.turn = snap["turn"]
        b.winner = snap["winner"]
        b.turn_no = snap["turn_no"]
        b.chain_active = snap["chain_active"]
        b.chain_banned = set(snap["chain_banned"])
        b.chain_starter = snap["chain_starter"]
        b.chain_current = snap["chain_current"]
        self.selected = snap["selected"]
        self.remaining = dict(snap["remaining"])
        self.running = snap["running"]
        self._last = time.monotonic()
        self._set_log(snap["log"])
        self.redraw()
        # undo back to an AI seat's turn: let that seat play on
        self._maybe_ai()
        self._maybe_kibitz()

    def _snapshot(self):
        b = self.board
        return {
            "grid": copy.deepcopy(b.grid),
            "turn": b.turn,
            "winner": b.winner,
            "turn_no": b.turn_no,
            "chain_active": b.chain_active,
            "chain_banned": set(b.chain_banned),
            "chain_starter": b.chain_starter,
            "chain_current": b.chain_current,
            "selected": self.selected,
            "remaining": dict(self.remaining),
            "running": self.running,
            "log": self.log.get("1.0", "end-1c"),
        }

    # -- log helpers -----------------------------------------------------
    def _append_log(self, line):
        self.log.config(state=tk.NORMAL)
        self.log.insert(tk.END, line + "\n")
        self.log.see(tk.END)
        self.log.config(state=tk.DISABLED)

    def _set_log(self, text):
        self.log.config(state=tk.NORMAL)
        self.log.delete("1.0", "end")
        if text:
            self.log.insert("1.0", text)
        self.log.config(state=tk.DISABLED)


def main():
    board = halma8.Halma8()
    HalmaGUI(board).run()


if __name__ == "__main__":
    main()
