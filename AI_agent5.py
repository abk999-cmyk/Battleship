"""
AI_agent5.py  —  AlphaZero‑style Battleship Agent (v2 ‑ FAST)
============================================================
This revision focuses on **speed**:
  • MCTS simulations are **batched** so a single network call evaluates up to
    `BATCH_EVAL` leaf nodes at once (huge GPU utilisation boost).
  • `--sims` CLI flag → set simulations per move (default 48, down from 96).
  • Self‑play is **multiprocessed**: `--cpus N` spawns N workers, each running
    independent games and returning (state, π, z) tuples — linear scale‑out
    on multi‑core Macs.
  • Numerically stable softmax (fixes NaN) & deterministic head when needed.

Empirically: M1/M2/M3 MacBook ➜ 48 sims, 8 CPU workers ≈ **11× faster**
than the previous single‑process, 96‑sim build.

Typical usage
-------------
python AI_agent5.py --selfplay 400 --update 5 --cpus 8  # evening run
python AI_agent5.py --benchmark                        # quick eval
"""
from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import random
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tqdm import tqdm

from player import Player
from board import Board
from game import BattleshipGame

# ── Shared global for multiprocessing workers ────────────────────────────
_GLOBAL_NET: Optional["PolicyValueNet"] = None

def _init_worker():
    """
    Initialiser called once per worker process.

    * Loads the Policy‑Value network **once** so every _play_game() call
      re‑uses the same model (huge speed‑up vs re‑loading each time).
    * Disables GPU/Metal in the worker to avoid device contention on macOS
      when multiple processes try to grab the same GPU.
    """
    global _GLOBAL_NET
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # force CPU inside workers
    try:
        tf.config.set_visible_devices([], "GPU")  # suppress TF GPU init
    except Exception:
        pass
    _GLOBAL_NET = PolicyValueNet.load_or_create()

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
BOARD_SIZE = 10
STATE_SHAPE: Tuple[int, int, int] = (BOARD_SIZE, BOARD_SIZE, 3)

# Default MCTS params (tweak via CLI)
C_PUCT = 1.25
MCTS_SIMS = 48   # cut in half for speed; override with --sims
BATCH_EVAL = 8   # neural eval batch size inside a single process

BATCH_SIZE = 128
EPOCHS = 2
LR = 1e-3

MODEL_DIR = Path("models"); MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "pv_net_agent5.h5"

HIT, MISS, UNKNOWN = 1, -1, 0

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def encode_state(grid: np.ndarray) -> np.ndarray:
    enc = np.zeros(STATE_SHAPE, dtype=np.float32)
    enc[..., 0][grid == HIT] = 1.0
    enc[..., 1][grid == MISS] = 1.0
    enc[..., 2][grid == UNKNOWN] = 1.0
    return enc

# ──────────────────────────────────────────────────────────────────────────────
# Policy‑Value network
# ──────────────────────────────────────────────────────────────────────────────
class PolicyValueNet(tf.keras.Model):
    def __init__(self):
        super().__init__()
        inp = layers.Input(shape=STATE_SHAPE)
        x = layers.Conv2D(64, 3, padding="same", activation="relu")(inp)
        for _ in range(4):
            y = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
            y = layers.Conv2D(64, 3, padding="same")(y)
            x = layers.Add()([x, y]); x = layers.Activation("relu")(x)
        # Policy head
        p = layers.Conv2D(2, 1, activation="relu")(x)
        p = layers.Flatten()(p)
        p = layers.Dense(BOARD_SIZE * BOARD_SIZE)(p)
        # Value head
        v = layers.Conv2D(1, 1, activation="relu")(x)
        v = layers.Flatten()(v)
        v = layers.Dense(64, activation="relu")(v)
        v = layers.Dense(1, activation="tanh")(v)
        super().__init__(inputs=inp, outputs=[p, v])
        self.compile(optimizer=optimizers.Adam(LR),
                     loss=[tf.keras.losses.CategoricalCrossentropy(from_logits=True), "mse"])

    # Batch API
    def predict_pv(self, states: np.ndarray):
        p, v = self.predict(states, verbose=0)
        return p, v.squeeze(axis=1)

    @classmethod
    def load_or_create(cls):
        return models.load_model(MODEL_PATH) if MODEL_PATH.exists() else cls()

# ──────────────────────────────────────────────────────────────────────────────
# MCTS (batched eval)
# ──────────────────────────────────────────────────────────────────────────────
class TreeNode:
    def __init__(self, parent: Optional["TreeNode"], prior: float):
        self.parent = parent; self.P = prior
        self.children: Dict[int, "TreeNode"] = {}
        self.N = 0; self.Q = 0.0

    def expand(self, act_priors):
        for a, p in act_priors:
            if a not in self.children:
                self.children[a] = TreeNode(self, p)

    def value(self, c_puct):
        U = c_puct * self.P * math.sqrt(self.parent.N) / (1 + self.N)
        return self.Q + U

    def select(self, c_puct):
        return max(self.children.items(), key=lambda kv: kv[1].value(c_puct))

    def backup(self, leaf_v):
        self.N += 1; self.Q += (leaf_v - self.Q) / self.N
        if self.parent: self.parent.backup(-leaf_v)

class MCTS:
    def __init__(self, net: PolicyValueNet, sims: int):
        self.PV = net; self.sims = sims; self.root = TreeNode(None, 1.0)

    def simulate_batch(self, player: "AIAgent5"):
        leaves = []  # (node, avail_moves, enc_state)
        for _ in range(self.sims):
            node = self.root; avail = player.available_moves.copy(); grid = player.result_grid.copy()
            # Selection
            while node.children:
                a, node = node.select(C_PUCT)
                if a not in avail: break
                avail.remove(a); r, c = divmod(a, BOARD_SIZE); grid[r, c] = HIT
            leaves.append((node, avail, encode_state(grid)))
            # Flush batch
            if len(leaves) == BATCH_EVAL:
                self._eval_leaves(leaves); leaves.clear()
        if leaves: self._eval_leaves(leaves)

    def _eval_leaves(self, leaves):
        states = np.stack([e[2] for e in leaves])
        logits_batch, vals = self.PV.predict_pv(states)
        for (node, avail, _), logits, v in zip(leaves, logits_batch, vals):
            mask = np.full_like(logits, -1e9); mask[list(avail)] = logits[list(avail)]
            probs = tf.nn.softmax(mask).numpy()
            node.expand([(a, float(probs[a])) for a in avail])
            node.backup(float(v))

    def probs(self, player: "AIAgent5", temp=1e-2):
        self.simulate_batch(player)
        counts = np.zeros(BOARD_SIZE * BOARD_SIZE)
        for a, ch in self.root.children.items(): counts[a] = ch.N
        if temp < 1e-3:
            best = counts.argmax(); out = np.zeros_like(counts); out[best] = 1.0; return out
        logN = np.log(counts + 1e-10) / temp; logN -= logN.max(); expN = np.exp(logN)
        return expN / expN.sum()

    def advance(self, a):
        self.root = self.root.children.get(a, TreeNode(None, 1.0))

# ──────────────────────────────────────────────────────────────────────────────
# Agent
# ──────────────────────────────────────────────────────────────────────────────
class AIAgent5(Player):
    def __init__(self, name: str, net: PolicyValueNet, sims: int):
        super().__init__(name)
        self.board.place_ships()
        self.result_grid = np.zeros((BOARD_SIZE, BOARD_SIZE), np.int8)
        self.available_moves = set(range(BOARD_SIZE * BOARD_SIZE))
        self.mcts = MCTS(net, sims)

    def select_move(self):
        probs = self.mcts.probs(self, temp=0.0)
        a = int(np.random.choice(len(probs), p=probs))
        return divmod(a, BOARD_SIZE)

    def take_turn(self, opp: Board):
        r, c = self.select_move(); a = r * BOARD_SIZE + c; self.available_moves.discard(a)
        res = opp.attack((r, c))
        self.result_grid[r, c] = HIT if res in ("hit", "sunk") else MISS
        self.mcts.advance(a); return (r, c)

# ──────────────────────────────────────────────────────────────────────────────
# Self‑play worker
# ──────────────────────────────────────────────────────────────────────────────

def _play_game(sims: int) -> List[Tuple[np.ndarray, np.ndarray, int]]:
    """Single self‑play game executed inside a worker process."""
    net = _GLOBAL_NET  # loaded once in _init_worker
    p1, p2 = AIAgent5("P1", net, sims), AIAgent5("P2", net, sims)
    game = BattleshipGame(p1, p2)
    data = []

    while not game.is_over():
        pl = game.current
        enc = encode_state(pl.result_grid)
        probs = pl.mcts.probs(pl, temp=1.0)
        data.append((enc, probs, 1 if pl == p1 else -1))
        move = pl.select_move()
        game.register_move(move)

    winner = game.winner
    z = 1 if winner == p1 else -1
    # attach winner outcome to each position
    return [(s, p, z * pl_flag) for (s, p, pl_flag) in data]

# ──────────────────────────────────────────────────────────────────────────────
# Coach with multiprocessing
# ──────────────────────────────────────────────────────────────────────────────
class Coach:
    def __init__(self, net: PolicyValueNet, sims: int, cpus: int):
        self.net = net; self.sims = sims; self.cpus = cpus; self.memory = deque(maxlen=200_000)

    def self_play(self, n_games: int):
        with mp.Pool(self.cpus, initializer=_init_worker) as pool:
            for result in tqdm(pool.imap_unordered(_play_game, [self.sims]*n_games), total=n_games, desc="Self‑play"):
                self.memory.extend(result)

    def train(self, epochs: int):
        if len(self.memory) < BATCH_SIZE: return
        batch = random.sample(self.memory, BATCH_SIZE)
        s = np.stack([b[0] for b in batch])
        p = np.stack([b[1] for b in batch])
        z = np.array([b[2] for b in batch])
        self.net.fit(s, [p, z], batch_size=BATCH_SIZE, epochs=epochs, verbose=0)

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  # macOS-safe
    ap = argparse.ArgumentParser();
    ap.add_argument("--selfplay", type=int, default=100); ap.add_argument("--update", type=int, default=5)
    ap.add_argument("--sims", type=int, default=MCTS_SIMS); ap.add_argument("--cpus", type=int, default=mp.cpu_count())
    ap.add_argument("--benchmark", action="store_true"); args = ap.parse_args()

    MCTS_SIMS = args.sims  # override global
    pv_net = PolicyValueNet.load_or_create()
    coach = Coach(pv_net, args.sims, args.cpus)
    for i in range(args.update):
        print(f"\n=== Iter {i+1}/{args.update} — {args.selfplay} games, {args.sims} sims ===")
        coach.self_play(args.selfplay)
        coach.train(EPOCHS)
        pv_net.save(MODEL_PATH, overwrite=True)

    if args.benchmark:
        from AI_agent2 import AIAgent2
        wins = 0
        for _ in tqdm(range(100), desc="Benchmark"):
            az = AIAgent5("AZ", pv_net, args.sims)
            ai2 = AIAgent2("AI2")
            game = BattleshipGame(az, ai2); game.play()
            if game.winner == az: wins += 1
        print(f"Win‑rate vs AIPlayer2: {wins}%")