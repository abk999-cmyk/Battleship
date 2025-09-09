# generate_dataset.py
"""
Multi‑process dataset generator for Battleship.

Produces a pickled stream of (state_tensor, ship_grid) samples where:
  • state_tensor  : 10×10×3  float16   (miss / hit / unknown)
  • ship_grid     : 10×10     int8     (1 = ship present)

Usage:
    python generate_dataset.py --games 50000 --workers 8 --out data/battleship_supervised.pkl
"""

import argparse, os, pickle, multiprocessing as mp, tqdm, numpy as np
from ..core.game import BattleshipGame
from ..agents.AI_agent import AIPlayer


# ---------------- utility -------------------------------------------------
def encode_state(result_grid: np.ndarray) -> np.ndarray:
    """3‑plane encoding  (miss, hit, unknown)  returned as float16."""
    miss = (result_grid == -1)
    hit  = (result_grid ==  1)
    unk  = (result_grid ==  0)
    return np.stack([miss, hit, unk], axis=-1).astype(np.float16)   # 10×10×3


def encode_true(board) -> np.ndarray:
    """Binary mask of true ship squares (int8)."""
    g = np.zeros((10, 10), dtype=np.int8)
    for (r, c) in board.ship_lookup.keys():
        g[r, c] = 1
    return g


# ---------------- worker --------------------------------------------------
def one_game(_):
    """Play a self‑play game and return a list of samples."""
    g = BattleshipGame(AIPlayer("A"), AIPlayer("B"))
    samples = []
    while not g.is_over() and g.turns < 100:          # cap 100 moves
        p = g.current
        s = encode_state(p.turn_board_state())
        y = encode_true(p.board)
        samples.append((s, y))
        g.step()
    return samples


# ---------------- main ----------------------------------------------------
def main(total_games: int, workers: int, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with mp.Pool(workers) as pool, open(out_path, "wb") as fout:
        for result in tqdm.tqdm(pool.imap_unordered(one_game, range(total_games)),
                                total=total_games, ncols=80):
            for sample in result:
                pickle.dump(sample, fout)
    print(f"✔ finished {total_games} games → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--games",   type=int, default=50_000)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    parser.add_argument("--out",     type=str, default="data/battleship_supervised.pkl")
    args = parser.parse_args()

    main(args.games, args.workers, args.out)