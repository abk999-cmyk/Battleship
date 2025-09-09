"""
rl_finetune.py  –  Multi‑process REINFORCE fine‑tuning
------------------------------------------------------
• Needs models/battleship_heatmap.h5 (supervised model)
• Spawns W worker processes (default: all CPU cores)
• Aggregates gradients every --batch games (vectorized loss)
• Tqdm progress bar shows overall status
"""

import argparse, os, multiprocessing as mp, numpy as np, tensorflow as tf
from functools import partial
from tqdm import tqdm
from tensorflow.keras.models import load_model, save_model

from ..core.game import BattleshipGame
from ..agents.AI_agent import AIPlayer


# ---------------- helper --------------------------------------------------
def encode_state(grid):
    m = (grid == -1); h = (grid == 1); u = (grid == 0)
    return np.stack([m, h, u], -1).astype(np.float16)   # 10×10×3


def choose_move(logits, avail):
    flat = logits.numpy().ravel()
    mask = np.full_like(flat, -1e9, dtype=np.float32)
    for r, c in avail:
        mask[r * 10 + c] = flat[r * 10 + c]
    p = tf.nn.softmax(mask).numpy()
    idx = np.random.choice(len(p), p=p)
    return divmod(idx, 10)


# ---------------- worker --------------------------------------------------
_model = None
def init_worker(model_path):
    global _model
    _model = load_model(model_path, compile=False)

def play_game(_):
    players = [AIPlayer("A"), AIPlayer("B")]
    for p in players:
        p.nn_model = _model
    g = BattleshipGame(players[0], players[1])
    traj = []
    while not g.is_over():
        cur = g.current
        s = encode_state(cur.result_grid)
        logits = _model(np.expand_dims(s,0), training=False)[0]
        move = choose_move(logits, cur.available_moves)
        if cur is players[0]:
            traj.append((s, move[0]*10+move[1]))
        g.step_manual(move)
    r = 1.0 if g.winner is players[0] else 0.0
    return traj, r


# ---------------- vectorized reinforce loss ------------------------------
@tf.function
def reinforce_loss(model, traj_batch, R_batch):
    states, actions, rewards = [], [], []
    for tb, R in zip(traj_batch, R_batch):
        for s, a in tb:
            states.append(s); actions.append(a); rewards.append(R)
    S = tf.convert_to_tensor(states, dtype=tf.float16)         # (N,10,10,3)
    A = tf.convert_to_tensor(actions, dtype=tf.int32)
    R = tf.convert_to_tensor(rewards, dtype=tf.float32)
    logits = model(S, training=True)                           # (N,10,10,1)
    logits = tf.reshape(logits, [tf.shape(S)[0], 100])
    logp = tf.nn.log_softmax(logits)
    idx  = tf.stack([tf.range(tf.shape(S)[0]), A], axis=1)
    logp_a = tf.gather_nd(logp, idx)
    return -tf.reduce_sum(logp_a * R)


# ---------------- main ----------------------------------------------------
def main(args):
    model_path = "models/battleship_heatmap.h5"
    assert os.path.exists(model_path), "Pretrained model not found!"
    model = load_model(model_path, compile=False)
    opt   = tf.keras.optimizers.Adam(1e-4)

    pool = mp.Pool(args.workers, initializer=init_worker, initargs=(model_path,))
    pbar = tqdm(total=args.games, desc="RL‑FineTune", ncols=80)

    buf_traj, buf_R, processed = [], [], 0
    for traj, R in pool.imap_unordered(play_game, range(args.games)):
        buf_traj.append(traj); buf_R.append(R); processed += 1
        pbar.update(1)

        if processed % args.batch == 0:
            with tf.GradientTape() as tape:
                loss = reinforce_loss(model, buf_traj, buf_R)
            grads = tape.gradient(loss, model.trainable_variables)
            opt.apply_gradients(zip(grads, model.trainable_variables))
            buf_traj.clear(); buf_R.clear()
            pbar.set_postfix(loss=f"{loss.numpy():.4f}")

            if processed % 1000 == 0:
                ckpt = f"models/rl_finetune_{processed}.h5"
                save_model(model, ckpt)

    pool.close(); pool.join()
    out_path = "models/battleship_heatmap_finetuned.h5"
    save_model(model, out_path)
    print(f"\n✔ Fine‑tuning complete → {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--games",   type=int, default=20000)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--batch",   type=int, default=256)
    args = ap.parse_args()
    main(args)