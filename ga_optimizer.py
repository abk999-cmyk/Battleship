# ga_optimizer.py  – Memory‑safe Genetic Algorithm tuner for AIAgent3
# =============================================================================
# This rewrite fixes the runaway RAM usage that paused PyCharm (~110 GB!) by
#   • returning *only* scalar fitness values from worker processes so the main
#     process never receives huge data blobs;
#   • giving each Pool‑worker a finite lifetime (maxtasksperchild) so that
#     NumPy/TensorFlow allocations are periodically released back to the OS;
#   • forcing explicit garbage‑collection after every game;
#   • storing *just two* populations in RAM (current + next) instead of keeping
#     the entire generational history.
# Drop this file in your repo root (same level as game.py) and run:
#     $ python ga_optimizer.py --gens 60 --pop 50 --cpus 8
# -----------------------------------------------------------------------------
import argparse, json, os, random, math, time, gc, multiprocessing as mp
import numpy as np
from pathlib import Path
from types import SimpleNamespace
from collections import namedtuple
from tqdm import tqdm

from game import BattleshipGame
from AI_agent3 import AIAgent3
from AI_agent2 import AIPlayer2     as aiplayer2
import logging

# Set seed for reproducibility
random.seed(1337)
np.random.seed(1337)

# -----------------------------------------------------------------------------
# Configurable parameters (overridden by CLI flags)
# -----------------------------------------------------------------------------
OPPONENTS          = [AIAgent3]
GAMES_PER_OPP      = 40            # ⟵ each chromosome plays 100× each opponent
MUTATION_STD       = 0.12         # σ for Gaussian mutation (per‑gene)
CROSSOVER_RATE     = 0.25         # probability of uniform crossover per gene
ELITISM_FRACTION   = 0.1          # keep top‑N genomes unchanged each gen
WEIGHT_BOUNDS      = (0.0, 1.0)   # clamp meta‑weights inside this range
SAVE_EVERY         = 5            # gens between `ga_weights.json` checkpoints

META_KEYS = [                # must match AIAgent3.meta_weights order
    'density', 'neural', 'montecarlo', 'information_gain', 'opponent_model'
]
Chrom = namedtuple('Chrom', META_KEYS + ['fitness'])

# -----------------------------------------------------------------------------
# Helper – evaluation function (executed in worker processes)
# -----------------------------------------------------------------------------

def _evaluate(args):
    """Evaluate *one* chromosome; returns (fitness, weights_dict, win_rate, avg_moves)."""
    weights, seed = args
    random.seed(seed)
    
    # Disable AIAgent3 logging to prevent memory overload
    logging.getLogger('AI_Agent3').setLevel(logging.CRITICAL)
    
    games  = 0
    wins   = 0
    moves  = 0

    for Opp in OPPONENTS:
        for _ in range(GAMES_PER_OPP):
            pl1 = AIAgent3("GA-Candidate", continuous_learning=False, opponent_modeling=False)
            pl1.meta_weights.update(weights)
            
            # Disable logging for AIAgent3 opponents as well
            if Opp == AIAgent3:
                opp = Opp("Opponent", continuous_learning=False, opponent_modeling=False)
                # Ensure opponent's logger is also disabled
                if hasattr(opp, 'logger'):
                    opp.logger.setLevel(logging.CRITICAL)
            else:
                opp = Opp("Opponent")
            
            g   = BattleshipGame(pl1, opp)
            g.play()
            games += 1
            if g.winner is pl1:
                wins += 1
                moves += g.move_count
            # hard‑release Board grids & NumPy buffers
            del g, pl1, opp
            gc.collect()

    win_rate     = wins / games
    avg_moves    = moves /(2*wins) if wins else 200 
    fitness      = 100 - avg_moves   # higher is better 
    return fitness, weights, win_rate, avg_moves

# -----------------------------------------------------------------------------
# Genetic operators
# -----------------------------------------------------------------------------

def random_chrom():
    w = {k: random.uniform(*WEIGHT_BOUNDS) for k in META_KEYS}
    return Chrom(**w, fitness=None)

def mutate(weights):
    new = {}
    for k, v in weights.items():
        if random.random() < 0.9:   # 90 % genes mutate lightly
            v += random.gauss(0, MUTATION_STD)
        new[k] = min(max(v, WEIGHT_BOUNDS[0]), WEIGHT_BOUNDS[1])
    return new

def crossover(a, b):
    child = {}
    for k in META_KEYS:
        child[k] = a[k] if random.random() > CROSSOVER_RATE else b[k]
    return child

# -----------------------------------------------------------------------------
# Main GA loop
# -----------------------------------------------------------------------------

def evolve(pop_size: int, gens: int, cpus: int, return_stats: bool = False):
    """
    Run GA evolution.
    
    Args:
        pop_size: Population size
        gens: Number of generations
        cpus: Number of CPU workers
        return_stats: If True, return detailed stats of best chromosome
        
    Returns:
        If return_stats=True: dict with weights, win_rate, avg_moves, fitness
        Otherwise: None
    """
    # --- initialise population ---------------------------------------------
    population = [random_chrom() for _ in range(pop_size)]

    # multiprocessing pool – workers die after 30 tasks to recycle mem
    ctx  = mp.get_context("spawn")
    pool = ctx.Pool(processes=cpus, maxtasksperchild=30)
    
    best_stats = None

    for gen in range(1, gens + 1):
        # Evaluate population ------------------------------------------------
        jobs = [( {k: getattr(ch, k) for k in META_KEYS}, random.randrange(1<<30) )
                for ch in population]
        results = []
        for fitness, weights, win_rate, avg_moves in tqdm(pool.imap_unordered(_evaluate, jobs),
                                     total=pop_size, ncols=80,
                                     desc=f"Gen {gen}/{gens}"):
            results.append((Chrom(**weights, fitness=fitness), win_rate, avg_moves))

        # Sort by fitness (desc) --------------------------------------------
        results.sort(key=lambda c: c[0].fitness, reverse=True)
        best = results[0][0]
        best_win_rate = results[0][1]
        best_avg_moves = results[0][2]
        
        print(f"🧬  Gen {gen}   best fitness = {best.fitness:.2f}   win_rate = {best_win_rate:.3f}   avg_moves = {best_avg_moves:.2f}")
        print(f"   weights = { {k: getattr(best,k) for k in META_KEYS} }")
        
        # Store best stats for final generation
        if gen == gens:
            best_stats = {
                'weights': {k: getattr(best, k) for k in META_KEYS},
                'win_rate': best_win_rate,
                'avg_moves': best_avg_moves,
                'fitness': best.fitness
            }

        # Save checkpoint ----------------------------------------------------
        if gen % SAVE_EVERY == 0 or gen == gens:
            out_path = Path("models") / "ga_weights.json"
            out_path.parent.mkdir(exist_ok=True)
            with out_path.open("w") as fh:
                json.dump({k: getattr(best, k) for k in META_KEYS}, fh, indent=2)
            print(f"💾  Saved checkpoint → {out_path}")

        # Create next generation --------------------------------------------
        elite_cnt = max(1, int(ELITISM_FRACTION * pop_size))
        next_pop  = [r[0] for r in results[:elite_cnt]]       # carry elites
        # remainder via tournament selection + variation
        while len(next_pop) < pop_size:
            pa, pb = random.sample([r[0] for r in results[:pop_size//2]], 2)
            child_w = mutate(crossover({k:getattr(pa,k) for k in META_KEYS},
                                       {k:getattr(pb,k) for k in META_KEYS}))
            next_pop.append(Chrom(**child_w, fitness=None))

        population = next_pop

    pool.close(); pool.join()
    print("✔ GA optimisation finished.")
    
    if return_stats:
        return best_stats
    return None

# -----------------------------------------------------------------------------
# CLI entry‑point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop",  type=int, default=40,
                    help="population size")
    ap.add_argument("--gens", type=int, default=50,
                    help="number of generations")
    ap.add_argument("--cpus", type=int, default=max(mp.cpu_count()//2, 1),
                    help="workers for evaluation pool")
    args = ap.parse_args()

    evolve(pop_size=args.pop, gens=args.gens, cpus=args.cpus)
