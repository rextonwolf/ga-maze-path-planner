"""
GA Maze Path Planner (WingMan / NIDAR - Genetic Algorithm module)

A Genetic Algorithm evolves a sequence of moves (U/D/L/R) that takes a drone
from START to GOAL through a grid maze with obstacles. Compared against BFS
(the optimal baseline).

Run:  python ga_maze.py
"""
import random
from collections import deque

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# ---------------- Maze (1 = wall, 0 = free) ----------------
MAZE = np.array([
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [1, 1, 0, 1, 0, 1, 0, 1, 1, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [1, 1, 1, 1, 1, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 1, 1, 1, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 1, 1, 1, 1, 1, 0],
])
START, GOAL = (0, 0), (9, 9)
MOVES = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}  # U D L R

# ---------------- GA parameters ----------------
POP_SIZE = 200
GENOME_LEN = 60
GENERATIONS = 300
MUTATION_RATE = 0.05
TOURNAMENT_K = 5
ELITES = 4
SEED = 42


def simulate(genome):
    """Follow the moves; blocked moves are ignored. Returns path up to the goal."""
    r, c = START
    path = [(r, c)]
    for g in genome:
        dr, dc = MOVES[g]
        nr, nc = r + dr, c + dc
        if 0 <= nr < MAZE.shape[0] and 0 <= nc < MAZE.shape[1] and MAZE[nr, nc] == 0:
            r, c = nr, nc
            path.append((r, c))
            if (r, c) == GOAL:
                break
    return path


def fitness(genome):
    path = simulate(genome)
    end = path[-1]
    dist = abs(end[0] - GOAL[0]) + abs(end[1] - GOAL[1])
    score = -dist                                  # closer to goal is better
    score -= 0.1 * (len(genome) - len(set(path)))  # penalise wasted/revisited moves
    if end == GOAL:
        score += 100 - len(path)                   # big bonus, shorter is better
    return score


def tournament(pop, fits):
    idx = random.sample(range(len(pop)), TOURNAMENT_K)
    return pop[max(idx, key=lambda i: fits[i])]


def crossover(a, b):
    p = random.randint(1, GENOME_LEN - 1)
    return a[:p] + b[p:]


def mutate(g):
    return [random.randint(0, 3) if random.random() < MUTATION_RATE else x for x in g]


def run_ga():
    random.seed(SEED)
    pop = [[random.randint(0, 3) for _ in range(GENOME_LEN)] for _ in range(POP_SIZE)]
    best_hist, avg_hist = [], []
    best, best_fit = None, -1e9
    for gen in range(GENERATIONS):
        fits = [fitness(g) for g in pop]
        i = int(np.argmax(fits))
        if fits[i] > best_fit:
            best, best_fit = pop[i][:], fits[i]
        best_hist.append(best_fit)
        avg_hist.append(float(np.mean(fits)))
        order = sorted(range(len(pop)), key=lambda k: fits[k], reverse=True)
        new_pop = [pop[k][:] for k in order[:ELITES]]
        while len(new_pop) < POP_SIZE:
            new_pop.append(mutate(crossover(tournament(pop, fits), tournament(pop, fits))))
        pop = new_pop
    return best, best_hist, avg_hist


def bfs():
    q, seen = deque([(START, [START])]), {START}
    while q:
        (r, c), path = q.popleft()
        if (r, c) == GOAL:
            return path
        for dr, dc in MOVES.values():
            nr, nc = r + dr, c + dc
            if (0 <= nr < MAZE.shape[0] and 0 <= nc < MAZE.shape[1]
                    and MAZE[nr, nc] == 0 and (nr, nc) not in seen):
                seen.add((nr, nc))
                q.append(((nr, nc), path + [(nr, nc)]))
    return None


def draw(ax, path, title):
    ax.imshow(MAZE, cmap="Greys")
    ys, xs = zip(*path)
    ax.plot(xs, ys, "r-o", ms=4)
    ax.scatter(*START[::-1], c="green", s=120, label="Start")
    ax.scatter(*GOAL[::-1], c="blue", s=120, label="Goal")
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])


if __name__ == "__main__":
    best, best_hist, avg_hist = run_ga()
    ga_path = simulate(best)
    bfs_path = bfs()
    reached = ga_path[-1] == GOAL

    print(f"GA reached goal : {reached}")
    print(f"GA path length  : {len(ga_path) - 1} steps")
    print(f"BFS (optimal)   : {len(bfs_path) - 1} steps")
    if reached:
        print(f"GA is {(len(ga_path) - 1) / (len(bfs_path) - 1):.2f}x the optimal length")

    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    draw(axs[0], ga_path, f"GA path ({len(ga_path) - 1} steps)")
    draw(axs[1], bfs_path, f"BFS optimal ({len(bfs_path) - 1} steps)")
    axs[2].plot(best_hist, label="Best fitness")
    axs[2].plot(avg_hist, label="Average fitness")
    axs[2].set_title("GA convergence")
    axs[2].set_xlabel("Generation"); axs[2].set_ylabel("Fitness"); axs[2].legend()
    plt.tight_layout()
    plt.savefig("ga_result.png", dpi=120)
    if matplotlib.get_backend().lower() != "agg":
        plt.show()
