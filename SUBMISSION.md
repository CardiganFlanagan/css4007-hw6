# HW6 submission

## AI-use declaration

Which tools did you use, and for what?

- **Claude (Anthropic, Claude Opus 5.5)**: I used it to check my Part A arithmetic, to help write `gridworld.py`, `part_b.py` and `part_c.py`, and to draft the explanations in this file. I ran all the code myself, compared the solver output with my paper work, and checked that every number in this file matches the files in `outputs/`.

## Setup and reproduction

Python 3.13, `gymnasium==1.4.0`, `numpy==2.5.4` (see `requirements.txt`).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python part_b.py                   # FrozenLake experiments -> outputs/b_*
python part_c.py                   # rule checks, solver, simulations -> outputs/c_*
```

Both scripts are deterministic, so running them again gives the same files.

| File | Contents |
| --- | --- |
| `gridworld.py` | Grid World model (transitions read from `data/maps.json`), value iteration, policy extraction, map printing |
| `part_b.py` | FrozenLake episode runner and the four experiment conditions |
| `part_c.py` | Rule checks, solving both maps, simulating the computed policies |
| `outputs/b_episodes.csv`, `outputs/b_summary.md`, `outputs/b_traces/` | Part B results |
| `outputs/c_rule_checks.txt`, `outputs/c_paper_solution.txt`, `outputs/c_extended_solution.txt`, `outputs/c_episodes.csv`, `outputs/c_summary.md` | Part C results |

## Part A — Paper calculations

Scan: [`part_a/part_a.jpg`](part_a/part_a.jpg) (committed before any code).

**1. The problem as an MDP.**
- **States:** the 10 non-wall tiles. `G` (1,4) and `H` (2,4) are terminal states with value 0. The wall (2,2) is not a state.
- **Actions:** up, right, down and left at every nonterminal tile.
- **Transitions:** from `S`, `.` and `M` the chosen move always happens. From `I`, the agent goes the chosen way with probability 0.8 and each perpendicular way with probability 0.1. A move into the wall or off the map leaves the agent where it is.
- **Rewards:** the reward of the tile where the move ends: −1 for floor, start or ice, −3 for mud, +10 for the goal, −10 for the hole. Bumping into a wall still pays the reward of the tile you stay on.
- **Goal of the agent:** maximize the expected discounted sum of rewards, `r0 + 0.9 r1 + 0.9² r2 + …`. Roughly: reach `G` quickly, avoid paying for mud, and don't fall into `H`.

**2. V1** (V0 = 0 everywhere, so V1 is the best expected immediate reward):

```text
 -1   -1   10    G
 -1    #   -1    H
 -1   -1   -1   -1
```

- (1,3): moving right enters `G`, which gives +10.
- (1,2) mud: right or left gives −1. Up or down bumps and stays on mud, which gives −3.
- (2,3) ice: left gives 0.8(−1) + 0.1(−1) + 0.1(−1) = −1. Up gives 0.8(−1) + 0.1(−1, bounce off wall) + 0.1(−10) = −1.9.
- Every other tile has some move that pays −1.

**3. V2 using only V1.**

(1,2) mud:
- up (bump, stay on M): −3 + 0.9(−1) = −3.9
- right → (1,3): −1 + 0.9(10) = **8.0**
- down (wall, stay on M): −3 + 0.9(−1) = −3.9
- left → (1,1): −1 + 0.9(−1) = −1.9
- **V2(1,2) = 8.0**

(1,3):
- up (bump, stay): −1 + 0.9(10) = 8.0
- right → G: +10 + 0.9(0) = **10.0**
- down → ice: −1 + 0.9(−1) = −1.9
- left → mud: −3 + 0.9(−1) = −3.9
- **V2(1,3) = 10.0**

(2,3) ice, written as the probability-weighted sum:
- up: 0.8[−1 + 0.9(10)] + 0.1[−1 + 0.9(−1)] (left: wall, stay) + 0.1[−10 + 0] (right: H) = 6.4 − 0.19 − 1.0 = **5.21**
- right: 0.8[−10] + 0.1[−1 + 0.9(10)] (up) + 0.1[−1 + 0.9(−1)] (down) = −8 + 0.8 − 0.19 = −7.39
- down: 0.8[−1 + 0.9(−1)] + 0.1[−1 + 0.9(−1)] (left: stay) + 0.1[−10] (right: H) = −1.52 − 0.19 − 1.0 = −2.71
- left: 0.8[−1 + 0.9(−1)] (wall, stay) + 0.1[−1 + 0.9(10)] (up) + 0.1[−1 + 0.9(−1)] (down) = −1.52 + 0.8 − 0.19 = −0.91
- **V2(2,3) = 5.21**

**4. Best actions after two rounds:** (1,2) → right, (1,3) → right, (2,3) ↑ up. There are no ties at these three tiles.

Two rounds do not prove the policy is optimal. Information about the goal spreads only one step per round. After two rounds most of the map still has a flat −1 and many tied actions; for example, all four actions are tied at `S`. The values are also far from converged: the largest change in round 2 is still 9.0. A later round can make a different action better. My solver shows this at `S`: the best action is "up" in rounds 4–5 and becomes "right" from round 6 onward. Only when values stop changing (the Bellman equation holds) is the greedy policy optimal.

## Part B — FrozenLake

### Fixed policy, written before the experiments

Gymnasium action codes are 0 = left, 1 = down, 2 = right, 3 = up. States are numbered row by row:

```text
 0  1  2  3      S F F F
 4  5  6  7      F H F H
 8  9 10 11      F F F H
12 13 14 15      H F F G
```

| State | 0 | 1 | 2 | 3 | 4 | 6 | 8 | 9 | 10 | 13 | 14 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Action | down | left | down | left | down | down | right | down | down | right | right |

Reasoning: the shortest safe route is 0 → 4 → 8 → 9 → 13 → 14 → 15, which avoids hole 5 on the left and hole 12 at the bottom-left. States that are not on this route move toward it or toward the goal without stepping into a neighboring hole. From 1, "down" would fall into 5, so the policy goes left back to the route. 2 and 6 go down toward 10, then 10 goes down to 14. 3 goes left, because "down" leads to hole 7. I designed the policy for the map as drawn and did not try to handle slipping. The policy is the `FIXED_POLICY` dict in `part_b.py` and is the same in every condition.

Seeds: episode `i` of batch `b` is reset with `1000·b + i`. The random policy draws actions from `numpy.random.default_rng(b)`, created fresh for each batch, so the slippery and non-slippery random runs start from the same action stream. Episodes stop on `terminated` or `truncated`, with at most 100 steps. Final state 15 means goal; any other terminal state means hole.

### Results

| Slippery? | Policy | Batch seed | Goals /50 | Holes /50 | Time limits /50 | Success % | Mean steps |
| --- | --- | --- | --- | --- | --- | --- | --- |
| no | random | 11 | 0 | 50 | 0 | 0% | 7.94 |
| no | random | 22 | 1 | 49 | 0 | 2% | 7.16 |
| no | random | 33 | 2 | 48 | 0 | 4% | 6.66 |
| no | fixed | 11 | 50 | 0 | 0 | 100% | 6.00 |
| no | fixed | 22 | 50 | 0 | 0 | 100% | 6.00 |
| no | fixed | 33 | 50 | 0 | 0 | 100% | 6.00 |
| yes | random | 11 | 1 | 49 | 0 | 2% | 7.98 |
| yes | random | 22 | 3 | 47 | 0 | 6% | 7.66 |
| yes | random | 33 | 1 | 49 | 0 | 2% | 7.16 |
| yes | fixed | 11 | 1 | 49 | 0 | 2% | 5.22 |
| yes | fixed | 22 | 2 | 48 | 0 | 4% | 6.00 |
| yes | fixed | 33 | 1 | 49 | 0 | 2% | 5.14 |

Data: [`outputs/b_episodes.csv`](outputs/b_episodes.csv) has one row per episode, 600 rows, including the full state and action sequence. Traces are in [`outputs/b_traces/`](outputs/b_traces/):
- Successful slippery episode: [`slippery_fixed_goal.txt`](outputs/b_traces/slippery_fixed_goal.txt), seed 11020, 14 steps.
- Failed slippery episode: [`slippery_fixed_hole.txt`](outputs/b_traces/slippery_fixed_hole.txt), seed 11000.
- Non-slippery fixed run: [`not_slippery_fixed_goal.txt`](outputs/b_traces/not_slippery_fixed_goal.txt).

Outcomes never observed (listed in `never_observed.txt`): no time-limit episode in any condition, and no hole for the fixed policy without slipping.

**What slipping changed.** Without slipping, the fixed policy is deterministic: the same 6 moves every time, 150/150 goals. With `is_slippery=True`, FrozenLake moves the intended way with probability only 1/3, and to each perpendicular side with 1/3. My route goes right next to holes. At state 4 the action "down" slides right into hole 5 one time in three. At state 8, "right" slides down into hole 12. Success dropped to 4/150 (2.7%), close to the random policy's 5/150 (3.3%). Mean steps also dropped (about 5.5 vs 6.0), because most episodes end early in a hole. A policy that is good on a deterministic map can be useless under slip. A slip-aware policy would choose actions whose sideways outcomes cannot reach a hole, for example "left" at state 4, which never falls in. That same policy would stand still forever on the non-slippery map, though, so one fixed policy cannot be good in both settings.

**Consistency of batches.** The deterministic fixed run is identical in all batches. In every random or slippery condition the goal counts vary between batches (0–3 out of 50), while the conclusion stays the same: about 2–4% success. With such small counts, one batch of 50 can shift the success rate by several percentage points.

**Why one success is not enough.** A single episode is one sample of a random process. The slippery fixed policy did reach the goal in episode 11020, but in 146 out of 150 runs it fell into a hole. One episode does not show the probability of success; many seeded runs estimate it.

**Why this is not training.** The policies never change. The fixed policy is a lookup table written before the runs, and the random policy ignores rewards completely. Nothing uses the rewards to update values or actions, so the 150th episode is played exactly like the first. These runs measure how good a given policy is; they do not improve it.

## Part C — Custom solver

The model reads tile rules from `data/maps.json`: reward, whether the tile is terminal or traversable, and deterministic or slippery movement. `GridWorld.transitions()` builds the probabilities from those rules. Value iteration (`GridWorld.value_iteration`) only calls `transitions()` and knows nothing about specific letters. Adding `D` and `W` required no code change, only their entries in the JSON. The model raises an error if a map contains a letter with no rules. Each round computes all new values from the previous round's values only.

### Compare code with paper

From [`outputs/c_paper_solution.txt`](outputs/c_paper_solution.txt):

```text
V1:
  -1.000   -1.000   10.000        G
  -1.000        #   -1.000        H
  -1.000   -1.000   -1.000   -1.000

(1,2) [M]: up=-3.9000, right=8.0000, down=-3.9000, left=-1.9000 -> V2=8.0000, best: right
(1,3) [.]: up=8.0000, right=10.0000, down=-1.9000, left=-3.9000 -> V2=10.0000, best: right
(2,3) [I]: up=5.2100, right=-7.3900, down=-2.7100, left=-0.9100 -> V2=5.2100, best: up
```

All V1 values, all twelve V2 action values and the three best actions match my paper work exactly. There were no discrepancies to correct.

### Values, policies and convergence

Stopping rule: stop when the largest |V_{k+1}(s) − V_k(s)| is below 1e-6, with a safeguard of 10,000 iterations. Both maps converged long before the safeguard.

| Map | Iterations | Final max change |
| --- | --- | --- |
| paper `.M.G / .#IH / S...` | 12 | 2.17e-07 |
| extended `.M.G / .#IH / SD.W` | 11 | 2.41e-07 |

The policy is extracted greedily from the final values. **Tie-breaking:** actions within 1e-9 of the best Q-value count as tied. All tied actions are printed, and the simulation uses the first one in the order up, right, down, left. With the converged values there are no ties on either map.

Paper map ([`c_paper_solution.txt`](outputs/c_paper_solution.txt)):

```text
   4.200    8.000   10.000        G        >    >    >    G
   2.780        #    5.824        H        ^    #    ^    H
   1.536    2.818    4.242    2.818        >    >    ^    <
```

Extended map ([`c_extended_solution.txt`](outputs/c_extended_solution.txt)):

```text
   4.200    8.000   10.000        G        >    >    >    G
   2.780        #    5.824        H        ^    #    ^    H
   1.502    2.818    4.242    2.818        ^    >    ^    <
```

### Simulation results

Three batches of 50 episodes from `S`, using the same seed scheme as Part B and a 100-step limit. In my simulator, episode `e` uses `numpy.random.default_rng(e)` for slips. Discounted return is `r0 + 0.9 r1 + 0.9² r2 + …`. Episodes are in [`outputs/c_episodes.csv`](outputs/c_episodes.csv).

| Map | Batch seed | Goals /50 | Holes /50 | Time limits /50 | Mean steps | Mean discounted return |
| --- | --- | --- | --- | --- | --- | --- |
| paper | 11 | 47 | 3 | 0 | 5.04 | 2.2297 |
| paper | 22 | 44 | 6 | 0 | 4.96 | 1.4424 |
| paper | 33 | 42 | 8 | 0 | 4.88 | 0.9700 |
| extended | 11 | 50 | 0 | 0 | 5.00 | 1.5020 |
| extended | 22 | 50 | 0 | 0 | 5.00 | 1.5020 |
| extended | 33 | 50 | 0 | 0 | 5.00 | 1.5020 |

### Explain your results

**What changed with sand and water.** Only the start changed its action. On the paper map, `S` goes right along the bottom row and up through the ice: S → (3,2) → (3,3) → ice → (1,3) → G. Its Q-value for right is 1.536, versus 1.502 for going up the left column through the mud. On the extended map, (3,2) is sand, so the first step costs −2 instead of −1. Q(S, right) drops by exactly 1, to 0.536, and the agent switches to the left route S → (2,1) → (1,1) → mud → (1,3) → G. V*(S) drops from 1.536 to 1.502. All other values and arrows are identical on both maps. From (3,2) and (3,3) the best moves lead away from the new floors, and a tile's own cost only matters when you enter it. Water at (3,4) changed Q((3,3), right) from 1.536 to −2.464, but "up" was already best there, so the arrow did not change. V(3,4) stayed 2.818 because the best move from water is left, back onto ordinary floor.

**Where cost and risk mattered.**
- The agent pays for mud willingly: from (1,1) "right" into mud (4.20) beats every alternative, because mud is the only way to the goal along the top row.
- The two routes from `S` differ by only 0.034 in expected value. The bottom route is shorter in floor cost but crosses the ice, where 10% of moves "up" slide right into the hole.
- Value iteration maximizes expected return, so on the paper map it accepts that risk. The simulation shows its cost: 17/150 paper-map episodes fell into the hole (11.3%, close to the theoretical 0.1/0.9 ≈ 11.1%, since a slip into the wall just repeats the attempt). On the extended map there were zero holes.
- On ice the policy picks "up" (5.82). "Left" (4.48) is perfectly safe, because both of its slip outcomes avoid the hole, but slower; "right" (−6.92) goes straight toward the hole. The policy takes a 10% hole risk instead of the safe detour, because the detour costs more on average.

**How simulation relates to the computed values.** V*(S) is an expectation over infinitely many possible futures with an infinite horizon. A batch of 50 episodes gives only a sample mean.
- On the extended map the policy never touches ice, so every episode is the same 5-step path and the return is exactly V*(S) = 1.502 in every batch.
- On the paper map the batch means range from 0.97 to 2.23, depending on how many holes the batch happened to contain. The mean over all 150 episodes is 1.547, close to V*(S) = 1.536, as it should be.
- The horizon difference does not matter here. Every episode reached a terminal state within 6 steps, so the 100-step limit never cut an episode, and after termination all rewards are 0. The finite sum therefore equals the full infinite sum for each episode.
- What separates simulation from V* is sampling noise. A policy that loops for a long time would add a second gap, because rewards after step 100 would be missing.

### Rule checks

Saved in [`outputs/c_rule_checks.txt`](outputs/c_rule_checks.txt), run on the extended map before solving. All checks pass.

- Moving off the map from (1,1) leaves the agent on (1,1) with reward −1.
- Moving down from mud into the wall leaves the agent on mud with reward −3.
- The wall (2,2) is not a state.
- All four ice actions have outcome probabilities summing to 1, with each slip resolved separately against walls. For example, "left" from ice gives stay 0.8, up 0.1, down 0.1. 100,000 simulated ice moves "up" gave frequencies 0.802 / 0.099 / 0.099.
- Entering mud pays −3, sand −2, water −5; bumping off the map from water pays −5. Moves from sand and water are deterministic. Entering G pays exactly +10, and entering H from water pays −10.
- `G` and `H` loop to themselves with reward 0 for every action, and their values stay 0 after value iteration.
