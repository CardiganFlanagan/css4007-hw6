import json
from pathlib import Path

DATA = Path(__file__).parent / "data" / "maps.json"

MOVES = {"up": (-1, 0), "right": (0, 1), "down": (1, 0), "left": (0, -1)}
PERPENDICULAR = {
    "up": ("left", "right"),
    "down": ("left", "right"),
    "left": ("up", "down"),
    "right": ("up", "down"),
}
ARROWS = {"up": "^", "right": ">", "down": "v", "left": "<"}


def load_config(path=DATA):
    with open(path) as f:
        return json.load(f)


class GridWorld:
    def __init__(self, rows, tiles, actions, discount):
        self.grid = [list(r) for r in rows]
        self.tiles = tiles
        self.actions = actions
        self.gamma = discount
        self.n_rows = len(self.grid)
        self.n_cols = len(self.grid[0])
        unknown = {c for row in self.grid for c in row} - set(tiles)
        if unknown:
            raise ValueError(f"no rules for tiles {sorted(unknown)}")
        self.states = [
            (r, c)
            for r in range(self.n_rows)
            for c in range(self.n_cols)
            if tiles[self.grid[r][c]].get("traversable", True)
        ]
        self.start = next(s for s in self.states if self.tile(s) == "S")

    @classmethod
    def from_config(cls, config, map_key):
        return cls(config[map_key], config["tiles"], config["actions"], config["discount"])

    def tile(self, s):
        return self.grid[s[0]][s[1]]

    def rule(self, s):
        return self.tiles[self.tile(s)]

    def is_terminal(self, s):
        return self.rule(s).get("terminal", False)

    def reward(self, s):
        return self.rule(s)["reward"]

    def destination(self, s, direction):
        dr, dc = MOVES[direction]
        r, c = s[0] + dr, s[1] + dc
        if not (0 <= r < self.n_rows and 0 <= c < self.n_cols):
            return s
        if not self.tiles[self.grid[r][c]].get("traversable", True):
            return s
        return (r, c)

    def direction_probs(self, s, action):
        rule = self.rule(s)
        if rule.get("movement") == "slippery":
            side = rule["perpendicular_each"]
            a, b = PERPENDICULAR[action]
            return [(action, rule["intended"]), (a, side), (b, side)]
        return [(action, 1.0)]

    def transitions(self, s, action):
        if self.is_terminal(s):
            return [(1.0, s, 0.0)]
        out = {}
        for direction, p in self.direction_probs(s, action):
            nxt = self.destination(s, direction)
            prob, _ = out.get(nxt, (0.0, None))
            out[nxt] = (prob + p, self.reward(nxt))
        return [(p, nxt, r) for nxt, (p, r) in out.items()]

    def q_value(self, V, s, action):
        return sum(p * (r + self.gamma * V[nxt]) for p, nxt, r in self.transitions(s, action))

    def bellman_update(self, V):
        new = {}
        for s in self.states:
            if self.is_terminal(s):
                new[s] = 0.0
            else:
                new[s] = max(self.q_value(V, s, a) for a in self.actions)
        return new

    def value_iteration(self, tolerance, max_iterations, initial=0.0):
        V = {s: float(initial) for s in self.states}
        history = [V]
        delta = float("inf")
        for k in range(1, max_iterations + 1):
            new = self.bellman_update(V)
            delta = max(abs(new[s] - V[s]) for s in self.states)
            V = new
            history.append(V)
            if delta < tolerance:
                return V, k, delta, True, history
        return V, max_iterations, delta, False, history

    def greedy_actions(self, V, s, tie_tol=1e-9):
        q = {a: self.q_value(V, s, a) for a in self.actions}
        best = max(q.values())
        return [a for a in self.actions if q[a] >= best - tie_tol], q

    def policy(self, V, tie_tol=1e-9):
        pi, ties = {}, {}
        for s in self.states:
            if self.is_terminal(s):
                continue
            best, _ = self.greedy_actions(V, s, tie_tol)
            pi[s] = best[0]
            ties[s] = best
        return pi, ties

    def value_map(self, V, fmt="{:8.3f}"):
        lines = []
        for r in range(self.n_rows):
            cells = []
            for c in range(self.n_cols):
                t = self.grid[r][c]
                if (r, c) in V and not self.is_terminal((r, c)):
                    cells.append(fmt.format(V[(r, c)]))
                else:
                    cells.append(f"{t:>8}")
            lines.append(" ".join(cells))
        return "\n".join(lines)

    def policy_map(self, pi, ties=None):
        lines = []
        for r in range(self.n_rows):
            cells = []
            for c in range(self.n_cols):
                s = (r, c)
                if s in pi:
                    marks = "".join(ARROWS[a] for a in (ties[s] if ties else [pi[s]]))
                    cells.append(f"{marks:>4}")
                else:
                    cells.append(f"{self.grid[r][c]:>4}")
            lines.append(" ".join(cells))
        return "\n".join(lines)

    def step(self, s, action, rng):
        outcomes = self.transitions(s, action)
        u = rng.random()
        acc = 0.0
        for p, nxt, r in outcomes:
            acc += p
            if u < acc:
                return nxt, r
        p, nxt, r = outcomes[-1]
        return nxt, r
