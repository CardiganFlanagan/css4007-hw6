import csv
from pathlib import Path

import numpy as np

from gridworld import GridWorld, load_config

OUT = Path(__file__).parent / "outputs"


def pos(s):
    return f"({s[0] + 1},{s[1] + 1})"


def paper_comparison(world, history, config):
    lines = ["V1 (every nonterminal tile):", world.value_map(history[1]), ""]
    lines.append("V2 at the requested positions, all four actions computed from V1:")
    for r, c in config["paper_v2_positions"]:
        s = (r - 1, c - 1)
        best, q = world.greedy_actions(history[1], s)
        parts = ", ".join(f"{a}={q[a]:.4f}" for a in world.actions)
        lines.append(f"  {pos(s)} [{world.tile(s)}]: {parts} -> V2={max(q.values()):.4f}, best: {'/'.join(best)}")
    return "\n".join(lines)


def solve(world, config):
    sv = config["solver"]
    V, iters, delta, converged, history = world.value_iteration(sv["tolerance"], sv["max_iterations"], config["initial_value"])
    pi, ties = world.policy(V)
    return V, iters, delta, converged, history, pi, ties


def solution_report(name, world, V, iters, delta, converged, pi, ties):
    status = "converged" if converged else "DID NOT CONVERGE (safeguard reached)"
    lines = [
        f"Map: {name}",
        "\n".join("  " + "".join(row) for row in world.grid),
        f"Status: {status} after {iters} iterations, final max |change| = {delta:.3e}",
        "",
        "Values V*:",
        world.value_map(V),
        "",
        "Policy (all tied actions shown; ties = within 1e-9 of best):",
        world.policy_map(pi, ties),
        "",
        "Policy used for simulation (first tied action in order up, right, down, left):",
        world.policy_map(pi),
        "",
        "Q-values per state:",
    ]
    for s in world.states:
        if world.is_terminal(s):
            continue
        _, q = world.greedy_actions(V, s)
        lines.append(f"  {pos(s)} [{world.tile(s)}] " + "  ".join(f"{a}={q[a]:9.4f}" for a in world.actions))
    return "\n".join(lines)


def simulate(world, pi, config, map_name):
    sim = config["simulation"]
    rows = []
    for batch_seed in sim["batch_seeds"]:
        for i in range(sim["episodes_per_batch"]):
            episode_seed = 1000 * batch_seed + i
            rng = np.random.default_rng(episode_seed)
            s = world.start
            total, discounted, outcome, steps = 0.0, 0.0, "time_limit", 0
            path = [pos(s)]
            for t in range(sim["max_steps"]):
                s, r = world.step(s, pi[s], rng)
                total += r
                discounted += world.gamma ** t * r
                steps = t + 1
                path.append(pos(s))
                if world.is_terminal(s):
                    outcome = "goal" if world.tile(s) == "G" else "hole"
                    break
            rows.append({
                "map": map_name,
                "batch_seed": batch_seed,
                "episode_seed": episode_seed,
                "steps": steps,
                "total_reward": round(total, 6),
                "discounted_return": round(discounted, 6),
                "outcome": outcome,
                "path": " ".join(path),
            })
    return rows


def summarize(rows):
    out = []
    for key in sorted({(r["map"], r["batch_seed"]) for r in rows}, key=lambda k: (k[0] != "paper", k[1])):
        b = [r for r in rows if (r["map"], r["batch_seed"]) == key]
        out.append({
            "map": key[0],
            "batch_seed": key[1],
            "goals": sum(r["outcome"] == "goal" for r in b),
            "holes": sum(r["outcome"] == "hole" for r in b),
            "time_limits": sum(r["outcome"] == "time_limit" for r in b),
            "mean_steps": np.mean([r["steps"] for r in b]),
            "mean_discounted_return": np.mean([r["discounted_return"] for r in b]),
            "n": len(b),
        })
    return out


def rule_checks(config):
    w = GridWorld.from_config(config, "extended_map")
    lines = []

    def check(label, ok):
        lines.append(f"[{'PASS' if ok else 'FAIL'}] {label}")
        return ok

    def show(s, a):
        return ", ".join(f"p={p:.1f}->{pos(n)} r={r}" for p, n, r in w.transitions(s, a))

    corner = (0, 0)
    t = w.transitions(corner, "up")
    lines.append(f"(1,1) '.' up (off map): {show(corner, 'up')}")
    check("moving off the map stays in place and pays the floor reward -1", t == [(1.0, corner, -1)])

    mud = (0, 1)
    t = w.transitions(mud, "down")
    lines.append(f"(1,2) 'M' down (into wall #): {show(mud, 'down')}")
    check("bumping into the wall from mud stays on mud and pays mud reward -3", t == [(1.0, mud, -3)])

    check("wall (2,2) is not a state", (1, 1) not in w.states)

    ice = (1, 2)
    for a in w.actions:
        t = w.transitions(ice, a)
        lines.append(f"(2,3) 'I' {a}: {show(ice, a)}")
        check(f"ice '{a}' probabilities sum to 1", abs(sum(p for p, _, _ in t) - 1) < 1e-12)
    t = {n: (p, r) for p, n, r in w.transitions(ice, "left")}
    check("ice 'left' slip into wall merges with intended stay: stay p=0.8, up 0.1, down 0.1",
          abs(t[ice][0] - 0.8) < 1e-12 and abs(t[(0, 2)][0] - 0.1) < 1e-12 and abs(t[(2, 2)][0] - 0.1) < 1e-12)

    check("entering mud (1,2) from (1,1) pays -3", w.transitions((0, 0), "right") == [(1.0, (0, 1), -3)])
    check("entering sand (3,2) from S pays -2", w.transitions((2, 0), "right") == [(1.0, (2, 1), -2)])
    check("entering water (3,4) from (3,3) pays -5", w.transitions((2, 2), "right") == [(1.0, (2, 3), -5)])
    check("bumping off the map from water stays on water and pays -5", w.transitions((2, 3), "down") == [(1.0, (2, 3), -5)])
    check("moves from sand are deterministic", all(len(w.transitions((2, 1), a)) == 1 for a in w.actions))
    check("moves from water are deterministic", all(len(w.transitions((2, 3), a)) == 1 for a in w.actions))
    check("entering G from (1,3) pays exactly +10", w.transitions((0, 2), "right") == [(1.0, (0, 3), 10)])
    check("water (3,4) up into H pays -10", w.transitions((2, 3), "up") == [(1.0, (1, 3), -10)])

    goal, hole = (0, 3), (1, 3)
    for s in (goal, hole):
        check(f"terminal {pos(s)} self-loops with reward 0 for every action",
              all(w.transitions(s, a) == [(1.0, s, 0.0)] for a in w.actions))
    V, *_ = w.value_iteration(1e-6, 10000)
    check("terminal values stay 0 after value iteration", V[goal] == 0 and V[hole] == 0)

    rng = np.random.default_rng(0)
    counts = {}
    for _ in range(100000):
        n, _ = w.step(ice, "up", rng)
        counts[n] = counts.get(n, 0) + 1
    freq = ", ".join(f"{pos(n)}: {c / 100000:.3f}" for n, c in sorted(counts.items()))
    lines.append(f"Sampled 100000 ice 'up' moves with the simulator: {freq}")
    check("simulator ice frequencies within 0.01 of 0.8/0.1/0.1",
          abs(counts[(0, 2)] / 1e5 - 0.8) < 0.01 and abs(counts[ice] / 1e5 - 0.1) < 0.01 and abs(counts[hole] / 1e5 - 0.1) < 0.01)
    return "\n".join(lines)


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    config = load_config()
    OUT.mkdir(exist_ok=True)

    checks = rule_checks(config)
    (OUT / "c_rule_checks.txt").write_text(checks + "\n")
    print(checks, "\n")
    if "FAIL" in checks:
        raise SystemExit("rule checks failed")

    all_rows = []
    for name, key in [("paper", "paper_map"), ("extended", "extended_map")]:
        world = GridWorld.from_config(config, key)
        V, iters, delta, converged, history, pi, ties = solve(world, config)
        report = solution_report(name, world, V, iters, delta, converged, pi, ties)
        if name == "paper":
            report = paper_comparison(world, history, config) + "\n\n" + report
        (OUT / f"c_{name}_solution.txt").write_text(report + "\n")
        print(report, "\n")
        all_rows += simulate(world, pi, config, name)

    write_csv(OUT / "c_episodes.csv", all_rows)
    summary = summarize(all_rows)
    lines = ["| Map | Batch seed | Goals /50 | Holes /50 | Time limits /50 | Mean steps | Mean discounted return |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for s in summary:
        lines.append(f"| {s['map']} | {s['batch_seed']} | {s['goals']} | {s['holes']} | {s['time_limits']} | "
                     f"{s['mean_steps']:.2f} | {s['mean_discounted_return']:.4f} |")
    table = "\n".join(lines)
    (OUT / "c_summary.md").write_text(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
