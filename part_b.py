import csv
import re
from pathlib import Path

import gymnasium as gym
import numpy as np

OUT = Path(__file__).parent / "outputs"

LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3
ACTION_NAMES = {LEFT: "left", DOWN: "down", RIGHT: "right", UP: "up"}

FIXED_POLICY = {
    0: DOWN,
    1: LEFT,
    2: DOWN,
    3: LEFT,
    4: DOWN,
    6: DOWN,
    8: RIGHT,
    9: DOWN,
    10: DOWN,
    13: RIGHT,
    14: RIGHT,
}

BATCH_SEEDS = [11, 22, 33]
EPISODES = 50
MAX_STEPS = 100
GOAL = 15
HIGHLIGHT = re.compile(r"\x1b\[41m(.)\x1b\[0m")


def clean_frame(frame):
    return HIGHLIGHT.sub("@", frame).strip("\n")


def run_episode(env, policy_name, episode_seed, rng, record=False):
    state, _ = env.reset(seed=episode_seed)
    frames = [clean_frame(env.render())] if record else []
    states, actions = [state], []
    total, outcome, steps = 0.0, "time_limit", 0
    for t in range(MAX_STEPS):
        if policy_name == "random":
            action = int(rng.integers(env.action_space.n))
        else:
            action = FIXED_POLICY[state]
        state, reward, terminated, truncated, _ = env.step(action)
        total += reward
        steps = t + 1
        states.append(state)
        actions.append(action)
        if record:
            frames.append(clean_frame(env.render()))
        if terminated:
            outcome = "goal" if state == GOAL else "hole"
            break
        if truncated:
            break
    return {
        "steps": steps,
        "total_reward": total,
        "outcome": outcome,
        "states": states,
        "actions": actions,
        "frames": frames,
    }


def run_all():
    rows, traces = [], {}
    for slippery in (False, True):
        env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=slippery, render_mode="ansi", max_episode_steps=MAX_STEPS)
        for policy_name in ("random", "fixed"):
            condition = f"{'slippery' if slippery else 'not_slippery'}_{policy_name}"
            for batch_seed in BATCH_SEEDS:
                rng = np.random.default_rng(batch_seed)
                for i in range(EPISODES):
                    episode_seed = 1000 * batch_seed + i
                    ep = run_episode(env, policy_name, episode_seed, rng, record=True)
                    rows.append({
                        "condition": condition,
                        "is_slippery": slippery,
                        "policy": policy_name,
                        "batch_seed": batch_seed,
                        "episode_seed": episode_seed,
                        "steps": ep["steps"],
                        "total_reward": ep["total_reward"],
                        "outcome": ep["outcome"],
                        "states": " ".join(map(str, ep["states"])),
                        "actions": " ".join(ACTION_NAMES[a] for a in ep["actions"]),
                    })
                    for outcome in ("goal", "hole", "time_limit"):
                        key = (condition, outcome)
                        if ep["outcome"] == outcome and key not in traces:
                            traces[key] = (batch_seed, episode_seed, ep)
        env.close()
    return rows, traces


def summarize(rows):
    lines = ["| Slippery? | Policy | Batch seed | Goals /50 | Holes /50 | Time limits /50 | Success % | Mean steps |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for slippery in (False, True):
        for policy in ("random", "fixed"):
            for seed in BATCH_SEEDS:
                b = [r for r in rows if r["is_slippery"] == slippery and r["policy"] == policy and r["batch_seed"] == seed]
                goals = sum(r["outcome"] == "goal" for r in b)
                holes = sum(r["outcome"] == "hole" for r in b)
                limits = sum(r["outcome"] == "time_limit" for r in b)
                lines.append(f"| {'yes' if slippery else 'no'} | {policy} | {seed} | {goals} | {holes} | {limits} | "
                             f"{100 * goals / len(b):.0f}% | {np.mean([r['steps'] for r in b]):.2f} |")
    return "\n".join(lines)


def write_trace(path, condition, batch_seed, episode_seed, ep):
    lines = [f"condition={condition} batch_seed={batch_seed} episode_seed={episode_seed} "
             f"outcome={ep['outcome']} steps={ep['steps']} total_reward={ep['total_reward']}",
             "@ marks the agent; the line in brackets is the action just taken.", ""]
    for t, frame in enumerate(ep["frames"]):
        lines.append(f"t={t} state={ep['states'][t]}")
        lines.append(frame)
        lines.append("")
    path.write_text("\n".join(lines))


def main():
    OUT.mkdir(exist_ok=True)
    rows, traces = run_all()
    with open(OUT / "b_episodes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    table = summarize(rows)
    (OUT / "b_summary.md").write_text(table + "\n")
    print(table)

    trace_dir = OUT / "b_traces"
    trace_dir.mkdir(exist_ok=True)
    for old in trace_dir.glob("*.txt"):
        old.unlink()
    missing = []
    for condition in sorted({r["condition"] for r in rows}):
        for outcome in ("goal", "hole", "time_limit"):
            if (condition, outcome) in traces:
                batch_seed, episode_seed, ep = traces[(condition, outcome)]
                write_trace(trace_dir / f"{condition}_{outcome}.txt", condition, batch_seed, episode_seed, ep)
            else:
                missing.append(f"{condition}: no {outcome} episode in 150 runs")
    (trace_dir / "never_observed.txt").write_text("\n".join(missing) + "\n")
    print("\n".join(missing))


if __name__ == "__main__":
    main()
