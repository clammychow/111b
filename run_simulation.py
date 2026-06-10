"""Foraging strategy diffusion ABM on a ring."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
np.seterr(all="raise")

Strategy = Literal["risky", "safe"]

DEFAULT_PAYOFFS: dict[str, list[tuple[float, float]]] = {
    "risky": [(0.40, 8.0), (0.60, -2.0)],
    "safe": [(0.50, 4.0), (0.50, 0.0)],
}


@dataclass
class SimulationConfig:
    n_agents: int = 100
    n_timesteps: int = 500
    seed: int = 42
    beta: float = 1.0 # copy sensitivity to payoff differences
    food_decay: float = 2.0
    payoffs: dict[str, list[tuple[float, float]]] = field(
        default_factory=lambda: copy.deepcopy(DEFAULT_PAYOFFS)
    )
    # None allows overrides for invariance verification tests
    copy_probability: float | None = None
    fixed_food: float | None = None
    n_risky: int | None = None

    def __post_init__(self):
        if self.n_agents < 3:
            raise ValueError("n_agents must be >= 3")
        if self.n_timesteps <= 0:
            raise ValueError("n_timesteps must be positive")
        if self.beta < 0:
            raise ValueError("beta cannot be negative")
        if self.n_risky is not None:
            if self.n_risky > self.n_agents:
                raise  ValueError("n_risky cannot be greater than n_agents")

# Post-run data - for plotting
@dataclass
class SimulationResult:
    risky_pct_trajectory: np.ndarray
    final_risky_pct: float
    n_risky_final: int
    n_safe_final: int
    config: SimulationConfig


@dataclass
class TrajectoryBatchResult:
    mean_trajectory: np.ndarray
    std_trajectory: np.ndarray
    mean_final_risky_pct: float
    std_final_risky_pct: float
    mean_n_risky_final: float
    mean_n_safe_final: float
    n_runs: int


@dataclass
class SweepResult:
    start_risky_pct: np.ndarray
    mean_final_risky_pct: np.ndarray
    std_final_risky_pct: np.ndarray
    n_runs_per_ratio: int


# Controls agent behavior
class Agent:
    def __init__(self, strategy: Strategy, food: float = 0.0) -> None:
        self.strategy = strategy
        self.food = food

    # compute and add payoffs/decay to current food
    def forage(self, rng: np.random.Generator, payoffs: dict[str, list[tuple[float, float]]]) -> None:
        outcomes = payoffs[self.strategy]
        probs = [p for p, _ in outcomes]
        deltas = [d for _, d in outcomes]
        delta = deltas[rng.choice(len(deltas), p=probs)]
        self.food = max(0.0, self.food + delta)

    def apply_decay(self, decay_amount: float) -> None:
        self.food = max(0.0, self.food - decay_amount)

    # computes copy probability; does not directly update strategy
    def copy_probability(
        self,
        neighbor: Agent,
        beta: float,
        copy_override: float | None,
    ) -> float:
        if neighbor.strategy == self.strategy:
            return 0.0
        delta = neighbor.food - self.food
        if delta <= 0.0:
            return 0.0
        if copy_override is not None:
            return copy_override
        # no sensitivity -> probability becomes 0, not 0.5
        if beta == 0.0:
            return 0.0
        return 1.0 / (1.0 + np.exp(-beta * delta))


# Controls simulation behavior
class ForagingABM:
    def __init__(self, agents: list[Agent], config: SimulationConfig) -> None:
        self.agents = agents
        self.config = config
        self.n = len(agents)

    def neighbor_left(self, i: int) -> int:
        return (i - 1) % self.n

    def neighbor_right(self, i: int) -> int:
        return (i + 1) % self.n

    def step(self, rng: np.random.Generator) -> None:
        cfg = self.config

        if cfg.fixed_food is not None:
            for agent in self.agents:
                agent.food = cfg.fixed_food
        else:
            for agent in self.agents:
                agent.forage(rng, cfg.payoffs)
            for agent in self.agents:
                agent.apply_decay(cfg.food_decay)

        # updates all agent strategies at once
        new_strategies: list[Strategy] = []
        for i, agent in enumerate(self.agents):
            left = self.neighbor_left(i)
            right = self.neighbor_right(i)
            neighbor_idx = left if rng.random() < 0.5 else right
            neighbor = self.agents[neighbor_idx]
            prob = agent.copy_probability(neighbor, cfg.beta, cfg.copy_probability)
            if rng.random() < prob:
                new_strategies.append(neighbor.strategy)
            else:
                new_strategies.append(agent.strategy)

        for agent, strategy in zip(self.agents, new_strategies):
            agent.strategy = strategy

    # appends % risky at each new timestep to track run trajectory
    def run(self, rng: np.random.Generator) -> np.ndarray:
        trajectory = [risky_pct(self.agents)]
        for _ in range(self.config.n_timesteps):
            self.step(rng)
            trajectory.append(risky_pct(self.agents))
        return np.array(trajectory, dtype=float)


def risky_pct(agents: list[Agent]) -> float:
    """Percentage of agents with the risky strategy."""
    if not agents:
        raise ValueError("No agents in ring")
    n_risky = sum(1 for a in agents if a.strategy == "risky")
    return 100.0 * n_risky / len(agents)


def n_risky_from_start_pct(start_pct: float, n_agents: int) -> int:
    """Convert initial % risky agents to a risky agent count."""
    return max(1, min(n_agents - 1, round(start_pct / 100.0 * n_agents)))


def init_agents(config: SimulationConfig, rng: np.random.Generator) -> list[Agent]:
    if config.n_risky is not None:
        n_risky = config.n_risky
    else:
        n_risky = config.n_agents // 2
    strategies: list[Strategy] = ["risky"] * n_risky + ["safe"] * (config.n_agents - n_risky)
    rng.shuffle(strategies)
    return [Agent(strategy=s) for s in strategies]


def run_simulation(config: SimulationConfig) -> SimulationResult:
    rng = np.random.default_rng(config.seed)
    agents = init_agents(config, rng)
    abm = ForagingABM(agents, config)
    trajectory = abm.run(rng)
    n_risky = sum(1 for a in agents if a.strategy == "risky")
    n_safe = len(agents) - n_risky
    return SimulationResult(
        risky_pct_trajectory=trajectory,
        final_risky_pct=float(trajectory[-1]),
        n_risky_final=n_risky,
        n_safe_final=n_safe,
        config=config,
    )


def average_ratio_trajectory(config: SimulationConfig, n_runs: int = 30) -> TrajectoryBatchResult:
    trajectories = []
    final_ratios = []
    final_risky = []
    final_safe = []
    for i in range(n_runs):
        run_config = copy.deepcopy(config)
        run_config.n_risky = config.n_agents // 2
        run_config.seed = config.seed + i
        result = run_simulation(run_config)
        trajectories.append(result.risky_pct_trajectory)
        final_ratios.append(result.final_risky_pct)
        final_risky.append(result.n_risky_final)
        final_safe.append(result.n_safe_final)

    stacked = np.stack(trajectories, axis=0)
    final_arr = np.array(final_ratios, dtype=float)
    return TrajectoryBatchResult(
        mean_trajectory=stacked.mean(axis=0),
        std_trajectory=stacked.std(axis=0),
        mean_final_risky_pct=float(final_arr.mean()),
        std_final_risky_pct=float(final_arr.std()),
        mean_n_risky_final=float(np.mean(final_risky)),
        mean_n_safe_final=float(np.mean(final_safe)),
        n_runs=n_runs,
    )


# Sensitivity analysis to determine robustness against starting composition
def starting_ratio_sweep(
    config: SimulationConfig,
    start_risky_pct: np.ndarray,
    n_runs_per_ratio: int = 30,
) -> SweepResult:
    start_risky_pct = np.asarray(start_risky_pct, dtype=float)
    mean_final = np.zeros(len(start_risky_pct))
    std_final = np.zeros(len(start_risky_pct))

    for idx, pct in enumerate(start_risky_pct):
        finals = []
        for run_i in range(n_runs_per_ratio):
            run_config = copy.deepcopy(config)
            run_config.n_risky = n_risky_from_start_pct(float(pct), config.n_agents)
            run_config.seed = config.seed + idx * 1000 + run_i
            result = run_simulation(run_config)
            finals.append(result.final_risky_pct)
        finals_arr = np.array(finals, dtype=float)
        mean_final[idx] = finals_arr.mean()
        std_final[idx] = finals_arr.std()

    return SweepResult(
        start_risky_pct=start_risky_pct,
        mean_final_risky_pct=mean_final,
        std_final_risky_pct=std_final,
        n_runs_per_ratio=n_runs_per_ratio,
    )


def _ylim_from_values(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 2.0
    ymin, ymax = float(finite.min()), float(finite.max())
    if ymin == ymax:
        ymin -= 0.1
        ymax += 0.1
    return ymin, ymax


def save_diagnostic_figures(
    trajectory_batch: TrajectoryBatchResult,
    sweep_result: SweepResult,
    output_dir: str | Path = "results",
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    timesteps = np.arange(len(trajectory_batch.mean_trajectory))

    # ratio_trajectory.png
    fig, ax = plt.subplots(figsize=(8, 5))
    mean = trajectory_batch.mean_trajectory
    std = trajectory_batch.std_trajectory
    ax.plot(timesteps, mean, label="Mean % risky agents")
    ax.fill_between(timesteps, mean - std, mean + std, alpha=0.3, label="±1 std")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("% risky agents")
    ax.set_ylim(0, 100)
    ax.set_title("Average risky-agent trajectory (50/50 start)")
    ax.legend()
    fig.tight_layout()
    p1 = out / "ratio_trajectory.png"
    fig.savefig(p1)
    plt.close(fig)
    paths.append(p1)

    # final_ratio.png
    fig, ax = plt.subplots(figsize=(6, 5))
    mean_pct = trajectory_batch.mean_final_risky_pct
    ax.bar(
        ["Risky", "Safe"],
        [trajectory_batch.mean_n_risky_final, trajectory_batch.mean_n_safe_final],
        color=["#e74c3c", "#3498db"],
    )
    ax.set_ylabel("Mean agent count")
    ax.set_title(f"Final composition (mean risky = {mean_pct:.1f}%)")
    fig.tight_layout()
    p2 = out / "final_ratio.png"
    fig.savefig(p2)
    plt.close(fig)
    paths.append(p2)

    # starting_ratio_sweep.png
    fig, ax = plt.subplots(figsize=(8, 5))
    x = sweep_result.start_risky_pct
    y = sweep_result.mean_final_risky_pct
    yerr = sweep_result.std_final_risky_pct
    ax.errorbar(x, y, yerr=yerr, fmt="o-", capsize=3)
    ymin, ymax = _ylim_from_values(y)
    margin = max(1.0, (ymax - ymin) * 0.1)
    ax.set_xlim(0, 100)
    ax.set_ylim(max(0.0, ymin - margin), min(100.0, ymax + margin))
    ax.set_xlabel("Initial % risky agents")
    ax.set_ylabel("Mean final % risky agents")
    ax.set_title("Starting composition sweep")
    fig.tight_layout()
    p3 = out / "starting_ratio_sweep.png"
    fig.savefig(p3)
    plt.close(fig)
    paths.append(p3)

    return paths


def main() -> None:
    config = SimulationConfig()
    trajectory_batch = average_ratio_trajectory(config, n_runs=30)
    sweep_result = starting_ratio_sweep(
        config,
        start_risky_pct=np.linspace(10, 90, 20),
        n_runs_per_ratio=30,
    )
    paths = save_diagnostic_figures(trajectory_batch, sweep_result, "results")
    for path in paths:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
