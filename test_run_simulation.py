import copy
import unittest

import numpy as np

from run_simulation import (
    Agent,
    ForagingABM,
    SimulationConfig,
    average_ratio_trajectory,
    init_agents,
    n_risky_from_start_pct,
    run_simulation,
    risky_pct,
    starting_ratio_sweep,
)


class TestRunSimulation(unittest.TestCase):
    def setUp(self):
        self.abm_mini = SimulationConfig(
            n_agents = 10, 
            n_timesteps = 10, 
            n_risky = 5,
            seed = 5
            )
        self.abm_no_imitation = SimulationConfig(
            n_agents = 10, 
            n_timesteps = 10, 
            n_risky = 5, 
            copy_probability = 0
            )
        self.abm_fixed_food = SimulationConfig(
            n_agents = 10, 
            n_timesteps = 10, 
            n_risky = 5, 
            fixed_food = 10
            )
        self.abm_homogenous = SimulationConfig(
            n_agents = 10, 
            n_timesteps = 10, 
            n_risky = 0
            )
        self.abm_risky_worse = SimulationConfig(
            n_agents=10,
            n_timesteps=20,
            n_risky=5,
            payoffs={"risky": [(0.4, 4), (0.6, -4)], "safe": [(0.5, 4), (0.5, 0)]}
        )
        self.test_ring = [Agent("risky"), Agent("safe"), Agent("safe")]
            
    def test_no_imitation(self):
        run = run_simulation(self.abm_no_imitation)
        self.assertEqual(run.n_risky_final, 5)
        self.assertEqual(run.n_safe_final, 5)
        self.assertEqual(run.risky_pct_trajectory[0], run.risky_pct_trajectory[9])
    
    def test_fixed_food(self):
        run = run_simulation(self.abm_fixed_food)
        self.assertEqual(run.n_risky_final, 5)
        self.assertEqual(run.n_safe_final, 5)
        self.assertEqual(run.risky_pct_trajectory[0], run.risky_pct_trajectory[9])

    def test_homogenous_start(self):
        run = run_simulation(self.abm_homogenous)
        self.assertEqual(run.n_risky_final, 0)
        self.assertEqual(run.n_safe_final, 10)
        self.assertEqual(run.risky_pct_trajectory[0], run.risky_pct_trajectory[9])

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            SimulationConfig(n_agents = 2)
        with self.assertRaises(ValueError):
            SimulationConfig(n_agents = 3, n_risky = 7)
        with self.assertRaises(ValueError):
            SimulationConfig(beta = -1)
        with self.assertRaises(ValueError):
            SimulationConfig(n_timesteps = -1)

    def test_risky_worse_EV(self):
        run = run_simulation(self.abm_risky_worse)
        self.assertGreater(run.n_safe_final, run.n_risky_final)

    def test_food_floor(self):
        config = SimulationConfig(
            n_agents=3,
            n_timesteps=10,
            n_risky=1,
            payoffs={"risky": [(1.0, -100.0)], "safe": [(1.0, -100.0)]},
            food_decay=0.0,
        )
        rng = np.random.default_rng(0)
        agents = init_agents(config, rng)
        abm = ForagingABM(agents, config)
        for _ in range(config.n_timesteps):
            abm.step(rng)
            for agent in abm.agents:
                self.assertGreaterEqual(agent.food, 0.0)

    def test_food_decay(self):
        config = SimulationConfig(
            n_agents=3,
            n_timesteps=5,
            n_risky=1,
            payoffs={"risky": [(1.0, 0.0)], "safe": [(1.0, 0.0)]},
            food_decay=2.0,
        )
        rng = np.random.default_rng(0)
        agents = init_agents(config, rng)
        agent = agents[0]
        agent.food = 10.0
        abm = ForagingABM(agents, config)
        abm.step(rng)
        self.assertEqual(agent.food, 8.0)
        agent.food = 1.0
        abm.step(rng)
        self.assertEqual(agent.food, 0.0)

    def test_trajectory_shape(self):
        config = self.abm_mini
        run = run_simulation(config)
        self.assertEqual(len(run.risky_pct_trajectory), config.n_timesteps + 1)

    def test_average_trajectory_shape(self):
        config = self.abm_mini
        batch = average_ratio_trajectory(config, n_runs=5)
        self.assertEqual(batch.mean_trajectory.shape, (config.n_timesteps + 1,))
        self.assertEqual(batch.std_trajectory.shape, (config.n_timesteps + 1,))
        self.assertAlmostEqual(batch.mean_trajectory[0], 50.0)

    def test_sweep_shape(self):
        start_pct = np.array([25.0, 50.0, 75.0])
        sweep = starting_ratio_sweep(self.abm_mini, start_pct, n_runs_per_ratio=3)
        self.assertEqual(len(sweep.start_risky_pct), 3)
        self.assertEqual(len(sweep.mean_final_risky_pct), 3)
        self.assertEqual(len(sweep.std_final_risky_pct), 3)

    def test_reproducibility(self):
        r1 = run_simulation(copy.deepcopy(self.abm_mini))
        r2 = run_simulation(copy.deepcopy(self.abm_mini))
        np.testing.assert_array_equal(r1.risky_pct_trajectory, r2.risky_pct_trajectory)

    def test_risky_pct(self):
        agents = [Agent("risky"), Agent("safe"), Agent("safe")]
        self.assertAlmostEqual(risky_pct(agents), 100/3)
        self.assertEqual(risky_pct([Agent("safe")]), 0)
        self.assertEqual(risky_pct([Agent("risky")]), 100)


if __name__ == "__main__":
    unittest.main()
