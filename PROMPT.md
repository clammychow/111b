# Project Brief: Foraging Strategy Diffusion Based on Success Imitation
I am creating an agent-based model that simulates the spread of different foraging strategies among agents that copy each other based on current observed success. I want to write a single Python script from scratch that runs this simulation and saves the visualized results to a separate folder. Please read this entire brief first and produce a detailed plan including file tree milestones, test/verification strategies, and risks. Do not write any code yet.

## Scientific Goal
Demonstrate how the dominance of one foraging strategy can emerge from an individual social imitation rule.

### Model Assumptions:
This model is structured as a ring system where agents can only interact with the agents on either side of them. The population does not fluctuate. Agents have 2 state variables: strategy and accumulated food. The outcome I am trying to measure is how the ratio of risky to safe agents changes over time.

At each timestep:
1. Every agent rolls for a food payoff to update their accumulated food count. 
   - Risky agents have a 40% chance of gaining 8 food, 60% chance of losing 2 food.
   - Safe agents have a 50% chance of gaining 4 food, 50% chance of gaining 0 food.
   - Food is accumulated over time and cannot go below 0.
2. Every agent experiences a decay of 2 food.
3. After accumulated food is updated, all agents randomly choose a neighbor to observe.
   - If the neighbor has more accumulated food than AND an opposite strategy from the agent, the probability that the agent copies the neighbor's strategy increases as the food difference between the agent and the neighbor increases. Use the softmax rule and account for zero division.
   - If the neighbor has equal/less food OR the same strategy, the probability that the agent copies the neighbor is 0.

## Engineering Goals
1. One `run_simulation.py` script that includes an Agent class to track each agent's current accumulated food and strategy
2. Function(s) to track and return the ratio of risky to safe agents over time for plotting the average ratio trajectory over multiple runs. All runs should start with 50% risky, 50% safe.
3. A `starting_ratio_sweep` function that varies the starting ratio of risky to safe agents, averages the final ratio of risky to safe agents over multiple runs, and returns the results for plotting
4. Automatic saving of all diagnostic figures to the specified directory when I do a demo run; no GUI windows
5. Configurable parameters to allow model verification and edge-case checks. Ensure the following checks can be straightforward to perform:
- Fixing copy probability to 0
- Fixing all agents' food amounts to a constant
- Allowing all agents to start with the same strategy
- Changing the expected values and variances of risky and safe payoffs

## Repository Layout
```
abm-project/
    # I will provide these:
    PROMPT.md
    SKILL.md or SUBAGENT.md
    Dockerfile
    README.md
    # You provide these:
    PLAN.md
    run_simulation.py
    results/
        ratio_trajectory.png
        final_ratio.png
        starting_ratio_sweep.png
```

The entire demo should run and produce the figures under `results/` when I call `run_simulation.py`

## Before Writing Any Code
Create a PLAN.md and include:
1. A brief description of the class hierarchy and how data flows from Agent through the abm into the plotting functions
2. Proposed architecture for agent state variables, history tracking, and parameter sweep design
3. Confirmation of the shapes returned by the simulation and parameter sweep
4. A list of any edge cases and implementation ambiguities worth handling explicitly
5. A description of how the required verification checks can be performed

Do not write any code yet. Stop after drafting the plan and wait for confirmation.
