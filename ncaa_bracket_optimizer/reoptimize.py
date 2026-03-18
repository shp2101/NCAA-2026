#!/usr/bin/env python3
"""Re-run optimization using saved simulation results (no re-simulating)."""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(__file__))

from config import NUM_SIMULATIONS, POOL_SIZE
from model_v2 import WinProbabilityModelV2 as WinProbabilityModel
from simulator import BracketSimulator
from optimizer import BracketOptimizer

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Load existing data
with open(os.path.join(DATA_DIR, "merged_team_data.json")) as f:
    team_data = json.load(f)
with open(os.path.join(DATA_DIR, "tournament_field.json")) as f:
    field = json.load(f)

# Build model
model = WinProbabilityModel(team_data)

# Quick simulation (100K to be fast but still accurate)
sim = BracketSimulator(model, field)
results = sim.simulate(100000)

# Optimize with improved divergence
prize_structure = {1: 1.0, 2: 1.0, 3: 1.0}
opt = BracketOptimizer(sim, None, None, prize_structure)
opt.compute_leverage_scores()

bracket_a, bracket_b = opt.generate_portfolio()
opt.save_brackets(bracket_a, bracket_b)

# Pool simulation
pool_results = opt.evaluate_portfolio(bracket_a, bracket_b, n_pool_sims=10000)
