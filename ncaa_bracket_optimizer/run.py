#!/usr/bin/env python3
"""
NCAA Bracket Optimizer - Main Orchestration Script
===================================================

USAGE:
  # Step 1: Collect data (run after Selection Sunday)
  python run.py collect

  # Step 2: Run full pipeline (collect + simulate + optimize)
  python run.py full

  # Step 3: Update with public picks and re-optimize
  python run.py optimize --public-picks data/public_pick_pcts.json

  # Step 4: Run pool contest simulation
  python run.py pool-sim

  # Quick test with mock data
  python run.py test

ESPN Tournament Challenge Scoring:
  Round of 64:  10 pts
  Round of 32:  20 pts
  Sweet 16:     40 pts
  Elite Eight:  80 pts
  Final Four:   160 pts
  Championship: 320 pts
"""

import sys
import os
import json
import argparse

# Add parent dir to path
sys.path.insert(0, os.path.dirname(__file__))

from config import NUM_SIMULATIONS, POOL_SIZE

# Use V2 modules (enhanced with CBBpy + Four Factors + calibrated model)
from data_collector_v2 import (
    collect_all_data_v2, load_tournament_field, load_public_pick_pcts,
    load_group_picks, save_tournament_field, get_barttorvik_data,
    load_historical_tournament_data,
)
from model_v2 import WinProbabilityModelV2 as WinProbabilityModel
from simulator import BracketSimulator
from optimizer import BracketOptimizer

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)


def create_mock_field():
    """
    Create a mock tournament field for testing the pipeline.
    Uses realistic 2025-ish teams.
    """
    mock_field = {
        "South": {
            "1": "Houston", "16": "Norfolk St.",
            "8": "Wisconsin", "9": "Drake",
            "5": "Clemson", "12": "McNeese",
            "4": "Auburn", "13": "Vermont",
            "6": "BYU", "11": "San Diego St.",
            "3": "Kentucky", "14": "Colgate",
            "7": "Texas", "10": "Colorado St.",
            "2": "Marquette", "15": "Montana St.",
        },
        "East": {
            "1": "Duke", "16": "American",
            "8": "Florida", "9": "UCF",
            "5": "Michigan St.", "12": "Grand Canyon",
            "4": "Arizona", "13": "High Point",
            "6": "Illinois", "11": "New Mexico",
            "3": "Wisconsin", "14": "Oakland",
            "7": "St. Mary's", "10": "Nevada",
            "2": "Iowa St.", "15": "Robert Morris",
        },
        "Midwest": {
            "1": "Kansas", "16": "Fairleigh Dickinson",
            "8": "TCU", "9": "Mississippi St.",
            "5": "Gonzaga", "12": "UC Irvine",
            "4": "Purdue", "13": "Iona",
            "6": "Creighton", "11": "Dayton",
            "3": "Baylor", "14": "Troy",
            "7": "Texas A&M", "10": "Memphis",
            "2": "Tennessee", "15": "Grambling",
        },
        "West": {
            "1": "Connecticut", "16": "Wagner",
            "8": "Northwestern", "9": "Boise St.",
            "5": "San Diego St.", "12": "Yale",
            "4": "North Carolina", "13": "Samford",
            "6": "Clemson", "11": "Oregon",
            "3": "Marquette", "14": "Morehead St.",
            "7": "Dayton", "10": "Colorado",
            "2": "Alabama", "15": "Longwood",
        },
    }

    filepath = os.path.join(DATA_DIR, "tournament_field.json")
    with open(filepath, "w") as f:
        json.dump(mock_field, f, indent=2)
    print(f"[MAIN] Mock tournament field saved to {filepath}")
    return mock_field


def create_mock_team_data(field):
    """Create mock team data with realistic efficiency margins for testing."""
    import random
    random.seed(42)

    team_data = {}
    # Assign realistic metrics based on seed
    seed_to_eff = {
        1: (28, 4), 2: (24, 4), 3: (20, 4), 4: (17, 4),
        5: (14, 4), 6: (12, 4), 7: (10, 4), 8: (8, 4),
        9: (6, 4), 10: (5, 4), 11: (4, 5), 12: (3, 5),
        13: (0, 4), 14: (-2, 4), 15: (-5, 4), 16: (-10, 5),
    }

    for region in field:
        for seed_str, team in field[region].items():
            seed = int(seed_str)
            mean_eff, std_eff = seed_to_eff[seed]
            eff = mean_eff + random.gauss(0, std_eff)

            # Offensive and defensive split
            adj_oe = 105 + eff * 0.5 + random.gauss(0, 2)
            adj_de = 105 - eff * 0.5 + random.gauss(0, 2)
            barthag = 1 / (1 + 10 ** (-eff / 10))

            team_data[team] = {
                "name": team,
                "adj_oe": round(adj_oe, 1),
                "adj_de": round(adj_de, 1),
                "eff_margin": round(eff, 1),
                "barthag": round(barthag, 4),
                "tempo": round(67 + random.gauss(0, 3), 1),
                "srs": round(eff * 0.8 + random.gauss(0, 2), 1),
                "barttorvik_rank": None,
            }

    filepath = os.path.join(DATA_DIR, "merged_team_data.json")
    with open(filepath, "w") as f:
        json.dump(team_data, f, indent=2)
    print(f"[MAIN] Mock team data saved ({len(team_data)} teams)")
    return team_data


def run_collect(field=None):
    """Step 1: Collect all data."""
    print("\n" + "=" * 60)
    print("  STEP 1: DATA COLLECTION (V2)")
    print("=" * 60)
    if field is None:
        field = load_tournament_field()
    if field:
        team_data = collect_all_data_v2(field, use_cbbpy=True, quick_mode=True)
    else:
        # No field yet - just get Barttorvik data
        barttorvik = get_barttorvik_data()
        historical = load_historical_tournament_data()
        team_data = {**barttorvik, "__model_params__": {
            "calibrated_logistic_coeff": historical.get("calibrated_logistic_coeff", 0.15),
            "historical_rates": {},
        }}
    return team_data


def run_full(n_sims=None, use_mock=False):
    """Run the full pipeline: collect → simulate → optimize."""
    if n_sims is None:
        n_sims = NUM_SIMULATIONS

    print("\n" + "=" * 60)
    print("  NCAA BRACKET OPTIMIZER - FULL PIPELINE")
    print(f"  Simulations: {n_sims:,}")
    print(f"  Pool Size: {POOL_SIZE}")
    print("=" * 60)

    # Step 1: Get team data
    if use_mock:
        field = create_mock_field()
        team_data = create_mock_team_data(field)
        # Add model params for mock data
        historical = load_historical_tournament_data()
        team_data["__model_params__"] = {
            "calibrated_logistic_coeff": historical.get("calibrated_logistic_coeff", 0.15),
            "historical_rates": {
                "sweet16": historical.get("sweet16_rates", {}),
                "elite8": historical.get("elite8_rates", {}),
                "final4": historical.get("final4_rates", {}),
                "championship": historical.get("championship_rates", {}),
            },
        }
    else:
        field = load_tournament_field()
        if field is None:
            print("\n[ERROR] No tournament field found!")
            print("Please create data/tournament_field.json with the bracket")
            print("or run 'python run.py test' to use mock data.")
            return
        team_data = run_collect(field)

    # Step 2: Build model
    print("\n" + "-" * 40)
    print("  Building win probability model...")
    print("-" * 40)
    model = WinProbabilityModel(team_data)

    # Step 3: Simulate
    print("\n" + "-" * 40)
    print("  Running Monte Carlo simulation...")
    print("-" * 40)
    sim = BracketSimulator(model, field)
    results = sim.simulate(n_sims)

    sim.print_advancement_table()
    sim.print_expected_points_table()

    # Step 4: Optimize (portfolio approach with top-3 payouts)
    print("\n" + "-" * 40)
    print("  Optimizing bracket portfolio for contest EV...")
    print("-" * 40)
    public_picks = load_public_pick_pcts()
    group_picks = load_group_picks()

    # Prize structure — will be updated when Sanjay confirms amounts
    prize_structure = {1: 1.0, 2: 1.0, 3: 1.0}  # equal weight for now

    opt = BracketOptimizer(sim, public_picks, group_picks, prize_structure)
    opt.compute_leverage_scores()

    # Generate coordinated portfolio (different champions, divergent picks)
    bracket_a, bracket_b = opt.generate_portfolio()
    opt.save_brackets(bracket_a, bracket_b)

    # Step 5: Pool simulation (top-3 tracking)
    print("\n" + "-" * 40)
    print("  Simulating pool contest (top-3 optimization)...")
    print("-" * 40)
    n_pool = 25000 if not use_mock else 3000
    pool_results = opt.evaluate_portfolio(bracket_a, bracket_b, n_pool_sims=n_pool)

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"\n  Results saved in: {DATA_DIR}/")
    print(f"  - simulation_results.json (advancement probabilities)")
    print(f"  - final_brackets.json (machine-readable brackets)")
    print(f"  - brackets_readable.txt (human-readable brackets)")
    print(f"  - pool_simulation_results.json (top-3 analysis)")

    return bracket_a, bracket_b, pool_results


def run_optimize(public_picks_file=None, group_picks_file=None):
    """Re-run optimization with updated pick data."""
    print("\n  RE-OPTIMIZING with updated pick data...")

    # Load existing simulation results
    sim_file = os.path.join(DATA_DIR, "simulation_results.json")
    team_file = os.path.join(DATA_DIR, "merged_team_data.json")
    field_file = os.path.join(DATA_DIR, "tournament_field.json")

    if not os.path.exists(sim_file):
        print("[ERROR] No simulation results found. Run 'python run.py full' first.")
        return

    with open(team_file) as f:
        team_data = json.load(f)
    with open(field_file) as f:
        field = json.load(f)

    model = WinProbabilityModel(team_data)
    sim = BracketSimulator(model, field)

    # Re-run simulation
    results = sim.simulate()

    public_picks = load_public_pick_pcts(public_picks_file)
    group_picks = load_group_picks(group_picks_file)

    prize_structure = {1: 1.0, 2: 1.0, 3: 1.0}  # update with actual amounts
    opt = BracketOptimizer(sim, public_picks, group_picks, prize_structure)
    opt.compute_leverage_scores()

    bracket_a, bracket_b = opt.generate_portfolio()
    opt.save_brackets(bracket_a, bracket_b)
    pool_results = opt.evaluate_portfolio(bracket_a, bracket_b, n_pool_sims=25000)

    return bracket_a, bracket_b, pool_results


def run_test():
    """Quick test with mock data and fewer simulations."""
    print("\n  RUNNING TEST with mock data (1000 sims)...")
    return run_full(n_sims=1000, use_mock=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NCAA Bracket Optimizer")
    parser.add_argument("command", choices=["collect", "full", "optimize", "pool-sim", "test"],
                        help="Command to run")
    parser.add_argument("--sims", type=int, default=None, help="Number of simulations")
    parser.add_argument("--public-picks", type=str, default=None, help="Path to public pick data")
    parser.add_argument("--group-picks", type=str, default=None, help="Path to group pick data")

    args = parser.parse_args()

    if args.command == "collect":
        run_collect()
    elif args.command == "full":
        run_full(n_sims=args.sims)
    elif args.command == "optimize":
        run_optimize(args.public_picks, args.group_picks)
    elif args.command == "test":
        run_test()
    else:
        parser.print_help()
