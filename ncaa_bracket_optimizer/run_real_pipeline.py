#!/usr/bin/env python3
"""
Run the full NCAA Bracket Optimizer pipeline with REAL 2026 data.

Data sources:
1. Sports Reference 2026 (real efficiency margins, Barthag, adj O/D)
2. Market Consensus (Polymarket + sportsbook blend)
3. Historical tournament data (1985-2024 for model calibration)
"""

import sys
import os
import json
import math

sys.path.insert(0, os.path.dirname(__file__))

from config import NUM_SIMULATIONS, POOL_SIZE
from model_v2 import WinProbabilityModelV2 as WinProbabilityModel
from simulator import BracketSimulator
from optimizer import BracketOptimizer
from data_collector_v2 import load_historical_tournament_data

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def build_merged_team_data():
    """
    Merge Sports Reference data + Market Consensus into a single
    team data dict ready for the model.
    """
    print("=" * 60)
    print("  BUILDING MERGED TEAM DATA FROM REAL SOURCES")
    print("=" * 60)

    # Load Sports Reference data (our primary stats source)
    sr_path = os.path.join(DATA_DIR, "sports_reference_2026.json")
    with open(sr_path) as f:
        sr_data = json.load(f)
    print(f"  Sports Reference: {len(sr_data)} teams loaded")

    # Load market consensus
    market_path = os.path.join(DATA_DIR, "market_consensus.json")
    with open(market_path) as f:
        market_data = json.load(f)
    print(f"  Market Consensus: {len(market_data)} teams loaded")

    # Load historical data for calibration
    historical = load_historical_tournament_data()

    # Build merged team data
    team_data = {}
    for team_name, sr in sr_data.items():
        team_data[team_name] = {
            "name": team_name,
            "adj_oe": sr.get("adj_oe"),
            "adj_de": sr.get("adj_de"),
            "eff_margin": sr.get("eff_margin"),
            "barthag": sr.get("barthag"),
            "srs": sr.get("srs"),
            "sos": sr.get("sos"),
            "record": sr.get("record"),
            "conference": sr.get("conference"),
            "seed": sr.get("seed"),
            "region": sr.get("region"),
        }

    # Integrate market data
    aliases = {
        "UConn": "Connecticut", "Connecticut": "UConn",
        "St. John's": "St John's", "St John's": "St. John's",
        "Miami": "Miami (FL)", "Miami (FL)": "Miami",
        "Saint Louis": "St. Louis", "St. Louis": "Saint Louis",
        "Norhtern Iowa": "Northern Iowa",  # typo in market data
    }

    matched_market = 0
    for market_team, prob in market_data.items():
        matched = None
        if market_team in team_data:
            matched = market_team
        elif market_team in aliases and aliases[market_team] in team_data:
            matched = aliases[market_team]
        else:
            # Substring match
            for td_team in team_data:
                if market_team.lower() in td_team.lower() or td_team.lower() in market_team.lower():
                    matched = td_team
                    break

        if matched:
            team_data[matched]["market_champ_prob"] = prob / 100  # store as 0-1
            matched_market += 1

    print(f"  Market data matched: {matched_market} teams")

    # Add model parameters
    team_data["__model_params__"] = {
        "calibrated_logistic_coeff": historical.get("calibrated_logistic_coeff", 0.17),
        "historical_rates": {
            "sweet16": historical.get("sweet16_rates", {}),
            "elite8": historical.get("elite8_rates", {}),
            "final4": historical.get("final4_rates", {}),
            "championship": historical.get("championship_rates", {}),
        }
    }

    # Save
    merged_path = os.path.join(DATA_DIR, "merged_team_data.json")
    with open(merged_path, "w") as f:
        json.dump(team_data, f, indent=2)
    print(f"\n  Merged team data saved: {len(team_data) - 1} teams")
    print(f"  Calibrated logistic coeff: {historical.get('calibrated_logistic_coeff', 0.17)}")

    # Print top teams by efficiency margin
    sorted_teams = sorted(
        [(t, d) for t, d in team_data.items() if t != "__model_params__"],
        key=lambda x: x[1].get("eff_margin", 0) or 0,
        reverse=True,
    )
    print(f"\n  Top 15 Teams by Efficiency Margin:")
    for i, (team, data) in enumerate(sorted_teams[:15]):
        em = data.get("eff_margin", 0)
        barthag = data.get("barthag", 0)
        mkt = data.get("market_champ_prob", 0) * 100
        print(f"    {i+1:>2}. {team:<22} EM: {em:>+6.1f}  Barthag: {barthag:.4f}  Market: {mkt:>5.1f}%")

    return team_data


def run_pipeline(n_sims=500000):
    """Run the full pipeline with real data."""

    # Step 1: Build merged data
    team_data = build_merged_team_data()

    # Step 2: Load field
    field_path = os.path.join(DATA_DIR, "tournament_field.json")
    with open(field_path) as f:
        field = json.load(f)

    # Step 3: Build model
    print("\n" + "-" * 60)
    print("  Building win probability model...")
    print("-" * 60)
    model = WinProbabilityModel(team_data)

    # Show a few key matchup probabilities
    print("\n  Sample Matchup Probabilities:")
    matchups = [
        ("Duke", "Siena", 1, 16),
        ("Duke", "UConn", 1, 2),
        ("Michigan", "Iowa State", 1, 2),
        ("Arizona", "Purdue", 1, 2),
        ("Florida", "Houston", 1, 2),
        ("Duke", "Michigan", 1, 1),
        ("Duke", "Arizona", 1, 1),
    ]
    for team_a, team_b, seed_a, seed_b in matchups:
        prob = model.win_probability(team_a, team_b, seed_a, seed_b)
        print(f"    ({seed_a}) {team_a} vs ({seed_b}) {team_b}: "
              f"{team_a} {prob:.1%} | {team_b} {1-prob:.1%}")

    # Step 4: Simulate
    print("\n" + "-" * 60)
    print(f"  Running {n_sims:,} Monte Carlo tournament simulations...")
    print("-" * 60)
    sim = BracketSimulator(model, field)
    results = sim.simulate(n_sims)

    sim.print_advancement_table(top_n=30)
    sim.print_expected_points_table(top_n=30)

    # Step 5: Compare model vs market
    _compare_model_vs_market(sim, team_data)

    # Step 6: Optimize portfolio
    print("\n" + "-" * 60)
    print("  Optimizing bracket portfolio for contest EV...")
    print("-" * 60)

    # Prize structure: equal weight for now (will update when Sanjay confirms)
    prize_structure = {1: 1.0, 2: 1.0, 3: 1.0}

    opt = BracketOptimizer(sim, None, None, prize_structure)
    opt.compute_leverage_scores()

    bracket_a, bracket_b = opt.generate_portfolio()
    opt.save_brackets(bracket_a, bracket_b)

    # Step 7: Pool simulation
    print("\n" + "-" * 60)
    print("  Simulating pool contest (top-3 optimization)...")
    print("-" * 60)
    pool_results = opt.evaluate_portfolio(bracket_a, bracket_b, n_pool_sims=10000)

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"\n  Results saved in: {DATA_DIR}/")
    print(f"  - merged_team_data.json")
    print(f"  - simulation_results.json")
    print(f"  - final_brackets.json")
    print(f"  - brackets_readable.txt")
    print(f"  - pool_simulation_results.json")

    return bracket_a, bracket_b, pool_results


def _compare_model_vs_market(sim, team_data):
    """Show where our model disagrees with the market."""
    print(f"\n{'='*80}")
    print(f"  MODEL vs MARKET - Where Do We Disagree?")
    print(f"{'='*80}")
    print(f"  {'Team':<22} {'Model':>7} {'Market':>7} {'Diff':>7} {'Signal':<20}")
    print(f"  {'-'*22} {'-'*7} {'-'*7} {'-'*7} {'-'*20}")

    comparisons = []
    for team, probs in sim.advancement_probs.items():
        model_champ = probs.get(6, 0) * 100
        # Find market prob
        market_prob = 0
        td = team_data.get(team, {})
        if isinstance(td, dict):
            market_prob = td.get("market_champ_prob", 0) * 100

        if model_champ > 0.5 or market_prob > 0.5:
            diff = model_champ - market_prob
            comparisons.append((team, model_champ, market_prob, diff))

    comparisons.sort(key=lambda x: abs(x[3]), reverse=True)

    for team, model_p, market_p, diff in comparisons[:20]:
        if abs(diff) < 0.3:
            signal = "~ Aligned"
        elif diff > 2:
            signal = "^^ MODEL LIKES MORE"
        elif diff > 0.5:
            signal = "^ Model slightly higher"
        elif diff < -2:
            signal = "vv MARKET LIKES MORE"
        elif diff < -0.5:
            signal = "v Market slightly higher"
        else:
            signal = "~ Close"

        print(f"  {team:<22} {model_p:>6.1f}% {market_p:>6.1f}% {diff:>+6.1f}% {signal}")

    print(f"\n  Key: Large positive diff = our model sees value the market doesn't")
    print(f"       Large negative diff = market knows something we might be missing")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sims", type=int, default=500000)
    args = parser.parse_args()
    run_pipeline(args.sims)
