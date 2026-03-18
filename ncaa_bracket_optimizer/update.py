#!/usr/bin/env python3
"""
NCAA Bracket Optimizer - Iterative Update Script
=================================================

Run this whenever you have new data to plug in.
Designed to be re-run multiple times over the week.

WORKFLOW:
  Day 1 (Selection Sunday - TODAY):
    python update.py --step initial
    → Scrapes team stats, market odds, runs 500K sims, generates V1 brackets

  Day 2-3 (Mon-Tue):
    python update.py --step market
    → Re-fetches market odds (they shift as people bet)
    → Re-optimizes brackets with updated market data

  Day 3-4 (Tue-Wed, once ESPN picks open):
    python update.py --step picks
    → Scrapes ESPN "Who Picked Whom" data
    → THIS IS THE BIG ONE - real leverage data
    → Re-generates brackets with actual public pick %

  Day 4 (Wed, before lock):
    python update.py --step final
    → Final re-scrape of all data
    → 500K sims + 25K pool sims
    → Generates FINAL Bracket A and Bracket B

  Anytime:
    python update.py --step market-only
    → Quick: just re-fetch market odds and show comparison to model

MANUAL DATA ENTRY:
  If scraping fails, you can manually update these files:
    data/polymarket_odds.json      - Polymarket championship %
    data/sportsbook_odds.json      - Sportsbook implied probabilities
    data/public_pick_pcts.json     - ESPN Who Picked Whom
    data/group_picks.json          - Your private group's picks
"""

import sys
import os
import json
import argparse
import time

sys.path.insert(0, os.path.dirname(__file__))

from config import NUM_SIMULATIONS, POOL_SIZE
from data_collector_v2 import (
    collect_all_data_v2, load_tournament_field, load_public_pick_pcts,
    load_group_picks, get_barttorvik_data, load_historical_tournament_data,
)
from model_v2 import WinProbabilityModelV2 as WinProbabilityModel
from simulator import BracketSimulator
from optimizer import BracketOptimizer
from market_data import (
    get_market_consensus, get_espn_who_picked_whom,
    get_polymarket_odds, get_sportsbook_odds, load_market_consensus,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def step_initial(n_sims=500000):
    """
    STEP 1: Initial run after bracket is announced.
    Collects all data, runs full simulation, generates V1 brackets.
    """
    print("\n" + "=" * 70)
    print("  INITIAL RUN - Selection Sunday")
    print("  Collecting data + 500K sims + market odds")
    print("=" * 70)

    field = load_tournament_field()
    if not field:
        print("[ERROR] No tournament field found! Update data/tournament_field.json")
        return

    # Collect team stats (Barttorvik + CBBpy)
    team_data = collect_all_data_v2(field, use_cbbpy=True, quick_mode=True)

    # Get market odds
    market = get_market_consensus()

    # Integrate market odds into team data
    team_data = _integrate_market_data(team_data, market)

    # Build model and simulate
    model = WinProbabilityModel(team_data)
    sim = BracketSimulator(model, field)
    results = sim.simulate(n_sims)

    sim.print_advancement_table(top_n=30)
    sim.print_expected_points_table(top_n=30)

    # Show model vs market comparison
    _compare_model_vs_market(sim, market)

    # Optimize
    public_picks = load_public_pick_pcts()
    prize_structure = _load_prize_structure()

    opt = BracketOptimizer(sim, public_picks, None, prize_structure)
    opt.compute_leverage_scores()
    bracket_a, bracket_b = opt.generate_portfolio()
    opt.save_brackets(bracket_a, bracket_b)

    # Pool simulation
    pool = opt.evaluate_portfolio(bracket_a, bracket_b, n_pool_sims=10000)

    print("\n[UPDATE] Initial brackets generated! These will be refined as more data comes in.")
    print("[UPDATE] Next step: run 'python update.py --step market' tomorrow to update odds.")


def step_market(n_sims=500000):
    """
    STEP 2: Re-fetch market odds and re-optimize.
    Run daily as odds shift.
    """
    print("\n" + "=" * 70)
    print("  MARKET UPDATE - Refreshing odds data")
    print("=" * 70)

    field = load_tournament_field()
    team_data = _load_team_data()

    # Re-fetch market data
    market = get_market_consensus()
    team_data = _integrate_market_data(team_data, market)

    # Re-simulate with updated data
    model = WinProbabilityModel(team_data)
    sim = BracketSimulator(model, field)
    results = sim.simulate(n_sims)

    _compare_model_vs_market(sim, market)

    # Re-optimize
    public_picks = load_public_pick_pcts()
    prize_structure = _load_prize_structure()

    opt = BracketOptimizer(sim, public_picks, None, prize_structure)
    opt.compute_leverage_scores()
    bracket_a, bracket_b = opt.generate_portfolio()
    opt.save_brackets(bracket_a, bracket_b)

    pool = opt.evaluate_portfolio(bracket_a, bracket_b, n_pool_sims=10000)


def step_picks(n_sims=500000):
    """
    STEP 3: ESPN Who Picked Whom data is available.
    THIS IS THE BIG UPDATE - real leverage data.
    """
    print("\n" + "=" * 70)
    print("  PICKS UPDATE - ESPN Who Picked Whom data")
    print("  This is the most important update!")
    print("=" * 70)

    field = load_tournament_field()
    team_data = _load_team_data()

    # Get ESPN pick data
    espn_picks = get_espn_who_picked_whom()
    if espn_picks is None:
        print("\n[WARNING] ESPN WPW data not available yet.")
        print("You can manually enter it in data/public_pick_pcts.json")
        print("Format: {\"Team Name\": {\"r64\": 0.95, \"r32\": 0.80, ...}}")
        return

    # Re-fetch market data too
    market = get_market_consensus()
    team_data = _integrate_market_data(team_data, market)

    # Re-simulate
    model = WinProbabilityModel(team_data)
    sim = BracketSimulator(model, field)
    results = sim.simulate(n_sims)

    _compare_model_vs_market(sim, market)

    # Re-optimize WITH REAL pick data
    group_picks = load_group_picks()
    prize_structure = _load_prize_structure()

    opt = BracketOptimizer(sim, espn_picks, group_picks, prize_structure)
    opt.compute_leverage_scores()
    bracket_a, bracket_b = opt.generate_portfolio()
    opt.save_brackets(bracket_a, bracket_b)

    pool = opt.evaluate_portfolio(bracket_a, bracket_b, n_pool_sims=25000)

    print("\n[UPDATE] Brackets updated with REAL ESPN pick data!")
    print("[UPDATE] Leverage calculations are now based on actual public picks.")


def step_final(n_sims=500000):
    """
    STEP 4: Final bracket generation before lock.
    Maximum sims, maximum data, final answer.
    """
    print("\n" + "=" * 70)
    print("  FINAL BRACKETS - Lock-in ready")
    print(f"  {n_sims:,} tournament sims + 25K pool sims")
    print("=" * 70)

    field = load_tournament_field()

    # Full fresh data collection
    team_data = collect_all_data_v2(field, use_cbbpy=True, quick_mode=False)

    market = get_market_consensus()
    team_data = _integrate_market_data(team_data, market)

    # Full simulation
    model = WinProbabilityModel(team_data)
    sim = BracketSimulator(model, field)
    results = sim.simulate(n_sims)

    sim.print_advancement_table(top_n=40)
    sim.print_expected_points_table(top_n=40)

    _compare_model_vs_market(sim, market)

    # Full optimization
    public_picks = load_public_pick_pcts()
    group_picks = load_group_picks()
    prize_structure = _load_prize_structure()

    opt = BracketOptimizer(sim, public_picks, group_picks, prize_structure)
    opt.compute_leverage_scores()
    bracket_a, bracket_b = opt.generate_portfolio()
    opt.save_brackets(bracket_a, bracket_b)

    pool = opt.evaluate_portfolio(bracket_a, bracket_b, n_pool_sims=25000)

    print("\n" + "=" * 70)
    print("  FINAL BRACKETS READY!")
    print("  Enter these into ESPN Tournament Challenge")
    print("=" * 70)


def step_market_only():
    """Quick market check - no re-simulation."""
    print("\n  Quick market odds update...")
    market = get_market_consensus()

    # Load existing simulation
    field = load_tournament_field()
    team_data = _load_team_data()

    if team_data:
        model = WinProbabilityModel(team_data)
        sim = BracketSimulator(model, field)
        # Quick sim just for comparison
        sim.simulate(n_sims=10000)
        _compare_model_vs_market(sim, market)


# ============================================================
#  Helper Functions
# ============================================================

def _integrate_market_data(team_data, market):
    """Add market-implied probabilities to team data."""
    if not market:
        return team_data

    # Fuzzy match market team names to our team data
    aliases = {
        "UConn": "Connecticut", "Connecticut": "UConn",
        "St. John's": "St John's", "St John's": "St. John's",
        "Miami": "Miami (FL)",
    }

    for market_team, prob in market.items():
        matched = None
        if market_team in team_data:
            matched = market_team
        elif market_team in aliases and aliases[market_team] in team_data:
            matched = aliases[market_team]
        else:
            # Substring match
            for td_team in team_data:
                if td_team == "__model_params__":
                    continue
                if market_team.lower() in td_team.lower() or td_team.lower() in market_team.lower():
                    matched = td_team
                    break

        if matched:
            team_data[matched]["market_champ_prob"] = prob / 100  # store as 0-1

    return team_data


def _compare_model_vs_market(sim, market):
    """
    Show where our model disagrees with the market.
    Big disagreements are either edges or blind spots.
    """
    if not market:
        return

    print(f"\n{'='*80}")
    print(f"  MODEL vs MARKET - Where Do We Disagree?")
    print(f"{'='*80}")
    print(f"  {'Team':<22} {'Model':>7} {'Market':>7} {'Diff':>7} {'Signal':<20}")
    print(f"  {'-'*22} {'-'*7} {'-'*7} {'-'*7} {'-'*20}")

    comparisons = []
    for team, probs in sim.advancement_probs.items():
        model_champ = probs.get(6, 0) * 100  # as percentage
        market_prob = market.get(team, 0)

        if model_champ > 0.5 or market_prob > 0.5:
            diff = model_champ - market_prob
            comparisons.append((team, model_champ, market_prob, diff))

    comparisons.sort(key=lambda x: abs(x[3]), reverse=True)

    for team, model_p, market_p, diff in comparisons[:20]:
        if abs(diff) < 0.3:
            signal = "≈ Aligned"
        elif diff > 2:
            signal = "⬆ MODEL LIKES MORE"
        elif diff > 0.5:
            signal = "↑ Model slightly higher"
        elif diff < -2:
            signal = "⬇ MARKET LIKES MORE"
        elif diff < -0.5:
            signal = "↓ Market slightly higher"
        else:
            signal = "≈ Close"

        print(f"  {team:<22} {model_p:>6.1f}% {market_p:>6.1f}% {diff:>+6.1f}% {signal}")

    print(f"\n  Key: Large positive diff = our model sees value the market doesn't")
    print(f"       Large negative diff = market knows something we might be missing")


def _load_team_data():
    """Load existing merged team data."""
    path = os.path.join(DATA_DIR, "merged_team_data.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def _load_prize_structure():
    """Load prize structure or return default."""
    path = os.path.join(DATA_DIR, "prize_structure.json")
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
            # Convert string keys to int
            return {int(k): v for k, v in data.items()}
    # Default: equal weight for top 3
    return {1: 1.0, 2: 1.0, 3: 1.0}


def set_prize_structure(first, second, third):
    """Set the prize structure for optimization."""
    structure = {"1": first, "2": second, "3": third}
    path = os.path.join(DATA_DIR, "prize_structure.json")
    with open(path, "w") as f:
        json.dump(structure, f, indent=2)
    print(f"[UPDATE] Prize structure set: 1st=${first}, 2nd=${second}, 3rd=${third}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NCAA Bracket Optimizer - Update")
    parser.add_argument("--step", required=True,
                        choices=["initial", "market", "picks", "final", "market-only"],
                        help="Which update step to run")
    parser.add_argument("--sims", type=int, default=500000,
                        help="Number of tournament simulations")
    parser.add_argument("--prize", nargs=3, type=float, default=None,
                        help="Prize structure: first second third (e.g., --prize 100 50 25)")

    args = parser.parse_args()

    if args.prize:
        set_prize_structure(args.prize[0], args.prize[1], args.prize[2])

    if args.step == "initial":
        step_initial(args.sims)
    elif args.step == "market":
        step_market(args.sims)
    elif args.step == "picks":
        step_picks(args.sims)
    elif args.step == "final":
        step_final(args.sims)
    elif args.step == "market-only":
        step_market_only()
