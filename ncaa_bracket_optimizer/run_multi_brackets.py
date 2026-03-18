#!/usr/bin/env python3
"""
Generate multiple bracket categories from the same simulation:
1. Bracket C + D (20-person pool)
2. Bracket E (Extra Contrarian / Chaos)
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(__file__))

from model_v2 import WinProbabilityModelV2 as WinProbabilityModel
from simulator import BracketSimulator
from data_collector_v2 import load_historical_tournament_data

# We need to monkey-patch POOL_SIZE in optimizer since it imports at module level
import config
import optimizer as opt_module

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def build_merged_team_data():
    sr_path = os.path.join(DATA_DIR, "sports_reference_2026.json")
    with open(sr_path) as f:
        sr_data = json.load(f)

    market_path = os.path.join(DATA_DIR, "market_consensus.json")
    with open(market_path) as f:
        market_data = json.load(f)

    historical = load_historical_tournament_data()

    team_data = {}
    for team_name, sr in sr_data.items():
        team_data[team_name] = {
            "name": team_name, "adj_oe": sr.get("adj_oe"), "adj_de": sr.get("adj_de"),
            "eff_margin": sr.get("eff_margin"), "barthag": sr.get("barthag"),
            "srs": sr.get("srs"), "sos": sr.get("sos"), "record": sr.get("record"),
            "conference": sr.get("conference"), "seed": sr.get("seed"), "region": sr.get("region"),
        }

    aliases = {
        "UConn": "Connecticut", "Connecticut": "UConn",
        "St. John's": "St John's", "St John's": "St. John's",
        "Miami": "Miami (FL)", "Miami (FL)": "Miami",
        "Saint Louis": "St. Louis", "St. Louis": "Saint Louis",
        "Norhtern Iowa": "Northern Iowa",
    }

    for market_team, prob in market_data.items():
        matched = None
        if market_team in team_data:
            matched = market_team
        elif market_team in aliases and aliases[market_team] in team_data:
            matched = aliases[market_team]
        else:
            for td_team in team_data:
                if market_team.lower() in td_team.lower() or td_team.lower() in market_team.lower():
                    matched = td_team
                    break
        if matched:
            team_data[matched]["market_champ_prob"] = prob / 100

    team_data["__model_params__"] = {
        "calibrated_logistic_coeff": historical.get("calibrated_logistic_coeff", 0.17),
        "historical_rates": {}
    }
    return team_data


def run_multi_brackets(n_sims=500000):
    print("=" * 60)
    print("  MULTI-BRACKET GENERATOR")
    print("=" * 60)

    team_data = build_merged_team_data()
    print(f"  Teams loaded: {len(team_data) - 1}")

    field_path = os.path.join(DATA_DIR, "tournament_field.json")
    with open(field_path) as f:
        field = json.load(f)

    model = WinProbabilityModel(team_data)
    sim = BracketSimulator(model, field)
    sim.simulate(n_sims)
    sim.print_advancement_table(top_n=10)

    results = {}

    # ============================================================
    # Category 1: 20-person pool
    # ============================================================
    print("\n" + "=" * 60)
    print("  CATEGORY: 20-PERSON POOL")
    print("=" * 60)

    # Monkey-patch POOL_SIZE in the optimizer module
    opt_module.POOL_SIZE = 20

    from optimizer import BracketOptimizer
    opt_20 = BracketOptimizer(sim, None, None, {1: 1.0, 2: 1.0, 3: 1.0})
    opt_20.compute_leverage_scores()
    bracket_c, bracket_d = opt_20.generate_portfolio()

    bracket_c["label"] = "BRACKET C — 20-Person Pool (Model Favorite)"
    bracket_d["label"] = "BRACKET D — 20-Person Pool (Contrarian)"

    pool_20_results = opt_20.evaluate_portfolio(bracket_c, bracket_d, n_pool_sims=10000)

    results["pool_20"] = {
        "bracket_c": bracket_c,
        "bracket_d": bracket_d,
        "pool_results": pool_20_results,
    }

    # ============================================================
    # Category 2: Extra Contrarian (Chaos Bracket)
    # ============================================================
    print("\n" + "=" * 60)
    print("  CATEGORY: EXTRA CONTRARIAN (CHAOS BRACKET)")
    print("=" * 60)

    opt_module.POOL_SIZE = 30
    opt_chaos = BracketOptimizer(sim, None, None, {1: 1.0, 2: 1.0, 3: 1.0})
    opt_chaos.compute_leverage_scores()

    # Find best non-Duke, non-Michigan champion
    champ_candidates = []
    for team, rounds in opt_chaos.leverage_scores.items():
        champ_data = rounds.get(6, {})
        if champ_data.get("model_prob", 0) > 0.01:
            champ_candidates.append({
                "team": team,
                "model_prob": champ_data["model_prob"],
                "leverage": champ_data["leverage"],
                "contest_ev": champ_data["contest_ev"],
            })
    champ_candidates.sort(key=lambda x: x["contest_ev"], reverse=True)

    chaos_champ = None
    for c in champ_candidates:
        if c["team"] not in ("Duke", "Michigan"):
            chaos_champ = c
            break

    print(f"\n  Chaos champion: {chaos_champ['team']} "
          f"(model: {chaos_champ['model_prob']:.1%}, leverage: {chaos_champ['leverage']:.2f}x)")

    # Generate with moderately aggressive settings (not so extreme leverage_mult
    # that it breaks the EV calculation, but still very contrarian)
    bracket_e = opt_chaos._generate_bracket(
        upset_threshold=0.18,       # Pick upsets at just 18% probability
        leverage_mult=1.8,          # Strong contrarian boost (normal is 0.8-1.5)
        label="BRACKET E — Extra Contrarian (Chaos)",
        forced_champion=chaos_champ["team"],
    )

    # Evaluate chaos bracket paired with bracket A in a 30-person pool
    brackets_path = os.path.join(DATA_DIR, "final_brackets.json")
    with open(brackets_path) as f:
        saved = json.load(f)
    bracket_a_saved = saved["bracket_a"]

    chaos_pool = opt_chaos.evaluate_portfolio(bracket_a_saved, bracket_e, n_pool_sims=10000)

    results["chaos"] = {
        "bracket_e": bracket_e,
        "pool_results": chaos_pool,
    }

    # Restore
    opt_module.POOL_SIZE = 30

    # Save
    output_path = os.path.join(DATA_DIR, "multi_bracket_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"  MULTI-BRACKET SUMMARY")
    print(f"{'=' * 60}")

    print(f"\n  20-Person Pool:")
    print(f"    Bracket C: {bracket_c['champion']} (EV: {bracket_c['expected_points']:.0f})")
    print(f"    Bracket D: {bracket_d['champion']} (EV: {bracket_d['expected_points']:.0f})")
    print(f"    C FF: {', '.join(bracket_c['final_four'])}")
    print(f"    D FF: {', '.join(bracket_d['final_four'])}")
    print(f"    Either wins: {pool_20_results['either_win_pct']:.1%}")
    print(f"    Either top 3: {pool_20_results['either_top3_pct']:.1%}")

    print(f"\n  Extra Contrarian:")
    print(f"    Bracket E: {bracket_e['champion']} (EV: {bracket_e['expected_points']:.0f})")
    print(f"    E FF: {', '.join(bracket_e['final_four'])}")
    print(f"    Paired w/ A wins: {chaos_pool['either_win_pct']:.1%}")
    print(f"    Paired w/ A top 3: {chaos_pool['either_top3_pct']:.1%}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sims", type=int, default=500000)
    args = parser.parse_args()
    run_multi_brackets(args.sims)
