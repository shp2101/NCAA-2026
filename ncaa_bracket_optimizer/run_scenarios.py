#!/usr/bin/env python3
"""
Run 5 bracket optimization scenarios:

1. ORIGINAL (50-person pool, top 3 paid) — already computed
2. MOST CONTRARIAN — max leverage, willing to sacrifice expected points for uniqueness
3. 30-PERSON POOL — smaller pool = less need for extreme contrarian plays
4. 40-PERSON POOL — middle ground
5. ORIGINAL 50-person re-run for direct comparison

All scenarios share the same 100K simulation base (same advancement probs),
but differ in optimization strategy and pool simulation.
"""

import sys
import os
import json
import copy

sys.path.insert(0, os.path.dirname(__file__))

from config import ESPN_SCORING, SEED_MATCHUPS_R64, LEVERAGE_WEIGHT
from model_v2 import WinProbabilityModelV2 as WinProbabilityModel
from simulator import BracketSimulator
from optimizer import BracketOptimizer

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


class ContrianOptimizer(BracketOptimizer):
    """
    Maximum contrarian optimizer — picks the highest-leverage team
    at every decision point. Designed to win big or lose big.
    Champion is the team with highest leverage ratio (model prob / public pick pct).
    """

    def generate_portfolio(self):
        """Override: pick champions purely by leverage, not contest EV."""
        print("\n[CONTRARIAN] Generating MAX LEVERAGE bracket portfolio...")

        # Rank by pure leverage (model_prob / public_pct), not contest EV
        champ_candidates = []
        for team, rounds in self.leverage_scores.items():
            champ_data = rounds.get(6, {})
            if champ_data.get("model_prob", 0) > 0.02:  # at least 2% model prob
                champ_candidates.append({
                    "team": team,
                    "model_prob": champ_data["model_prob"],
                    "public_pct": champ_data["public_pct"],
                    "leverage": champ_data["leverage"],
                    "contest_ev": champ_data["contest_ev"],
                })

        # Sort by pure leverage ratio (most undervalued by public)
        champ_candidates.sort(key=lambda x: x["leverage"], reverse=True)

        print(f"\n  Top Champions by PURE LEVERAGE (most contrarian):")
        for i, c in enumerate(champ_candidates[:8]):
            print(f"    {i+1}. {c['team']:<22} Model: {c['model_prob']:>5.1%}  "
                  f"Public: {c['public_pct']:>5.1%}  "
                  f"Leverage: {c['leverage']:>5.2f}x")

        # Bracket A: highest leverage champion
        best_champ = champ_candidates[0]
        print(f"\n[CONTRARIAN] Bracket A champion: {best_champ['team']} "
              f"(leverage: {best_champ['leverage']:.2f}x)")

        bracket_a = self._generate_bracket(
            upset_threshold=0.20,  # very aggressive upset picking
            leverage_mult=LEVERAGE_WEIGHT * 2.0,  # extreme leverage weighting
            label="BRACKET A (Max Contrarian)",
            forced_champion=best_champ["team"],
        )

        # Bracket B: 2nd highest leverage from different region
        a_region = self._get_team_region(best_champ["team"])
        alt_champ = None
        for cand in champ_candidates[1:]:
            if self._get_team_region(cand["team"]) != a_region:
                alt_champ = cand
                break
        if alt_champ is None and len(champ_candidates) > 1:
            alt_champ = champ_candidates[1]

        if alt_champ:
            print(f"[CONTRARIAN] Bracket B champion: {alt_champ['team']} "
                  f"(leverage: {alt_champ['leverage']:.2f}x)")

            forced_ff = self._find_alternative_ff_picks(
                bracket_a, alt_champ["team"], best_champ["team"]
            )

            bracket_b = self._generate_bracket(
                upset_threshold=0.18,  # even more aggressive
                leverage_mult=LEVERAGE_WEIGHT * 2.5,  # extreme
                label="BRACKET B (Max Contrarian Alt)",
                forced_champion=alt_champ["team"],
                avoid_picks=self._get_key_picks(bracket_a),
                forced_regional_winners=forced_ff,
            )
        else:
            bracket_b = bracket_a  # fallback

        self._report_portfolio_divergence(bracket_a, bracket_b)
        return bracket_a, bracket_b


def run_all_scenarios():
    """Run all 5 scenarios."""

    # Load data
    with open(os.path.join(DATA_DIR, "merged_team_data.json")) as f:
        team_data = json.load(f)
    with open(os.path.join(DATA_DIR, "tournament_field.json")) as f:
        field = json.load(f)

    # Build model
    model = WinProbabilityModel(team_data)

    # Run shared simulation (100K for speed — same probabilities for all scenarios)
    sim = BracketSimulator(model, field)
    results = sim.simulate(100000)

    sim.print_advancement_table(top_n=15)

    scenarios = {}

    # ================================================================
    # SCENARIO 1: Original (50-person pool, top 3)
    # ================================================================
    print("\n" + "=" * 70)
    print("  SCENARIO 1: ORIGINAL (50-person pool, top 3 paid)")
    print("=" * 70)

    opt1 = BracketOptimizer(sim, None, None, {1: 1.0, 2: 1.0, 3: 1.0})
    opt1.compute_leverage_scores()
    a1, b1 = opt1.generate_portfolio()
    pool1 = opt1.evaluate_portfolio(a1, b1, n_pool_sims=10000)
    opt1.save_brackets(a1, b1)

    scenarios["1_original_50"] = {
        "label": "Original (50-person pool)",
        "pool_size": 50,
        "bracket_a": a1, "bracket_b": b1,
        "results": pool1,
    }

    # ================================================================
    # SCENARIO 2: MOST CONTRARIAN (50-person pool, max leverage)
    # ================================================================
    print("\n" + "=" * 70)
    print("  SCENARIO 2: MOST CONTRARIAN (max leverage picks)")
    print("=" * 70)

    opt2 = ContrianOptimizer(sim, None, None, {1: 1.0, 2: 1.0, 3: 1.0})
    opt2.compute_leverage_scores()
    a2, b2 = opt2.generate_portfolio()
    pool2 = opt2.evaluate_portfolio(a2, b2, n_pool_sims=10000)

    scenarios["2_contrarian"] = {
        "label": "Most Contrarian",
        "pool_size": 50,
        "bracket_a": a2, "bracket_b": b2,
        "results": pool2,
    }

    # ================================================================
    # SCENARIO 3: 30-PERSON POOL
    # ================================================================
    print("\n" + "=" * 70)
    print("  SCENARIO 3: 30-PERSON POOL (top 3 paid)")
    print("=" * 70)

    # Monkey-patch pool size for this scenario
    import config
    orig_pool_size = config.POOL_SIZE
    config.POOL_SIZE = 30

    # Re-import optimizer to pick up new pool size
    opt3 = BracketOptimizer(sim, None, None, {1: 1.0, 2: 1.0, 3: 1.0})
    opt3.compute_leverage_scores()
    a3, b3 = opt3.generate_portfolio()
    pool3 = opt3.evaluate_portfolio(a3, b3, n_pool_sims=10000)

    scenarios["3_pool_30"] = {
        "label": "30-Person Pool",
        "pool_size": 30,
        "bracket_a": a3, "bracket_b": b3,
        "results": pool3,
    }

    config.POOL_SIZE = orig_pool_size  # restore

    # ================================================================
    # SCENARIO 4: 40-PERSON POOL
    # ================================================================
    print("\n" + "=" * 70)
    print("  SCENARIO 4: 40-PERSON POOL (top 3 paid)")
    print("=" * 70)

    config.POOL_SIZE = 40

    opt4 = BracketOptimizer(sim, None, None, {1: 1.0, 2: 1.0, 3: 1.0})
    opt4.compute_leverage_scores()
    a4, b4 = opt4.generate_portfolio()
    pool4 = opt4.evaluate_portfolio(a4, b4, n_pool_sims=10000)

    scenarios["4_pool_40"] = {
        "label": "40-Person Pool",
        "pool_size": 40,
        "bracket_a": a4, "bracket_b": b4,
        "results": pool4,
    }

    config.POOL_SIZE = orig_pool_size  # restore

    # ================================================================
    # SCENARIO 5: 50-PERSON POOL (duplicate run for comparison stability)
    # ================================================================
    print("\n" + "=" * 70)
    print("  SCENARIO 5: 50-PERSON POOL (re-run for comparison)")
    print("=" * 70)

    config.POOL_SIZE = 50

    opt5 = BracketOptimizer(sim, None, None, {1: 1.0, 2: 1.0, 3: 1.0})
    opt5.compute_leverage_scores()
    a5, b5 = opt5.generate_portfolio()
    pool5 = opt5.evaluate_portfolio(a5, b5, n_pool_sims=10000)

    scenarios["5_pool_50_v2"] = {
        "label": "50-Person Pool (v2)",
        "pool_size": 50,
        "bracket_a": a5, "bracket_b": b5,
        "results": pool5,
    }

    # ================================================================
    # COMPARISON TABLE
    # ================================================================
    print("\n\n")
    print("=" * 100)
    print("  SCENARIO COMPARISON — ALL 5 BRACKETS")
    print("=" * 100)

    header = (f"  {'Scenario':<30} {'Pool':>4} {'Champ A':<12} {'Champ B':<12} "
              f"{'Win%':>6} {'Top3%':>6} {'Win Edge':>8} {'T3 Edge':>8} "
              f"{'FF Overlap':>10}")
    print(header)
    print("  " + "-" * 96)

    for key in sorted(scenarios.keys()):
        sc = scenarios[key]
        r = sc["results"]
        a = sc["bracket_a"]
        b = sc["bracket_b"]
        ff_a = set(a.get("final_four", []))
        ff_b = set(b.get("final_four", []))
        overlap = len(ff_a & ff_b)
        pool = sc["pool_size"]
        random_win = 1 / pool
        random_t3 = 3 / pool

        print(f"  {sc['label']:<30} {pool:>4} {a['champion']:<12} {b['champion']:<12} "
              f"{r['either_win_pct']:>5.1%} {r['either_top3_pct']:>5.1%} "
              f"{r['either_win_pct']/random_win:>7.1f}x {r['either_top3_pct']/random_t3:>7.1f}x "
              f"{overlap}/4")

    print("=" * 100)

    # Detail each scenario's brackets
    print("\n\n")
    print("=" * 100)
    print("  BRACKET DETAILS BY SCENARIO")
    print("=" * 100)

    for key in sorted(scenarios.keys()):
        sc = scenarios[key]
        a = sc["bracket_a"]
        b = sc["bracket_b"]

        print(f"\n  --- {sc['label']} (Pool: {sc['pool_size']}) ---")
        print(f"  Bracket A: {a['champion']} champion")
        print(f"    FF: {', '.join(a.get('final_four', []))}")
        print(f"    Expected Points: {a.get('expected_points', 0):.0f}")

        print(f"  Bracket B: {b['champion']} champion")
        print(f"    FF: {', '.join(b.get('final_four', []))}")
        print(f"    Expected Points: {b.get('expected_points', 0):.0f}")

        r = sc["results"]
        print(f"  Results: Win {r['either_win_pct']:.1%} | Top 3 {r['either_top3_pct']:.1%} | "
              f"Avg Best Pos {r['avg_best_position']:.1f} | Median {r['median_best_position']:.0f}")

    # Save all scenarios
    save_data = {}
    for key, sc in scenarios.items():
        save_data[key] = {
            "label": sc["label"],
            "pool_size": sc["pool_size"],
            "bracket_a_champion": sc["bracket_a"]["champion"],
            "bracket_a_ff": sc["bracket_a"].get("final_four", []),
            "bracket_a_expected_pts": sc["bracket_a"].get("expected_points", 0),
            "bracket_b_champion": sc["bracket_b"]["champion"],
            "bracket_b_ff": sc["bracket_b"].get("final_four", []),
            "bracket_b_expected_pts": sc["bracket_b"].get("expected_points", 0),
            "results": sc["results"],
        }

    with open(os.path.join(DATA_DIR, "all_scenarios.json"), "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  All scenarios saved to {DATA_DIR}/all_scenarios.json")

    return scenarios


if __name__ == "__main__":
    run_all_scenarios()
