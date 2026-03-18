"""
NCAA Bracket Optimizer - Contest EV Optimizer (V2)
==================================================

Upgrades over V1:
1. Top-3 finish optimization (not just 1st place)
2. Joint portfolio optimization (Bracket A + B cover different scenarios)
3. Configurable prize structure with EV calculation
4. Support for real ESPN public pick % and group-specific picks
5. Higher-volume pool contest simulation (25K+)

Key insight: With 2 entries in a 50-person pool paying top 3,
the optimal strategy is NOT two independent brackets. It's a
coordinated portfolio where:
  - Bracket A: "Model Favorite" — best expected points with moderate leverage
  - Bracket B: "Contrarian Upside" — different champion/FF, high leverage picks
  - They MUST diverge on champion and at least 1-2 Final Four teams
"""

import numpy as np
import json
import os
from collections import defaultdict
from config import (
    ESPN_SCORING, POOL_SIZE, NUM_BRACKET_CANDIDATES,
    SAFE_BRACKET_UPSET_THRESHOLD, SWING_BRACKET_UPSET_THRESHOLD,
    LEVERAGE_WEIGHT, SEED_MATCHUPS_R64,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


class BracketOptimizer:
    """
    Optimizes a portfolio of 2 brackets for maximum EV in a pool contest
    with top-3 payouts.
    """

    def __init__(self, simulator, public_picks=None, group_picks=None, prize_structure=None):
        """
        Parameters:
            simulator: BracketSimulator (already run)
            public_picks: dict of {team: {round: pick_pct}} or None
            group_picks: dict of group member picks or None
            prize_structure: dict like {1: 100, 2: 50, 3: 25} or None
                           If None, uses equal weight for top 3
        """
        self.sim = simulator
        self.model = simulator.model
        self.field = simulator.field
        self.adv_probs = dict(simulator.advancement_probs)
        self.public_picks = public_picks
        self.group_picks = group_picks

        # Prize structure: maps finishing position to payout
        # Default: equal weight for top 3 (optimize for P(top 3))
        self.prize_structure = prize_structure or {1: 1.0, 2: 1.0, 3: 1.0}

        # Use provided ESPN public picks, or estimate from seed data
        if self.public_picks is not None:
            print(f"[OPT] Using REAL ESPN public pick data ({len(self.public_picks)} teams)")
            # Fill in any missing teams with seed-based estimates
            for region in self.field:
                for seed_str, team in self.field[region].items():
                    if team not in self.public_picks:
                        seed = int(seed_str)
                        self.public_picks[team] = {
                            "r64": self._chalk_pick_rate(seed, 1),
                            "r32": self._chalk_pick_rate(seed, 2),
                            "s16": self._chalk_pick_rate(seed, 3),
                            "e8": self._chalk_pick_rate(seed, 4),
                            "f4": self._chalk_pick_rate(seed, 5),
                            "champ": self._chalk_pick_rate(seed, 6),
                        }
        else:
            self.public_picks = self._estimate_public_picks()

        # If we have group picks, blend them with national estimates
        if self.group_picks:
            self._integrate_group_picks()

    def _estimate_public_picks(self):
        """Estimate public pick % from historical seed tendencies."""
        print("[OPT] Estimating public pick percentages from seed data...")
        picks = {}
        for region in self.field:
            for seed_str, team in self.field[region].items():
                seed = int(seed_str)
                picks[team] = {
                    "r64": self._chalk_pick_rate(seed, 1),
                    "r32": self._chalk_pick_rate(seed, 2),
                    "s16": self._chalk_pick_rate(seed, 3),
                    "e8": self._chalk_pick_rate(seed, 4),
                    "f4": self._chalk_pick_rate(seed, 5),
                    "champ": self._chalk_pick_rate(seed, 6),
                }
        return picks

    def _chalk_pick_rate(self, seed, round_num):
        """Estimated public pick rate by seed and round."""
        public_rates = {
            1: {1: 0.99, 2: 0.92, 3: 0.75, 4: 0.55, 5: 0.30, 6: 0.15},
            2: {1: 0.95, 2: 0.80, 3: 0.55, 4: 0.30, 5: 0.12, 6: 0.05},
            3: {1: 0.90, 2: 0.65, 3: 0.35, 4: 0.15, 5: 0.05, 6: 0.02},
            4: {1: 0.85, 2: 0.55, 3: 0.25, 4: 0.10, 5: 0.03, 6: 0.01},
            5: {1: 0.70, 2: 0.40, 3: 0.15, 4: 0.05, 5: 0.02, 6: 0.005},
            6: {1: 0.68, 2: 0.35, 3: 0.12, 4: 0.04, 5: 0.015, 6: 0.004},
            7: {1: 0.60, 2: 0.28, 3: 0.08, 4: 0.03, 5: 0.01, 6: 0.003},
            8: {1: 0.50, 2: 0.18, 3: 0.05, 4: 0.02, 5: 0.005, 6: 0.001},
            9: {1: 0.50, 2: 0.18, 3: 0.05, 4: 0.02, 5: 0.005, 6: 0.001},
            10: {1: 0.40, 2: 0.15, 3: 0.05, 4: 0.02, 5: 0.005, 6: 0.001},
            11: {1: 0.32, 2: 0.12, 3: 0.05, 4: 0.02, 5: 0.008, 6: 0.002},
            12: {1: 0.30, 2: 0.10, 3: 0.03, 4: 0.01, 5: 0.003, 6: 0.001},
            13: {1: 0.15, 2: 0.04, 3: 0.01, 4: 0.003, 5: 0.001, 6: 0.0003},
            14: {1: 0.10, 2: 0.02, 3: 0.005, 4: 0.001, 5: 0.0003, 6: 0.0001},
            15: {1: 0.05, 2: 0.01, 3: 0.002, 4: 0.0005, 5: 0.0001, 6: 0.00003},
            16: {1: 0.01, 2: 0.002, 3: 0.0003, 4: 0.0001, 5: 0.00002, 6: 0.000005},
        }
        return public_rates.get(seed, {}).get(round_num, 0.01)

    def _integrate_group_picks(self):
        """
        If we have actual group pick data, blend it with national estimates.
        Group-specific data is more valuable since that's who we're competing against.
        """
        print("[OPT] Integrating group-specific pick data...")
        # TODO: When group data is available, compute per-team pick rates
        # from actual group brackets and weight them 70/30 vs national data
        pass

    # ============================================================
    #  Leverage Computation
    # ============================================================

    def compute_leverage_scores(self):
        """Compute leverage = Model_Prob / Public_Pick_Pct for each team/round."""
        round_map = {1: "r64", 2: "r32", 3: "s16", 4: "e8", 5: "f4", 6: "champ"}

        leverage = {}
        for team in self.adv_probs:
            leverage[team] = {}
            public = self.public_picks.get(team, {})

            for round_num in range(1, 7):
                model_prob = self.adv_probs[team].get(round_num, 0)
                public_pct = max(public.get(round_map[round_num], 0.01), 0.001)

                lev = model_prob / public_pct
                # Cap leverage to prevent extreme contrarian distortion
                # In a 20-30 person pool, >5x leverage provides diminishing returns
                capped_lev = min(lev, 5.0)
                leverage[team][round_num] = {
                    "model_prob": model_prob,
                    "public_pct": public_pct,
                    "leverage": capped_lev,
                    "ev_points": model_prob * ESPN_SCORING[round_num],
                    "contest_ev": model_prob * ESPN_SCORING[round_num] * (capped_lev ** LEVERAGE_WEIGHT),
                }

        self.leverage_scores = leverage
        self._print_leverage_report()
        return leverage

    def _print_leverage_report(self):
        """Print teams with highest leverage in later rounds."""
        print(f"\n{'='*80}")
        print(f"  LEVERAGE REPORT - Best Contrarian Picks")
        print(f"{'='*80}")

        round_names = {3: "Sweet 16", 4: "Elite 8", 5: "Final Four", 6: "Champion"}
        for round_num, round_name in round_names.items():
            print(f"\n  --- {round_name} ---")
            teams = sorted(
                [(t, d[round_num]) for t, d in self.leverage_scores.items()
                 if d[round_num]["model_prob"] > 0.02],
                key=lambda x: x[1]["contest_ev"], reverse=True,
            )
            for team, data in teams[:8]:
                print(f"  {team:<25} Model: {data['model_prob']:>5.1%}  "
                      f"Public: {data['public_pct']:>5.1%}  "
                      f"Leverage: {data['leverage']:>5.2f}x  "
                      f"Contest EV: {data['contest_ev']:>7.1f}")

    # ============================================================
    #  Portfolio Bracket Generation
    # ============================================================

    def generate_portfolio(self):
        """
        Generate an optimized portfolio of 2 brackets.

        Bracket A ("Model Favorite"):
          - Best champion by contest EV (model prob × leverage)
          - Moderate leverage picks throughout
          - Designed to consistently finish top 3

        Bracket B ("Contrarian Upside"):
          - DIFFERENT champion than Bracket A
          - Higher leverage throughout, especially in late rounds
          - Designed to win the pool outright when chalk busts

        The two brackets are constrained to diverge on:
          - Champion (mandatory)
          - At least 1 Final Four team (mandatory)
          - Several Sweet 16 / Elite 8 picks
        """
        print("\n[OPT] Generating optimized bracket portfolio...")

        # Step 1: Rank champion candidates by contest EV
        champ_candidates = self._rank_champion_candidates()

        # Step 2: Generate Bracket A with top champion
        best_champ = champ_candidates[0]
        print(f"\n[OPT] Bracket A champion: {best_champ['team']} "
              f"(model: {best_champ['model_prob']:.1%}, "
              f"leverage: {best_champ['leverage']:.2f}x)")

        bracket_a = self._generate_bracket(
            upset_threshold=SAFE_BRACKET_UPSET_THRESHOLD,
            leverage_mult=0.8,  # moderate leverage
            label="BRACKET A (Model Favorite)",
            forced_champion=best_champ["team"],
        )

        # Step 3: Generate Bracket B with different champion
        # Pick the best champion candidate that ISN'T Bracket A's champion
        # and ideally from a different region
        bracket_a_champ_region = self._get_team_region(best_champ["team"])
        alt_champ = None
        for cand in champ_candidates[1:]:
            # Prefer different region for maximum divergence
            cand_region = self._get_team_region(cand["team"])
            if cand_region != bracket_a_champ_region:
                alt_champ = cand
                break
        if alt_champ is None and len(champ_candidates) > 1:
            alt_champ = champ_candidates[1]

        if alt_champ:
            print(f"[OPT] Bracket B champion: {alt_champ['team']} "
                  f"(model: {alt_champ['model_prob']:.1%}, "
                  f"leverage: {alt_champ['leverage']:.2f}x)")

            # Find best alternative FF picks for divergence
            forced_ff = self._find_alternative_ff_picks(
                bracket_a, alt_champ["team"], best_champ["team"]
            )

            # Use higher leverage multiplier for more contrarian picks
            bracket_b = self._generate_bracket(
                upset_threshold=SWING_BRACKET_UPSET_THRESHOLD,
                leverage_mult=LEVERAGE_WEIGHT * 1.3,  # extra contrarian boost
                label="BRACKET B (Contrarian Upside)",
                forced_champion=alt_champ["team"],
                avoid_picks=self._get_key_picks(bracket_a),
                forced_regional_winners=forced_ff,
            )
        else:
            bracket_b = self._generate_bracket(
                upset_threshold=SWING_BRACKET_UPSET_THRESHOLD,
                leverage_mult=LEVERAGE_WEIGHT * 1.3,
                label="BRACKET B (Contrarian Upside)",
            )

        # Step 4: Report portfolio divergence
        self._report_portfolio_divergence(bracket_a, bracket_b)

        return bracket_a, bracket_b

    def _rank_champion_candidates(self):
        """Rank teams by champion contest EV."""
        candidates = []
        for team, rounds in self.leverage_scores.items():
            champ_data = rounds.get(6, {})
            if champ_data.get("model_prob", 0) > 0.01:
                candidates.append({
                    "team": team,
                    "model_prob": champ_data["model_prob"],
                    "public_pct": champ_data["public_pct"],
                    "leverage": champ_data["leverage"],
                    "contest_ev": champ_data["contest_ev"],
                    "ev_points": champ_data["ev_points"],
                })

        candidates.sort(key=lambda x: x["contest_ev"], reverse=True)

        print(f"\n  Top Champion Candidates (by Contest EV):")
        for i, c in enumerate(candidates[:8]):
            print(f"    {i+1}. {c['team']:<22} Model: {c['model_prob']:>5.1%}  "
                  f"Public: {c['public_pct']:>5.1%}  "
                  f"Leverage: {c['leverage']:>5.2f}x  "
                  f"Contest EV: {c['contest_ev']:>7.1f}")

        return candidates

    def _get_team_region(self, team_name):
        """Find which region a team is in."""
        for region in self.field:
            for seed, team in self.field[region].items():
                if team == team_name:
                    return region
        return None

    def _get_key_picks(self, bracket):
        """Extract key picks from a bracket (all late-round picks) to avoid in the other."""
        avoid_teams = set(bracket.get("final_four", []))
        # Also include S16 and E8 picks for broader divergence
        for region in self.field:
            for round_num in [3, 4]:  # S16 and E8
                picks = bracket.get("rounds", {}).get(region, {}).get(round_num, [])
                avoid_teams.update(picks)
        return {
            "champion": bracket.get("champion"),
            "final_four": avoid_teams,  # now includes S16 + E8 + FF teams
        }

    def _find_alternative_ff_picks(self, bracket_a, alt_champion, a_champion):
        """
        Find the best alternative regional winner for 1-2 regions where
        Bracket B should diverge from Bracket A.

        Strategy: In regions NOT containing Bracket B's champion,
        find the 2nd-best team by contest EV with decent leverage.
        Force that team as the regional winner in Bracket B.
        """
        alt_champ_region = self._get_team_region(alt_champion)
        a_champ_region = self._get_team_region(a_champion)
        a_ff = set(bracket_a.get("final_four", []))

        forced_ff = {}  # {region: team_name_to_force}

        # For each region, find the best alternative E8 pick
        for region in self.field:
            # Don't mess with the champion's region
            if region == alt_champ_region:
                continue

            # Get Bracket A's regional winner
            a_regional_winner = bracket_a["rounds"][region].get(4, [None])[0]
            if not a_regional_winner:
                continue

            # Find the 2nd-best team in this region by E8 contest EV
            best_alt = None
            best_alt_ev = 0

            for seed_str, team in self.field[region].items():
                if team == a_regional_winner:
                    continue  # skip the team Bracket A already picked

                e8_data = self.leverage_scores.get(team, {}).get(4, {})
                contest_ev = e8_data.get("contest_ev", 0)
                model_prob = e8_data.get("model_prob", 0)

                # Only consider teams with at least 5% E8 probability
                if model_prob >= 0.05 and contest_ev > best_alt_ev:
                    best_alt = team
                    best_alt_ev = contest_ev

            if best_alt:
                # Only force the alternative if it has at least half the EV
                a_e8_data = self.leverage_scores.get(a_regional_winner, {}).get(4, {})
                a_ev = a_e8_data.get("contest_ev", 0)

                if best_alt_ev >= a_ev * 0.25:  # at least 25% of the top EV
                    forced_ff[region] = best_alt
                    alt_data = self.leverage_scores.get(best_alt, {}).get(4, {})
                    print(f"[OPT] Bracket B divergence: {region} → {best_alt} "
                          f"(model E8: {alt_data.get('model_prob', 0):.1%}, "
                          f"leverage: {alt_data.get('leverage', 0):.2f}x) "
                          f"instead of {a_regional_winner}")

        # Limit to 1-2 forced swaps to keep bracket quality
        if len(forced_ff) > 2:
            # Keep the two swaps with highest EV
            sorted_swaps = sorted(
                forced_ff.items(),
                key=lambda x: self.leverage_scores.get(x[1], {}).get(4, {}).get("contest_ev", 0),
                reverse=True
            )
            forced_ff = dict(sorted_swaps[:2])

        return forced_ff

    def _generate_bracket(self, upset_threshold, leverage_mult, label="",
                          forced_champion=None, avoid_picks=None,
                          forced_regional_winners=None):
        """
        Generate a single bracket with optional constraints.

        If forced_champion is set, the bracket works backward to ensure
        that team wins the championship.

        If avoid_picks is set, the bracket tries to diverge from those picks
        in the Final Four and Elite 8.
        """
        bracket = {"label": label, "rounds": {}, "picks": [], "expected_points": 0}
        total_expected = 0
        avoid = avoid_picks or {}
        forced_rw = forced_regional_winners or {}

        for region in self.field:
            bracket["rounds"][region] = {}

            # Check if there's a forced regional winner for this region
            forced_rw_name = forced_rw.get(region)

            teams = []
            for higher_seed, lower_seed in SEED_MATCHUPS_R64:
                team_high = self.field[region].get(str(higher_seed))
                team_low = self.field[region].get(str(lower_seed))
                teams.append({"name": team_high, "seed": higher_seed})
                teams.append({"name": team_low, "seed": lower_seed})

            # Round of 64
            round_winners = []
            for i in range(0, len(teams), 2):
                forced = (self._must_advance(teams[i], teams[i+1], forced_champion)
                          or self._must_advance(teams[i], teams[i+1], forced_rw_name))
                winner = self._pick_winner(
                    teams[i], teams[i+1], round_num=1,
                    upset_threshold=upset_threshold,
                    leverage_mult=leverage_mult,
                    forced_team=forced,
                )
                round_winners.append(winner)
                total_expected += self.sim.get_expected_points_per_pick(winner["name"], 1)
            bracket["rounds"][region][1] = [w["name"] for w in round_winners]

            # Round of 32
            r32_winners = []
            for i in range(0, len(round_winners), 2):
                forced = (self._must_advance(round_winners[i], round_winners[i+1], forced_champion)
                          or self._must_advance(round_winners[i], round_winners[i+1], forced_rw_name))
                winner = self._pick_winner(
                    round_winners[i], round_winners[i+1], round_num=2,
                    upset_threshold=upset_threshold,
                    leverage_mult=leverage_mult,
                    forced_team=forced,
                )
                r32_winners.append(winner)
                total_expected += self.sim.get_expected_points_per_pick(winner["name"], 2)
            bracket["rounds"][region][2] = [w["name"] for w in r32_winners]

            # Sweet 16
            s16_winners = []
            for i in range(0, len(r32_winners), 2):
                forced = (self._must_advance(r32_winners[i], r32_winners[i+1], forced_champion)
                          or self._must_advance(r32_winners[i], r32_winners[i+1], forced_rw_name))
                winner = self._pick_winner(
                    r32_winners[i], r32_winners[i+1], round_num=3,
                    upset_threshold=upset_threshold,
                    leverage_mult=leverage_mult,
                    forced_team=forced,
                    avoid_team=avoid.get("final_four"),
                )
                s16_winners.append(winner)
                total_expected += self.sim.get_expected_points_per_pick(winner["name"], 3)
            bracket["rounds"][region][3] = [w["name"] for w in s16_winners]

            # Elite 8
            e8_forced = (self._must_advance(s16_winners[0], s16_winners[1], forced_champion)
                         or self._must_advance(s16_winners[0], s16_winners[1], forced_rw_name))

            elite_winner = self._pick_winner(
                s16_winners[0], s16_winners[1], round_num=4,
                upset_threshold=upset_threshold,
                leverage_mult=leverage_mult,
                forced_team=e8_forced,
                avoid_team=avoid.get("final_four"),
            )
            total_expected += self.sim.get_expected_points_per_pick(elite_winner["name"], 4)
            bracket["rounds"][region][4] = [elite_winner["name"]]
            bracket["picks"].append(elite_winner)

        # Final Four
        ff = bracket["picks"]

        semi1 = self._pick_winner(ff[0], ff[1], round_num=5,
                                   upset_threshold=upset_threshold,
                                   leverage_mult=leverage_mult,
                                   forced_team=self._must_advance(ff[0], ff[1], forced_champion))
        total_expected += self.sim.get_expected_points_per_pick(semi1["name"], 5)

        semi2 = self._pick_winner(ff[2], ff[3], round_num=5,
                                   upset_threshold=upset_threshold,
                                   leverage_mult=leverage_mult,
                                   forced_team=self._must_advance(ff[2], ff[3], forced_champion))
        total_expected += self.sim.get_expected_points_per_pick(semi2["name"], 5)

        champ = self._pick_winner(semi1, semi2, round_num=6,
                                   upset_threshold=upset_threshold,
                                   leverage_mult=leverage_mult,
                                   forced_team=self._must_advance(semi1, semi2, forced_champion))
        total_expected += self.sim.get_expected_points_per_pick(champ["name"], 6)

        bracket["final_four"] = [ff[i]["name"] for i in range(4)]
        bracket["finalist_1"] = semi1["name"]
        bracket["finalist_2"] = semi2["name"]
        bracket["champion"] = champ["name"]
        bracket["expected_points"] = total_expected

        self._print_bracket(bracket)
        return bracket

    def _must_advance(self, team_a, team_b, forced_champion):
        """If forced_champion is in this matchup, they must win."""
        if forced_champion is None:
            return None
        if team_a["name"] == forced_champion:
            return team_a
        if team_b["name"] == forced_champion:
            return team_b
        return None

    def _pick_winner(self, team_a, team_b, round_num, upset_threshold,
                     leverage_mult, forced_team=None, avoid_team=None):
        """
        Pick the winner of a matchup.

        Parameters:
            forced_team: if set, this team must win (for champion path)
            avoid_team: set of team names to slightly penalize (for portfolio divergence)
        """
        if forced_team is not None:
            return forced_team

        prob_a = self.model.win_probability(
            team_a["name"], team_b["name"],
            team_a.get("seed"), team_b.get("seed")
        )

        lev_a = self.leverage_scores.get(team_a["name"], {}).get(round_num, {})
        lev_b = self.leverage_scores.get(team_b["name"], {}).get(round_num, {})

        leverage_a = lev_a.get("leverage", 1.0)
        leverage_b = lev_b.get("leverage", 1.0)

        # Leverage matters more in later rounds where point values are higher
        # and differentiation matters more. In early rounds (R64-S16), mostly
        # pick the better team; leverage kicks in for E8+ picks.
        if round_num <= 2:
            round_leverage_boost = 0.3  # minimal leverage in R64/R32
        elif round_num == 3:
            round_leverage_boost = 0.6  # moderate in S16
        else:
            round_leverage_boost = 1.0 + (round_num - 4) * 0.2  # full leverage E8+

        score_a = prob_a * (leverage_a ** (leverage_mult * round_leverage_boost))
        score_b = (1 - prob_a) * (leverage_b ** (leverage_mult * round_leverage_boost))

        # Portfolio divergence: penalize picks that duplicate the other bracket
        # Stronger penalty in later rounds (where divergence matters most)
        if avoid_team and isinstance(avoid_team, set):
            # Scale penalty: 20% in S16, 35% in E8, 40% in FF
            avoid_penalty = max(0.35, 0.15 + round_num * 0.05)
            if team_a["name"] in avoid_team and team_b["name"] not in avoid_team:
                score_a *= (1 - avoid_penalty)
            elif team_b["name"] in avoid_team and team_a["name"] not in avoid_team:
                score_b *= (1 - avoid_penalty)

        if score_a >= score_b:
            return team_a
        else:
            if (1 - prob_a) >= upset_threshold or leverage_b > 2.0:
                return team_b
            else:
                return team_a

    def _report_portfolio_divergence(self, bracket_a, bracket_b):
        """Report how the two brackets differ."""
        print(f"\n{'='*60}")
        print(f"  PORTFOLIO DIVERGENCE REPORT")
        print(f"{'='*60}")

        champ_a = bracket_a["champion"]
        champ_b = bracket_b["champion"]
        print(f"  Champions: {champ_a} vs {champ_b} {'✓ DIFFERENT' if champ_a != champ_b else '✗ SAME'}")

        ff_a = set(bracket_a["final_four"])
        ff_b = set(bracket_b["final_four"])
        shared_ff = ff_a & ff_b
        unique_a = ff_a - ff_b
        unique_b = ff_b - ff_a
        print(f"  Final Four overlap: {len(shared_ff)}/4 shared")
        if unique_a:
            print(f"    Only in A: {', '.join(unique_a)}")
        if unique_b:
            print(f"    Only in B: {', '.join(unique_b)}")

        # Count total pick differences across all rounds
        total_diff = 0
        total_picks = 0
        for region in self.field:
            for round_num in range(1, 5):
                picks_a = set(bracket_a["rounds"][region].get(round_num, []))
                picks_b = set(bracket_b["rounds"][region].get(round_num, []))
                total_picks += len(picks_a)
                total_diff += len(picks_a ^ picks_b) // 2  # symmetric diff / 2

        print(f"  Total pick differences: {total_diff}/{total_picks} "
              f"({total_diff/total_picks*100:.0f}% divergence)")
        print(f"{'='*60}")

    # ============================================================
    #  Pool Contest Simulation (Top-3 Optimization)
    # ============================================================

    def evaluate_portfolio(self, bracket_a, bracket_b, n_pool_sims=25000):
        """
        Simulate the pool contest to estimate:
        - P(either bracket finishes top 3)
        - P(either bracket wins)
        - Expected payout given prize structure
        - Comparison to a random entrant

        This is the key function — it tells us our actual edge.
        """
        print(f"\n[OPT] Simulating pool contest...")
        print(f"  Iterations: {n_pool_sims:,}")
        print(f"  Pool size: {POOL_SIZE}")
        print(f"  Prize structure: {self.prize_structure}")

        # Track results
        a_positions = []  # finishing position of bracket A in each sim
        b_positions = []
        best_positions = []  # best of A or B

        a_wins = 0
        b_wins = 0
        either_wins = 0
        a_top3 = 0
        b_top3 = 0
        either_top3 = 0
        total_payout = 0.0

        for sim_num in range(n_pool_sims):
            if (sim_num + 1) % 5000 == 0:
                print(f"  Pool sim {sim_num + 1:,}/{n_pool_sims:,}...")

            # Simulate one tournament outcome
            sim_result = self.sim._simulate_tournament()

            # Score our brackets
            score_a = self._score_bracket(bracket_a, sim_result)
            score_b = self._score_bracket(bracket_b, sim_result)

            # Score all opponents
            opponent_scores = []
            for _ in range(POOL_SIZE - 2):  # -2 because we have 2 entries
                opp_bracket = self._generate_random_public_bracket()
                opp_score = self._score_bracket(opp_bracket, sim_result)
                opponent_scores.append(opp_score)

            # Determine positions
            all_scores = sorted(opponent_scores, reverse=True)

            pos_a = 1 + sum(1 for s in all_scores if s > score_a)
            pos_b = 1 + sum(1 for s in all_scores if s > score_b)
            # Our entries also compete with each other
            if score_b > score_a:
                pos_a += 1
            elif score_a > score_b:
                pos_b += 1

            best_pos = min(pos_a, pos_b)

            a_positions.append(pos_a)
            b_positions.append(pos_b)
            best_positions.append(best_pos)

            # Track wins and top-3
            if pos_a == 1:
                a_wins += 1
            if pos_b == 1:
                b_wins += 1
            if best_pos == 1:
                either_wins += 1

            if pos_a <= 3:
                a_top3 += 1
            if pos_b <= 3:
                b_top3 += 1
            if best_pos <= 3:
                either_top3 += 1

            # Calculate payout for this sim
            payout = 0
            for entry_pos in [pos_a, pos_b]:
                if entry_pos in self.prize_structure:
                    payout += self.prize_structure[entry_pos]
            total_payout += payout

        # Calculate statistics
        n = n_pool_sims
        avg_pos_a = np.mean(a_positions)
        avg_pos_b = np.mean(b_positions)
        avg_best = np.mean(best_positions)
        median_best = np.median(best_positions)

        # Distribution of best finishing position
        pos_distribution = defaultdict(int)
        for p in best_positions:
            pos_distribution[p] += 1

        print(f"\n{'='*60}")
        print(f"  POOL CONTEST RESULTS ({n:,} simulations)")
        print(f"{'='*60}")

        print(f"\n  --- Win Probabilities ---")
        print(f"    Bracket A wins:      {a_wins/n:>6.1%}")
        print(f"    Bracket B wins:      {b_wins/n:>6.1%}")
        print(f"    Either wins:         {either_wins/n:>6.1%}")
        print(f"    Random entrant wins: {1/POOL_SIZE:>6.1%}")
        print(f"    Win edge over random:{either_wins/n / (1/POOL_SIZE):>6.1f}x")

        print(f"\n  --- Top 3 Probabilities ---")
        print(f"    Bracket A top 3:     {a_top3/n:>6.1%}")
        print(f"    Bracket B top 3:     {b_top3/n:>6.1%}")
        print(f"    Either top 3:        {either_top3/n:>6.1%}")
        print(f"    Random top 3:        {3/POOL_SIZE:>6.1%}")
        print(f"    Top-3 edge:          {either_top3/n / (3/POOL_SIZE):>6.1f}x")

        print(f"\n  --- Finishing Position ---")
        print(f"    Avg position (A):    {avg_pos_a:.1f}")
        print(f"    Avg position (B):    {avg_pos_b:.1f}")
        print(f"    Avg best position:   {avg_best:.1f}")
        print(f"    Median best position:{median_best:.0f}")

        print(f"\n  --- Position Distribution (best of A/B) ---")
        for pos in sorted(pos_distribution.keys()):
            if pos <= 10:
                pct = pos_distribution[pos] / n
                bar = "█" * int(pct * 100)
                prize_label = f" ← ${self.prize_structure[pos]}" if pos in self.prize_structure else ""
                print(f"    #{pos:<3} {pct:>5.1%} {bar}{prize_label}")

        if total_payout > 0:
            avg_payout = total_payout / n
            print(f"\n  --- Expected Payout ---")
            print(f"    Average payout per contest: ${avg_payout:.2f}")
            random_ev = sum(self.prize_structure.values()) / POOL_SIZE
            print(f"    Random entrant EV:          ${random_ev:.2f}")
            print(f"    Our edge:                   ${avg_payout - random_ev:+.2f} ({avg_payout/random_ev:.1f}x)")

        print(f"{'='*60}")

        results = {
            "a_win_pct": a_wins / n,
            "b_win_pct": b_wins / n,
            "either_win_pct": either_wins / n,
            "a_top3_pct": a_top3 / n,
            "b_top3_pct": b_top3 / n,
            "either_top3_pct": either_top3 / n,
            "avg_position_a": avg_pos_a,
            "avg_position_b": avg_pos_b,
            "avg_best_position": avg_best,
            "median_best_position": float(median_best),
            "expected_payout": total_payout / n,
            "n_simulations": n,
        }

        # Save results
        with open(os.path.join(DATA_DIR, "pool_simulation_results.json"), "w") as f:
            json.dump(results, f, indent=2)

        return results

    def _score_bracket(self, bracket, sim_result):
        """Score a bracket against actual tournament results."""
        score = 0
        for region in self.field:
            if region not in bracket.get("rounds", {}):
                continue
            sim_rounds = sim_result.get("rounds", {}).get(region, {})
            for round_num in range(1, 5):
                picks = bracket["rounds"][region].get(round_num, [])
                actual = sim_rounds.get(round_num, [])
                for pick in picks:
                    if pick in actual:
                        score += ESPN_SCORING[round_num]

        sim_ff = sim_result.get("final_four", [])
        for team in bracket.get("final_four", []):
            if team in sim_ff:
                score += ESPN_SCORING[5]

        if bracket.get("champion") == sim_result.get("champion"):
            score += ESPN_SCORING[6]

        return score

    def _generate_random_public_bracket(self):
        """Generate a bracket mimicking typical public behavior."""
        bracket = {"rounds": {}, "picks": []}

        for region in self.field:
            bracket["rounds"][region] = {}
            teams = []
            for higher_seed, lower_seed in SEED_MATCHUPS_R64:
                team_high = self.field[region].get(str(higher_seed))
                team_low = self.field[region].get(str(lower_seed))
                teams.append({"name": team_high, "seed": higher_seed})
                teams.append({"name": team_low, "seed": lower_seed})

            # R64
            rw = []
            for i in range(0, len(teams), 2):
                chalk = self._chalk_pick_rate(teams[i]["seed"], 1)
                rw.append(teams[i] if np.random.random() < chalk else teams[i+1])
            bracket["rounds"][region][1] = [w["name"] for w in rw]

            # R32
            r32 = []
            for i in range(0, len(rw), 2):
                p = self._public_matchup_prob(rw[i], rw[i+1])
                r32.append(rw[i] if np.random.random() < p else rw[i+1])
            bracket["rounds"][region][2] = [w["name"] for w in r32]

            # S16
            s16 = []
            for i in range(0, len(r32), 2):
                p = self._public_matchup_prob(r32[i], r32[i+1])
                s16.append(r32[i] if np.random.random() < p else r32[i+1])
            bracket["rounds"][region][3] = [w["name"] for w in s16]

            # E8
            p = self._public_matchup_prob(s16[0], s16[1])
            ew = s16[0] if np.random.random() < p else s16[1]
            bracket["rounds"][region][4] = [ew["name"]]
            bracket["picks"].append(ew)

        ff = bracket["picks"]
        p = self._public_matchup_prob(ff[0], ff[1])
        s1 = ff[0] if np.random.random() < p else ff[1]
        p = self._public_matchup_prob(ff[2], ff[3])
        s2 = ff[2] if np.random.random() < p else ff[3]
        p = self._public_matchup_prob(s1, s2)
        ch = s1 if np.random.random() < p else s2

        bracket["final_four"] = [f["name"] for f in ff]
        bracket["champion"] = ch["name"]
        return bracket

    def _public_matchup_prob(self, team_a, team_b):
        """Probability the public picks team_a over team_b."""
        seed_a = team_a.get("seed", 8)
        seed_b = team_b.get("seed", 8)
        if seed_a == seed_b:
            return 0.5
        seed_diff = seed_b - seed_a
        return 1 / (1 + np.exp(-0.25 * seed_diff))

    def _print_bracket(self, bracket):
        """Pretty print a bracket."""
        label = bracket["label"]
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
        for region in self.field:
            print(f"\n  --- {region} Region ---")
            for round_num in sorted(bracket["rounds"][region].keys()):
                round_names = {1: "R64", 2: "R32", 3: "S16", 4: "E8"}
                teams = bracket["rounds"][region][round_num]
                print(f"    {round_names.get(round_num, f'R{round_num}')}: {', '.join(teams)}")
        print(f"\n  Final Four: {', '.join(bracket['final_four'])}")
        print(f"  Finalists: {bracket['finalist_1']} vs {bracket['finalist_2']}")
        print(f"  Champion: {bracket['champion']}")
        print(f"  Expected Points: {bracket['expected_points']:.1f}")
        print(f"{'='*60}")

    def save_brackets(self, bracket_a, bracket_b):
        """Save the final brackets to files."""
        output = {"bracket_a": bracket_a, "bracket_b": bracket_b}
        filepath = os.path.join(DATA_DIR, "final_brackets.json")
        with open(filepath, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n[OPT] Brackets saved to {filepath}")

        txt_path = os.path.join(DATA_DIR, "brackets_readable.txt")
        with open(txt_path, "w") as f:
            for bracket in [bracket_a, bracket_b]:
                f.write(f"\n{'='*60}\n")
                f.write(f"  {bracket['label']}\n")
                f.write(f"{'='*60}\n\n")
                for region in self.field:
                    f.write(f"  --- {region} Region ---\n")
                    for round_num in sorted(bracket["rounds"][region].keys()):
                        round_names = {1: "R64", 2: "R32", 3: "S16", 4: "E8"}
                        teams = bracket["rounds"][region][round_num]
                        f.write(f"    {round_names.get(round_num)}: {', '.join(teams)}\n")
                    f.write("\n")
                f.write(f"  Final Four: {', '.join(bracket['final_four'])}\n")
                f.write(f"  Champion: {bracket['champion']}\n")
                f.write(f"  Expected Points: {bracket['expected_points']:.1f}\n\n")

        print(f"[OPT] Readable brackets saved to {txt_path}")
