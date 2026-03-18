"""
NCAA Bracket Optimizer - Monte Carlo Bracket Simulator
Simulates the entire tournament thousands of times to compute
advancement probabilities for every team.
"""

import numpy as np
import json
import os
from collections import defaultdict
from config import ESPN_SCORING, NUM_SIMULATIONS, SEED_MATCHUPS_R64, REGIONS

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


class BracketSimulator:
    """
    Monte Carlo simulator for the NCAA tournament.

    Simulates the full 64-team bracket (excluding First Four for simplicity)
    using the win probability model to determine game outcomes.

    Tracks:
    - Advancement probabilities for each team to each round
    - Championship probabilities
    - Expected points per pick (for ESPN scoring)
    """

    def __init__(self, model, tournament_field):
        """
        Parameters:
            model: WinProbabilityModel instance
            tournament_field: dict of {region: {seed: team_name}}
        """
        self.model = model
        self.field = tournament_field
        self.regions = list(tournament_field.keys())

        # Build the bracket structure
        self.bracket = self._build_bracket()

        # Results storage
        self.advancement_probs = defaultdict(lambda: defaultdict(float))
        self.championship_counts = defaultdict(int)
        self.simulation_results = []

    def _build_bracket(self):
        """
        Build the initial bracket structure from the tournament field.
        Returns a list of regional brackets, each containing the 16 teams
        in matchup order.
        """
        bracket = {}
        for region in self.regions:
            teams = []
            for higher_seed, lower_seed in SEED_MATCHUPS_R64:
                team_high = self.field[region].get(str(higher_seed), f"#{higher_seed} {region}")
                team_low = self.field[region].get(str(lower_seed), f"#{lower_seed} {region}")
                teams.append({
                    "name": team_high,
                    "seed": higher_seed,
                })
                teams.append({
                    "name": team_low,
                    "seed": lower_seed,
                })
            bracket[region] = teams
        return bracket

    def simulate(self, n_sims=None):
        """
        Run the full Monte Carlo simulation.

        Parameters:
            n_sims: number of simulations (default from config)

        Returns:
            dict: advancement probabilities for each team
        """
        if n_sims is None:
            n_sims = NUM_SIMULATIONS

        print(f"\n[SIM] Running {n_sims:,} tournament simulations...")

        # Reset counters
        self.advancement_probs = defaultdict(lambda: defaultdict(float))
        self.championship_counts = defaultdict(int)
        self.simulation_results = []

        for sim_num in range(n_sims):
            if (sim_num + 1) % 5000 == 0:
                print(f"  Simulation {sim_num + 1:,}/{n_sims:,}...")

            result = self._simulate_tournament()
            self.simulation_results.append(result)

        # Convert counts to probabilities
        for team in self.advancement_probs:
            for round_num in self.advancement_probs[team]:
                self.advancement_probs[team][round_num] /= n_sims

        # Sort championship probs
        champ_probs = {
            team: count / n_sims
            for team, count in self.championship_counts.items()
        }
        champ_probs = dict(sorted(champ_probs.items(), key=lambda x: -x[1]))

        print(f"\n[SIM] Simulation complete!")
        print(f"\n  Top 10 Championship Probabilities:")
        for i, (team, prob) in enumerate(champ_probs.items()):
            if i >= 10:
                break
            print(f"    {i+1}. {team}: {prob:.1%}")

        # Save results
        results = {
            "advancement_probs": {
                team: dict(rounds) for team, rounds in self.advancement_probs.items()
            },
            "championship_probs": champ_probs,
            "n_simulations": n_sims,
        }
        with open(os.path.join(DATA_DIR, "simulation_results.json"), "w") as f:
            json.dump(results, f, indent=2)

        return results

    def _simulate_tournament(self):
        """
        Simulate a single tournament. Returns the bracket result.
        """
        result = {"rounds": {}, "champion": None}

        # Phase 1: Simulate each region through Elite 8
        final_four = []
        for region in self.regions:
            teams = self.bracket[region].copy()
            regional_results = {}

            # Round of 64 (8 games per region)
            round_winners = []
            for i in range(0, len(teams), 2):
                winner = self._simulate_game(teams[i], teams[i+1], round_num=1)
                round_winners.append(winner)
                self.advancement_probs[winner["name"]][1] += 1

            # Track both R64 participants as having made the tournament
            for team in teams:
                self.advancement_probs[team["name"]][0] += 1

            regional_results[1] = [w["name"] for w in round_winners]

            # Round of 32 (4 games per region)
            r32_winners = []
            for i in range(0, len(round_winners), 2):
                winner = self._simulate_game(round_winners[i], round_winners[i+1], round_num=2)
                r32_winners.append(winner)
                self.advancement_probs[winner["name"]][2] += 1
            regional_results[2] = [w["name"] for w in r32_winners]

            # Sweet 16 (2 games per region)
            s16_winners = []
            for i in range(0, len(r32_winners), 2):
                winner = self._simulate_game(r32_winners[i], r32_winners[i+1], round_num=3)
                s16_winners.append(winner)
                self.advancement_probs[winner["name"]][3] += 1
            regional_results[3] = [w["name"] for w in s16_winners]

            # Elite 8 (1 game per region)
            elite_winner = self._simulate_game(s16_winners[0], s16_winners[1], round_num=4)
            self.advancement_probs[elite_winner["name"]][4] += 1
            regional_results[4] = [elite_winner["name"]]

            final_four.append(elite_winner)
            result["rounds"][region] = regional_results

        # Phase 2: Final Four
        # Semifinal 1: Region 0 vs Region 1
        semi1_winner = self._simulate_game(final_four[0], final_four[1], round_num=5)
        self.advancement_probs[semi1_winner["name"]][5] += 1

        # Semifinal 2: Region 2 vs Region 3
        semi2_winner = self._simulate_game(final_four[2], final_four[3], round_num=5)
        self.advancement_probs[semi2_winner["name"]][5] += 1

        # Championship
        champion = self._simulate_game(semi1_winner, semi2_winner, round_num=6)
        self.advancement_probs[champion["name"]][6] += 1
        self.championship_counts[champion["name"]] += 1

        result["champion"] = champion["name"]
        result["final_four"] = [ff["name"] for ff in final_four]

        return result

    def _simulate_game(self, team_a, team_b, round_num=1):
        """
        Simulate a single game using the model's win probability.

        Parameters:
            team_a: dict with "name" and "seed"
            team_b: dict with "name" and "seed"
            round_num: round number (for potential round adjustments)

        Returns:
            dict: the winning team
        """
        prob_a = self.model.win_probability(
            team_a["name"], team_b["name"],
            team_a.get("seed"), team_b.get("seed")
        )

        # Optional: Add small random variance to account for "March Madness"
        # This slightly increases upset frequency to match historical rates
        chaos_factor = 0.02  # 2% randomness
        prob_a = prob_a * (1 - chaos_factor) + np.random.random() * chaos_factor

        if np.random.random() < prob_a:
            return team_a
        else:
            return team_b

    def get_advancement_probs(self):
        """Return formatted advancement probabilities."""
        return dict(self.advancement_probs)

    def get_expected_points_per_pick(self, team_name, round_num):
        """
        Calculate expected ESPN points for picking a team to win in a given round.

        E[points] = P(team reaches round) × ESPN_SCORING[round]
        """
        prob = self.advancement_probs.get(team_name, {}).get(round_num, 0)
        points = ESPN_SCORING.get(round_num, 0)
        return prob * points

    def get_all_expected_points(self):
        """
        Get expected points for every team in every round.
        Returns dict of {team: {round: expected_points}}
        """
        expected = {}
        for team in self.advancement_probs:
            expected[team] = {}
            for round_num in range(1, 7):
                expected[team][round_num] = self.get_expected_points_per_pick(team, round_num)
            expected[team]["total"] = sum(expected[team].values())

        return dict(sorted(expected.items(), key=lambda x: -x[1].get("total", 0)))

    def print_advancement_table(self, top_n=25):
        """Print a formatted table of advancement probabilities."""
        print(f"\n{'='*90}")
        print(f"  ADVANCEMENT PROBABILITIES (Top {top_n})")
        print(f"{'='*90}")
        print(f"  {'Team':<25} {'R64':>6} {'R32':>6} {'S16':>6} {'E8':>6} {'F4':>6} {'Champ':>6}")
        print(f"  {'-'*25} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")

        # Sort by championship probability
        sorted_teams = sorted(
            self.advancement_probs.items(),
            key=lambda x: x[1].get(6, 0),
            reverse=True,
        )

        for team, probs in sorted_teams[:top_n]:
            r1 = probs.get(1, 0)
            r2 = probs.get(2, 0)
            r3 = probs.get(3, 0)
            r4 = probs.get(4, 0)
            r5 = probs.get(5, 0)
            r6 = probs.get(6, 0)
            print(f"  {team:<25} {r1:>5.1%} {r2:>5.1%} {r3:>5.1%} {r4:>5.1%} {r5:>5.1%} {r6:>5.1%}")

    def print_expected_points_table(self, top_n=25):
        """Print expected ESPN points per pick."""
        exp = self.get_all_expected_points()

        print(f"\n{'='*100}")
        print(f"  EXPECTED ESPN POINTS PER PICK (Top {top_n})")
        print(f"{'='*100}")
        print(f"  {'Team':<25} {'R64':>7} {'R32':>7} {'S16':>7} {'E8':>7} {'F4':>7} {'Champ':>7} {'TOTAL':>8}")
        print(f"  {'-'*25} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*8}")

        for i, (team, pts) in enumerate(exp.items()):
            if i >= top_n:
                break
            print(f"  {team:<25} {pts.get(1,0):>7.1f} {pts.get(2,0):>7.1f} {pts.get(3,0):>7.1f} "
                  f"{pts.get(4,0):>7.1f} {pts.get(5,0):>7.1f} {pts.get(6,0):>7.1f} {pts.get('total',0):>8.1f}")
