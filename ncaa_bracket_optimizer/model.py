"""
NCAA Bracket Optimizer - Win Probability Model
Computes game-level win probabilities using multiple factors.
"""

import math
import json
import os
from config import HISTORICAL_SEED_WIN_RATES_R64

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


class WinProbabilityModel:
    """
    Ensemble model combining:
    1. Adjusted efficiency margin differential (primary signal)
    2. Historical seed-vs-seed performance (prior)
    3. BPI/SRS as secondary signals
    4. Tempo adjustment for high-variance matchups

    The core formula uses a logistic function on the efficiency margin
    differential, calibrated to historical tournament outcomes.
    """

    # Logistic regression coefficient calibrated to tournament data
    # A 10-point efficiency margin differential ≈ 85% win probability
    LOGISTIC_COEFF = 0.15

    # How much to weight each component
    WEIGHTS = {
        "eff_margin": 0.55,     # Primary: Barttorvik/KenPom efficiency margin
        "seed_history": 0.15,   # Historical seed performance
        "srs": 0.10,            # Sports Reference SRS
        "bpi": 0.10,            # ESPN BPI
        "barthag": 0.10,        # Barttorvik power rating
    }

    def __init__(self, team_data):
        """
        Initialize with merged team data from data_collector.
        team_data: dict of {team_name: {metrics...}}
        """
        self.team_data = team_data
        self._validate_data()

    def _validate_data(self):
        """Check what data is available and adjust weights accordingly."""
        available = {"eff_margin": False, "srs": False, "bpi": False, "barthag": False}

        sample_count = 0
        for team, data in self.team_data.items():
            if data.get("eff_margin") is not None:
                available["eff_margin"] = True
            if data.get("srs") is not None:
                available["srs"] = True
            if data.get("bpi") is not None:
                available["bpi"] = True
            if data.get("barthag") is not None:
                available["barthag"] = True
            sample_count += 1
            if sample_count > 20:
                break

        # Redistribute weights if some data is missing
        total_available = sum(
            self.WEIGHTS[k] for k in available if available[k]
        ) + self.WEIGHTS["seed_history"]

        if total_available > 0:
            self.active_weights = {}
            for k, v in available.items():
                if v:
                    self.active_weights[k] = self.WEIGHTS[k] / total_available
                else:
                    self.active_weights[k] = 0
            self.active_weights["seed_history"] = self.WEIGHTS["seed_history"] / total_available
        else:
            # Fallback to seed history only
            self.active_weights = {k: 0 for k in self.WEIGHTS}
            self.active_weights["seed_history"] = 1.0

        print(f"[MODEL] Active weights: {self.active_weights}")

    def win_probability(self, team_a, team_b, seed_a=None, seed_b=None):
        """
        Compute P(team_a beats team_b).

        Parameters:
            team_a: name of team A
            team_b: name of team B
            seed_a: seed of team A (1-16)
            seed_b: seed of team B (1-16)

        Returns:
            float: probability that team_a wins (0 to 1)
        """
        data_a = self.team_data.get(team_a, {})
        data_b = self.team_data.get(team_b, {})

        probabilities = {}

        # 1. Efficiency margin model (logistic)
        eff_a = data_a.get("eff_margin")
        eff_b = data_b.get("eff_margin")
        if eff_a is not None and eff_b is not None:
            diff = eff_a - eff_b
            probabilities["eff_margin"] = self._logistic(diff, self.LOGISTIC_COEFF)
        else:
            probabilities["eff_margin"] = None

        # 2. Historical seed matchup
        if seed_a is not None and seed_b is not None:
            probabilities["seed_history"] = self._seed_based_prob(seed_a, seed_b)
        else:
            probabilities["seed_history"] = 0.5

        # 3. SRS-based model
        srs_a = data_a.get("srs")
        srs_b = data_b.get("srs")
        if srs_a is not None and srs_b is not None:
            diff = srs_a - srs_b
            probabilities["srs"] = self._logistic(diff, 0.12)
        else:
            probabilities["srs"] = None

        # 4. BPI-based model
        bpi_a = data_a.get("bpi")
        bpi_b = data_b.get("bpi")
        if bpi_a is not None and bpi_b is not None:
            diff = bpi_a - bpi_b
            probabilities["bpi"] = self._logistic(diff, 0.12)
        else:
            probabilities["bpi"] = None

        # 5. Barthag-based model
        barthag_a = data_a.get("barthag")
        barthag_b = data_b.get("barthag")
        if barthag_a is not None and barthag_b is not None:
            # Barthag is already a 0-1 rating; use log5 method
            probabilities["barthag"] = self._log5(barthag_a, barthag_b)
        else:
            probabilities["barthag"] = None

        # Weighted ensemble
        total_weight = 0
        weighted_prob = 0

        for key, prob in probabilities.items():
            if prob is not None:
                w = self.active_weights.get(key, 0)
                weighted_prob += w * prob
                total_weight += w

        if total_weight > 0:
            final_prob = weighted_prob / total_weight
        else:
            # Absolute fallback: coin flip
            final_prob = 0.5

        # Clamp to avoid 0% or 100%
        final_prob = max(0.005, min(0.995, final_prob))

        return final_prob

    def _logistic(self, x, coeff):
        """Standard logistic function."""
        return 1.0 / (1.0 + math.exp(-coeff * x))

    def _log5(self, rating_a, rating_b):
        """
        Log5 method for converting two team ratings to a win probability.
        Used for ratings already on a 0-1 scale (like Barthag).
        """
        if rating_a == 0 and rating_b == 0:
            return 0.5
        num = rating_a * (1 - rating_b)
        den = rating_a * (1 - rating_b) + rating_b * (1 - rating_a)
        if den == 0:
            return 0.5
        return num / den

    def _seed_based_prob(self, seed_a, seed_b):
        """
        Compute win probability based on historical seed matchup data.
        """
        key = (min(seed_a, seed_b), max(seed_a, seed_b))

        if key in HISTORICAL_SEED_WIN_RATES_R64:
            higher_seed_wins = HISTORICAL_SEED_WIN_RATES_R64[key]
            if seed_a == key[0]:  # team_a is the higher seed
                return higher_seed_wins
            else:
                return 1 - higher_seed_wins

        # For non-standard matchups (later rounds), use seed difference
        seed_diff = seed_b - seed_a  # positive if A is favored
        return self._logistic(seed_diff, 0.10)

    def get_team_power_rating(self, team_name):
        """
        Get a single composite power rating for a team (0-100 scale).
        Useful for quick comparisons.
        """
        data = self.team_data.get(team_name, {})

        components = []

        eff = data.get("eff_margin")
        if eff is not None:
            # Typical range: -15 to +35. Normalize to 0-100.
            normalized = max(0, min(100, (eff + 15) * 2))
            components.append(("eff_margin", normalized, 0.5))

        barthag = data.get("barthag")
        if barthag is not None:
            components.append(("barthag", barthag * 100, 0.3))

        srs = data.get("srs")
        if srs is not None:
            normalized = max(0, min(100, (srs + 15) * 2.5))
            components.append(("srs", normalized, 0.2))

        if not components:
            return 50.0  # default

        total_weight = sum(c[2] for c in components)
        return sum(c[1] * c[2] / total_weight for c in components)

    def print_matchup(self, team_a, team_b, seed_a=None, seed_b=None):
        """Pretty-print a matchup analysis."""
        prob = self.win_probability(team_a, team_b, seed_a, seed_b)
        print(f"\n{'='*50}")
        print(f"  {team_a} ({seed_a}) vs {team_b} ({seed_b})")
        print(f"  Win Prob: {team_a} {prob:.1%} | {team_b} {1-prob:.1%}")

        data_a = self.team_data.get(team_a, {})
        data_b = self.team_data.get(team_b, {})

        if data_a.get("eff_margin") and data_b.get("eff_margin"):
            print(f"  Eff Margin: {data_a['eff_margin']:.1f} vs {data_b['eff_margin']:.1f}")
        if data_a.get("barthag") and data_b.get("barthag"):
            print(f"  Barthag: {data_a['barthag']:.4f} vs {data_b['barthag']:.4f}")
        print(f"{'='*50}")

        return prob
