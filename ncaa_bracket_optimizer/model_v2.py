"""
NCAA Bracket Optimizer - Enhanced Win Probability Model (V2)
Uses calibrated logistic regression + expanded Four Factors metrics.

The "Four Factors" of basketball (Dean Oliver):
1. Shooting (eFG%)
2. Turnovers (TO Rate)
3. Rebounding (ORB%)
4. Free Throws (FT Rate)

These four factors explain ~90% of the variance in winning.
We add efficiency margin as the primary signal and use the
Four Factors as adjustment factors.
"""

import math
import json
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


class WinProbabilityModelV2:
    """
    Enhanced ensemble model:
    1. Adjusted efficiency margin (Barttorvik) — primary signal
    2. Four Factors differential — secondary signals
    3. Historical seed performance — Bayesian prior
    4. Barthag (log5 method) — cross-validation signal
    5. Calibrated logistic coefficient from historical data

    When CBBpy box score data is available, the Four Factors
    provide meaningful signal beyond raw efficiency margin,
    especially for identifying upset-prone teams (high TO rate,
    low FT%, poor defensive rebounding).
    """

    def __init__(self, team_data):
        """
        Parameters:
            team_data: dict from collect_all_data_v2 (includes __model_params__)
        """
        self.team_data = {k: v for k, v in team_data.items() if k != "__model_params__"}

        # Load calibrated parameters
        params = team_data.get("__model_params__", {})
        self.logistic_coeff = params.get("calibrated_logistic_coeff", 0.15)
        self.historical_rates = params.get("historical_rates", {})

        # Determine what data is available and set weights
        self._setup_weights()

        print(f"[MODEL v2] Initialized with {len(self.team_data)} teams")
        print(f"[MODEL v2] Calibrated logistic coefficient: {self.logistic_coeff}")
        print(f"[MODEL v2] Active components: {[k for k, v in self.component_weights.items() if v > 0]}")

    def _setup_weights(self):
        """Dynamically set component weights based on available data."""
        has_eff = any(d.get("eff_margin") is not None for d in self.team_data.values())
        has_barthag = any(d.get("barthag") is not None for d in self.team_data.values())
        has_four_factors = any(d.get("efg_pct") is not None for d in self.team_data.values())
        has_market = any(d.get("market_champ_prob") is not None for d in self.team_data.values())

        # Base weights (when all data available)
        #
        # Prediction market data REMOVED — we rely purely on
        # sports-specific statistical signals. The weight previously
        # allocated to market data is redistributed to eff_margin,
        # four_factors, and barthag.
        weights = {
            "eff_margin": 0.45 if has_eff else 0,
            "four_factors": 0.22 if has_four_factors else 0,
            "barthag": 0.13 if has_barthag else 0,
            "seed_history": 0.10,
            "market": 0,  # DISABLED — no prediction market influence
            "upset_factors": 0.10 if has_four_factors else 0,
        }

        # Normalize
        total = sum(weights.values())
        self.component_weights = {k: v/total for k, v in weights.items()} if total > 0 else {"seed_history": 1.0}

    def win_probability(self, team_a, team_b, seed_a=None, seed_b=None):
        """
        Compute P(team_a beats team_b).

        Uses a weighted ensemble of multiple sub-models.
        """
        data_a = self.team_data.get(team_a, {})
        data_b = self.team_data.get(team_b, {})

        components = {}

        # 1. Efficiency margin model (primary signal)
        eff_a = data_a.get("eff_margin")
        eff_b = data_b.get("eff_margin")
        if eff_a is not None and eff_b is not None:
            diff = eff_a - eff_b
            components["eff_margin"] = self._logistic(diff, self.logistic_coeff)

        # 2. Four Factors model
        if data_a.get("efg_pct") is not None and data_b.get("efg_pct") is not None:
            ff_score = self._four_factors_score(data_a, data_b)
            components["four_factors"] = ff_score

        # 3. Barthag (log5 method)
        barthag_a = data_a.get("barthag")
        barthag_b = data_b.get("barthag")
        if barthag_a is not None and barthag_b is not None:
            components["barthag"] = self._log5(barthag_a, barthag_b)

        # 4. Historical seed performance
        if seed_a is not None and seed_b is not None:
            components["seed_history"] = self._seed_based_prob(seed_a, seed_b)

        # 5. Market-implied probability (prediction markets + sportsbooks)
        market_a = data_a.get("market_champ_prob")
        market_b = data_b.get("market_champ_prob")
        if market_a is not None and market_b is not None and (market_a + market_b) > 0:
            # Use log5 on market championship probabilities
            # This isn't a direct matchup prob, but the relative strength
            # implied by the market is a strong signal
            components["market"] = self._log5(market_a, market_b)
        elif market_a is not None and market_a > 0:
            # Only have market data for one team - still useful
            components["market"] = min(0.85, 0.5 + market_a * 2)
        elif market_b is not None and market_b > 0:
            components["market"] = max(0.15, 0.5 - market_b * 2)

        # 6. Upset factors (adjustments for tournament-specific factors)
        if data_a.get("efg_pct") is not None and data_b.get("efg_pct") is not None:
            components["upset_factors"] = self._upset_factor_score(data_a, data_b, seed_a, seed_b)

        # Weighted ensemble
        total_weight = 0
        weighted_prob = 0

        for key, prob in components.items():
            w = self.component_weights.get(key, 0)
            if w > 0:
                weighted_prob += w * prob
                total_weight += w

        if total_weight > 0:
            final_prob = weighted_prob / total_weight
        else:
            final_prob = 0.5

        # Clamp
        return max(0.005, min(0.995, final_prob))

    def _four_factors_score(self, data_a, data_b):
        """
        Compute win probability from Dean Oliver's Four Factors.
        Each factor is weighted by its importance to winning.

        Standard weights:
        - eFG%: 40% (shooting is king)
        - TO Rate: 25% (taking care of the ball)
        - ORB%: 20% (second chances)
        - FT Rate: 15% (getting to the line)
        """
        score = 0.5  # start at even

        # eFG% differential (40% weight)
        efg_a = data_a.get("efg_pct", 50)
        efg_b = data_b.get("efg_pct", 50)
        opp_efg_a = data_a.get("opp_efg_pct", 50)
        opp_efg_b = data_b.get("opp_efg_pct", 50)
        # Team A's offense vs Team B's defense and vice versa
        efg_adv = (efg_a - opp_efg_b) - (efg_b - opp_efg_a)
        score += 0.40 * self._normalize_factor(efg_adv, scale=10)

        # Turnover rate differential (25% weight) - lower is better
        to_a = data_a.get("to_rate", 18)
        to_b = data_b.get("to_rate", 18)
        opp_to_a = data_a.get("opp_to_rate", 18)
        opp_to_b = data_b.get("opp_to_rate", 18)
        to_adv = (to_b - to_a) + (opp_to_a - opp_to_b)  # lower TO for A is better
        score += 0.25 * self._normalize_factor(to_adv, scale=8)

        # ORB% (20% weight)
        orb_a = data_a.get("orb_pct", 28)
        orb_b = data_b.get("orb_pct", 28)
        orb_adv = orb_a - orb_b
        score += 0.20 * self._normalize_factor(orb_adv, scale=10)

        # FT Rate (15% weight)
        ft_rate_a = data_a.get("ft_rate", 30)
        ft_rate_b = data_b.get("ft_rate", 30)
        ft_adv = ft_rate_a - ft_rate_b
        score += 0.15 * self._normalize_factor(ft_adv, scale=15)

        return max(0.05, min(0.95, score))

    def _upset_factor_score(self, data_a, data_b, seed_a, seed_b):
        """
        Identify upset potential through tournament-specific factors:
        - High TO rate teams are upset-prone (pressure in big games)
        - Low FT% teams choke in close games
        - High 3P% teams have variance (can win/lose any game)
        - Experience matters (but we may not have this data)

        Returns P(team_a wins) adjusted for upset factors.
        """
        # Start at baseline from seeds
        if seed_a and seed_b:
            baseline = self._seed_based_prob(seed_a, seed_b)
        else:
            baseline = 0.5

        adjustment = 0

        # High TO rate makes the favorite more vulnerable
        to_a = data_a.get("to_rate", 18)
        to_b = data_b.get("to_rate", 18)

        # If team_a is favored but has high turnovers, reduce their edge
        if baseline > 0.5 and to_a > 20:  # above average TO rate
            adjustment -= 0.03 * (to_a - 18) / 5

        # If team_b is underdog but forces TOs, increase upset chance
        opp_to_b = data_b.get("opp_to_rate", 18)
        if baseline > 0.5 and opp_to_b > 20:
            adjustment -= 0.02 * (opp_to_b - 18) / 5

        # Free throw shooting in close games
        ft_a = data_a.get("ft_pct", 72)
        ft_b = data_b.get("ft_pct", 72)
        if ft_a < 68:  # poor FT shooting
            adjustment -= 0.02
        if ft_b > 78:  # excellent FT shooting
            adjustment -= 0.01

        # 3-point variance (high 3PT% teams have higher variance = more upsets both ways)
        three_a = data_a.get("three_pct", 34)
        three_b = data_b.get("three_pct", 34)
        if baseline > 0.5 and three_b > 38:  # underdog shoots well from 3
            adjustment -= 0.02

        return max(0.05, min(0.95, baseline + adjustment))

    def _normalize_factor(self, diff, scale=10):
        """Convert a raw stat differential to a -0.5 to 0.5 adjustment."""
        return max(-0.5, min(0.5, diff / (2 * scale)))

    def _logistic(self, x, coeff):
        """Standard logistic function."""
        return 1.0 / (1.0 + math.exp(-coeff * x))

    def _log5(self, rating_a, rating_b):
        """Log5 method for ratings on 0-1 scale."""
        if rating_a == 0 and rating_b == 0:
            return 0.5
        num = rating_a * (1 - rating_b)
        den = rating_a * (1 - rating_b) + rating_b * (1 - rating_a)
        return num / den if den != 0 else 0.5

    def _seed_based_prob(self, seed_a, seed_b):
        """Win probability from historical seed matchup data."""
        # Use calibrated coefficient on seed difference
        seed_diff = seed_b - seed_a  # positive if A is favored
        return self._logistic(seed_diff, 0.17)  # tuned to historical data

    def get_team_profile(self, team_name):
        """Get a formatted profile of a team's metrics."""
        data = self.team_data.get(team_name, {})
        if not data:
            return f"{team_name}: No data available"

        lines = [f"\n{'='*50}", f"  {team_name}", f"{'='*50}"]

        if data.get("eff_margin"):
            lines.append(f"  Efficiency Margin: {data['eff_margin']:+.1f}")
        if data.get("adj_oe"):
            lines.append(f"  Adj Off Eff: {data['adj_oe']:.1f}  |  Adj Def Eff: {data.get('adj_de', 'N/A')}")
        if data.get("barthag"):
            lines.append(f"  Barthag: {data['barthag']:.4f}")
        if data.get("efg_pct"):
            lines.append(f"  eFG%: {data['efg_pct']:.1f}%  (opp: {data.get('opp_efg_pct', 'N/A')}%)")
        if data.get("to_rate"):
            lines.append(f"  TO Rate: {data['to_rate']:.1f}%  (forced: {data.get('opp_to_rate', 'N/A')}%)")
        if data.get("orb_pct"):
            lines.append(f"  ORB%: {data['orb_pct']:.1f}%")
        if data.get("ft_rate"):
            lines.append(f"  FT Rate: {data['ft_rate']:.1f}%  |  FT%: {data.get('ft_pct', 'N/A')}%")
        if data.get("three_pct"):
            lines.append(f"  3PT%: {data['three_pct']:.1f}%")
        if data.get("barttorvik_rank"):
            lines.append(f"  Barttorvik Rank: #{int(data['barttorvik_rank'])}")

        return "\n".join(lines)

    def print_matchup(self, team_a, team_b, seed_a=None, seed_b=None):
        """Detailed matchup analysis."""
        prob = self.win_probability(team_a, team_b, seed_a, seed_b)

        print(f"\n{'='*60}")
        print(f"  MATCHUP: ({seed_a}) {team_a} vs ({seed_b}) {team_b}")
        print(f"  Win Probability: {team_a} {prob:.1%} | {team_b} {1-prob:.1%}")
        print(f"{'='*60}")

        data_a = self.team_data.get(team_a, {})
        data_b = self.team_data.get(team_b, {})

        metrics = [
            ("Eff Margin", "eff_margin", "{:+.1f}"),
            ("eFG%", "efg_pct", "{:.1f}%"),
            ("TO Rate", "to_rate", "{:.1f}%"),
            ("ORB%", "orb_pct", "{:.1f}%"),
            ("FT Rate", "ft_rate", "{:.1f}%"),
            ("FT%", "ft_pct", "{:.1f}%"),
            ("3PT%", "three_pct", "{:.1f}%"),
        ]

        for label, key, fmt in metrics:
            val_a = data_a.get(key)
            val_b = data_b.get(key)
            if val_a is not None and val_b is not None:
                str_a = fmt.format(val_a)
                str_b = fmt.format(val_b)
                print(f"  {label:<12}  {str_a:>8}  vs  {str_b:<8}")

        return prob
