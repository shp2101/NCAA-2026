"""
NCAA Bracket Optimizer - Configuration
ESPN Tournament Challenge scoring and simulation parameters.
"""

# ESPN Tournament Challenge Scoring
ESPN_SCORING = {
    1: 10,   # Round of 64
    2: 20,   # Round of 32
    3: 40,   # Sweet 16
    4: 80,   # Elite Eight
    5: 160,  # Final Four
    6: 320,  # Championship
}

# Total possible points: 10*32 + 20*16 + 40*8 + 80*4 + 160*2 + 320*1 = 1920
TOTAL_POSSIBLE_POINTS = sum(
    points * (64 // (2 ** round_num))
    for round_num, points in ESPN_SCORING.items()
)

# Simulation parameters
NUM_SIMULATIONS = 500000  # Monte Carlo iterations
NUM_BRACKET_CANDIDATES = 5000  # Candidate brackets to evaluate for EV

# Pool parameters
POOL_SIZE = 30
NUM_ENTRIES = 2

# Strategy parameters
SAFE_BRACKET_UPSET_THRESHOLD = 0.35   # Pick upset if model prob > 35%
SWING_BRACKET_UPSET_THRESHOLD = 0.25  # Pick upset if model prob > 25%

# Leverage multiplier: how much to weight "contrarian value"
# Higher = more upset-heavy when public is ignoring a team
LEVERAGE_WEIGHT = 1.5

# Historical seed win rates (Round of 64) - based on 1985-2024 data
HISTORICAL_SEED_WIN_RATES_R64 = {
    (1, 16): 0.993,
    (2, 15): 0.938,
    (3, 14): 0.851,
    (4, 13): 0.793,
    (5, 12): 0.649,
    (6, 11): 0.625,
    (7, 10): 0.609,
    (8, 9):  0.510,
}

# Historical Sweet 16 rates by seed
HISTORICAL_SWEET16_RATE = {
    1: 0.82, 2: 0.63, 3: 0.48, 4: 0.38,
    5: 0.22, 6: 0.20, 7: 0.15, 8: 0.10,
    9: 0.08, 10: 0.12, 11: 0.14, 12: 0.10,
    13: 0.03, 14: 0.02, 15: 0.01, 16: 0.005,
}

# Historical Elite 8 rates by seed
HISTORICAL_ELITE8_RATE = {
    1: 0.55, 2: 0.35, 3: 0.22, 4: 0.16,
    5: 0.08, 6: 0.08, 7: 0.05, 8: 0.04,
    9: 0.03, 10: 0.04, 11: 0.06, 12: 0.03,
    13: 0.01, 14: 0.005, 15: 0.003, 16: 0.001,
}

# Historical Final Four rates by seed
HISTORICAL_FINAL4_RATE = {
    1: 0.32, 2: 0.16, 3: 0.10, 4: 0.06,
    5: 0.04, 6: 0.03, 7: 0.02, 8: 0.02,
    9: 0.01, 10: 0.02, 11: 0.03, 12: 0.01,
    13: 0.003, 14: 0.001, 15: 0.001, 16: 0.0,
}

# Barttorvik T-Rank URL
BARTTORVIK_URL = "https://barttorvik.com/trank.php"
BARTTORVIK_TEAM_URL = "https://barttorvik.com/team.php"

# NCAA API
NCAA_STATS_URL = "https://www.ncaa.com/rankings/basketball-men/d1"

# KenPom (if accessible)
KENPOM_URL = "https://kenpom.com/"

# Regions in bracket order
REGIONS = ["South", "East", "Midwest", "West"]

# Seeds in bracket matchup order (Round of 64)
SEED_MATCHUPS_R64 = [
    (1, 16), (8, 9), (5, 12), (4, 13),
    (6, 11), (3, 14), (7, 10), (2, 15),
]
