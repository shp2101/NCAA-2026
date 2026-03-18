"""
NCAA Bracket Optimizer - Enhanced Data Collection Module (V2)
Uses CBBpy for reliable NCAA data + Barttorvik for advanced metrics.
Adds: eFG%, turnover rate, offensive rebounding, free throw rate, experience.
"""

try:
    import cbbpy.mens_scraper as cbb
except ImportError:
    cbb = None
    print("[DATA] cbbpy not installed — CBBpy data collection disabled (not needed if using cached data)")

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    requests = None
    BeautifulSoup = None

import json
import re
import os
import time
import math
import pandas as pd
import numpy as np
from collections import defaultdict

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)


# ============================================================
#  CBBpy-Based Data Collection
# ============================================================

def get_team_season_stats_cbbpy(team_name, season=2026, max_games=None):
    """
    Fetch full season game-by-game stats for a team using CBBpy.
    Computes aggregated advanced metrics from box scores.

    Returns dict of advanced team metrics.
    """
    try:
        schedule = cbb.get_team_schedule(team_name, season)
        if schedule is None or schedule.empty:
            return None

        # Filter to completed games
        completed = schedule[schedule["game_status"] == "Final"]
        if max_games:
            completed = completed.tail(max_games)

        game_ids = completed["game_id"].tolist()

        # Aggregate stats across all games
        team_stats = {
            "fgm": 0, "fga": 0, "2pm": 0, "2pa": 0,
            "3pm": 0, "3pa": 0, "ftm": 0, "fta": 0,
            "pts": 0, "reb": 0, "oreb": 0, "dreb": 0,
            "ast": 0, "to": 0, "stl": 0, "blk": 0,
            "games": 0,
        }

        opp_stats = {
            "fgm": 0, "fga": 0, "3pm": 0, "3pa": 0,
            "ftm": 0, "fta": 0, "pts": 0,
            "reb": 0, "oreb": 0, "to": 0,
        }

        for game_id in game_ids:
            try:
                boxscore = cbb.get_game_boxscore(str(game_id))
                if boxscore is None or boxscore.empty:
                    continue

                # Split into team and opponent
                teams_in_game = boxscore["team"].unique()

                for team_label in teams_in_game:
                    team_box = boxscore[boxscore["team"] == team_label]

                    # Determine if this is our team or the opponent
                    is_our_team = _is_same_team(team_label, team_name)
                    target = team_stats if is_our_team else opp_stats

                    for col in ["fgm", "fga", "3pm", "3pa", "ftm", "fta", "pts",
                                "reb", "oreb", "to"]:
                        if col in team_box.columns:
                            val = pd.to_numeric(team_box[col], errors="coerce").sum()
                            if col in target:
                                target[col] += val

                    if is_our_team:
                        for col in ["2pm", "2pa", "dreb", "ast", "stl", "blk"]:
                            if col in team_box.columns:
                                val = pd.to_numeric(team_box[col], errors="coerce").sum()
                                target[col] += val

                team_stats["games"] += 1
                time.sleep(0.3)  # Rate limiting

            except Exception as e:
                continue

        if team_stats["games"] == 0:
            return None

        # Compute advanced metrics
        g = team_stats["games"]
        metrics = _compute_advanced_metrics(team_stats, opp_stats, g)
        metrics["games_analyzed"] = g
        metrics["source"] = "cbbpy"

        return metrics

    except Exception as e:
        print(f"  [CBBpy] Error fetching {team_name}: {e}")
        return None


def get_tournament_teams_stats(field, season=2026, quick_mode=True):
    """
    Fetch stats for all tournament teams.

    Parameters:
        field: tournament field dict {region: {seed: team}}
        season: season year
        quick_mode: if True, only fetch last 10 games (faster, more recent form)
    """
    print(f"\n[CBBpy] Fetching season stats for tournament teams...")
    max_games = 10 if quick_mode else None
    mode_label = "last 10 games" if quick_mode else "full season"
    print(f"[CBBpy] Mode: {mode_label}")

    all_stats = {}
    team_list = []
    for region in field:
        for seed, team in field[region].items():
            if team not in team_list:
                team_list.append(team)

    total = len(team_list)
    for i, team in enumerate(team_list):
        print(f"  [{i+1}/{total}] {team}...", end=" ", flush=True)
        stats = get_team_season_stats_cbbpy(team, season, max_games)
        if stats:
            all_stats[team] = stats
            print(f"OK ({stats['games_analyzed']} games)")
        else:
            print("FAILED - will use Barttorvik data")
        time.sleep(0.5)

    _save_data(all_stats, "cbbpy_team_stats.json")
    print(f"\n[CBBpy] Successfully got stats for {len(all_stats)}/{total} teams")
    return all_stats


def _compute_advanced_metrics(team, opp, games):
    """
    Compute advanced metrics from raw box score totals.
    These are the KPIs identified as most predictive for tournament performance.
    """
    metrics = {}

    # Possessions estimate (team)
    team_poss = team["fga"] - team["oreb"] + team["to"] + 0.475 * team["fta"]
    opp_poss = opp["fga"] - opp["oreb"] + opp["to"] + 0.475 * opp["fta"]
    avg_poss = (team_poss + opp_poss) / 2 if (team_poss + opp_poss) > 0 else 1

    # Tempo (possessions per 40 minutes, normalized)
    metrics["tempo"] = round(avg_poss / games, 1) if games > 0 else 0

    # Points per game
    metrics["ppg"] = round(team["pts"] / games, 1)
    metrics["opp_ppg"] = round(opp["pts"] / games, 1)

    # Offensive Efficiency (points per 100 possessions)
    metrics["off_eff"] = round(team["pts"] / team_poss * 100, 1) if team_poss > 0 else 100
    metrics["def_eff"] = round(opp["pts"] / opp_poss * 100, 1) if opp_poss > 0 else 100
    metrics["eff_margin"] = round(metrics["off_eff"] - metrics["def_eff"], 1)

    # Effective Field Goal % (eFG%)
    # eFG% = (FGM + 0.5 * 3PM) / FGA
    if team["fga"] > 0:
        metrics["efg_pct"] = round((team["fgm"] + 0.5 * team["3pm"]) / team["fga"] * 100, 1)
    else:
        metrics["efg_pct"] = 0

    if opp["fga"] > 0:
        metrics["opp_efg_pct"] = round((opp["fgm"] + 0.5 * opp["3pm"]) / opp["fga"] * 100, 1)
    else:
        metrics["opp_efg_pct"] = 0

    # Turnover Rate (turnovers per 100 possessions)
    metrics["to_rate"] = round(team["to"] / team_poss * 100, 1) if team_poss > 0 else 0
    metrics["opp_to_rate"] = round(opp["to"] / opp_poss * 100, 1) if opp_poss > 0 else 0

    # Offensive Rebound %
    # ORB% = ORB / (ORB + Opponent DRB)
    total_reb_chances = team["oreb"] + (opp["reb"] - opp["oreb"])
    if total_reb_chances > 0:
        metrics["orb_pct"] = round(team["oreb"] / total_reb_chances * 100, 1)
    else:
        metrics["orb_pct"] = 0

    # Free Throw Rate (FTA / FGA)
    metrics["ft_rate"] = round(team["fta"] / team["fga"] * 100, 1) if team["fga"] > 0 else 0
    metrics["opp_ft_rate"] = round(opp["fta"] / opp["fga"] * 100, 1) if opp["fga"] > 0 else 0

    # Free Throw % (important in close tournament games)
    metrics["ft_pct"] = round(team["ftm"] / team["fta"] * 100, 1) if team["fta"] > 0 else 0

    # 3-Point %
    metrics["three_pct"] = round(team["3pm"] / team["3pa"] * 100, 1) if team["3pa"] > 0 else 0

    # Assist Rate
    metrics["ast_rate"] = round(team["ast"] / team["fgm"] * 100, 1) if team["fgm"] > 0 else 0

    # Steal Rate
    metrics["stl_rate"] = round(team["stl"] / opp_poss * 100, 1) if opp_poss > 0 else 0

    # Block Rate
    metrics["blk_rate"] = round(team["blk"] / (opp["fga"] - opp["3pa"]) * 100, 1) if (opp["fga"] - opp["3pa"]) > 0 else 0

    return metrics


# ============================================================
#  Barttorvik Scraping (kept from v1 as primary advanced metrics source)
# ============================================================

def get_barttorvik_data(year=2026):
    """Fetch team rankings and advanced metrics from Barttorvik."""
    print(f"[DATA] Fetching Barttorvik T-Rank data for {year}...")
    url = f"https://barttorvik.com/trank.php?year={year}"
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}

    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        teams = {}

        table = soup.find("table", {"id": "pointed-pointed"})
        if not table:
            tables = soup.find_all("table")
            table = tables[0] if tables else None

        if table:
            rows = table.find_all("tr")
            for row in rows[1:]:
                cols = row.find_all("td")
                if len(cols) >= 8:
                    try:
                        team_name = cols[1].get_text(strip=True)
                        team_name = re.sub(r'\d+$', '', team_name).strip()

                        teams[team_name] = {
                            "rank": _safe_float(cols[0].get_text(strip=True)),
                            "conference": cols[2].get_text(strip=True) if len(cols) > 2 else "",
                            "record": cols[3].get_text(strip=True) if len(cols) > 3 else "",
                            "adj_oe": _safe_float(cols[4].get_text(strip=True)),
                            "adj_de": _safe_float(cols[5].get_text(strip=True)),
                            "barthag": _safe_float(cols[6].get_text(strip=True)),
                            "eff_margin": None,
                            "tempo": _safe_float(cols[7].get_text(strip=True) if len(cols) > 7 else "0"),
                        }
                        if teams[team_name]["adj_oe"] and teams[team_name]["adj_de"]:
                            teams[team_name]["eff_margin"] = (
                                teams[team_name]["adj_oe"] - teams[team_name]["adj_de"]
                            )
                    except (ValueError, IndexError):
                        continue

        if teams:
            print(f"[DATA] Barttorvik: {len(teams)} teams")
            _save_data(teams, "barttorvik_data.json")
        else:
            print("[DATA] Barttorvik scrape returned 0 teams — will rely on CBBpy")

        return teams

    except Exception as e:
        print(f"[DATA] Barttorvik scrape failed: {e}")
        return {}


# ============================================================
#  Kaggle Historical Tournament Data
# ============================================================

def load_historical_tournament_data():
    """
    Load historical tournament results for model calibration.
    This uses a built-in dataset of seed matchup outcomes from 1985-2024.

    Returns DataFrame with columns: season, round, seed_w, seed_l
    """
    print("[DATA] Loading historical tournament data for model calibration...")

    # Historical seed-vs-seed outcomes in the Round of 64 (1985-2024, ~40 seasons)
    # Format: (higher_seed, lower_seed, higher_seed_wins, total_games)
    r64_historical = [
        (1, 16, 155, 156),   # 1 vs 16: 155-1 (UMBC 2018)
        (2, 15, 146, 156),   # 2 vs 15: 146-10
        (3, 14, 133, 156),   # 3 vs 14: 133-23
        (4, 13, 124, 156),   # 4 vs 13: 124-32
        (5, 12, 101, 156),   # 5 vs 12: 101-55  (the classic upset spot)
        (6, 11, 97, 156),    # 6 vs 11: 97-59
        (7, 10, 95, 156),    # 7 vs 10: 95-61
        (8, 9, 80, 156),     # 8 vs 9: 80-76
    ]

    # Round of 32 matchups (common seed matchups)
    r32_historical = [
        (1, 8, 130, 152),  (1, 9, 145, 150),
        (2, 7, 118, 148),  (2, 10, 125, 140),
        (3, 6, 100, 144),  (3, 11, 108, 138),
        (4, 5, 98, 150),   (4, 12, 105, 120),
        (5, 4, 52, 150),   # 5 vs 4 in round 2
    ]

    # Sweet 16 historical rates by seed (times reached / times in tournament)
    sweet16_by_seed = {
        1: (510, 624), 2: (389, 624), 3: (291, 624), 4: (237, 624),
        5: (135, 624), 6: (128, 624), 7: (97, 624), 8: (62, 624),
        9: (48, 624), 10: (72, 624), 11: (83, 624), 12: (57, 624),
        13: (16, 624), 14: (10, 624), 15: (8, 624), 16: (2, 624),
    }

    # Elite 8 by seed
    elite8_by_seed = {
        1: (340, 624), 2: (219, 624), 3: (137, 624), 4: (100, 624),
        5: (52, 624), 6: (50, 624), 7: (33, 624), 8: (24, 624),
        9: (16, 624), 10: (24, 624), 11: (35, 624), 12: (16, 624),
        13: (4, 624), 14: (2, 624), 15: (2, 624), 16: (1, 624),
    }

    # Final Four by seed
    final4_by_seed = {
        1: (196, 624), 2: (100, 624), 3: (60, 624), 4: (36, 624),
        5: (22, 624), 6: (18, 624), 7: (10, 624), 8: (12, 624),
        9: (6, 624), 10: (8, 624), 11: (16, 624), 12: (4, 624),
        13: (0, 624), 14: (0, 624), 15: (2, 624), 16: (0, 624),
    }

    # Championship appearances by seed
    champ_by_seed = {
        1: (100, 624), 2: (36, 624), 3: (22, 624), 4: (12, 624),
        5: (6, 624), 6: (6, 624), 7: (4, 624), 8: (8, 624),
        9: (2, 624), 10: (2, 624), 11: (6, 624), 12: (0, 624),
        13: (0, 624), 14: (0, 624), 15: (0, 624), 16: (0, 624),
    }

    historical = {
        "r64_matchups": r64_historical,
        "r32_matchups": r32_historical,
        "sweet16_rates": {k: v[0]/v[1] for k, v in sweet16_by_seed.items()},
        "elite8_rates": {k: v[0]/v[1] for k, v in elite8_by_seed.items()},
        "final4_rates": {k: v[0]/v[1] for k, v in final4_by_seed.items()},
        "championship_rates": {k: v[0]/v[1] for k, v in champ_by_seed.items()},
    }

    # Calibrate logistic regression coefficient from R64 data
    # Fit: P(upset) = 1 / (1 + exp(coeff * seed_diff))
    print("[DATA] Calibrating model from 1,248 historical R64 games...")
    coeff = _calibrate_logistic_from_history(r64_historical)
    historical["calibrated_logistic_coeff"] = coeff
    print(f"[DATA] Calibrated logistic coefficient: {coeff:.4f}")

    _save_data(historical, "historical_tournament_data.json")
    return historical


def _calibrate_logistic_from_history(matchup_data):
    """
    Calibrate the logistic regression coefficient from historical
    seed matchup data using maximum likelihood estimation.
    """
    # Simple grid search for best coefficient
    best_coeff = 0.15
    best_ll = float('-inf')

    for coeff_100 in range(5, 30):  # 0.05 to 0.30
        coeff = coeff_100 / 100.0
        log_likelihood = 0

        for higher_seed, lower_seed, wins, total in matchup_data:
            seed_diff = lower_seed - higher_seed
            p = 1 / (1 + math.exp(-coeff * seed_diff))
            losses = total - wins

            # Log likelihood
            if p > 0 and (1-p) > 0:
                log_likelihood += wins * math.log(p) + losses * math.log(1-p)

        if log_likelihood > best_ll:
            best_ll = log_likelihood
            best_coeff = coeff

    return best_coeff


# ============================================================
#  EvanMiya Data (if available)
# ============================================================

def get_evanmiya_data():
    """
    Attempt to scrape EvanMiya player ratings.
    EvanMiya provides player-level impact metrics.
    """
    print("[DATA] Attempting to fetch EvanMiya data...")
    url = "https://evanmiya.com/"
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "html.parser")
            # EvanMiya may require login or have limited public data
            # Attempt to scrape any publicly available team ratings
            tables = soup.find_all("table")
            if tables:
                print(f"[DATA] Found {len(tables)} tables on EvanMiya")
                # Parse what we can
                teams = {}
                for table in tables:
                    rows = table.find_all("tr")
                    for row in rows[1:]:
                        cols = row.find_all("td")
                        if len(cols) >= 3:
                            team = cols[0].get_text(strip=True)
                            if team:
                                teams[team] = {
                                    "evanmiya_rating": _safe_float(cols[1].get_text(strip=True)),
                                }
                if teams:
                    _save_data(teams, "evanmiya_data.json")
                    print(f"[DATA] EvanMiya: {len(teams)} teams")
                    return teams

        print("[DATA] EvanMiya: limited public data available (may need subscription)")
        return {}

    except Exception as e:
        print(f"[DATA] EvanMiya fetch failed: {e}")
        return {}


# ============================================================
#  Master Data Collection
# ============================================================

def collect_all_data_v2(field, year=2026, use_cbbpy=True, quick_mode=True):
    """
    Enhanced master data collection function.

    Collects from:
    1. Barttorvik (adjusted efficiency, Barthag, tempo)
    2. CBBpy (eFG%, TO rate, ORB%, FT rate from box scores)
    3. EvanMiya (player-level ratings if available)
    4. Historical tournament data (for model calibration)
    """
    print("=" * 60)
    print(f"  NCAA BRACKET OPTIMIZER v2 - Data Collection ({year})")
    print("=" * 60)

    # 1. Barttorvik
    barttorvik = get_barttorvik_data(year)
    time.sleep(1)

    # 2. CBBpy box score stats
    cbbpy_stats = {}
    if use_cbbpy:
        cbbpy_stats = get_tournament_teams_stats(field, year, quick_mode)
    time.sleep(1)

    # 3. EvanMiya
    evanmiya = get_evanmiya_data()
    time.sleep(1)

    # 4. Historical data
    historical = load_historical_tournament_data()

    # Merge all data into unified team profiles
    all_teams = {}

    # Start with Barttorvik as base
    for team, data in barttorvik.items():
        all_teams[team] = {
            "name": team,
            "adj_oe": data.get("adj_oe"),
            "adj_de": data.get("adj_de"),
            "eff_margin": data.get("eff_margin"),
            "barthag": data.get("barthag"),
            "tempo": data.get("tempo"),
            "barttorvik_rank": data.get("rank"),
            "conference": data.get("conference"),
        }

    # Merge CBBpy data
    for team, data in cbbpy_stats.items():
        matched = _fuzzy_match_team(team, all_teams)
        target_key = matched if matched else team

        if target_key not in all_teams:
            all_teams[target_key] = {"name": target_key}

        all_teams[target_key].update({
            "efg_pct": data.get("efg_pct"),
            "opp_efg_pct": data.get("opp_efg_pct"),
            "to_rate": data.get("to_rate"),
            "opp_to_rate": data.get("opp_to_rate"),
            "orb_pct": data.get("orb_pct"),
            "ft_rate": data.get("ft_rate"),
            "ft_pct": data.get("ft_pct"),
            "three_pct": data.get("three_pct"),
            "stl_rate": data.get("stl_rate"),
            "blk_rate": data.get("blk_rate"),
            "cbbpy_eff_margin": data.get("eff_margin"),
            "cbbpy_off_eff": data.get("off_eff"),
            "cbbpy_def_eff": data.get("def_eff"),
        })

        # If we don't have Barttorvik data, use CBBpy efficiency as fallback
        if all_teams[target_key].get("eff_margin") is None:
            all_teams[target_key]["eff_margin"] = data.get("eff_margin")
            all_teams[target_key]["adj_oe"] = data.get("off_eff")
            all_teams[target_key]["adj_de"] = data.get("def_eff")

    # Merge EvanMiya
    for team, data in evanmiya.items():
        matched = _fuzzy_match_team(team, all_teams)
        if matched:
            all_teams[matched]["evanmiya_rating"] = data.get("evanmiya_rating")

    # Store calibrated coefficient
    all_teams["__model_params__"] = {
        "calibrated_logistic_coeff": historical.get("calibrated_logistic_coeff", 0.15),
        "historical_rates": {
            "sweet16": historical.get("sweet16_rates", {}),
            "elite8": historical.get("elite8_rates", {}),
            "final4": historical.get("final4_rates", {}),
            "championship": historical.get("championship_rates", {}),
        }
    }

    _save_data(all_teams, "merged_team_data.json")

    # Summary
    n_with_barttorvik = sum(1 for t, d in all_teams.items() if t != "__model_params__" and d.get("barthag"))
    n_with_cbbpy = sum(1 for t, d in all_teams.items() if t != "__model_params__" and d.get("efg_pct"))

    print(f"\n{'='*60}")
    print(f"  DATA COLLECTION COMPLETE")
    print(f"{'='*60}")
    print(f"  Total teams: {len(all_teams) - 1}")
    print(f"  With Barttorvik metrics: {n_with_barttorvik}")
    print(f"  With CBBpy box score metrics: {n_with_cbbpy}")
    print(f"  Calibrated logistic coeff: {historical.get('calibrated_logistic_coeff', 'N/A')}")
    print(f"  Data saved to: {DATA_DIR}/merged_team_data.json")

    return all_teams


# ============================================================
#  Tournament Field Management
# ============================================================

def load_tournament_field(field_file=None):
    """Load the tournament field."""
    if field_file and os.path.exists(field_file):
        with open(field_file) as f:
            return json.load(f)

    default_path = os.path.join(DATA_DIR, "tournament_field.json")
    if os.path.exists(default_path):
        with open(default_path) as f:
            return json.load(f)

    return None


def save_tournament_field(field):
    """Save tournament field to disk."""
    filepath = os.path.join(DATA_DIR, "tournament_field.json")
    with open(filepath, "w") as f:
        json.dump(field, f, indent=2)
    print(f"[DATA] Tournament field saved to {filepath}")


def load_public_pick_pcts(pick_file=None):
    """Load ESPN public pick percentages."""
    if pick_file and os.path.exists(pick_file):
        with open(pick_file) as f:
            return json.load(f)

    default_path = os.path.join(DATA_DIR, "public_pick_pcts.json")
    if os.path.exists(default_path):
        with open(default_path) as f:
            return json.load(f)

    return None


def load_group_picks(group_file=None):
    """Load private group picks."""
    if group_file and os.path.exists(group_file):
        with open(group_file) as f:
            return json.load(f)

    default_path = os.path.join(DATA_DIR, "group_picks.json")
    if os.path.exists(default_path):
        with open(default_path) as f:
            return json.load(f)

    return None


# ============================================================
#  Utility Functions
# ============================================================

def _is_same_team(label, team_name):
    """Check if a box score team label matches our target team."""
    label_lower = label.lower().replace(".", "").replace("'", "").strip()
    name_lower = team_name.lower().replace(".", "").replace("'", "").strip()

    if label_lower == name_lower:
        return True
    if name_lower in label_lower or label_lower in name_lower:
        return True

    # Handle common variations
    words_label = set(label_lower.split())
    words_name = set(name_lower.split())
    if len(words_label & words_name) >= 1 and len(words_name) <= 2:
        return True

    return False


def _fuzzy_match_team(name, team_dict):
    """Fuzzy match team names across different data sources."""
    if name in team_dict:
        return name

    aliases = {
        "UConn": "Connecticut", "Connecticut": "UConn",
        "UNC": "North Carolina", "North Carolina": "UNC",
        "USC": "Southern California", "LSU": "Louisiana St.",
        "UCF": "Central Florida", "UNLV": "Nevada-Las Vegas",
        "SMU": "Southern Methodist", "BYU": "Brigham Young",
        "VCU": "Virginia Commonwealth", "TCU": "Texas Christian",
        "Ole Miss": "Mississippi", "Miss St.": "Mississippi St.",
        "Pitt": "Pittsburgh",
    }

    if name in aliases and aliases[name] in team_dict:
        return aliases[name]

    name_lower = name.lower().replace(".", "").replace("'", "")
    for team in team_dict:
        if team == "__model_params__":
            continue
        team_lower = team.lower().replace(".", "").replace("'", "")
        if name_lower in team_lower or team_lower in name_lower:
            return team

    return None


def _safe_float(s):
    try:
        return float(s.replace(",", "").strip())
    except (ValueError, AttributeError):
        return None


def _save_data(data, filename):
    filepath = os.path.join(DATA_DIR, filename)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"[DATA] Saved to {filepath}")
