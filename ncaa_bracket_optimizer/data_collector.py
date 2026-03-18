"""
NCAA Bracket Optimizer - Data Collection Module
Scrapes team stats from Barttorvik, sports-reference, and other public sources.
"""

import requests
import json
import re
import os
import time
from bs4 import BeautifulSoup
import csv
from io import StringIO

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)


def get_barttorvik_data(year=2026):
    """
    Fetch team rankings and advanced metrics from Barttorvik.
    Returns dict keyed by team name with metrics.
    """
    print(f"[DATA] Fetching Barttorvik T-Rank data for {year}...")
    url = f"https://barttorvik.com/trank.php?year={year}&sort=&lastx=0&hession=All&cut=All&type=pointed&quad=5&mingames=0&top=0&start=20251101&end=20260501&ression=All&defession=All&venue=All&team=&conession=All&session=All&begin=20251101&end2=20260501&sitession=SQ&iession=All"

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    }

    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        teams = {}

        # Parse the table - Barttorvik uses a specific table structure
        table = soup.find("table", {"id": "pointed-pointed"})
        if not table:
            # Try alternative table IDs
            tables = soup.find_all("table")
            if tables:
                table = tables[0]

        if table:
            rows = table.find_all("tr")
            for row in rows[1:]:  # skip header
                cols = row.find_all("td")
                if len(cols) >= 15:
                    try:
                        team_name = cols[1].get_text(strip=True)
                        # Clean up team name
                        team_name = re.sub(r'\d+$', '', team_name).strip()

                        teams[team_name] = {
                            "rank": _safe_float(cols[0].get_text(strip=True)),
                            "conference": cols[2].get_text(strip=True),
                            "record": cols[3].get_text(strip=True),
                            "adj_oe": _safe_float(cols[4].get_text(strip=True)),  # Adjusted Offensive Efficiency
                            "adj_de": _safe_float(cols[5].get_text(strip=True)),  # Adjusted Defensive Efficiency
                            "barthag": _safe_float(cols[6].get_text(strip=True)),  # Power Rating (0-1)
                            "eff_margin": None,  # computed below
                            "tempo": _safe_float(cols[7].get_text(strip=True) if len(cols) > 7 else "0"),
                            "sos": _safe_float(cols[14].get_text(strip=True) if len(cols) > 14 else "0"),
                        }
                        # Efficiency margin = AdjOE - AdjDE
                        if teams[team_name]["adj_oe"] and teams[team_name]["adj_de"]:
                            teams[team_name]["eff_margin"] = (
                                teams[team_name]["adj_oe"] - teams[team_name]["adj_de"]
                            )
                    except (ValueError, IndexError):
                        continue

        if teams:
            print(f"[DATA] Successfully scraped {len(teams)} teams from Barttorvik")
            _save_data(teams, "barttorvik_data.json")
        else:
            print("[DATA] Warning: Could not parse Barttorvik table, will try CSV export...")
            teams = _try_barttorvik_csv(year)

        return teams

    except Exception as e:
        print(f"[DATA] Barttorvik scrape failed: {e}")
        print("[DATA] Trying alternative data source...")
        return _try_barttorvik_csv(year)


def _try_barttorvik_csv(year=2026):
    """Try the Barttorvik CSV/JSON endpoint as fallback."""
    try:
        url = f"https://barttorvik.com/getadvstats.php?year={year}&type=pointed"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=30)

        if resp.status_code == 200:
            data = resp.json() if resp.headers.get('content-type', '').find('json') >= 0 else None
            if data:
                teams = {}
                for row in data:
                    team_name = row.get("team", row.get("Team", ""))
                    if team_name:
                        teams[team_name] = {
                            "rank": row.get("rk", row.get("Rank", 0)),
                            "adj_oe": row.get("adjoe", row.get("AdjOE", 0)),
                            "adj_de": row.get("adjde", row.get("AdjDE", 0)),
                            "barthag": row.get("barthag", row.get("Barthag", 0)),
                            "eff_margin": (row.get("adjoe", 0) or 0) - (row.get("adjde", 0) or 0),
                            "tempo": row.get("tempo", row.get("Tempo", 0)),
                        }
                print(f"[DATA] Got {len(teams)} teams from Barttorvik API")
                _save_data(teams, "barttorvik_data.json")
                return teams
    except Exception as e:
        print(f"[DATA] Barttorvik CSV fallback also failed: {e}")

    return {}


def get_sports_reference_data(year=2026):
    """
    Fetch basic team stats from Sports Reference (basketball-reference).
    """
    print(f"[DATA] Fetching Sports Reference data for {year}...")
    url = f"https://www.sports-reference.com/cbb/seasons/men/{year}-ratings.html"
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        teams = {}
        table = soup.find("table", {"id": "ratings"})
        if table:
            tbody = table.find("tbody")
            if tbody:
                for row in tbody.find_all("tr"):
                    if row.get("class") and "thead" in row.get("class", []):
                        continue
                    cols = row.find_all("td")
                    if len(cols) >= 10:
                        try:
                            team_link = cols[0].find("a")
                            team_name = team_link.get_text(strip=True) if team_link else cols[0].get_text(strip=True)

                            teams[team_name] = {
                                "conference": cols[1].get_text(strip=True),
                                "wins": _safe_int(cols[2].get_text(strip=True)),
                                "losses": _safe_int(cols[3].get_text(strip=True)),
                                "srs": _safe_float(cols[6].get_text(strip=True)),  # Simple Rating System
                                "sos": _safe_float(cols[7].get_text(strip=True)),
                                "off_rtg": _safe_float(cols[10].get_text(strip=True) if len(cols) > 10 else "0"),
                                "def_rtg": _safe_float(cols[11].get_text(strip=True) if len(cols) > 11 else "0"),
                            }
                        except (ValueError, IndexError):
                            continue

        if teams:
            print(f"[DATA] Successfully scraped {len(teams)} teams from Sports Reference")
            _save_data(teams, "sports_reference_data.json")

        return teams

    except Exception as e:
        print(f"[DATA] Sports Reference scrape failed: {e}")
        return {}


def get_ncaa_net_rankings():
    """
    Fetch NCAA NET rankings (the official selection committee ranking system).
    """
    print("[DATA] Fetching NCAA NET rankings...")
    url = "https://www.ncaa.com/rankings/basketball-men/d1/ncaa-mens-basketball-net-rankings"
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        rankings = {}
        table = soup.find("table")
        if table:
            rows = table.find_all("tr")
            for row in rows[1:]:
                cols = row.find_all("td")
                if len(cols) >= 3:
                    try:
                        rank = _safe_int(cols[0].get_text(strip=True))
                        team = cols[1].get_text(strip=True)
                        record = cols[2].get_text(strip=True) if len(cols) > 2 else ""
                        rankings[team] = {
                            "net_rank": rank,
                            "record": record,
                        }
                    except (ValueError, IndexError):
                        continue

        if rankings:
            print(f"[DATA] Got {len(rankings)} teams from NET rankings")
            _save_data(rankings, "net_rankings.json")

        return rankings

    except Exception as e:
        print(f"[DATA] NET rankings fetch failed: {e}")
        return {}


def get_espn_bpi():
    """
    Fetch ESPN's Basketball Power Index (BPI) data.
    """
    print("[DATA] Fetching ESPN BPI data...")
    url = "https://www.espn.com/mens-college-basketball/bpi"
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        teams = {}
        # ESPN BPI page structure
        table = soup.find("table", {"class": "Table"})
        if table:
            rows = table.find_all("tr")
            for row in rows[1:]:
                cols = row.find_all("td")
                if len(cols) >= 5:
                    try:
                        team = cols[1].get_text(strip=True)
                        teams[team] = {
                            "bpi": _safe_float(cols[3].get_text(strip=True)),
                            "bpi_off": _safe_float(cols[4].get_text(strip=True) if len(cols) > 4 else "0"),
                            "bpi_def": _safe_float(cols[5].get_text(strip=True) if len(cols) > 5 else "0"),
                        }
                    except (ValueError, IndexError):
                        continue

        if teams:
            print(f"[DATA] Got {len(teams)} teams from ESPN BPI")
            _save_data(teams, "espn_bpi.json")

        return teams
    except Exception as e:
        print(f"[DATA] ESPN BPI fetch failed: {e}")
        return {}


def load_tournament_field(field_file=None):
    """
    Load the tournament field (68 teams with seeds and regions).
    This should be populated after Selection Sunday.

    Expected format (JSON):
    {
        "South": {
            "1": "Team A",
            "2": "Team B",
            ...
            "16": "Team P"
        },
        "East": {...},
        "Midwest": {...},
        "West": {...}
    }
    """
    if field_file and os.path.exists(field_file):
        with open(field_file) as f:
            return json.load(f)

    default_path = os.path.join(DATA_DIR, "tournament_field.json")
    if os.path.exists(default_path):
        with open(default_path) as f:
            return json.load(f)

    print("[DATA] No tournament field found. Please create data/tournament_field.json")
    print("[DATA] after Selection Sunday with the bracket.")
    return None


def load_public_pick_pcts(pick_file=None):
    """
    Load ESPN public pick percentages for each team to advance to each round.

    Expected format (JSON):
    {
        "Team A": {
            "r64": 0.98,
            "r32": 0.85,
            "s16": 0.60,
            "e8": 0.35,
            "f4": 0.18,
            "champ": 0.08
        },
        ...
    }
    """
    if pick_file and os.path.exists(pick_file):
        with open(pick_file) as f:
            return json.load(f)

    default_path = os.path.join(DATA_DIR, "public_pick_pcts.json")
    if os.path.exists(default_path):
        with open(default_path) as f:
            return json.load(f)

    print("[DATA] No public pick data found. Will use seed-based estimates.")
    return None


def load_group_picks(group_file=None):
    """
    Load picks from your private ESPN group for more targeted optimization.

    Expected format (JSON):
    {
        "player1": {
            "champion": "Team A",
            "final_four": ["Team A", "Team B", "Team C", "Team D"],
            ...
        }
    }
    """
    if group_file and os.path.exists(group_file):
        with open(group_file) as f:
            return json.load(f)

    default_path = os.path.join(DATA_DIR, "group_picks.json")
    if os.path.exists(default_path):
        with open(default_path) as f:
            return json.load(f)

    return None


def collect_all_data(year=2026):
    """
    Master function to collect all available data and merge into unified dataset.
    """
    print("=" * 60)
    print(f"  NCAA BRACKET OPTIMIZER - Data Collection ({year})")
    print("=" * 60)

    barttorvik = get_barttorvik_data(year)
    time.sleep(2)  # Be polite with requests

    sports_ref = get_sports_reference_data(year)
    time.sleep(2)

    net_rankings = get_ncaa_net_rankings()
    time.sleep(2)

    espn_bpi = get_espn_bpi()

    # Merge all data into unified team profiles
    all_teams = {}

    # Start with Barttorvik as the base (best advanced metrics)
    for team, data in barttorvik.items():
        all_teams[team] = {
            "name": team,
            "adj_oe": data.get("adj_oe"),
            "adj_de": data.get("adj_de"),
            "eff_margin": data.get("eff_margin"),
            "barthag": data.get("barthag"),
            "tempo": data.get("tempo"),
            "barttorvik_rank": data.get("rank"),
        }

    # Merge Sports Reference data
    for team, data in sports_ref.items():
        matched = _fuzzy_match_team(team, all_teams)
        if matched:
            all_teams[matched]["srs"] = data.get("srs")
            all_teams[matched]["sos_sr"] = data.get("sos")
            all_teams[matched]["off_rtg_sr"] = data.get("off_rtg")
            all_teams[matched]["def_rtg_sr"] = data.get("def_rtg")
        else:
            all_teams[team] = {"name": team, **data}

    # Merge NET rankings
    for team, data in net_rankings.items():
        matched = _fuzzy_match_team(team, all_teams)
        if matched:
            all_teams[matched]["net_rank"] = data.get("net_rank")

    # Merge ESPN BPI
    for team, data in espn_bpi.items():
        matched = _fuzzy_match_team(team, all_teams)
        if matched:
            all_teams[matched]["bpi"] = data.get("bpi")
            all_teams[matched]["bpi_off"] = data.get("bpi_off")
            all_teams[matched]["bpi_def"] = data.get("bpi_def")

    # Save merged dataset
    _save_data(all_teams, "merged_team_data.json")

    print(f"\n[DATA] Final merged dataset: {len(all_teams)} teams")
    print(f"[DATA] Data sources collected:")
    print(f"  - Barttorvik: {len(barttorvik)} teams")
    print(f"  - Sports Reference: {len(sports_ref)} teams")
    print(f"  - NET Rankings: {len(net_rankings)} teams")
    print(f"  - ESPN BPI: {len(espn_bpi)} teams")

    return all_teams


# ---- Utility Functions ----

def _safe_float(s):
    """Safely convert string to float."""
    try:
        return float(s.replace(",", "").strip())
    except (ValueError, AttributeError):
        return None

def _safe_int(s):
    """Safely convert string to int."""
    try:
        return int(s.replace(",", "").strip())
    except (ValueError, AttributeError):
        return None

def _fuzzy_match_team(name, team_dict):
    """
    Simple fuzzy matching for team names across different sources.
    E.g., "UConn" vs "Connecticut", "UNC" vs "North Carolina"
    """
    # Direct match
    if name in team_dict:
        return name

    # Common aliases
    aliases = {
        "UConn": "Connecticut",
        "Connecticut": "UConn",
        "UNC": "North Carolina",
        "North Carolina": "UNC",
        "USC": "Southern California",
        "LSU": "Louisiana St.",
        "UCF": "Central Florida",
        "UNLV": "Nevada-Las Vegas",
        "SMU": "Southern Methodist",
        "BYU": "Brigham Young",
        "VCU": "Virginia Commonwealth",
        "TCU": "Texas Christian",
    }

    if name in aliases and aliases[name] in team_dict:
        return aliases[name]

    # Substring match
    name_lower = name.lower().replace(".", "").replace("'", "")
    for team in team_dict:
        team_lower = team.lower().replace(".", "").replace("'", "")
        if name_lower in team_lower or team_lower in name_lower:
            return team

    return None

def _save_data(data, filename):
    """Save data to JSON file."""
    filepath = os.path.join(DATA_DIR, filename)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[DATA] Saved to {filepath}")
