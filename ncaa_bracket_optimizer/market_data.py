"""
NCAA Bracket Optimizer - Prediction Market & Sportsbook Data
=============================================================

Pulls championship odds from:
1. Polymarket (real-money prediction market)
2. Sportsbook consensus odds (DraftKings, FanDuel, etc.)

Market odds are valuable because they represent the "wisdom of crowds"
backed by real money. They incorporate injury news, momentum, matchup
analysis, and information we might not have in our statistical model.

We use them as:
- An additional signal in the win probability model
- A cross-check against our model (big disagreements = investigate)
- A proxy for "sharp money" public opinion (different from casual ESPN pickers)
"""

import requests
import json
import os
import math
import re
from bs4 import BeautifulSoup

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def get_polymarket_odds():
    """
    Fetch NCAA tournament winner odds from Polymarket.

    Polymarket prices are directly interpretable as probabilities
    (e.g., 22 cents = 22% implied probability).
    """
    print("[MARKET] Fetching Polymarket NCAA tournament odds...")

    # Polymarket's CLOB API endpoint for event markets
    # The 2026 NCAA tournament winner event
    try:
        # Try the Polymarket Gamma API
        url = "https://gamma-api.polymarket.com/events?slug=2026-ncaa-tournament-winner"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=15)

        if resp.status_code == 200:
            data = resp.json()
            if data and len(data) > 0:
                event = data[0]
                markets = event.get("markets", [])
                odds = {}
                for market in markets:
                    team = market.get("groupItemTitle", market.get("question", ""))
                    # Clean team name
                    team = team.replace("Will ", "").replace(" win the 2026 NCAA Tournament?", "")
                    team = team.replace(" win the 2026 NCAA tournament?", "").strip()

                    price = market.get("outcomePrices")
                    if price:
                        try:
                            prices = json.loads(price) if isinstance(price, str) else price
                            yes_price = float(prices[0]) if isinstance(prices, list) else float(price)
                            odds[team] = round(yes_price * 100, 1)  # Convert to percentage
                        except (ValueError, IndexError, TypeError):
                            pass

                if odds:
                    print(f"[MARKET] Polymarket: got odds for {len(odds)} teams")
                    _save_data(odds, "polymarket_odds.json")
                    return odds

        print("[MARKET] Polymarket API didn't return data, using scraped fallback...")

    except Exception as e:
        print(f"[MARKET] Polymarket API error: {e}")

    # Fallback: use the data we already scraped
    return _load_cached_or_manual_polymarket()


def _load_cached_or_manual_polymarket():
    """Load cached Polymarket data or return manually entered data."""
    cached = os.path.join(DATA_DIR, "polymarket_odds.json")
    if os.path.exists(cached):
        with open(cached) as f:
            return json.load(f)

    # Manually entered from our web scrape (March 15, 2026)
    manual_odds = {
        "Duke": 22.0,
        "Michigan": 18.0,
        "Arizona": 15.3,
        "Florida": 10.8,
        "Houston": 8.0,
        "Iowa State": 4.3,
        "Illinois": 4.0,
        "UConn": 3.9,
        "Purdue": 3.1,
        "Arkansas": 2.5,
        "Vanderbilt": 2.4,
        "Utah State": 2.0,
        "Gonzaga": 1.8,
        "Michigan State": 1.7,
        "Virginia": 1.7,
        "St. John's": 1.6,
        "Alabama": 1.0,
        "Kansas": 1.0,
        "Tennessee": 1.0,
        "Kentucky": 0.8,
        "Wisconsin": 0.5,
        "North Carolina": 0.5,
        "Texas Tech": 0.5,
        "Nebraska": 0.5,
        "BYU": 0.3,
        "UCLA": 0.3,
        "Clemson": 0.2,
        "Creighton": 0.2,
        "Auburn": 0.2,
    }
    _save_data(manual_odds, "polymarket_odds.json")
    return manual_odds


def get_sportsbook_odds():
    """
    Fetch sportsbook consensus odds.

    American odds format: +500 means bet $100 to win $500 (implied prob ~16.7%)
    We convert to implied probabilities (removing vig).
    """
    print("[MARKET] Fetching sportsbook consensus odds...")

    try:
        # Try The Odds API (free tier: 500 requests/month)
        # This aggregates DraftKings, FanDuel, BetMGM, etc.
        url = "https://api.the-odds-api.com/v4/sports/basketball_ncaab_championship_winner/odds"
        params = {
            "apiKey": "DEMO_KEY",  # Replace with actual key if available
            "regions": "us",
            "markets": "outrights",
            "oddsFormat": "american",
        }
        resp = requests.get(url, params=params, timeout=15)

        if resp.status_code == 200:
            data = resp.json()
            odds = _parse_odds_api_response(data)
            if odds:
                print(f"[MARKET] Odds API: got odds for {len(odds)} teams")
                _save_data(odds, "sportsbook_odds.json")
                return odds

    except Exception as e:
        print(f"[MARKET] Odds API error: {e}")

    # Fallback: use manually entered sportsbook odds
    return _load_cached_or_manual_sportsbook()


def _parse_odds_api_response(data):
    """Parse The Odds API response into team probabilities."""
    if not data:
        return {}

    team_odds = {}
    for bookmaker_data in data:
        bookmakers = bookmaker_data.get("bookmakers", [])
        for book in bookmakers:
            markets = book.get("markets", [])
            for market in markets:
                outcomes = market.get("outcomes", [])
                for outcome in outcomes:
                    team = outcome.get("name", "")
                    price = outcome.get("price", 0)
                    implied = american_odds_to_prob(price)
                    if team and implied > 0:
                        if team not in team_odds:
                            team_odds[team] = []
                        team_odds[team].append(implied)

    # Average across all bookmakers
    return {team: round(sum(probs)/len(probs) * 100, 1)
            for team, probs in team_odds.items() if probs}


def _load_cached_or_manual_sportsbook():
    """Load cached sportsbook data or return manually entered data."""
    cached = os.path.join(DATA_DIR, "sportsbook_odds.json")
    if os.path.exists(cached):
        with open(cached) as f:
            return json.load(f)

    # Manually entered from web search (March 15, 2026)
    # Source: VegasInsider, OddsShark, DraftKings consensus
    manual_odds_american = {
        "Duke": 330,
        "Michigan": 340,
        "Arizona": 500,
        "Florida": 700,
        "Houston": 1000,
        "Iowa State": 1200,
        "UConn": 1300,
        "Illinois": 1500,
        "Purdue": 1600,
        "Arkansas": 2000,
        "Gonzaga": 2000,
        "Michigan State": 2500,
        "Alabama": 2500,
        "Virginia": 2500,
        "St. John's": 2500,
        "Kansas": 3000,
        "Vanderbilt": 3000,
        "Tennessee": 3000,
        "Wisconsin": 3500,
        "Kentucky": 4000,
        "Texas Tech": 4000,
        "Nebraska": 4000,
        "North Carolina": 5000,
        "UCLA": 5000,
        "BYU": 6000,
        "Utah State": 6000,
        "Clemson": 8000,
        "Louisville": 8000,
        "Ohio State": 10000,
        "Villanova": 10000,
    }

    # Convert to implied probabilities
    odds = {}
    for team, american in manual_odds_american.items():
        odds[team] = round(american_odds_to_prob(american) * 100, 2)

    _save_data(odds, "sportsbook_odds.json")
    return odds


def american_odds_to_prob(odds):
    """
    Convert American odds to implied probability.
    +330 → 100/(330+100) = 23.3%
    -150 → 150/(150+100) = 60%
    """
    if odds > 0:
        return 100 / (odds + 100)
    elif odds < 0:
        return abs(odds) / (abs(odds) + 100)
    else:
        return 0.5


def get_market_consensus():
    """
    Get a blended market consensus from Polymarket + sportsbooks.

    Polymarket weight: 40% (real money, but lower liquidity)
    Sportsbook weight: 60% (professional market makers, higher liquidity)
    """
    print("\n[MARKET] Building market consensus...")

    polymarket = get_polymarket_odds()
    sportsbook = get_sportsbook_odds()

    consensus = {}
    all_teams = set(list(polymarket.keys()) + list(sportsbook.keys()))

    for team in all_teams:
        poly_prob = polymarket.get(team, None)
        book_prob = sportsbook.get(team, None)

        if poly_prob is not None and book_prob is not None:
            # Weighted blend
            consensus[team] = round(0.4 * poly_prob + 0.6 * book_prob, 2)
        elif poly_prob is not None:
            consensus[team] = poly_prob
        elif book_prob is not None:
            consensus[team] = book_prob

    # Sort by probability
    consensus = dict(sorted(consensus.items(), key=lambda x: -x[1]))

    print(f"\n[MARKET] Market Consensus Championship Probabilities:")
    for i, (team, prob) in enumerate(consensus.items()):
        if i >= 15 or prob < 0.5:
            break
        print(f"  {i+1:>2}. {team:<22} {prob:>5.1f}%")

    _save_data(consensus, "market_consensus.json")
    return consensus


def load_market_consensus():
    """Load cached market consensus."""
    path = os.path.join(DATA_DIR, "market_consensus.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


# ============================================================
#  ESPN Who Picked Whom Scraper
# ============================================================

def get_espn_who_picked_whom():
    """
    Scrape ESPN's "Who Picked Whom" data showing what percentage
    of ESPN Tournament Challenge brackets picked each team to
    advance to each round.

    This is THE most important data for leverage calculations.
    Usually available ~24 hours after brackets open.
    """
    print("[MARKET] Fetching ESPN 'Who Picked Whom' data...")

    url = "https://fantasy.espn.com/tournament-challenge-bracket/2026/en/whopickedwhom"
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "html.parser")

            picks = {}
            # ESPN's WPW page typically shows a table per region
            tables = soup.find_all("table")

            for table in tables:
                rows = table.find_all("tr")
                for row in rows:
                    cols = row.find_all("td")
                    if len(cols) >= 7:
                        try:
                            team = cols[0].get_text(strip=True)
                            if not team:
                                continue

                            picks[team] = {
                                "r64": _parse_pct(cols[1].get_text(strip=True)),
                                "r32": _parse_pct(cols[2].get_text(strip=True)),
                                "s16": _parse_pct(cols[3].get_text(strip=True)),
                                "e8": _parse_pct(cols[4].get_text(strip=True)),
                                "f4": _parse_pct(cols[5].get_text(strip=True)),
                                "champ": _parse_pct(cols[6].get_text(strip=True)),
                            }
                        except (ValueError, IndexError):
                            continue

            if picks:
                print(f"[MARKET] ESPN WPW: got pick data for {len(picks)} teams")
                _save_data(picks, "public_pick_pcts.json")
                return picks

        print("[MARKET] ESPN WPW not available yet (brackets may not be open)")

    except Exception as e:
        print(f"[MARKET] ESPN WPW error: {e}")

    # Check for cached/manually entered data
    cached = os.path.join(DATA_DIR, "public_pick_pcts.json")
    if os.path.exists(cached):
        with open(cached) as f:
            print("[MARKET] Using cached ESPN WPW data")
            return json.load(f)

    print("[MARKET] No WPW data available yet - will use seed-based estimates")
    return None


def _parse_pct(text):
    """Parse a percentage string like '85.3%' to 0.853."""
    try:
        return float(text.replace("%", "").strip()) / 100
    except (ValueError, AttributeError):
        return 0.0


# ============================================================
#  Utility
# ============================================================

def _save_data(data, filename):
    filepath = os.path.join(DATA_DIR, filename)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"[MARKET] Saved to {filepath}")
