# NCAA Bracket Optimizer — Selection Sunday Report
**March 15, 2026 | V1 Brackets (Pre-ESPN "Who Picked Whom")**

---

## Model Configuration

| Component | Weight | Source |
|-----------|--------|--------|
| Efficiency Margin (logistic) | 40% | Sports Reference 2026 |
| Barthag (log5) | 10% | Sports Reference 2026 |
| Seed History | 10% | 1,248 historical R64 games (1985-2024) |
| Market Consensus | 12% | Polymarket (40%) + Sportsbooks (60%) |
| *Four Factors* | *0% — not yet active* | *Pending CBBpy box score pull* |
| *Upset Factors* | *0% — not yet active* | *Pending CBBpy data* |

Calibrated logistic coefficient: **0.17** (MLE from historical data)
Simulations: **500,000** tournament runs + **10,000** pool contest sims

---

## Championship Probabilities (Model)

| Rank | Team | Model | Market | Diff | Signal |
|------|------|-------|--------|------|--------|
| 1 | **Duke** | 30.6% | 22.6% | +8.0% | Model likes more |
| 2 | **Michigan** | 27.9% | 20.6% | +7.2% | Model likes more |
| 3 | Arizona | 14.3% | 16.2% | -1.9% | Aligned |
| 4 | Florida | 6.0% | 11.8% | -5.8% | Market likes more |
| 5 | Houston | 4.8% | 8.4% | -3.6% | Market likes more |
| 6 | Iowa State | 4.1% | 6.2% | -2.1% | Market likes more |
| 7 | Illinois | 3.9% | 5.3% | -1.4% | Close |
| 8 | Purdue | 2.8% | 4.7% | -1.9% | Market likes more |

**Key insight:** Our model is significantly more bullish on Duke and Michigan than the market. This is driven by their elite efficiency margins (+44.5 and +44.6 respectively). The market may be undervaluing the gap between tier 1 (Duke/Michigan) and tier 2 (Arizona/Florida/Houston).

---

## Bracket A — "Model Favorite" (Champion: Duke)

| Region | S16 | E8 | Final Four |
|--------|-----|-----|------------|
| East | Duke, Michigan State | **Duke** | **Duke** |
| South | Florida, Illinois | **Illinois** | **Illinois** |
| West | Arizona, Purdue | **Arizona** | **Arizona** |
| Midwest | Michigan, Iowa State | **Michigan** | **Michigan** |

**Championship:** Duke over Michigan
**Expected Points:** 1,114

---

## Bracket B — "Contrarian Upside" (Champion: Michigan)

| Region | S16 | E8 | Final Four |
|--------|-----|-----|------------|
| East | Duke, Michigan State | **Duke** | **Duke** |
| South | Vanderbilt, Houston | **Houston** | **Houston** |
| West | Arizona, Purdue | **Purdue** | **Purdue** |
| Midwest | Michigan, Iowa State | **Michigan** | **Michigan** |

**Championship:** Michigan over Duke
**Expected Points:** 1,067

---

## Portfolio Divergence

| | Bracket A | Bracket B |
|--|-----------|-----------|
| Champion | Duke | Michigan |
| Final Four | Duke, Illinois, Arizona, Michigan | Duke, Houston, Purdue, Michigan |
| FF Overlap | 2/4 shared (Duke, Michigan) |
| Unique to B | Houston (South), Purdue (West) |

**Why Houston in Bracket B?** 28.8% E8 probability, 2-seed with elite defense (adj DE 86.6). If Florida or Illinois stumble, Houston is the South's most likely alternative winner.

**Why Purdue in Bracket B?** 22.3% E8 probability, 2-seed with the #1 offensive efficiency in the country (adj OE 132.1). If Arizona has an off night, Purdue has the firepower to capitalize.

---

## Pool Contest Simulation

| Metric | Our Portfolio | Random Entrant |
|--------|--------------|----------------|
| Win pool | **29.4%** | 2.0% |
| Top 3 finish | **56.5%** | 6.0% |
| Edge over random | **14.7x** (win) / **9.4x** (top 3) |
| Median best finish | **3rd place** |
| Avg best finish | 5.1 |

**Position distribution (best of A or B):**
- 1st place: 29.4%
- 2nd place: 16.2%
- 3rd place: 10.8%
- Top 5: 69.7%

---

## What Changes Before Lock

1. **ESPN "Who Picked Whom"** (Tue/Wed) — replaces seed-based estimates with actual public pick %. This is the single biggest upgrade — it transforms our leverage calculations from estimates to precise data.

2. **CBBpy Box Scores** — adds Four Factors (eFG%, TO Rate, ORB%, FT Rate) which currently aren't active. These add nuance beyond raw efficiency margin, especially for identifying upset-prone teams.

3. **Market Odds Refresh** — Polymarket and sportsbook odds will shift as more money comes in. Re-pulling daily.

4. **Prize Structure** — once you confirm the dollar amounts for 1st/2nd/3rd, the optimizer can weight accordingly (e.g., if 1st is 60% of the pot, we'd lean Bracket A more aggressive).

5. **Group Picks** — if we can see your ESPN group's brackets before lock, we can specifically exploit their blind spots.

---

## Next Steps

```
# Tomorrow (Mon): refresh market odds
python update.py --step market

# Tue/Wed: ESPN picks become available (THE BIG ONE)
python update.py --step picks

# Wed before lock: final optimization
python update.py --step final --prize [1st] [2nd] [3rd]
```
