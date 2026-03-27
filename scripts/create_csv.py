import json
import csv
import os
from datetime import datetime

INPUT_DIR = "data/json"
OUTPUT_DIR = "data/csv"

CURRENT_YEAR = str(datetime.now().year)

CURRENT_FILE = os.path.join(OUTPUT_DIR, "ipl_current_season.csv")
OTHER_FILE = os.path.join(OUTPUT_DIR, "ipl_matches.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

current_season_rows = []
other_seasons_rows = []

for file in os.listdir(INPUT_DIR):
    if file.endswith(".json"):
        with open(os.path.join(INPUT_DIR, file), "r", encoding="utf-8") as f:
            data = json.load(f)

        info = data.get("info", {})

        teams = info.get("teams", ["NA", "NA"])
        team1, team2 = teams[0], teams[1]

        city = info.get("city", "NA")
        venue = info.get("venue", "NA")
        season = str(info.get("season", "NA"))

        toss_winner = info.get("toss", {}).get("winner", "NA")
        toss_decision = info.get("toss", {}).get("decision", "NA")

        outcome = info.get("outcome", {})
        winner = outcome.get("winner")
        if not winner:
            continue

        win_by_runs = outcome.get("by", {}).get("runs", 0)
        win_by_wickets = outcome.get("by", {}).get("wickets", 0)

        row = [
            team1, team2, city, venue, season,
            toss_winner, toss_decision, winner,
            win_by_runs, win_by_wickets
        ]

        if CURRENT_YEAR in season:
            current_season_rows.append(row)
        else:
            other_seasons_rows.append(row)

header = ['team1', 'team2', 'city', 'venue', 'season','toss_winner', 'toss_decision', 'winner','win_by_runs', 'win_by_wickets']

# Write current season CSV
with open(CURRENT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(current_season_rows)

# Write other seasons CSV
with open(OTHER_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(other_seasons_rows)

print(f"Current season CSV: {CURRENT_FILE}")
print(f"Other seasons CSV: {OTHER_FILE}")
