import json
import csv
import os

INPUT_DIR = "data/json"
OUTPUT_DIR = "data/csv"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "ipl_matches.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

rows = []

for file in os.listdir(INPUT_DIR):
    if file.endswith(".json"):
        with open(os.path.join(INPUT_DIR, file), "r", encoding="utf-8") as f:
            data = json.load(f)

        info = data.get("info", {})

        teams = info.get("teams", ["NA", "NA"])
        team1, team2 = teams[0], teams[1]

        city = info.get("city", "NA")
        venue = info.get("venue", "NA")
        season = info.get("season", "NA")

        toss_winner = info.get("toss", {}).get("winner", "NA")
        toss_decision = info.get("toss", {}).get("decision", "NA")

        outcome = info.get("outcome", {})
        winner = outcome.get("winner")
        if not winner:
            continue

        win_by_runs = outcome.get("by", {}).get("runs", 0)
        win_by_wickets = outcome.get("by", {}).get("wickets", 0)

        rows.append([team1, team2, city, venue, season, toss_winner, toss_decision, winner, win_by_runs, win_by_wickets])

header = ['team1', 'team2', 'city', 'venue', 'season', 'toss_winner', 'toss_decision', 'winner', 'win_by_runs', 'win_by_wickets']

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)

print(f"CSV created: {OUTPUT_FILE}")
