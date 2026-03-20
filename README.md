### IPL Dataset Pipeline

This project automates the process of collecting, updating, and preparing IPL match data for machine learning models that predict match winners.

#### Overview

The pipeline performs the following tasks:

1. Downloads latest IPL match data (JSON format) from CricSheet
2. Stores daily ZIP archives
3. Extracts only new JSON match files
4. Cleans up old ZIP files (keeps only the latest)
5. Converts JSON data into a structured CSV dataset
6. Fully automated using GitHub Actions

#### Data Source

* IPL data is sourced from:
  https://cricsheet.org/downloads/

#### Project Structure

```
data/
  json/        # Extracted match JSON files
  zips/        # Downloaded ZIP archives (latest only)
  csv/         # Final dataset for ML

scripts/
  download.py      # Downloads and extracts JSON files
  create_csv.py    # Converts JSON → CSV dataset

.github/workflows/
  download.yml     # Automates data download
  create_csv.yml   # Automates CSV creation
```

#### Automation Workflow

##### Data Download Workflow (`download.yml`)

Runs: Daily at 2 AM UTC

Steps:

* Downloads latest IPL ZIP file
* Saves ZIP with date (`ipl_YYYY-MM-DD.zip`)
* Extracts only **new JSON files**
* Logs update activity
* Deletes old ZIP files (keeps latest)
* Commits changes to repository

##### CSV Creation Workflow (`create_csv.yml`)

Triggered: Automatically after download workflow completes

Steps:

* Reads all JSON match files
* Extracts relevant match-level features
* Generates a clean CSV dataset
* Commits updated dataset

#### Generated Dataset

Output file:

```
data/csv/ipl_matches.csv
```

#### Features included:

| Column         | Description           |
| -------------- | --------------------- |
| team1          | First team            |
| team2          | Second team           |
| city           | Match city            |
| venue          | Stadium               |
| season         | IPL season            |
| toss_winner    | Toss winner           |
| toss_decision  | Bat/Field decision    |
| winner         | Match winner (target) |
| win_by_runs    | Win margin (runs)     |
| win_by_wickets | Win margin (wickets)  |

#### Purpose

This dataset is designed for:

* Match winner prediction models
* Classification tasks (ML/DL)
* Feature engineering experiments

#### Automation via GitHub Actions

No manual work required:

* Data updates daily
* Dataset always stays current
* CSV is regenerated automatically

#### Data Handling Rules

* Only **new JSON files** are added (no duplicates)
* Old ZIP files are removed automatically
* Matches without results are excluded from CSV

#### Future Improvements

* Add team performance features
* Include head-to-head statistics
* Build ML models (Logistic Regression, XGBoost)
* Deploy prediction API

#### Contribution

Feel free to fork the repo and improve:

* Feature engineering
* Model performance
* Automation workflows

#### License

This project uses publicly available cricket data from CricSheet.
