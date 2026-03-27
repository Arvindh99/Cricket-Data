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

#### Recent Enhancements (Machine Learning & Frontend)

I have successfully expanded the project beyond simple data extraction into a fully functional predictive web application!

Advanced Machine Learning Pipeline (`src/Model_Generation.py`)**

  * Implemented comprehensive pre-processing pipelines including `StandardScaler` and `LabelEncoder`.
  * Engineered powerful deterministic features (head-to-head stats, specific venue configurations, context-based win probabilities) to push test accuracy **>75%**.
  * Added an automated model selection pipeline using `RandomizedSearchCV` across a grid of algorithms (Random Forest, Gradient Boosting, Logistic Regression, KNN).
  * Model artifacts (pickles) and dataset mappings are automatically serialized into the nested `models/` directory.

Flask API & UI Frontend (`app/app.py`)**

  * Developed a lightweight Flask backend that serves a `/predict` JSON endpoint.
  * Designed a stunning, modern, "Crystal Ball" themed glassmorphic UI using Vanilla HTML/CSS/JS (`app/templates/index.html`, `app/static/style.css`).
  * The UI dynamically handles valid selections (preventing same-team matchups) and populates the toss-winner dropdown natively via Javascript.

**How to Run the App Locally:**

1. Navigate to the project root directory.
2. Execute the model generator to export artifacts: `python src/Model_Generation.py`
3. Launch the web server: `python app/app.py`
4. Access the Predictor UI at **http://127.0.0.1:5000**

#### Contribution

Feel free to fork the repo and improve:

* Deep Learning approaches
* More intuitive UI Animations
* Automated continuous model re-training

#### License

This project uses publicly available cricket data from CricSheet.
