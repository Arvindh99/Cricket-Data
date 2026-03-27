### IPL Match Winner Prediction

#### Live Demo: [Web Link](https://ipl-match-winner-prediction-0zdf.onrender.com)

#### Overview

This project is an **end-to-end machine learning pipeline and web application** that predicts the winner of IPL matches based on historical data.

It combines:

* Automated data ingestion
* Advanced feature engineering
* Model selection & training
* Interactive web interface for predictions

#### System Architecture

The project is divided into three major components:

##### 1. Data Pipeline

* Downloads IPL match data from CricSheet
* Extracts and processes JSON files
* Converts data into structured CSV datasets
* Fully automated using GitHub Actions

##### 2. Machine Learning Pipeline

* Feature engineering using:

  * Head-to-head stats
  * Venue-based performance
  * Contextual win probabilities

* Model selection using:

  * Random Forest
  * Gradient Boosting
  * Logistic Regression
  * KNN

* Achieved **>75% prediction accuracy**

#### 3. Web Application

* Backend: Flask API
* Frontend: HTML, CSS, JavaScript
* Interactive UI for match prediction

#### Features

* Predict match winner based on:

  * Teams
  * Venue
  * Toss winner & decision

* Dynamic UI with input validation
* Fully automated dataset updates
* Production deployment using **Render**

#### Project Structure

```
app/
  ├── static/
  ├── templates/
  └── app.py

data/
  ├── csv/
  │   ├── ipl_current_season.csv
  │   └── ipl_matches.csv
  ├── json/
  └── zips/

models/
  ├── mappings/
  ├── model/
  ├── preprocess/
  └── stats/

scripts/
  ├── create_csv.py
  └── download.py

src/
  └── Model_Generation.py

.github/workflows/
  ├── create_csv.yml
  └── update.yml
```

#### Automation Workflow

##### Data Update (`update.yml`)

* Runs daily
* Downloads latest IPL dataset
* Extracts new JSON files
* Removes old ZIP files

##### CSV Generation (`create_csv.yml`)

* Converts JSON → CSV
* Updates dataset automatically

#### Features

| Column         | Description          |
| -------------- | -------------------- |
| team1          | First team           |
| team2          | Second team          |
| city           | Match city           |
| venue          | Stadium              |
| season         | IPL season           |
| toss_winner    | Toss winner          |
| toss_decision  | Bat/Field decision   |
| winner         | Match winner         |
| win_by_runs    | Win margin (runs)    |
| win_by_wickets | Win margin (wickets) |

#### Machine Learning Pipeline

* Data preprocessing with:

  * Label Encoding
  * Feature scaling

* Feature engineering:

  * Team win rates
  * Venue performance
  * Context-based stats

* Model optimization using `RandomizedSearchCV`

#### Deployment

* Hosted on **Render**
* Uses **Gunicorn** for production
* Auto-deploy enabled via GitHub

#### Data Source

* IPL dataset: [https://cricsheet.org/downloads/](https://cricsheet.org/downloads/)

#### Contribution

Feel free to fork and improve:

* Model accuracy
* UI/UX
* Data pipeline efficiency

#### License

This project uses publicly available data from CricSheet.
