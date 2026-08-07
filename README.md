### IPL Data Pipeline Automation

#### Data Pipeline

This project includes an automated data pipeline that downloads and processes IPL match data from CricSheet.

#### Data Source

* IPL match data is downloaded from CricSheet.
* Source: CricSheet IPL datasets

#### Workflow Automation

The data update process is fully automated using GitHub Actions.

#### Data Update Workflow

* Runs automatically on a scheduled basis.
* Downloads the latest IPL dataset from CricSheet.
* Extracts JSON match files.
* Removes outdated ZIP files after extraction.

#### CSV Generation Workflow

* Converts downloaded JSON files into structured CSV datasets.
* Automatically updates the dataset whenever new match data is available.

#### Project Structure

```
data/
  ├── csv/
  ├── json/
  └── zips/

scripts/
  ├── download.py
  └── create_csv.py

.github/workflows/
  ├── update.yml
  └── create_csv.yml
```
