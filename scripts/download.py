import requests
import zipfile
import io
import os
from datetime import datetime

URL = "https://cricsheet.org/downloads/ipl_json.zip"

DATA_DIR = "data/json"
ZIP_DIR = "data/zips"
LOG_FILE = "data/update_log.txt"

def setup_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(ZIP_DIR, exist_ok=True)

def get_existing_files():
    return set(os.listdir(DATA_DIR))

def download_zip():
    response = requests.get(URL)
    response.raise_for_status()
    return response.content

def save_zip(content):
    today = datetime.utcnow().strftime("%Y-%m-%d")
    zip_path = os.path.join(ZIP_DIR, f"ipl_{today}.zip")
    
    with open(zip_path, "wb") as f:
        f.write(content)
    
    return zip_path

def extract_new_files(zip_content, existing_files):
    z = zipfile.ZipFile(io.BytesIO(zip_content))
    new_files_count = 0

    for file in z.namelist():
        if file.endswith(".json"):
            filename = os.path.basename(file)

            if filename not in existing_files:
                with z.open(file) as source, open(os.path.join(DATA_DIR, filename), "wb") as target:
                    target.write(source.read())
                new_files_count += 1

    return new_files_count

def write_log(new_count, total_files):
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    log_entry = (
        f"{now} | New files: {new_count} | Total files: {total_files}\n"
    )

    with open(LOG_FILE, "a") as f:
        f.write(log_entry)

def main():
    setup_dirs()

    existing_files = get_existing_files()

    zip_content = download_zip()
    save_zip(zip_content)

    new_files = extract_new_files(zip_content, existing_files)

    total_files = len(os.listdir(DATA_DIR))

    write_log(new_files, total_files)

    print(f"New files added: {new_files}")
    print(f"Total files: {total_files}")

if __name__ == "__main__":
    main()
