import requests
import zipfile
import io
import os

URL = "https://cricsheet.org/downloads/ipl_json.zip"

def download_and_extract():
    response = requests.get(URL)
    response.raise_for_status()

    z = zipfile.ZipFile(io.BytesIO(response.content))
    
    extract_path = "data"
    os.makedirs(extract_path, exist_ok=True)
    
    z.extractall(extract_path)
    print("Data downloaded and extracted.")

if __name__ == "__main__":
    download_and_extract()
