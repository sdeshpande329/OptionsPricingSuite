import sys
from pathlib import Path

# Ensure project root (the folder containing `src/`) is on sys.path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.data_downloader import DataDownloader

def download_data():
    data_downloader = DataDownloader()
    data_downloader.clean_data()

if __name__ == "__main__":
    download_data()