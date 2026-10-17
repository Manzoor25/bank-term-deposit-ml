# =============================================================
# WHAT THIS FILE DOES:
# Downloads the public UCI Bank Marketing file and loads it
# into a table (a pandas DataFrame). A DataFrame is like a
# spreadsheet that Python can filter and measure.
#
# HOW IT CONNECTS TO OTHER FILES:
# - It reads URLs and folder names from config.py
# - main.py calls ingest() when you run: python main.py download
# - The notebook and train.py call ingest() when they need rows
# - tests/test_ingest.py checks the unpacking without the internet
#
# WHY A SEPARATE FILE:
# Downloading is a different job from modeling. If the UCI link
# changes, this is the only file you should have to edit.
# =============================================================

import io
import urllib.request
import zipfile
from pathlib import Path

# pandas is the standard library for tables in Python data science.
# We import it as "pd" because that is the nickname every tutorial uses.
import pandas as pd

# config.py holds the shared settings. See that file's header for why.
from config import DATA_ZIP_URL, EXPECTED_COLUMNS, RAW_CSV_NAME, RAW_DIR


# --------------------------------------------------------------
# FUNCTION: download_file
# WHAT IT DOES: Copies a file from a web address onto the disk
# WHY WE NEED IT: The dataset is not stored in Git. Each person
#                 downloads the same public file.
# INPUTS: url (text, the web address), destination (a file path)
# OUTPUTS: The destination path, after the file has been written
#
# LEARNING NOTE: urllib is part of Python itself, so we do not
# add a package just to download one zip. The requests library
# is a popular alternative with a shorter spelling. Same job.
# --------------------------------------------------------------
def download_file(url, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)  # create data/raw if needed

    # Some data sites reject a blank User-Agent. We send a plain name
    # so the download looks like a normal program, not a broken script.
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "bank-term-deposit-ml/1.0 (teaching project)"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        destination.write_bytes(response.read())  # write the whole zip to disk
    return destination


# --------------------------------------------------------------
# FUNCTION: extract_bank_full_csv
# WHAT IT DOES: Pulls bank-full.csv out of the UCI zip
# WHY WE NEED IT: The published zip contains ANOTHER zip named
#                 bank.zip, and the CSV we want is inside that.
#                 This is a common beginner trap: unzip once and
#                 think the download failed.
# INPUTS: zip_path (the downloaded zip), output_csv_path (where to write)
# OUTPUTS: output_csv_path
#
# LEARNING NOTE: zipfile is also built into Python. We do not need
# WinZip or a terminal unzip command.
# --------------------------------------------------------------
def extract_bank_full_csv(zip_path, output_csv_path):
    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as outer_zip:
        names = outer_zip.namelist()  # every path stored inside the zip
        direct_csv = [name for name in names if name.endswith("bank-full.csv")]
        if direct_csv:
            output_csv_path.write_bytes(outer_zip.read(direct_csv[0]))
            return output_csv_path

        nested_zips = [name for name in names if name.endswith("bank.zip")]
        if not nested_zips:
            raise FileNotFoundError(
                "The download did not contain bank-full.csv or bank.zip. "
                "The UCI packaging may have changed. Look inside the zip "
                f"and update extract_bank_full_csv. Found: {names}"
            )

        # BytesIO lets us open the inner zip from memory. We do not
        # have to write bank.zip to disk just to open it.
        inner_bytes = io.BytesIO(outer_zip.read(nested_zips[0]))

    with zipfile.ZipFile(inner_bytes) as inner_zip:
        inner_names = [
            name for name in inner_zip.namelist() if name.endswith("bank-full.csv")
        ]
        if not inner_names:
            raise FileNotFoundError(
                "bank.zip did not contain bank-full.csv. "
                f"Found: {inner_zip.namelist()}"
            )
        output_csv_path.write_bytes(inner_zip.read(inner_names[0]))

    return output_csv_path


# --------------------------------------------------------------
# FUNCTION: load_bank_csv
# WHAT IT DOES: Reads the semicolon-separated CSV into a table
# WHY WE NEED IT: The UCI file uses ";" between columns, not commas.
#                 pandas' default would smash each row into one column.
# INPUTS: csv_path
# OUTPUTS: A pandas DataFrame with the expected column names
#
# LEARNING NOTE: You could also pass sep=";" at every call site.
# Doing it here means the rest of the project can forget that detail.
# --------------------------------------------------------------
def load_bank_csv(csv_path):
    frame = pd.read_csv(csv_path, sep=";")
    frame.columns = [str(column).strip() for column in frame.columns]

    missing = [column for column in EXPECTED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            "This CSV is missing columns the rest of the project expects: "
            f"{missing}. Check that you extracted bank-full.csv, not a different file."
        )
    return frame


# --------------------------------------------------------------
# FUNCTION: ingest
# WHAT IT DOES: Downloads the data if we do not already have it,
#               then returns the table
# WHY WE NEED IT: This is the one function other files call.
#                 They should not need to know about zips.
# INPUTS: force (True means "download again even if the file exists")
# OUTPUTS: The full bank-full table
# --------------------------------------------------------------
def ingest(force=False):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RAW_DIR / RAW_CSV_NAME

    if force or not csv_path.exists():
        zip_path = RAW_DIR / "bank-marketing.zip"
        download_file(DATA_ZIP_URL, zip_path)
        extract_bank_full_csv(zip_path, csv_path)

    return load_bank_csv(csv_path)


# --------------------------------------------------------------
# This block runs only when you execute this file directly:
#     python ingest.py
# It does NOT run when another file imports ingest.
# WHY: importing should not surprise you by downloading data.
# --------------------------------------------------------------
if __name__ == "__main__":
    table = ingest()
    yes_count = int((table["y"] == "yes").sum())
    print(f"Loaded {len(table)} rows from {RAW_DIR / RAW_CSV_NAME}")
    print(f"Subscribed (yes): {yes_count}. Did not subscribe (no): {len(table) - yes_count}")
