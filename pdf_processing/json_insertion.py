from google.oauth2 import service_account
from googleapiclient.discovery import build

from pathlib import Path
import os
import json
from termcolor import colored

"""
STEPS TO SET UP GOOGLE CLOUD API (for Google Sheets only)

1. Go to Google Cloud Console: https://console.cloud.google.com
2. Create a new project (or use an existing one)
3. Go to IAM & Admin -> Service Accounts
   a. Create a service account
   b. Create a JSON key for it and download it
4. Enable the "Google Sheets API" for your project
5. Save the JSON key file as pdf_processing/service-account.json
6. Share the target Google Sheet with the service account email (Editor)
"""

# Base paths relative to this file
BASE_DIR = Path(__file__).resolve().parent
SERVICE_ACCOUNT_FILE = BASE_DIR / "pdf_processing" / "service-account.json"
MATCHES_FILE = BASE_DIR / "pdf_processing" / "matches.json"
LOCAL_JSON_DIR = BASE_DIR / "finished_data"  # where *.json and cleaned_*.json live

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets"
]

# This spreadsheet must be shared with the service account email
SPREADSHEET_ID = "1aKZ_xFZNCEuihLo2dXzU07KwL5sDyAbD2Fod4FmE6FE"
TARGET_RANGE_TEMPLATE = "S1 Literature!M{row}:N{row}"  # two columns for links/paths


def build_sheets_service():
    creds = service_account.Credentials.from_service_account_file(
        str(SERVICE_ACCOUNT_FILE),
        scopes=SCOPES
    )
    return build("sheets", "v4", credentials=creds)


def get_title_links_from_local_json():
    """
    Scan LOCAL_JSON_DIR for .json files and build:
        { title: [cleaned_path, regular_path] }

    where 'title' is the base PDF name, e.g. 'bo1005-leketa-2019'.
    'cleaned_path' and 'regular_path' are local filesystem paths.
    """
    title_links = {}

    if not LOCAL_JSON_DIR.exists():
        raise FileNotFoundError(f"Local JSON directory does not exist: {LOCAL_JSON_DIR}")

    for filename in os.listdir(LOCAL_JSON_DIR):
        if not filename.endswith(".json"):
            continue

        file_path = (LOCAL_JSON_DIR / filename).resolve()
        file_path_str = str(file_path)

        base = filename[:-5]  # strip ".json"

        # cleaned version: cleaned_<title>.json
        if filename.startswith("cleaned_"):
            title = base[8:]  # strip "cleaned_"
            if title not in title_links:
                title_links[title] = [file_path_str, ""]
            else:
                title_links[title][0] = file_path_str
        else:
            # regular version: <title>.json
            title = base
            if title not in title_links:
                title_links[title] = ["", file_path_str]
            else:
                title_links[title][1] = file_path_str

    return title_links


def load_matches():
    with open(MATCHES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    print("Base dir:        ", BASE_DIR)
    print("Local JSON dir:  ", LOCAL_JSON_DIR)
    print("Matches file:    ", MATCHES_FILE)

    serviceSheets = build_sheets_service()
    title_links = get_title_links_from_local_json()
    matches = load_matches()

    manual_insertion = {}

    # Insert the local file paths into the spreadsheet
    for name, (clean_path, regular_path) in title_links.items():
        values = [[clean_path, regular_path]]

        # titles not found in matches.json get collected for manual handling
        if name not in matches:
            print(colored("Unable to find in matches.json:", "red"), name)
            manual_insertion[name] = [clean_path, regular_path]
            continue

        row = matches[name]["row"]
        print(f"Inserting into row {row} for title '{name}'")

        updated_cell = TARGET_RANGE_TEMPLATE.format(row=row)
        body = {"values": values}

        serviceSheets.spreadsheets().values().update(
            spreadsheetId=SPREADSHEET_ID,
            range=updated_cell,
            valueInputOption="RAW",
            body=body
        ).execute()

    # Optionally, write out anything that couldn't be matched
    if manual_insertion:
        out_path = BASE_DIR / "manual_insertion_needed.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(manual_insertion, f, indent=2, ensure_ascii=False)
        print("\nSaved titles needing manual insertion to:", out_path)


if __name__ == "__main__":
    main()
