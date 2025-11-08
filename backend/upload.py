import os
from pathlib import Path
import re
from datetime import datetime
import json
from langchain_core.documents import Document

from load import Storage, PDF, text_splitter

# --- Vector Store Setup ---
FAISS_PATH = "faiss_index"
storage = Storage(path=FAISS_PATH)

# --- Data Directories ---
ACTS_DIR = Path(__file__).parent.parent / "scraped_data/vermont_acts_2026"
JOURNALS_DIR = Path(__file__).parent.parent / "scraped_data/vermont_journals_2026"
TRANSCRIPTS_PATH = Path(__file__).parent.parent / "scraped_data/vermont_transcripts_clean.json"

# --- Metadata Extraction ---
def get_act_metadata(file_path: Path) -> dict:
    bill_name = file_path.parent.name
    chamber = 'house' if bill_name.startswith('H') else 'senate' if bill_name.startswith('S') else 'joint'
    
    return {
        "file_name": file_path.name,
        "source_url": f"https://legislature.vermont.gov/bill/status/2026/{bill_name}",
        "chamber": chamber,
        "journal_date": None,
        "bill_number": bill_name,
        "act_summary": "summary" in file_path.name.lower(),
        "as_enacted": "enacted" in file_path.name.lower(),
    }

def get_journal_metadata(file_path: Path) -> dict:
    filename = file_path.name
    chamber = 'senate' if filename.lower().startswith('s') else 'house' # Default to house for 'h' and 'j'
    if filename.lower().startswith('j'):
        chamber = 'joint'

    source_url = f"https://legislature.vermont.gov/Documents/2026/Docs/JOURNAL/{filename}"

    date_match = re.search(r'\d{2}(\d{2})(\d{2})', filename)
    journal_date = None
    if date_match:
        month, day = date_match.groups()
        year = 2026
        try:
            journal_date = datetime(year, int(month), int(day)).date()
        except ValueError:
            print(f"Warning: Could not parse date from filename {filename}")
            journal_date = None


    return {
        "file_name": filename,
        "source_url": source_url,
        "chamber": chamber,
        "journal_date": journal_date,
        "bill_number": None,
        "act_summary": None,
        "as_enacted": None,
    }

def get_transcript_metadata(transcript_entry: dict, chamber: str) -> dict:
    # The user specified to ignore committee name, but include date, time, and url
    date_str = transcript_entry.get("date")
    time_str = transcript_entry.get("time")
    
    journal_date = None
    if date_str:
        try:
            journal_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            print(f"Warning: Could not parse date '{date_str}' from transcript entry.")

    return {
        "file_name": transcript_entry.get("url"), # Using URL as a unique identifier for file_name
        "source_url": transcript_entry.get("url"),
        "chamber": chamber,
        "journal_date": journal_date,
        "meeting_time": time_str, 
        "bill_number": None,
        "act_summary": None,
        "as_enacted": None,
    }

# --- Main Upload Logic ---
def upload_files():
    # Process Acts
    print("Processing acts...")
    if not ACTS_DIR.exists():
        print(f"Warning: Directory not found: {ACTS_DIR}")
    else:
        act_files = list(ACTS_DIR.glob("**/*.pdf"))
        for pdf_path in act_files:
            print(f"Processing {pdf_path}...")
            metadata = get_act_metadata(pdf_path)
            PDF(str(pdf_path), storage, metadata)

    # Process Journals
    print("Processing journals...")
    if not JOURNALS_DIR.exists():
        print(f"Warning: Directory not found: {JOURNALS_DIR}")
    else:
        journal_files = list(JOURNALS_DIR.glob("*.pdf"))
        for pdf_path in journal_files:
            print(f"Processing {pdf_path}...")
            metadata = get_journal_metadata(pdf_path)
            PDF(str(pdf_path), storage, metadata)

    # Process Transcripts
    print("Processing transcripts...")
    if not TRANSCRIPTS_PATH.exists():
        print(f"Warning: File not found: {TRANSCRIPTS_PATH}")
    else:
        with open(TRANSCRIPTS_PATH, 'r') as f:
            all_transcripts_data = json.load(f)
        
        for chamber_name, committees in all_transcripts_data.items():
            for committee_abbr, transcript_list in committees.items():
                for transcript_entry in transcript_list:
                    if transcript_entry.get('transcript'):
                        doc = Document(page_content=transcript_entry['transcript'])
                        metadata = get_transcript_metadata(transcript_entry, chamber_name)
                        doc.metadata.update(metadata)
                        splits = text_splitter.split_documents([doc])
                        storage.add_documents(documents=splits)
                        print(f"Processing transcript from {transcript_entry.get('url')} (Chamber: {chamber_name}, Committee: {committee_abbr})")

    print("Upload complete.")

if __name__ == "__main__":
    upload_files()
