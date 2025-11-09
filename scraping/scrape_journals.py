import os
import requests
import time  # <-- Import time
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# --- Configuration ---
JOURNAL_PAGES = [
    "https://legislature.vermont.gov/house/service/2026/joint-assembly",
    "https://legislature.vermont.gov/house/service/2026/journal",
    "https://legislature.vermont.gov/senate/service/2026/journal"
]
BASE_URL = "https://legislature.vermont.gov/"
DOWNLOAD_DIR = "scraped_data/vermont_journals_2026"

# Pretend to be a real browser to avoid simple bot detection
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}
# ---------------------

def fetch_pdf_links_from_page(page_url, base_url):
    """
    Scrapes a single page to find all links to document PDFs.
    """
    print(f"Fetching links from {page_url}...")
    pdf_links = set()
    try:
        response = requests.get(page_url, headers=HEADERS)  # <-- Add headers
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Find all <a> tags where the href contains "/Documents/" and ".pdf"
        # This matches the pattern you provided.
        for link in soup.find_all("a", href=lambda h: h and "/documents/" in h.lower() and ".pdf" in h.lower()):
            href = link.get("href")
            # Clean up URL (remove any fragments like #page=1)
            cleaned_href = href.split("#")[0]
            # Convert relative URL to absolute
            absolute_url = urljoin(base_url, cleaned_href)
            pdf_links.add(absolute_url)
            
        print(f"Found {len(pdf_links)} PDF(s) on this page.")
        return pdf_links
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching page {page_url}: {e}")
        return set()

def download_pdfs(pdf_urls, download_dir):
    """
    Downloads all PDFs from the given set of URLs into the download directory.
    """
    os.makedirs(download_dir, exist_ok=True)

    existing_files = set(os.listdir(download_dir))
    to_download_urls = []
    skipped_count = 0

    for url in pdf_urls:
        filename = url.split("/")[-1]
        if filename in existing_files:
            print(f"Skipping {filename} (already exists).")
            skipped_count += 1
        else:
            to_download_urls.append(url)

    if not to_download_urls:
        if skipped_count == len(pdf_urls) and len(pdf_urls) > 0:
            print("All PDFs already exist. No new downloads needed.")
        else:
            print("No PDFs to download.")
        return

    print(f"\nStarting download of {len(to_download_urls)} new PDF(s)...")
    
    for url in to_download_urls:
        # Get filename from the URL
        filename = url.split("/")[-1]
        file_path = os.path.join(download_dir, filename)
            
        try:
            print(f"Downloading {filename}...")
            pdf_response = requests.get(url, headers=HEADERS)  # <-- Add headers
            pdf_response.raise_for_status()
            
            # Save the PDF content
            with open(file_path, "wb") as f:
                f.write(pdf_response.content)
            print(f"Successfully saved to {file_path}")
            
        except requests.exceptions.RequestException as e:
            print(f"Failed to download {url}. Error: {e}")
        except IOError as e:
            print(f"Failed to save {filename}. Error: {e}")

def main():
    """
    Main function to orchestrate the scraping.
    """
    print(f"Creating download directory at: {DOWNLOAD_DIR}")
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    
    all_pdf_urls = set()
    
    # Part 1: Get all PDF links from all pages
    for page_url in JOURNAL_PAGES:
        links_from_page = fetch_pdf_links_from_page(page_url, BASE_URL)
        all_pdf_urls.update(links_from_page)
        
    print(f"\nFound a total of {len(all_pdf_urls)} unique journal PDFs.")
    
    # Part 2: Download all found PDFs
    download_pdfs(all_pdf_urls, DOWNLOAD_DIR)
            
    print("\nJournal scraping complete.")

if __name__ == "__main__":
    # You'll need to install requests and beautifulsoup4:
    # pip install requests beautifulsoup4
    main()