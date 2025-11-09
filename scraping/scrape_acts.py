import os
import requests
import time
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# --- Configuration ---
STATUS_URL_TEMPLATE = "https://legislature.vermont.gov/bill/status/2026/{bill_name}"
BASE_URL = "https://legislature.vermont.gov/"
DOWNLOAD_DIR = "scraped_data/vermont_acts_2026"
# Stop after this many consecutive 404s
MAX_CONSECUTIVE_FAILURES = 1 # <-- Changed to 1 as requested
# ---------------------

def download_act_pdfs(bill_url, bill_name, download_dir):
    """
    Visits a single bill page and downloads PDFs from the 'act' tab.
    Returns a status code:
    - 200: Page found, processing complete (even if no act was found).
    - 404: Page not found.
    - 500: Other error.
    """
    try:
        response = requests.get(bill_url)
        
        # Check for other HTTP errors
        response.raise_for_status()
        
        # If we get here, the page exists (HTTP 200)
        soup = BeautifulSoup(response.text, "html.parser")

        # --- New check: Validate page has a bill title ---
        # Find <div class="bill-title">
        title_div = soup.find("div", class_="bill-title")
        if not title_div:
            print(f"--- {bill_name} is not a valid bill page (no title).")
            return 404 # Treat as failure to stop iteration
        # --- End of new check ---

        print(f"--- Checking {bill_name} ---")
        
        # Look for the specific div
        act_div = soup.find("div", id="act")
        
        if not act_div:
            print(f"No 'act' tab found for {bill_name}.")
            return 200 # Page existed, but no act

        # --- Modified PDF link finding logic ---
        # Find all PDF links within that div
        all_links = act_div.find_all("a", href=True)
        pdf_links = []
        for link in all_links:
            link_text = link.text.strip()
            href = link.get("href", "").lower()
            # Only grab the specific links requested
            if (link_text == "As Enacted" or link_text == "Act Summary") and ".pdf" in href:
                pdf_links.append(link)
        # --- End of modified logic ---
        
        if not pdf_links:
            print(f"'act' div found, but no PDF links inside for {bill_name}.")
            return 200 # Page existed, but no PDFs in the act div

        print(f"Found {len(pdf_links)} Act PDF(s) for {bill_name}. Downloading...")
        
        # Create a sub-directory for this bill's acts
        bill_act_dir = os.path.join(download_dir, bill_name)
        os.makedirs(bill_act_dir, exist_ok=True)
        
        for link in pdf_links:
            href = link.get("href")
            cleaned_href = href.split("#")[0]
            absolute_url = urljoin(BASE_URL, cleaned_href)
            filename = absolute_url.split("/")[-1]
            file_path = os.path.join(bill_act_dir, filename)
            
            try:
                print(f"Downloading {filename}...")
                pdf_response = requests.get(absolute_url)
                pdf_response.raise_for_status()
                
                with open(file_path, "wb") as f:
                    f.write(pdf_response.content)
                print(f"Successfully saved to {file_path}")
                
            except requests.exceptions.RequestException as e:
                print(f"Failed to download {absolute_url}. Error: {e}")
        
        return 200 # Success

    except requests.exceptions.RequestException as e:
        # Handle errors fetching the bill page itself
        print(f"Failed to fetch {bill_url}. Error: {e}")
        return 500 # General error

def iterate_and_scrape(bill_prefix, download_dir):
    """
    Iterates through bill numbers (e.g., H.1, H.2...) until
    MAX_CONSECUTIVE_FAILURES is reached.
    """
    print(f"\n--- Starting scrape for prefix '{bill_prefix}' ---")
    i = 1
    consecutive_failures = 0
    
    while consecutive_failures < MAX_CONSECUTIVE_FAILURES:
        bill_name = f"{bill_prefix}{i}"
        bill_url = STATUS_URL_TEMPLATE.format(bill_name=bill_name)
        
        status = download_act_pdfs(bill_url, bill_name, download_dir)
        
        if status == 404:
            consecutive_failures += 1
        else: # Any success or server error
            consecutive_failures = 0 # Reset on success
        # If status is 500 (other error), we also reset,
        # just in case it was a temporary server glitch.
        # else:  <-- Old logic
        #     consecutive_failures = 0 
            
        i += 1
        time.sleep(0.25) # Be polite
        
    print(f"--- Hit {consecutive_failures} consecutive failures. Stopping scrape for '{bill_prefix}' ---")


def main():
    """
    Main function to orchestrate the scraping.
    """
    print(f"Creating download directory at: {DOWNLOAD_DIR}")
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    
    # Iterate for House bills (H.1, H.2, ...)
    iterate_and_scrape("H.", DOWNLOAD_DIR)
    
    # Iterate for Senate bills (S.1, S.2, ...)
    iterate_and_scrape("S.", DOWNLOAD_DIR)
            
    print("\nScraping complete.")

if __name__ == "__main__":
    main()