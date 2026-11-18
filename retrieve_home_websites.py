import os
import json
import re
import time
import requests
from urllib.parse import urlparse

# =====================================================================
# GLOBAL PARAMETERS
# =====================================================================

# Path to your mounted Linux hard drive folder
OUTPUT_DIR = "" 

# Active URLs to visit. 
# 'https://www.ert.gr' acts as the master toggle for all ERT/Nerit variants.
ENABLED_URLS = []

# Connection Settings
TIMEOUT_SECONDS = 90
MAX_RETRIES = 1

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
}

# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

def is_valid_html(filepath):
    """
    Verifies the HTML file is complete and not truncated.
    Checks for the existence of fundamental closing tags that are 
    always present at the end of a completed web document transfer.
    """
    try:
        # errors='ignore' prevents crashes if a severed download split a UTF-8 character in half
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
            # If the download finished properly, it will have a closing html or body tag.
            # A Ctrl+C or severed connection will almost always cut off before these.
            if re.search(r'</\s*(html|body)\s*>', content, re.IGNORECASE):
                return True
            return False
    except Exception:
        return False

def get_representative_url(url):
    """Maps variant domains to their canonical representative URL."""
    lower_url = url.lower()
    if "ert.gr" in lower_url or "ertnews.gr" in lower_url or "nerit.gr" in lower_url:
        return "https://www.ert.gr"
    return url

def sanitize_folder_name(url):
    """Generates a clean directory name from the representative URL."""
    rep_url = get_representative_url(url)
    parsed = urlparse(rep_url)
    folder_name = parsed.netloc + parsed.path.replace('/', '_')
    return re.sub(r'[\\/*?:"<>|]', '_', folder_name)

def clean_timestamp(ts_string):
    """Converts '2014-08-05 15:21:39' to '20140805152139' format."""
    return ts_string.replace("-", "").replace(":", "").replace(" ", "")

def download_snapshot(source_url, timestamp_str):
    """Downloads snapshot safely. Returns 'CACHED', 'DOWNLOADED', or 'FAILED'."""
    formatted_ts = clean_timestamp(timestamp_str)
    wayback_url = f"https://web.archive.org/web/{formatted_ts}id_/{source_url}"
    
    folder_name = sanitize_folder_name(source_url)
    target_dir = os.path.join(OUTPUT_DIR, folder_name)
    os.makedirs(target_dir, exist_ok=True)
    
    filename = f"{formatted_ts}.html"
    filepath = os.path.join(target_dir, filename)
    tmp_filepath = filepath + ".tmp"
    
    # Check if the file exists AND is structurally complete
    if os.path.exists(filepath):
        if is_valid_html(filepath):
            print(f" -> Already cached and verified complete: {filename}")
            return "CACHED"
        else:
            print(f" -> Found corrupt/truncated file, re-downloading: {filename}")

    print(f" -> Downloading: {wayback_url}")
    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(wayback_url, headers=HEADERS, timeout=TIMEOUT_SECONDS)
            
            if response.status_code in [429, 503]:
                retry_after = response.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    sleep_time = int(retry_after) + 5
                    print(f"    [Attempt {attempt}/{MAX_RETRIES}] Blocked (429). Waiting {retry_after}s. Cooling down for {sleep_time}s...")
                else:
                    sleep_time = attempt * 60 
                    print(f"    [Attempt {attempt}/{MAX_RETRIES}] Blocked (429). Cooling down for {sleep_time} seconds...")
                
                time.sleep(sleep_time)
                continue
                
            response.raise_for_status()
            
            # Save to temporary file first
            with open(tmp_filepath, 'w', encoding='utf-8') as f:
                f.write(response.text)
                
            # Verify the freshly downloaded file is actually complete
            if not is_valid_html(tmp_filepath):
                os.remove(tmp_filepath)
                print(f"    [Attempt {attempt}/{MAX_RETRIES}] Downloaded file is corrupt/truncated. Retrying...")
                time.sleep(attempt * 5)
                continue
            
            # Atomic rename ensures the final .html file is absolutely perfect
            os.replace(tmp_filepath, filepath)
            print(f"    Success: Verified and saved to {filepath}")
            return "DOWNLOADED"
            
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            print(f"    [Attempt {attempt}/{MAX_RETRIES}] Connection Refused / Timeout: {e}")
            if attempt < MAX_RETRIES:
                sleep_time = attempt * 30
                print(f"    Cooling down for {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                return "FAILED"
                
        except requests.exceptions.RequestException as e:
            print(f"    [Attempt {attempt}/{MAX_RETRIES}] Network Failed: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(10)
            else:
                return "FAILED"
                
        except IOError as e:
            print(f"    !!! File Write Error: {e}")
            if os.path.exists(tmp_filepath):
                os.remove(tmp_filepath)
            return "FAILED"

# =====================================================================
# MAIN RUNNER
# =====================================================================

def process_snapshots(json_filepath):
    if not os.path.exists(json_filepath):
        print(f"Error: Could not find '{json_filepath}' in the current directory.")
        return

    with open(json_filepath, 'r', encoding='utf-8') as file:
        try:
            json_data = json.load(file)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            return

    total_downloaded = 0
    total_cached = 0
    total_failed = 0
    
    for month, entries in json_data.items():
        print(f"\n--- Processing period: {month} ---")
        for entry in entries:
            snapshots = entry.get("snapshots", {})
            for source_url, details in snapshots.items():
                
                rep_url = get_representative_url(source_url)
                if rep_url not in ENABLED_URLS:
                    continue
                
                timestamp = details.get("timestamp")
                if not timestamp:
                    continue
                    
                print(f"\nTarget: {source_url} | Snapshot Time: {timestamp}")
                
                status = download_snapshot(source_url, timestamp)
                
                if status == "DOWNLOADED":
                    total_downloaded += 1
                    time.sleep(15.0) 
                elif status == "CACHED":
                    total_cached += 1
                else:
                    total_failed += 1
                    time.sleep(5.0)
                
    print(f"\nTask Complete! Downloaded: {total_downloaded} | Cached: {total_cached} | Failed: {total_failed}")

if __name__ == "__main__":
    process_snapshots("wayback_snapshot_dates.json")
