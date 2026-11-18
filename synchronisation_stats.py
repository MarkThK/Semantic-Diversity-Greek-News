import os
import re
import glob
import json
from urllib.parse import urlparse

# --- CONFIGURATION ---
SNAPSHOTS_DIR = ""  # Set to your global snapshots path
INPUT_JSON = "refined_dataset/wayback_snapshot_dates.json" # Path to your current working JSON

def get_folder_name(url):
    """
    Generates a clean directory name, reducing ERT variants to a single folder
    to match the original download logic[cite: 4].
    """
    if any(x in url for x in ["ert.gr", "ertnews.gr", "nerit.gr"]):
        return "www.ert.gr"
        
    parsed = urlparse(url)
    folder_name = parsed.netloc + parsed.path.replace("/", "_")
    return re.sub(r'[\\/\*?:"<>|]', "_", folder_name)

def clean_timestamp(ts_string):
    """Strips formatting from the JSON timestamp to match snapshot filenames[cite: 4]."""
    return ts_string.replace("-", "").replace(":", "").replace(" ", "")

def is_file_corrupt(filepath):
    """
    Checks if an HTML file is an error page, abnormally small, 
    or lacks core website content tags[cite: 4].
    """
    try:
        if os.path.getsize(filepath) < 5120:
            return True
            
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            content_lower = content.lower()
            
            if "wayback machine hasn't archived that url" in content_lower:
                return True
            if "got an http 301 response at crawl time" in content_lower:
                return True
            if "502 bad gateway" in content_lower or "503 service unavailable" in content_lower:
                return True
            if "<a " not in content_lower:
                return True
            if not re.search(r"</\s*(html|body)\s*>", content, re.IGNORECASE):
                return True
                
        return False
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return True

def scan_snapshots_directory(base_dir):
    """
    Scans the snapshots directory tree to map all existing (folder, timestamp) files on disk[cite: 4].
    """
    disk_files = {}
    if not os.path.exists(base_dir):
        return disk_files

    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue
        for html_path in glob.glob(os.path.join(folder_path, "*.html")):
            filename = os.path.basename(html_path)
            ts_raw = filename.replace(".html", "")
            disk_files[(folder_name, ts_raw)] = html_path

    return disk_files

def main():
    if not os.path.exists(INPUT_JSON):
        print(f"Error: Dataset {INPUT_JSON} not found.")
        return

    print("Scanning hard drive and analyzing dataset... (Read-Only Mode)\n")

    # 1. Map physical files on disk[cite: 4]
    disk_snapshots = scan_snapshots_directory(SNAPSHOTS_DIR)
    matched_disk_keys = set()

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    # 2. Tracking delta statistics[cite: 4]
    stats = {
        "total_snapshots": 0,
        "valid_downloaded": 0,
        "corrupt_downloaded": 0,
        "missing_on_disk": 0,
        "download_flag_flipped_true": 0,
        "download_flag_flipped_false": 0,
        "sources_available_updated": 0,
        "untracked_snapshots_on_disk": 0
    }

    # 3. Compute statistics without modifying the dataset[cite: 4]
    for month_key, entries in json_data.items():
        for entry in entries:
            old_sources_avail = entry.get("sources_available", 0)
            valid_sources_count = 0

            for url, details in entry.get("snapshots", {}).items():
                stats["total_snapshots"] += 1
                folder_name = get_folder_name(url)
                ts_raw = clean_timestamp(details["timestamp"])
                key = (folder_name, ts_raw)

                prev_downloaded = details.get("is_downloaded", False)
                filepath = disk_snapshots.get(key)

                if filepath and os.path.exists(filepath):
                    matched_disk_keys.add(key)
                    
                    if not prev_downloaded:
                        stats["download_flag_flipped_true"] += 1

                    if is_file_corrupt(filepath):
                        stats["corrupt_downloaded"] += 1
                    else:
                        stats["valid_downloaded"] += 1
                        valid_sources_count += 1
                else:
                    stats["missing_on_disk"] += 1

                    if prev_downloaded:
                        stats["download_flag_flipped_false"] += 1

            if valid_sources_count != old_sources_avail:
                stats["sources_available_updated"] += 1

    # 4. Detect untracked snapshots on disk[cite: 4]
    untracked_keys = set(disk_snapshots.keys()) - matched_disk_keys
    stats["untracked_snapshots_on_disk"] = len(untracked_keys)

    # 5. Print Detailed Audit Report[cite: 4]
    print("=" * 50)
    print("           DATASET AUDIT & DELTA REPORT           ")
    print("                 (READ-ONLY MODE)                 ")
    print("=" * 50)
    print(f"Total Snapshots in Dataset:          {stats['total_snapshots']}")
    print(f"Valid Downloaded Snapshots:          {stats['valid_downloaded']}")
    print(f"Corrupt Snapshots Flagged:           {stats['corrupt_downloaded']}")
    print(f"Missing Snapshots (is_downloaded=0): {stats['missing_on_disk']}")
    print("-" * 50)
    print("Potential Status Transitions (If updated):")
    print(f"  • Newly Downloaded (False -> True): {stats['download_flag_flipped_true']}")
    print(f"  • Now Missing      (True -> False): {stats['download_flag_flipped_false']}")
    print(f"  • 'sources_available' Mismatches:   {stats['sources_available_updated']} dates")
    print("-" * 50)
    print(f"Untracked HTML Files on Disk:        {stats['untracked_snapshots_on_disk']}")
    if stats["untracked_snapshots_on_disk"] > 0:
        print("  Sample untracked snapshots found:")
        for folder, ts in list(untracked_keys)[:5]:
            print(f"    - {folder}/{ts}.html")
    print("=" * 50)

if __name__ == "__main__":
    main()
