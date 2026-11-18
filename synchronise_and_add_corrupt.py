import os
import re
import glob
import json
import pickle
import pandas as pd
from urllib.parse import urlparse

# --- CONFIGURATION ---
SNAPSHOTS_DIR = "" 
INPUT_DIR = "refined_dataset/historic"
OUTPUT_DIR = "refined_dataset"

JSON_FILE = "wayback_snapshot_dates.json"
CSV_FILE = "wayback_snapshot_dates.csv"
PKL_FILE = "wayback_snapshot_dates.pkl"

def get_folder_name(url):
    """
    Generates a clean directory name, reducing ERT variants to a single folder
    to match the original download logic.
    """
    if any(x in url for x in ["ert.gr", "ertnews.gr", "nerit.gr"]):
        return "www.ert.gr"
        
    parsed = urlparse(url)
    folder_name = parsed.netloc + parsed.path.replace("/", "_")
    return re.sub(r'[\\/\*?:"<>|]', "_", folder_name)

def clean_timestamp(ts_string):
    """Strips formatting from the JSON timestamp to match snapshot filenames."""
    return ts_string.replace("-", "").replace(":", "").replace(" ", "")

def is_file_corrupt(filepath):
    """
    Checks if an HTML file is an error page, abnormally small, 
    or lacks core website content tags.
    """
    try:
        # 1. Size Check: If it's under 5KB (5120 bytes), it is almost 
        # certainly a Wayback Machine error page or a 301 redirect.
        if os.path.getsize(filepath) < 5120:
            return True
            
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            content_lower = content.lower()
            
            # 2. Wayback Error Checks: Catch explicit archive failures
            if "wayback machine hasn't archived that url" in content_lower:
                return True
            if "got an http 301 response at crawl time" in content_lower:
                return True
            if "502 bad gateway" in content_lower or "503 service unavailable" in content_lower:
                return True
                
            # 3. Core Content Check: A proper news website must have links.
            # If there are no anchor tags, it is an empty shell.
            if "<a " not in content_lower:
                return True
                
            # 4. Terminal Tag Check: Ensure the HTML didn't truncate mid-download
            if not re.search(r"</\s*(html|body)\s*>", content, re.IGNORECASE):
                return True
                
        return False
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return True

def scan_snapshots_directory(base_dir):
    """
    Scans the snapshots directory tree to map all existing (folder, timestamp) files on disk.
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
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    input_json_path = os.path.join(INPUT_DIR, JSON_FILE)

    if not os.path.exists(input_json_path):
        print(f"Error: {input_json_path} not found.")
        return

    # 1. Map physical files on disk
    disk_snapshots = scan_snapshots_directory(SNAPSHOTS_DIR)
    matched_disk_keys = set()

    with open(input_json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    # 2. Tracking delta statistics
    stats = {
        "total_snapshots": 0,
        "valid_downloaded": 0,
        "corrupt_downloaded": 0,
        "missing_on_disk": 0,
        "download_flag_flipped_true": 0,   # Previously False -> now True
        "download_flag_flipped_false": 0,  # Previously True -> now False
        "sources_available_updated": 0,
        "untracked_snapshots_on_disk": 0
    }

    # 3. Update dataset hierarchy and compute deltas
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
                    details["is_downloaded"] = True

                    if not prev_downloaded:
                        stats["download_flag_flipped_true"] += 1

                    if is_file_corrupt(filepath):
                        details["is_corrupt"] = True
                        stats["corrupt_downloaded"] += 1
                    else:
                        details["is_corrupt"] = False
                        stats["valid_downloaded"] += 1
                        valid_sources_count += 1
                else:
                    details["is_downloaded"] = False
                    details["is_corrupt"] = False
                    stats["missing_on_disk"] += 1

                    if prev_downloaded:
                        stats["download_flag_flipped_false"] += 1

            # Check if active sources count changed based on valid downloaded files
            if valid_sources_count != old_sources_avail:
                stats["sources_available_updated"] += 1
                entry["sources_available"] = valid_sources_count

    # 4. Detect untracked snapshots on disk
    untracked_keys = set(disk_snapshots.keys()) - matched_disk_keys
    stats["untracked_snapshots_on_disk"] = len(untracked_keys)

    # 5. Export synchronized JSON[cite: 3]
    out_json_path = os.path.join(OUTPUT_DIR, JSON_FILE)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=4)
    print(f"[✓] Saved updated JSON: {out_json_path}")

    # 6. Export synchronized CSV[cite: 2, 3]
    export_records = []
    for month_key, entries in json_data.items():
        for entry in entries:
            for source_url, details in entry.get("snapshots", {}).items():
                export_records.append({
                    "YearMonth": month_key,
                    "PrimaryDate": entry["primary_date"],
                    "SourcesAvailable": entry["sources_available"],
                    "SourceURL": source_url,
                    "Timestamp": details["timestamp"],
                    "IsFallback": details.get("is_fallback", False),
                    "FallbackGapDays": details.get("gap_days", 0),
                    "IsDownloaded": details["is_downloaded"],
                    "IsCorrupt": details["is_corrupt"]
                })

    df = pd.DataFrame(export_records)
    out_csv_path = os.path.join(OUTPUT_DIR, CSV_FILE)
    df.to_csv(out_csv_path, index=False)
    print(f"[✓] Saved updated CSV:  {out_csv_path}")

    # 7. Export synchronized PKL[cite: 3]
    out_pkl_path = os.path.join(OUTPUT_DIR, PKL_FILE)
    with open(out_pkl_path, "wb") as f:
        pickle.dump(json_data, f)
    print(f"[✓] Saved updated PKL:  {out_pkl_path}")

    # 8. Detailed Delta & Audit Report
    print("\n" + "=" * 50)
    print("           DATASET AUDIT & DELTA REPORT           ")
    print("=" * 50)
    print(f"Total Snapshots in Dataset:          {stats['total_snapshots']}")
    print(f"Valid Downloaded Snapshots:          {stats['valid_downloaded']}")
    print(f"Corrupt Snapshots Flagged:           {stats['corrupt_downloaded']}")
    print(f"Missing Snapshots (is_downloaded=0): {stats['missing_on_disk']}")
    print("-" * 50)
    print("Status Transitions:")
    print(f"  • Newly Downloaded (False -> True): {stats['download_flag_flipped_true']}")
    print(f"  • Now Missing      (True -> False): {stats['download_flag_flipped_false']}")
    print(f"  • 'sources_available' Recounted:    {stats['sources_available_updated']} dates")
    print("-" * 50)
    print(f"Untracked HTML Files on Disk:        {stats['untracked_snapshots_on_disk']}")
    if stats["untracked_snapshots_on_disk"] > 0:
        print("  Sample untracked snapshots found:")
        for folder, ts in list(untracked_keys)[:5]:
            print(f"    - {folder}/{ts}.html")
    print("=" * 50)

if __name__ == "__main__":
    main()
