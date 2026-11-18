import json
import csv
import re
from collections import defaultdict

def normalize_site(url):
    """Strips protocol and trailing slashes to match folder names (e.g. www.newsit.gr)."""
    return re.sub(r'^https?://', '', url).rstrip('/').lower()

def normalize_timestamp(ts_str):
    """Extracts only digits to match txt filenames (e.g. 20140805152139)."""
    return re.sub(r'\D', '', str(ts_str))

def clean_extracted_data(extracted_json_path, wayback_json_path, csv_path):
    print("Loading wayback metadata...")
    try:
        with open(wayback_json_path, 'r', encoding='utf-8') as f:
            wayback_data = json.load(f)
    except Exception as e:
        print(f"Error loading {wayback_json_path}: {e}")
        return

    # Track every occurrence of a snapshot across all months/dates.
    # Key: (website, timestamp) -> List of booleans (True = clean usage, False = bad usage)
    snapshot_usage_history = defaultdict(list)

    # 1. Scan the entire wayback JSON across all months
    for month_key, entries_list in wayback_data.items():
        for entry in entries_list:
            month_is_bad = entry.get("month_incomplete", False)
            snapshots = entry.get("snapshots", {})
            
            for url, snap_info in snapshots.items():
                ts_raw = snap_info.get("timestamp", "")
                if not ts_raw:
                    continue
                    
                site_norm = normalize_site(url)
                ts_norm = normalize_timestamp(ts_raw)
                
                # Check for any bad flags in this specific appearance
                is_bad = (
                    month_is_bad or 
                    snap_info.get("source_incomplete_for_month", False) is True or
                    snap_info.get("is_corrupt", False) is True or
                    snap_info.get("is_downloaded", True) is False
                )
                
                # Record whether THIS usage was valid
                snapshot_usage_history[(site_norm, ts_norm)].append(not is_bad)

    # 2. Determine which snapshots are exclusively bad
    # A snapshot is kept if ANY usage was clean. It is deleted ONLY if ALL usages were bad.
    snapshots_to_delete = set()
    saved_by_fallback_count = 0

    for snap_key, usages in snapshot_usage_history.items():
        has_at_least_one_clean_use = any(usages)
        
        if not has_at_least_one_clean_use:
            snapshots_to_delete.add(snap_key)
        elif not all(usages):
            # It had at least one bad usage, but was saved by being valid elsewhere!
            saved_by_fallback_count += 1

    print(f"Total unique snapshots tracked: {len(snapshot_usage_history)}")
    print(f"Snapshots that had a bad flag but were SAVED because of a clean fallback use: {saved_by_fallback_count}")
    print(f"Snapshots with ALL bad flags marked for deletion: {len(snapshots_to_delete)}")

    # 3. Filter extracted data
    print("\nFiltering extracted data...")
    with open(extracted_json_path, 'r', encoding='utf-8') as f:
        extracted_data = json.load(f)

    filtered_data = []
    removed_articles = 0

    for row in extracted_data:
        site = normalize_site(str(row['website']))
        ts = normalize_timestamp(str(row['timestamp']))
        
        # Delete only if this snapshot appeared in wayback and had ZERO valid entries
        if (site, ts) in snapshots_to_delete:
            removed_articles += 1
            continue
            
        filtered_data.append(row)

    print(f"Removed {removed_articles} articles belonging to completely invalid snapshots.")
    print(f"Keeping {len(filtered_data)} valid articles.")

    # 4. Overwrite JSON and CSV
    if filtered_data:
        # Overwrite JSON
        with open(extracted_json_path, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, indent=4, ensure_ascii=False)
            
        # Overwrite CSV
        keys = filtered_data[0].keys()
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            dict_writer = csv.DictWriter(f, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(filtered_data)
            
        print("\nSuccessfully updated and overwritten extracted_data.json and extracted_data.csv!")
    else:
        print("\nWarning: All articles were filtered out!")

if __name__ == "__main__":
    # --- CONFIGURATION ---
    EXTRACTED_JSON = "extracted_data.json"
    EXTRACTED_CSV = "extracted_data.csv"
    WAYBACK_JSON = "../refined_dataset/wayback_snapshot_dates.json"
    
    clean_extracted_data(EXTRACTED_JSON, WAYBACK_JSON, EXTRACTED_CSV)