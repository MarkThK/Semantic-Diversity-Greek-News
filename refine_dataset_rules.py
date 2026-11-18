import os
import json
import pandas as pd
import pickle
from datetime import datetime

# Input / Output Paths
INPUT_JSON = "wayback_snapshot_dates_updated.json"
OUTDIR = "refined_dataset"

os.makedirs(OUTDIR, exist_ok=True)

OUT_JSON = os.path.join(OUTDIR, "wayback_snapshot_dates.json")
OUT_CSV = os.path.join(OUTDIR, "wayback_snapshot_dates.csv")
OUT_PKL = os.path.join(OUTDIR, "wayback_snapshot_dates.pkl")

def refine_dataset():
    print(f"Loading data from '{INPUT_JSON}'...")
    try:
        with open(INPUT_JSON, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: '{INPUT_JSON}' not found. Please ensure it is in the same directory.")
        return

    total_imputed = 0
    total_incomplete_months = 0
    total_incomplete_source_months = 0
    export_records = []

    print("Applying dataset refinement rules...")
    
    for month_key, days in data.items():
        # RULE 1: Incomplete Month Check
        # If the month has fewer than 4 scheduled primary dates overall.
        num_primary_dates = len(days)
        month_incomplete = (num_primary_dates < 4)
        
        if month_incomplete:
            total_incomplete_months += 1

        # Tag the month completion status in the JSON structure
        for day in days:
            day['month_incomplete'] = month_incomplete
            
        # Get all unique URLs scheduled in this month
        all_urls = set()
        for day in days:
            all_urls.update(day['snapshots'].keys())

        for url in all_urls:
            downloaded_snaps = []
            failed_snaps = []

            # Sort snapshots for this source by success/failure
            for i, day in enumerate(days):
                snap = day['snapshots'].get(url)
                if not snap:
                    continue
                
                pdate = datetime.strptime(day['primary_date'], '%Y-%m-%d').date()
                if snap.get('is_downloaded', False):
                    downloaded_snaps.append((pdate, i, snap))
                else:
                    failed_snaps.append((pdate, i, snap))

            # RULE 2: Incomplete Source Check
            # If a source has less than 3 snapshots successfully downloaded.
            num_downloaded = len(downloaded_snaps)
            source_incomplete = (num_downloaded < 3)
            
            if source_incomplete:
                total_incomplete_source_months += 1

            # RULE 3: Fallback Imputation (Reusing Closest Available)
            # Only impute if there are exactly 3 downloaded snapshots, 1 failed, and it's a 4-date month
            if not month_incomplete and num_downloaded == 3 and len(failed_snaps) == 1:
                failed_pdate, failed_idx, _ = failed_snaps[0]
                
                min_gap = float('inf')
                best_snap = None
                
                # Find the closest downloaded snapshot inside this month
                for down_pdate, _, down_snap in downloaded_snaps:
                    down_ts = datetime.strptime(down_snap['timestamp'], '%Y-%m-%d %H:%M:%S')
                    gap = abs((down_ts.date() - failed_pdate).days)
                    
                    if gap < min_gap:
                        min_gap = gap
                        best_snap = down_snap
                
                # Overwrite the failed snapshot with the reused fallback
                if best_snap:
                    days[failed_idx]['snapshots'][url] = {
                        "timestamp": best_snap['timestamp'],
                        "is_fallback": True,
                        "gap_days": min_gap,
                        "is_downloaded": True,
                        "is_reused_fallback": True, 
                        "source_incomplete_for_month": False
                    }
                    total_imputed += 1
                    source_incomplete = False # Ensure it evaluates as complete after imputation

            # Apply final source flags to the dictionary
            for day in days:
                if url in day['snapshots']:
                    is_reused = day['snapshots'][url].get('is_reused_fallback', False)
                    day['snapshots'][url]['is_reused_fallback'] = is_reused
                    day['snapshots'][url]['source_incomplete_for_month'] = source_incomplete

    # Flatten data for CSV export
    for month_key, days in data.items():
        for day in days:
            for url, snap in day['snapshots'].items():
                export_records.append({
                    "YearMonth": month_key,
                    "PrimaryDate": day['primary_date'],
                    "SourcesAvailable": day['sources_available'],
                    "SourceURL": url,
                    "Timestamp": snap.get('timestamp'),
                    "IsFallback": snap.get('is_fallback', False),
                    "FallbackGapDays": snap.get('gap_days'),
                    "IsDownloaded": snap.get('is_downloaded', False),
                    "MonthIncomplete": day['month_incomplete'],
                    "SourceIncomplete": snap.get('source_incomplete_for_month', False),
                    "IsReusedFallback": snap.get('is_reused_fallback', False)
                })

    # Save to JSON
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
        
    # Save to Pickle
    with open(OUT_PKL, 'wb') as f:
        pickle.dump(data, f)
        
    # Save to CSV
    df = pd.DataFrame(export_records)
    df.to_csv(OUT_CSV, index=False)

    print("\n======================================")
    print("      REFINEMENT STATISTICS           ")
    print("======================================")
    print(f"Total incomplete months flagged: {total_incomplete_months}")
    print(f"Total source-months flagged as incomplete: {total_incomplete_source_months}")
    print(f"Total missing snapshots successfully imputed: {total_imputed}")
    print(f"\nAll files saved successfully to directory: '{OUTDIR}'")
    print("======================================")

if __name__ == "__main__":
    refine_dataset()