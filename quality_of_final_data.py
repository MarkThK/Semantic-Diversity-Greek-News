import os
import json
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import calendar
from urllib.parse import urlparse

# --- CONFIGURATION ---
INPUT_JSON = "./refined_dataset/wayback_snapshot_dates.json"
OUTPUT_DIR = "statistics/quality_of_final_data"
REPORT_FILE = "problematic_snapshots_report.json"

def get_domain(url):
    """
    Extracts a clean domain name for the hover labels. 
    Groups ERT variants into a single identifier.
    """
    netloc = urlparse(url).netloc.replace("www.", "")
    if netloc in ["ert.gr", "nerit.gr", "ertnews.gr"]:
        return "ert_group"
    return netloc

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if not os.path.exists(INPUT_JSON):
        print(f"Error: {INPUT_JSON} not found.")
        return
        
    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # 1. Establish the baseline of expected sources across the entire dataset[cite: 1]
    expected_domains = set()
    for dates in data.values():
        for entry in dates:
            for url in entry.get("snapshots", {}).keys():
                expected_domains.add(get_domain(url))
                
    records = []
    problematic_report = {}
    
    # 2. Parse the JSON hierarchy to extract stats and detailed snapshot logs[cite: 1]
    for month_key, dates in data.items():
        year, month = month_key.split('-')
        
        total_snaps = 0
        valid_snaps = 0
        missing_snaps = 0
        corrupt_snaps = 0
        month_incomplete_flag = False
        
        # Sets for the graph tooltips
        missing_sites_on_disk = set()
        corrupt_sites_on_disk = set()
        absent_sites_from_json = set()
        
        # Detailed lists for the JSON report
        detailed_missing = []
        detailed_corrupt = []
        
        for entry in dates:
            if entry.get("month_incomplete", False):
                month_incomplete_flag = True
                
            entry_domains = set()
            
            for url, snap in entry.get("snapshots", {}).items():
                domain = get_domain(url)
                entry_domains.add(domain)
                total_snaps += 1
                
                is_dl = snap.get("is_downloaded", False)
                is_corrupt = snap.get("is_corrupt", False)
                
                if not is_dl:
                    missing_snaps += 1
                    missing_sites_on_disk.add(domain)
                    detailed_missing.append({
                        "url": url,
                        "timestamp": snap.get("timestamp")
                    })
                elif is_corrupt:
                    corrupt_snaps += 1
                    corrupt_sites_on_disk.add(domain)
                    detailed_corrupt.append({
                        "url": url,
                        "timestamp": snap.get("timestamp")
                    })
                else:
                    valid_snaps += 1
            
            # Check what expected sources were completely lacking from the JSON for this date[cite: 1]
            lacking_domains = expected_domains - entry_domains
            absent_sites_from_json.update(lacking_domains)
                    
        success_rate = (valid_snaps / total_snaps * 100) if total_snaps > 0 else 0
        fully_available = (success_rate == 100.0) and not month_incomplete_flag
        
        records.append({
            "YearMonth": month_key,
            "Year": int(year),
            "MonthNum": int(month),
            "Total": total_snaps,
            "Valid": valid_snaps,
            "Missing": missing_snaps,
            "Corrupt": corrupt_snaps,
            "MonthIncompleteFlag": month_incomplete_flag,
            "SuccessRate": round(success_rate, 2),
            "FullyAvailable": fully_available,
            "MissingSites": ", ".join(sorted(missing_sites_on_disk)) if missing_sites_on_disk else "None",
            "CorruptSites": ", ".join(sorted(corrupt_sites_on_disk)) if corrupt_sites_on_disk else "None",
            "AbsentSites": ", ".join(sorted(absent_sites_from_json)) if absent_sites_from_json else "None"
        })
        
        # If the month is not completely perfect, log the exact details into the report dictionary
        if not fully_available:
            problematic_report[month_key] = {
                "month_incomplete_flag": month_incomplete_flag,
                "lacking_domains_in_json": sorted(list(absent_sites_from_json)),
                "missing_snapshots_on_disk": detailed_missing,
                "corrupt_snapshots_on_disk": detailed_corrupt
            }
        
    df = pd.DataFrame(records)
    if df.empty: return
    
    # 3. Export the detailed JSON report
    report_path = os.path.join(OUTPUT_DIR, REPORT_FILE)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(problematic_report, f, indent=4)
    print(f"[✓] Detailed Problematic Snapshots Report saved to: {report_path}")
    
    # Calculate dynamic minimum for heatmap color scale
    min_success_rate = df['SuccessRate'].min()
    
    # =====================================================================
    # FIGURE 1: HIGH-CONTRAST HEATMAP
    # =====================================================================
    years = sorted(df['Year'].unique())
    months_str = [calendar.month_abbr[m] for m in range(1, 13)]
    
    z_data = []
    hover_data = []
    
    for y in years:
        z_row = []
        hover_row = []
        for m in range(1, 13):
            row = df[(df['Year'] == y) & (df['MonthNum'] == m)]
            if not row.empty:
                r = row.iloc[0]
                z_row.append(r['SuccessRate'])
                
                txt = f"<b>{calendar.month_abbr[m]} {y}</b><br>"
                txt += f"Success Rate: <b>{r['SuccessRate']}%</b><br>"
                txt += f"Valid: {r['Valid']} | Disk Missing: {r['Missing']} | Corrupt: {r['Corrupt']}<br>"
                
                if r['Missing'] > 0: txt += f"<br><i>Missing on disk:</i> {r['MissingSites']}"
                if r['Corrupt'] > 0: txt += f"<br><i>Corrupt on disk:</i> {r['CorruptSites']}"
                
                if r['MonthIncompleteFlag'] or r['AbsentSites'] != "None":
                    txt += f"<br><br><b>⚠️ Incomplete Month:</b>"
                    if r['AbsentSites'] != "None":
                        txt += f"<br><i>Lacking sources:</i> {r['AbsentSites']}"
                        
                hover_row.append(txt)
            else:
                z_row.append(None)
                hover_row.append("No Data Recorded")
                
        z_data.append(z_row)
        hover_data.append(hover_row)
    
    fig_heat = go.Figure(data=go.Heatmap(
        z=z_data, x=months_str, y=[str(y) for y in years], text=hover_data,
        hovertemplate="%{text}<extra></extra>", colorscale="RdYlGn",
        zmin=min_success_rate, zmax=100, xgap=2, ygap=2
    ))
    
    fig_heat.update_layout(
        title="<b>Dataset Health: Monthly Valid Snapshots (%)</b>", title_x=0.5,
        xaxis_side="top", template="plotly_white",
        yaxis=dict(autorange="reversed")
    )
    
    heat_path = os.path.join(OUTPUT_DIR, "monthly_availability_heatmap.html")
    fig_heat.write_html(heat_path)
    print(f"[✓] High-Contrast Heatmap saved to: {heat_path}")
    
    # =====================================================================
    # FIGURE 2: PROBLEMATIC BREAKDOWN 
    # =====================================================================
    prob_df = df[~df["FullyAvailable"]].sort_values(["Year", "MonthNum"]).copy()
    
    if not prob_df.empty:
        prob_df['HoverMissing'] = prob_df.apply(
            lambda r: f"<b>Disk Missing: {r['Missing']}</b><br>Sites: {r['MissingSites']}" + 
                      (f"<br><br><b>⚠️ Lacking in JSON:</b><br>{r['AbsentSites']}" if r['AbsentSites'] != "None" else ""), axis=1)
                      
        prob_df['HoverCorrupt'] = prob_df.apply(
            lambda r: f"<b>Corrupt: {r['Corrupt']}</b><br>Sites: {r['CorruptSites']}", axis=1)

        fig_bar = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig_bar.add_trace(go.Bar(
            x=prob_df["YearMonth"], y=prob_df["Missing"], 
            name="Missing on Disk", marker_color="#EF553B",
            text=prob_df["HoverMissing"], hovertemplate="%{text}<extra></extra>"
        ), secondary_y=False)
        
        fig_bar.add_trace(go.Bar(
            x=prob_df["YearMonth"], y=prob_df["Corrupt"], 
            name="Corrupt / Empty", marker_color="#FFA15A",
            text=prob_df["HoverCorrupt"], hovertemplate="%{text}<extra></extra>"
        ), secondary_y=False)
        
        fig_bar.add_trace(go.Scatter(
            x=prob_df["YearMonth"], y=prob_df["SuccessRate"],
            name="Success Rate (%)", mode="lines+markers",
            marker=dict(color="#00CC96", size=8), line=dict(width=3),
            hovertemplate="Success Rate: %{y}%<extra></extra>"
        ), secondary_y=True)
        
        incomplete_flags = prob_df[prob_df["MonthIncompleteFlag"]]
        if not incomplete_flags.empty:
            fig_bar.add_trace(go.Scatter(
                x=incomplete_flags["YearMonth"], 
                y=[prob_df["SuccessRate"].max() * 1.05] * len(incomplete_flags),
                mode="markers", name="Flagged 'Incomplete'",
                marker=dict(symbol="triangle-down", size=12, color="#333333"),
                hoverinfo="skip" 
            ), secondary_y=True)

        fig_bar.update_layout(
            title="<b>Problematic Months: Issue Breakdown & Lacking Websites</b>",
            title_x=0.5, barmode="stack", template="plotly_white",
            hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig_bar.update_yaxes(title_text="Number of Problematic Files", secondary_y=False)
        fig_bar.update_yaxes(title_text="Success Rate (%)", range=[min_success_rate - 5, 110], secondary_y=True)
        
        bar_path = os.path.join(OUTPUT_DIR, "problematic_months_breakdown.html")
        fig_bar.write_html(bar_path)
        print(f"[✓] Problematic Breakdown saved to: {bar_path}")

if __name__ == "__main__":
    main()