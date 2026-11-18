import os
import glob
import json
from datetime import datetime
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering
import warnings
import random

# Suppress minor sklearn warnings about small datasets when testing with 5 files
warnings.filterwarnings("ignore", category=UserWarning)

# --- CONFIGURATION ---
MASTER_DIR = "" 
OUTPUT_FILE = "ml_cluster_breaks.json"

def get_dom_string(filepath):
    """
    Parses HTML and flattens it into a single space-separated string of structural tags.
    Includes IDs and Classes to catch when developers reuse tags for different content.
    """
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        soup = BeautifulSoup(f.read(), 'lxml')
    
    # Safely ignored: metadata and inline text formatting
    ignore_tags = {'script', 'style', 'meta', 'link', 'head', 'noscript', 'svg', 'path', 'span', 'a', 'strong', 'em', 'b', 'i'}
    sequence = []

    # INSTEAD of parsing the whole page, only parse the main content area
    main_content = soup.find('main') or soup.find('div', id='content') or soup.body
    
    if not main_content:
        return ""
    
    for tag in soup.find_all(True):
        if tag.name in ignore_tags:
            continue
            
        # Build the structural signature (e.g., "div#news-feed.card.large")
        signature = tag.name
        
        tag_id = tag.get('id')
        if tag_id:
            signature += f"#{tag_id}"
            
        tag_classes = tag.get('class')
        if tag_classes:
            signature += f".{'.'.join(sorted(tag_classes))}"
            
        sequence.append(signature)
        
    return " ".join(sequence)

def format_timestamp(filename):
    """Converts YYYYMMDDHHMMSS.html to standard ISO 8601 datetime format."""
    ts_str = os.path.basename(filename).replace('.html', '')
    try:
        dt = datetime.strptime(ts_str, "%Y%m%d%H%M%S")
        return dt.isoformat()
    except ValueError:
        return ts_str

def analyze_site_clusters(html_files):
    """
    Converts HTML files to TF-IDF vectors and clusters them to find layout eras.
    """
    # 1. Convert all files into structural strings
    dom_strings = [get_dom_string(f) for f in html_files]
    
    # 2. Vectorize the strings using N-grams 
    # ngram_range=(1,2) tracks both individual elements and parent->child relationships
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    vectors = vectorizer.fit_transform(dom_strings)
    
    # 3. Agglomerative Clustering
    # Convert sparse TF-IDF matrix to dense array for AgglomerativeClustering
    dense_vectors = vectors.toarray()
    
    # distance_threshold=0.2 means: only create a new cluster if 
    # the vectors are more than 20% different. 
    # n_clusters=None is required when using a distance_threshold.
    clustering = AgglomerativeClustering(
        n_clusters=None, 
        distance_threshold=0.2, 
        metric='cosine', 
        linkage='average'
    )
    labels = clustering.fit_predict(dense_vectors)
    
    # 4. Find the chronological breaks
    breaks = []
    for i in range(1, len(labels)):
        prev_cluster = labels[i-1]
        curr_cluster = labels[i]
        
        if prev_cluster != curr_cluster:
            breaks.append({
                "time_of_break": format_timestamp(html_files[i]),
                "transition": f"Cluster {prev_cluster} -> Cluster {curr_cluster}",
                "previous_file": os.path.basename(html_files[i-1]),
                "changed_file": os.path.basename(html_files[i])
            })
            
    return labels, breaks

def main():
    if not os.path.exists(MASTER_DIR):
        print(f"Error: Directory {MASTER_DIR} not found.")
        return

    report = {}
    
    for site_dir in os.listdir(MASTER_DIR):
        site_path = os.path.join(MASTER_DIR, site_dir)
        if not os.path.isdir(site_path):
            continue
            
        print(f"\nAnalyzing website: {site_dir}")
        
        # Grab all HTML files and sort chronologically
        html_files = sorted(glob.glob(os.path.join(site_path, "*.html")))
        
        # =================================================================
        # TEST MODE LIMIT: Comment this out when ready to run the full dataset
        # html_files = html_files[-5:] 
        # html_files = random.sample(html_files, 5)
        #Chronological order restored for the break detection
        # html_files = sorted(html_files)
        # =================================================================
        
        if len(html_files) < 3:
            print("  Not enough snapshots to cluster (need at least 3 for ML).")
            continue
            
        print(f"  Clustering {len(html_files)} snapshots...")
        labels, breaks = analyze_site_clusters(html_files)
        
        unique_layouts = len(set(labels))
        print(f"  Found {unique_layouts} distinct layout variations.")
        
        if not breaks:
            print("  [✓] No layout breaks detected. The site stayed in one cluster.")
        else:
            for b in breaks:
                print(f"  [!] Break detected at {b['time_of_break']} ({b['transition']})")
                
        report[site_dir] = {
            "total_snapshots_analyzed": len(html_files),
            "distinct_layouts_found": unique_layouts,
            "layout_breaks": breaks
        }

    # Save results to JSON
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4)
        
    print(f"\nProcess complete. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
