import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances

def test_models_and_sources(csv_path="extracted_data.csv"):
    print("Loading data...")
    df = pd.read_csv(csv_path)
    
    # Parse dates and filter out bad rows
    df['timestamp'] = df['timestamp'].astype(str)
    df['datetime'] = pd.to_datetime(df['timestamp'], format='%Y%m%d%H%M%S', errors='coerce')
    df = df.dropna(subset=['datetime', 'headline'])
    df['year_month'] = df['datetime'].dt.strftime('%Y-%m')
    
    # Get the first 4 consecutive months
    unique_months = sorted(df['year_month'].unique())
    if len(unique_months) < 4:
        print(f"Warning: Found only {len(unique_months)} months in data. Using all of them.")
        target_months = unique_months
    else:
        target_months = unique_months[100:104] # Change this slice to test a different window
        
    df = df[df['year_month'].isin(target_months)].copy()
    print(f"Testing on {len(df)} articles across {len(target_months)} months: {', '.join(target_months)}\n")

    # Load Models
    print("Loading MPNet (Base)...")
    model_mpnet = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    
    print("Loading E5 (Large)...")
    model_e5 = SentenceTransformer('intfloat/multilingual-e5-large')

    # Generate Embeddings
    print("\nGenerating MPNet embeddings...")
    df['emb_mpnet'] = list(model_mpnet.encode(df['headline'].tolist(), show_progress_bar=True))
    
    print("Generating E5 embeddings (with prefix)...")
    prefixed_headlines = ["passage: " + h for h in df['headline'].tolist()]
    df['emb_e5'] = list(model_e5.encode(prefixed_headlines, show_progress_bar=True))

    # ---------------------------------------------------------
    # TASK 1: MONTHLY VARIANCE COMPARISON
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("MONTHLY DIVERSITY (VARIANCE) COMPARISON")
    print("="*50)
    
    monthly_stats = []
    for month in target_months:
        month_df = df[df['year_month'] == month]
        
        # MPNet Math
        emb_m = np.vstack(month_df['emb_mpnet'].values)
        cent_m = np.mean(emb_m, axis=0).reshape(1, -1)
        var_m = np.mean(cosine_distances(emb_m, cent_m))
        
        # E5 Math
        emb_e = np.vstack(month_df['emb_e5'].values)
        cent_e = np.mean(emb_e, axis=0).reshape(1, -1)
        var_e = np.mean(cosine_distances(emb_e, cent_e))
        
        monthly_stats.append({
            'Month': month,
            'Articles': len(month_df),
            'MPNet_Variance': round(var_m, 4),
            'E5_Variance': round(var_e, 4)
        })
        
    monthly_df = pd.DataFrame(monthly_stats)
    print(monthly_df.to_string(index=False))

    # ---------------------------------------------------------
    # TASK 2: NEWS SOURCE CENTROIDS
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("NEWS SOURCE DEVIATION FROM GLOBAL MEAN")
    print("="*50)
    
    # Calculate the overarching centroid of the entire 4-month period
    global_cent_m = np.mean(np.vstack(df['emb_mpnet'].values), axis=0).reshape(1, -1)
    global_cent_e = np.mean(np.vstack(df['emb_e5'].values), axis=0).reshape(1, -1)
    
    source_stats = []
    for website, group in df.groupby('website'):
        # MPNet Math
        w_emb_m = np.vstack(group['emb_mpnet'].values)
        w_cent_m = np.mean(w_emb_m, axis=0).reshape(1, -1)
        w_dist_m = cosine_distances(w_cent_m, global_cent_m)[0][0]
        
        # E5 Math
        w_emb_e = np.vstack(group['emb_e5'].values)
        w_cent_e = np.mean(w_emb_e, axis=0).reshape(1, -1)
        w_dist_e = cosine_distances(w_cent_e, global_cent_e)[0][0]
        
        source_stats.append({
            'Website': website,
            'Articles': len(group),
            'MPNet_Dist_to_Mean': round(w_dist_m, 4),
            'E5_Dist_to_Mean': round(w_dist_e, 4)
        })
        
    # Sort by how far they deviate from the center (using E5 as the sorting metric)
    source_df = pd.DataFrame(source_stats).sort_values('E5_Dist_to_Mean', ascending=False)
    print(source_df.to_string(index=False))
    print("\n(Note: Lower distance means the website's coverage is identical to the mainstream average.)")

if __name__ == "__main__":
    test_models_and_sources()