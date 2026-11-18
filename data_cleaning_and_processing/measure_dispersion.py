import os
import time
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances
from sklearn.cluster import KMeans
from scipy.stats import entropy

# =========================================================
# GLOBAL CONFIGURATION
# =========================================================
INPUT_CSV_PATH = "extracted_data.csv"
EMBEDDINGS_CSV_PATH = ""  # Global output path for embeddings
IMG_DIR = "./img"                                       # Subdirectory for plot images
METRICS_CSV_PATH = "./monthly_diversity_metrics.csv"    # Diversity summary metrics table
PAUSE_DURATION_SECONDS = 8                            # Pause duration at 25%, 50%, and 75%
SMOOTHING_WINDOW_MONTHS = 6                         # Determines how heavily to smooth the trendline


def run_diversity_analysis():
    # 1. Ensure Output Directory Exists
    os.makedirs(IMG_DIR, exist_ok=True)

    # 2. Hardware Acceleration Check
    if torch.cuda.is_available():
        device = "cuda"
        print("GPU Detected: Using NVIDIA CUDA")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("GPU Detected: Using Apple Silicon MPS")
    else:
        device = "cpu"
        print("No GPU detected: Falling back to CPU")

    # 3. Load and Prepare Data
    print(f"\nLoading dataset from '{INPUT_CSV_PATH}'...")
    df = pd.read_csv(INPUT_CSV_PATH)
    df['timestamp'] = df['timestamp'].astype(str)
    df['datetime'] = pd.to_datetime(
        df['timestamp'].str.extract(r'(\d{14})')[0], 
        format='%Y%m%d%H%M%S', 
        errors='coerce'
    )
    df = df.dropna(subset=['datetime', 'headline']).reset_index(drop=True)
    df['year_month'] = df['datetime'].dt.strftime('%Y-%m')
    
    months = sorted(df['year_month'].unique())
    total_articles = len(df)
    print(f"Loaded {total_articles} articles across {len(months)} months.")

    # 4 & 5. Check Cache or Generate Embeddings
    recompute_embeddings = True

    if os.path.exists(EMBEDDINGS_CSV_PATH):
        print(f"\nFound existing embeddings cache at '{EMBEDDINGS_CSV_PATH}'. Validating...")
        cached_df = pd.read_csv(EMBEDDINGS_CSV_PATH)
        
        if len(cached_df) == total_articles:
            dim_cols = [col for col in cached_df.columns if str(col).startswith('dim_')]
            if len(dim_cols) > 0:
                print(f"Cache is valid! Loading {len(dim_cols)}-dimensional embeddings from CSV...")
                embeddings_matrix = cached_df[dim_cols].values
                df['embedding'] = list(embeddings_matrix)
                recompute_embeddings = False
            else:
                print("Cache is corrupted (missing 'dim_' columns). Recomputing...")
        else:
            print(f"Cache row count mismatch (found {len(cached_df)}, expected {total_articles}). Recomputing...")

    if recompute_embeddings:
        print("\nLoading MPNet model...")
        model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2', device=device)

        headlines = df['headline'].tolist()
        q1 = int(total_articles * 0.25)
        q2 = int(total_articles * 0.50)
        q3 = int(total_articles * 0.75)

        slices = [
            (0, q1, "25%"),
            (q1, q2, "50%"),
            (q2, q3, "75%"),
            (q3, total_articles, "100%")
        ]

        all_embeddings = []
        print("\nComputing embeddings with pauses at 25%, 50%, and 75%...")

        for idx, (start_idx, end_idx, pct_label) in enumerate(slices):
            chunk = headlines[start_idx:end_idx]
            print(f"\nEncoding chunk {idx + 1}/4 (items {start_idx} to {end_idx}, reaching {pct_label})...")
            
            chunk_embeddings = model.encode(
                chunk, 
                show_progress_bar=True, 
                device=device,
                batch_size=64
            )
            all_embeddings.append(chunk_embeddings)

            if idx < len(slices) - 1:
                print(f"Reached ~{pct_label} completion. Pausing for {PAUSE_DURATION_SECONDS}s...")
                time.sleep(PAUSE_DURATION_SECONDS)

        embeddings_matrix = np.vstack(all_embeddings)
        df['embedding'] = list(embeddings_matrix)
        print("\nAll embeddings successfully computed!")

        print(f"\nSaving embeddings to '{EMBEDDINGS_CSV_PATH}'...")
        emb_dim_cols = [f"dim_{i}" for i in range(embeddings_matrix.shape[1])]
        emb_df = pd.DataFrame(embeddings_matrix, columns=emb_dim_cols)

        meta_cols = [col for col in ['website', 'timestamp', 'year_month', 'headline'] if col in df.columns]
        full_embeddings_export = pd.concat([df[meta_cols], emb_df], axis=1)

        full_embeddings_export.to_csv(EMBEDDINGS_CSV_PATH, index=False)
        print(f"Embeddings saved successfully ({full_embeddings_export.shape[0]} rows).")

    # 6. Calculate Monthly Diversity
    print("\nCalculating monthly metrics...")
    results = []
    
    for month in months:
        month_df = df[df['year_month'] == month]
        month_emb = np.vstack(month_df['embedding'].values)
        
        centroid = np.mean(month_emb, axis=0).reshape(1, -1)
        variance = np.mean(cosine_distances(month_emb, centroid))
        
        n_clusters = min(10, len(month_emb) // 2)
        if n_clusters > 1:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
            labels = kmeans.fit_predict(month_emb)
            _, counts = np.unique(labels, return_counts=True)
            shannon_ent = entropy(counts, base=2)
        else:
            shannon_ent = 0.0

        results.append({
            'Month': month,
            'Articles': len(month_df),
            'Variance': round(variance, 4),
            'Entropy': round(shannon_ent, 4)
        })

    results_df = pd.DataFrame(results)
    results_df['plot_date'] = pd.to_datetime(results_df['Month'])
    results_df['Var_Trend'] = results_df['Variance'].rolling(window=SMOOTHING_WINDOW_MONTHS, center=True, min_periods=1).mean().round(4)
    results_df['Ent_Trend'] = results_df['Entropy'].rolling(window=SMOOTHING_WINDOW_MONTHS, center=True, min_periods=1).mean().round(4)

    print("\n" + "=" * 55)
    print("MONTHLY DIVERSITY REPORT")
    print("=" * 55)
    print(results_df[['Month', 'Articles', 'Variance', 'Entropy']].to_string(index=False))
    
    export_df = results_df.drop(columns=['plot_date'])
    export_df.to_csv(METRICS_CSV_PATH, index=False)
    print(f"\nMetrics saved to '{METRICS_CSV_PATH}'")

    # 7. Generate Static Matplotlib Plots (PNG)
    print(f"\nGenerating and saving static PNG plots under '{IMG_DIR}/'...")

    def format_xaxis(ax):
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[7])) 
        ax.grid(True, which='major', linestyle='--', alpha=0.6)
        ax.grid(True, which='minor', linestyle=':', alpha=0.3)

    dates = results_df['plot_date']

    # Static Plot 1: Combined
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    color1 = 'tab:blue'
    ax1.plot(dates, results_df['Variance'], marker='o', linestyle='', color=color1, alpha=0.3, label='Var (Raw)')
    ax1.plot(dates, results_df['Var_Trend'], color=color1, linewidth=3, label='Var (Trend)')
    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('Semantic Variance (Distance)', color=color1, fontsize=12, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=color1)
    
    ax2 = ax1.twinx()  
    color2 = 'tab:red'
    ax2.plot(dates, results_df['Entropy'], marker='s', linestyle='', color=color2, alpha=0.3, label='Ent (Raw)')
    ax2.plot(dates, results_df['Ent_Trend'], color=color2, linewidth=3, linestyle='-', label='Ent (Trend)')
    ax2.set_ylabel('Shannon Entropy (Evenness)', color=color2, fontsize=12, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=color2)

    plt.title('News Topic Diversity: Variance vs. Entropy', fontsize=14, pad=15)
    format_xaxis(ax1)
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')
    fig1.autofmt_xdate()
    fig1.tight_layout()
    fig1.savefig(os.path.join(IMG_DIR, 'plot_1_combined.png'), dpi=300)
    plt.close(fig1)

    # Static Plot 2: Variance Only
    fig2, ax3 = plt.subplots(figsize=(12, 6))
    ax3.plot(dates, results_df['Variance'], marker='o', linestyle='-', color='tab:blue', alpha=0.3, label='Raw Monthly Data')
    ax3.plot(dates, results_df['Var_Trend'], color='tab:blue', linewidth=3, label=f'{SMOOTHING_WINDOW_MONTHS}-Month Trend')
    ax3.set_title('Semantic Variance (Topic Distance) Over Time', fontsize=14, pad=15)
    ax3.set_xlabel('Year', fontsize=12)
    ax3.set_ylabel('Variance', fontsize=12, fontweight='bold')
    format_xaxis(ax3)
    ax3.legend()
    fig2.autofmt_xdate()
    fig2.tight_layout()
    fig2.savefig(os.path.join(IMG_DIR, 'plot_2_variance_only.png'), dpi=300)
    plt.close(fig2)

    # Static Plot 3: Entropy Only
    fig3, ax4 = plt.subplots(figsize=(12, 6))
    ax4.plot(dates, results_df['Entropy'], marker='s', linestyle='-', color='tab:red', alpha=0.3, label='Raw Monthly Data')
    ax4.plot(dates, results_df['Ent_Trend'], color='tab:red', linewidth=3, label=f'{SMOOTHING_WINDOW_MONTHS}-Month Trend')
    ax4.set_title('Shannon Entropy (Topic Evenness) Over Time', fontsize=14, pad=15)
    ax4.set_xlabel('Year', fontsize=12)
    ax4.set_ylabel('Entropy', fontsize=12, fontweight='bold')
    format_xaxis(ax4)
    ax4.legend()
    fig3.autofmt_xdate()
    fig3.tight_layout()
    fig3.savefig(os.path.join(IMG_DIR, 'plot_3_entropy_only.png'), dpi=300)
    plt.close(fig3)

    # 8. Generate Interactive Plotly Plots (HTML)
    print(f"\nGenerating and saving interactive HTML plots under '{IMG_DIR}/'...")

    custom_data = np.stack(
        (results_df['Month'], results_df['Articles']), 
        axis=-1
    )

    # Interactive Plot 1: Combined Dual-Axis
    fig_html1 = make_subplots(specs=[[{"secondary_y": True}]])

    # Variance Traces
    fig_html1.add_trace(
        go.Scatter(
            x=results_df['plot_date'],
            y=results_df['Variance'],
            mode='markers',
            name='Variance (Raw)',
            marker=dict(color='rgba(31, 119, 180, 0.4)', size=7),
            customdata=custom_data,
            hovertemplate="<b>Month:</b> %{customdata[0]}<br><b>Articles:</b> %{customdata[1]}<br><b>Raw Variance:</b> %{y:.4f}<extra></extra>"
        ),
        secondary_y=False
    )
    fig_html1.add_trace(
        go.Scatter(
            x=results_df['plot_date'],
            y=results_df['Var_Trend'],
            mode='lines',
            name=f'Variance ({SMOOTHING_WINDOW_MONTHS}m Trend)',
            line=dict(color='rgb(31, 119, 180)', width=3),
            customdata=custom_data,
            hovertemplate="<b>Month:</b> %{customdata[0]}<br><b>Variance Trend:</b> %{y:.4f}<extra></extra>"
        ),
        secondary_y=False
    )

    # Entropy Traces
    fig_html1.add_trace(
        go.Scatter(
            x=results_df['plot_date'],
            y=results_df['Entropy'],
            mode='markers',
            name='Entropy (Raw)',
            marker=dict(color='rgba(214, 39, 40, 0.4)', size=7, symbol='square'),
            customdata=custom_data,
            hovertemplate="<b>Month:</b> %{customdata[0]}<br><b>Articles:</b> %{customdata[1]}<br><b>Raw Entropy:</b> %{y:.4f}<extra></extra>"
        ),
        secondary_y=True
    )
    fig_html1.add_trace(
        go.Scatter(
            x=results_df['plot_date'],
            y=results_df['Ent_Trend'],
            mode='lines',
            name=f'Entropy ({SMOOTHING_WINDOW_MONTHS}m Trend)',
            line=dict(color='rgb(214, 39, 40)', width=3, dash='dash'),
            customdata=custom_data,
            hovertemplate="<b>Month:</b> %{customdata[0]}<br><b>Entropy Trend:</b> %{y:.4f}<extra></extra>"
        ),
        secondary_y=True
    )

    fig_html1.update_layout(
        title='News Topic Diversity: Variance vs. Entropy (Interactive)',
        hovermode='x unified',
        template='plotly_white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )
    fig_html1.update_xaxes(title_text='Year', showgrid=True)
    fig_html1.update_yaxes(title_text='Semantic Variance (Distance)', secondary_y=False, title_font=dict(color='rgb(31, 119, 180)'))
    fig_html1.update_yaxes(title_text='Shannon Entropy (Evenness)', secondary_y=True, title_font=dict(color='rgb(214, 39, 40)'))
    fig_html1.write_html(os.path.join(IMG_DIR, 'plot_1_combined.html'))

    # Interactive Plot 2: Variance Only
    fig_html2 = go.Figure()
    fig_html2.add_trace(
        go.Scatter(
            x=results_df['plot_date'],
            y=results_df['Variance'],
            mode='markers',
            name='Raw Monthly Variance',
            marker=dict(color='rgba(31, 119, 180, 0.4)', size=8),
            customdata=custom_data,
            hovertemplate="<b>Month:</b> %{customdata[0]}<br><b>Articles:</b> %{customdata[1]}<br><b>Raw Variance:</b> %{y:.4f}<extra></extra>"
        )
    )
    fig_html2.add_trace(
        go.Scatter(
            x=results_df['plot_date'],
            y=results_df['Var_Trend'],
            mode='lines',
            name=f'{SMOOTHING_WINDOW_MONTHS}-Month Trend',
            line=dict(color='rgb(31, 119, 180)', width=3),
            customdata=custom_data,
            hovertemplate="<b>Month:</b> %{customdata[0]}<br><b>Trend Variance:</b> %{y:.4f}<extra></extra>"
        )
    )
    fig_html2.update_layout(
        title='Semantic Variance Over Time (Interactive)',
        xaxis_title='Year',
        yaxis_title='Semantic Variance (Distance)',
        hovermode='x unified',
        template='plotly_white'
    )
    fig_html2.write_html(os.path.join(IMG_DIR, 'plot_2_variance_only.html'))

    # Interactive Plot 3: Entropy Only
    fig_html3 = go.Figure()
    fig_html3.add_trace(
        go.Scatter(
            x=results_df['plot_date'],
            y=results_df['Entropy'],
            mode='markers',
            name='Raw Monthly Entropy',
            marker=dict(color='rgba(214, 39, 40, 0.4)', size=8, symbol='square'),
            customdata=custom_data,
            hovertemplate="<b>Month:</b> %{customdata[0]}<br><b>Articles:</b> %{customdata[1]}<br><b>Raw Entropy:</b> %{y:.4f}<extra></extra>"
        )
    )
    fig_html3.add_trace(
        go.Scatter(
            x=results_df['plot_date'],
            y=results_df['Ent_Trend'],
            mode='lines',
            name=f'{SMOOTHING_WINDOW_MONTHS}-Month Trend',
            line=dict(color='rgb(214, 39, 40)', width=3),
            customdata=custom_data,
            hovertemplate="<b>Month:</b> %{customdata[0]}<br><b>Trend Entropy:</b> %{y:.4f}<extra></extra>"
        )
    )
    fig_html3.update_layout(
        title='Shannon Entropy Over Time (Interactive)',
        xaxis_title='Year',
        yaxis_title='Shannon Entropy (Evenness)',
        hovermode='x unified',
        template='plotly_white'
    )
    fig_html3.write_html(os.path.join(IMG_DIR, 'plot_3_entropy_only.html'))

    print(f"- Saved: {os.path.join(IMG_DIR, 'plot_1_combined.html')}")
    print(f"- Saved: {os.path.join(IMG_DIR, 'plot_2_variance_only.html')}")
    print(f"- Saved: {os.path.join(IMG_DIR, 'plot_3_entropy_only.html')}")
    print("\nProcessing complete!")


if __name__ == "__main__":
    run_diversity_analysis()
