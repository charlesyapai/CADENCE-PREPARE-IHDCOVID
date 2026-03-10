
import os
import sys
import pandas as pd
import yaml
import logging
from datetime import datetime
import plotly.graph_objects as go
import plotly.io as pio
from matplotlib_venn import venn2, venn3
import matplotlib.pyplot as plt

# ==============================================================================
# 1. LOGGING & SETUP
# ==============================================================================

def setup_logger(name, output_dir):
    """Sets up a file and console logger."""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(name)

def load_config(config_path):
    """Loads YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def find_project_root(start_path, markers=["catalog.py", ".git", "configs"]):
    """
    Traverses up from start_path to find a directory containing one of the markers.
    Returns the path to the root directory.
    """
    current_path = os.path.abspath(start_path)
    while True:
        for marker in markers:
            if os.path.exists(os.path.join(current_path, marker)):
                return current_path
        
        parent = os.path.dirname(current_path)
        if parent == current_path: # Reached filesystem root
            # Fallback: Just return 2 levels up if we can't find markers? 
            # Or raise error? Let's assume 2 levels up from utils.py (src/utils.py -> project_root)
            # utils.py is in src, so dirname(dirname(utils)) is root.
            # But let's verify markers first.
             break
        current_path = parent
    
    # Fallback default
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ==============================================================================
# 2. VISUALIZATION HELPERS
# ==============================================================================

def plot_sankey(labels, source, target, value, title, output_path, node_colors=None, link_colors=None):
    """
    Generates a Sankey diagram using Plotly and saves as PNG.
    """
    if node_colors is None:
        node_colors = "blue"
    
    # Use light gray defaults if link_colors not provided
    if link_colors is None:
        link_colors = ["rgba(200, 200, 200, 0.5)"] * len(source)
        
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=30, # Increased padding to separation
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color=node_colors
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color=link_colors
        )
    )])

    fig.update_layout(
        title_text=title, 
        font=dict(size=14, color="black"), # Slightly reduced from 16 to help fit
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=1200, # Increased height
        width=2500, # Massive width to ensure separation
        margin=dict(l=50, r=500, t=50, b=50) # Huge right margin to force labels right
    )
    
    # Save as Static Image
    try:
        pio.write_image(fig, output_path, format='png', width=1200, height=800)
    except Exception as e:
        print(f"[WARN] Could not save Sankey as PNG (Kaleido missing?): {e}")
        # Fallback to HTML if PNG fails
        html_path = output_path.replace('.png', '.html')
        fig.write_html(html_path)
        print(f"[INFO] Saved as HTML instead: {html_path}")

def plot_venn2(set1, set2, labels, title, output_path):
    """
    Generates a 2-circle Venn diagram using matplotlib.
    """
    plt.figure(figsize=(8, 8))
    venn2([set1, set2], set_labels=labels)
    plt.title(title)
    plt.savefig(output_path)
    plt.close()

def plot_bar_chart(data_dict, title, xlabel, ylabel, output_path):
    """
    Simple QC Bar chart.
    """
    plt.figure(figsize=(10, 6))
    keys = list(data_dict.keys())
    values = list(data_dict.values())
    
    plt.bar(keys, values, color='skyblue')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    # Auto-rotate if many keys
    if len(keys) > 5:
        plt.xticks(rotation=45, ha='right')
        
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# ==============================================================================
# 3. DATAFRAME REPORTING
# ==============================================================================

def report_df_info(df, name, logger=None):
    """
    Logs basic info about a DataFrame.
    """
    msg = f"\n--- Report: {name} ---\n"
    msg += f"Rows: {len(df):,}\n"
    msg += f"Columns: {list(df.columns)}\n"
    msg += f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n"
    
    # Check for NaNs
    nan_counts = df.isna().sum()
    if nan_counts.sum() > 0:
        msg += "Missing Values:\n"
        for col, count in nan_counts[nan_counts > 0].items():
            msg += f"  - {col}: {count} ({count/len(df)*100:.1f}%)\n"
    else:
        msg += "No missing values.\n"
    
    if logger:
        logger.info(msg)
    else:
        print(msg)


# ==============================================================================
# 4. IO HELPERS
# ==============================================================================

def save_with_report(df, output_path, description, logger=None):
    """
    Saves a DataFrame to CSV and generates a sidecar text report.
    
    Args:
        df (pd.DataFrame): Data to save.
        output_path (str): Full path to the CSV file.
        description (str): Brief description for the report header.
        logger (logging.Logger, optional): Logger to record the action.
    """
    # 1. Save Data
    df.to_csv(output_path, index=False)
    
    # 2. Generate Report
    report_path = output_path.replace('.csv', '_report.txt')
    
    msg = f"DATASET REPORT: {description}\n"
    msg += "=" * 50 + "\n"
    msg += f"File: {os.path.basename(output_path)}\n"
    msg += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    msg += "-" * 50 + "\n"
    msg += f"Rows: {len(df):,}\n"
    msg += f"Columns: {len(df.columns)}\n"
    msg += "-" * 50 + "\n"
    msg += "Column Types:\n"
    for col, dtype in df.dtypes.items():
        msg += f"  - {col}: {dtype}\n"
        
    msg += "\nMissing Values:\n"
    nan_counts = df.isna().sum()
    if nan_counts.sum() > 0:
        for col, count in nan_counts[nan_counts > 0].items():
            msg += f"  - {col}: {count} ({count/len(df)*100:.1f}%)\n"
    else:
        msg += "  (None)\n"
        
    msg += "\nSample Data (First 3 rows):\n"
    msg += df.head(3).to_string()
    msg += "\n"
    
    with open(report_path, 'w') as f:
        f.write(msg)
        
    if logger:
        logger.info(f"Saved dataset: {output_path}")
        logger.info(f"Saved report:  {report_path}")

