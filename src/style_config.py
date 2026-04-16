"""
style_config.py
---------------
Shared dark Matplotlib theme and colour constants for all analysis notebooks.
Import at the top of every notebook (02–05) to keep figure styling consistent.
"""
import matplotlib.pyplot as plt

MPESA_GREEN = '#006600'
MPESA_RED   = '#CC0000'
PALETTE     = [
    MPESA_GREEN, MPESA_RED, '#FFD700', '#3498DB',
    '#9B59B6',   '#1ABC9C', '#E67E22', '#95A5A6',
]

def apply_dark_theme() -> None:
    """Apply consistent dark background theme to all Matplotlib figures."""
    plt.rcParams.update({
        'figure.facecolor': '#0F1117',
        'axes.facecolor':   '#1A1D27',
        'axes.edgecolor':   '#2A2D3A',
        'text.color':       '#E8E8E8',
        'axes.labelcolor':  '#E8E8E8',
        'xtick.color':      '#95A5A6',
        'ytick.color':      '#95A5A6',
        'axes.grid':        True,
        'grid.color':       '#2A2D3A',
        'grid.linewidth':   0.5,
        'font.family':      'DejaVu Sans',
    })
