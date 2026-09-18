from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "online_retail_II.csv"
ACCENT = "#6C5CE7"
ACCENT_2 = "#00CEC9"
PALETTE = ["#6C5CE7", "#00CEC9", "#FD79A8", "#FDCB6E", "#0984E3", "#00B894", "#E17055"]
FEATURE_COLS = ["Recency", "Frequency", "Monetary", "AvgBasket", "Tenure", "DistinctProducts"]

PLOTLY_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Inter Tight, Sora, sans-serif", size=13, color="#1a1a2e"),
    margin=dict(l=10, r=10, t=40, b=10),
    colorway=PALETTE,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
)
