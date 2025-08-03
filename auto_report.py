
import argparse
import time
import base64
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def build_dashboard(input_xlsx: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load ----
    xls = pd.ExcelFile(input_xlsx)
    sheet_name = 'deals' if 'deals' in xls.sheet_names else xls.sheet_names[0]
    deals = xls.parse(sheet_name)
    deals.columns = [str(c).strip() for c in deals.columns]

    # ---- Normalize ----
    def norm_stage(x: str) -> str:
        if not isinstance(x, str): return "Unspecified"
        s = x.strip().lower()
        if "closed" in s and "won" in s or s == "won": return "Closed Won"
        if "closed" in s and "lost" in s or s == "lost": return "Closed Lost"
        if "negoti" in s: return "Negotiation"
        if "proposal" in s or "quote" in s: return "Proposal"
        if "qualif" in s: return "Qualified"
        if "prospect" in s or "lead" in s: return "Lead"
        return x.strip().title()

    amount_col = "amount" if "amount" in deals.columns else None
    stage_col = "stage" if "stage" in deals.columns else None
    created_col = "created_date" if "created_date" in deals.columns else None
    close_col = "close_date" if "close_date" in deals.columns else None
    prob_col = "probability" if "probability" in deals.columns else None

    if amount_col is None or stage_col is None:
        raise ValueError("Input must include at least 'stage' and 'amount' columns in the 'deals' sheet.")

    deals[amount_col] = pd.to_numeric(deals[amount_col], errors="coerce")
    if created_col: deals[created_col] = pd.to_datetime(deals[created_col], errors="coerce")
    if close_col: deals[close_col] = pd.to_datetime(deals[close_col], errors="coerce")

    deals["_stage_norm"] = deals[stage_col].map(norm_stage)

    logical_order = ["Lead","Qualified","Proposal","Negotiation","Closed Won","Closed Lost"]
    present = [s for s in logical_order if s in deals["_stage_norm"].unique().tolist()]
    if len(present) < 3:
        present = deals["_stage_norm"].value_counts().index.tolist()

    counts = deals["_stage_norm"].value_counts()
    rows = []
    for i, stg in enumerate(present):
        sub = deals[deals["_stage_norm"] == stg]
        n = int(sub.shape[0])
        amt_sum = float(sub[amount_col].sum(skipna=True))
        avg_amt = float(sub[amount_col].mean(skipna=True)) if n>0 else np.nan
        conv_prev = np.nan
        if i>0:
            prev = present[i-1]
            prev_n = int(counts.get(prev, 0))
            if prev_n>0: conv_prev = n/prev_n
        rows.append({"Stage": stg, "Deals": n, "Amount (Sum)": amt_sum, "Avg Deal": avg_amt, "Conv from Prev": conv_prev})

    funnel_df = pd.DataFrame(rows)
    funnel_df["Stage"] = pd.Categorical(funnel_df["Stage"], present, ordered=True)
    funnel_df = funnel_df.sort_values("Stage")

    total_deals = int(deals.shape[0])
    won_mask = deals["_stage_norm"].eq("Closed Won") | deals[stage_col].astype(str).str.lower().eq("won")
    won_deals = int(won_mask.sum())
    closed_mask = deals["_stage_norm"].isin(["Closed Won","Closed Lost"]) | deals[stage_col].astype(str).str.lower().isin(["won","lost","closed won","closed lost"])
    closed_total = int(closed_mask.sum())
    win_rate = (won_deals / closed_total) if closed_total>0 else np.nan

    avg_deal_all = float(deals[amount_col].mean(skipna=True))
    open_mask = ~deals["_stage_norm"].isin(["Closed Won","Closed Lost"])
    open_pipeline = float(deals.loc[open_mask, amount_col].sum(skipna=True))

    stage_defaults = {"Lead":0.05,"Qualified":0.2,"Proposal":0.6,"Negotiation":0.75,"Closed Won":1.0,"Closed Lost":0.0}
    if prob_col and prob_col in deals.columns:
        deals["_prob"] = pd.to_numeric(deals[prob_col], errors="coerce")
    else:
        deals["_prob"] = np.nan
    deals["_prob_eff"] = deals["_prob"].where(deals["_prob"].notna(), deals["_stage_norm"].map(stage_defaults).fillna(0.25))
    weighted_pipeline = float(deals.loc[open_mask, amount_col].mul(deals.loc[open_mask, "_prob_eff"]).sum(skipna=True))

    if created_col and close_col and deals.loc[won_mask, [created_col, close_col]].notna().all(axis=None):
        cycle_days = (deals.loc[won_mask, close_col] - deals.loc[won_mask, created_col]).dt.days
        avg_cycle_days = float(cycle_days.mean()) if not cycle_days.empty else np.nan
    else:
        avg_cycle_days = np.nan

    summary_rows = [
        ["Total deals", total_deals],
        ["Won deals", won_deals],
        ["Win rate", win_rate],
        ["Avg deal (all)", avg_deal_all],
        ["Open pipeline (sum)", open_pipeline],
        ["Weighted pipeline (open)", weighted_pipeline],
        ["Avg sales cycle (days, won)", avg_cycle_days],
    ]
    summary_df = pd.DataFrame(summary_rows, columns=["Metric","Value"])

        # ---- Save CSVs ----
    funnel_csv = out_dir / "sales_funnel_dashboard.csv"
    summary_csv = out_dir / "sales_summary_metrics.csv"
    funnel_df.to_csv(funnel_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)


    # charts
    plt.figure(figsize=(8,5))
    funnel_df.plot(x="Stage", y="Deals", kind="barh", legend=False)
    plt.xlabel("Number of Deals"); plt.ylabel("Stage"); plt.title("Deals by Stage")
    plt.tight_layout()
    chart_counts = out_dir / "deals_funnel_counts.png"
    plt.savefig(chart_counts, dpi=160, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8,5))
    amt_series = deals.groupby("_stage_norm")[amount_col].sum().reindex(present).fillna(0)
    amt_series.plot(kind="barh", legend=False)
    plt.xlabel("Pipeline Amount (Sum)"); plt.ylabel("Stage"); plt.title("Pipeline Amount by Stage")
    plt.tight_layout()
    chart_amounts = out_dir / "deals_funnel_amounts.png"
    plt.savefig(chart_amounts, dpi=160, bbox_inches="tight")
    plt.close()

    def img_to_b64(path: Path) -> str:
        with open(path, "rb") as f: return base64.b64encode(f.read()).decode("utf-8")

    counts_b64 = img_to_b64(chart_counts)
    amounts_b64 = img_to_b64(chart_amounts)

    core = funnel_df[funnel_df["Stage"].isin(["Lead","Qualified","Proposal","Negotiation","Closed Won"])].copy()
    core_conv = core.dropna(subset=["Conv from Prev"])
    insight_line = ""
    if not core_conv.empty:
        worst = core_conv.loc[core_conv["Conv from Prev"].astype(float).idxmin()]
        insight_line = f"Lowest conversion appears at <strong>{worst['Stage']}</strong>: <strong>{worst['Conv from Prev']:.1%}</strong>."

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    methodology = """
    <h2>Methodology & Definitions</h2>
    <ul>
      <li><b>Stage Order:</b> Lead &rarr; Qualified &rarr; Proposal &rarr; Negotiation &rarr; Closed Won/Closed Lost.</li>
      <li><b>Win Rate:</b> Closed Won / (Closed Won + Closed Lost).</li>
      <li><b>Conv from Prev (point-in-time):</b> Stage_i &divide; Stage_{i-1}. For cohort accuracy, filter to deals created within the period.</li>
      <li><b>Open Pipeline:</b> Sum of amount for non-closed stages.</li>
      <li><b>Weighted Pipeline:</b> Sum of amount &times; probability; default stage probabilities applied if blank.</li>
      <li><b>Sales Cycle (days):</b> mean(close_date &minus; created_date) for Closed Won.</li>
      <li><b>Pipeline Coverage:</b> Pipeline closing in next 90 days &divide; ARR target.</li>
      <li><b>Ageing:</b> Days since created_date.</li>
    </ul>
    """

    html = f"""<!DOCTYPE html>
    <html lang="en"><head>
    <meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
    <title>Sales Funnel Dashboard</title>
    <style>
      :root {{ --bg:#0b0c10; --card:#15171c; --text:#e6e6e6; --muted:#a0a4ab; --accent:#8ab4f8; --border:#2a2e36; }}
      html,body{{margin:0;padding:0;font-family:Inter,system-ui,Arial,sans-serif;background:var(--bg);color:var(--text);}}
      .wrap{{max-width:1080px;margin:0 auto;padding:24px 16px 64px;}}
      h1,h2,h3{{margin:0 0 12px;line-height:1.2;}}
      h1{{font-size:28px}} h2{{font-size:22px;margin-top:28px}}
      .cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px;margin:16px 0 24px;}}
      .card{{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:14px;}}
      .kpi{{font-size:14px;color:var(--muted);}} .kpi b{{font-size:20px;color:var(--text);display:block;margin-top:4px;}}
      .section{{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:18px;margin-top:18px;}}
      .table{{width:100%;border-collapse:collapse;font-size:14px;color:var(--text);}}
      .table th,.table td{{text-align:left;padding:10px;border-bottom:1px solid var(--border);}}
      .table th{{color:var(--muted);font-weight:600;}}
      img{{width:100%;height:auto;border-radius:8px;display:block;}}
      .hint{{color:var(--muted);font-size:13px;}}
      a{{color:var(--accent);}}
    </style>
    </head>
    <body><div class="wrap">
    <h1>Sales Funnel Dashboard</h1>
    <p class="hint">Generated {now} from {input_xlsx.name}.</p>

    <div class="cards">
      <div class="card kpi">Win Rate<b>{(summary_df.loc[summary_df['Metric']=='Win rate','Value'].values[0]*100 if 'Win rate' in summary_df['Metric'].values else float('nan')):.1f}%</b></div>
      <div class="card kpi">Total Deals<b>{int(summary_df.loc[summary_df['Metric']=='Total deals','Value'].values[0])}</b></div>
      <div class="card kpi">Open Pipeline (sum)<b>{float(summary_df.loc[summary_df['Metric']=='Open pipeline (sum)','Value'].values[0]):,.0f}</b></div>
      <div class="card kpi">Weighted Pipeline (open)<b>{float(summary_df.loc[summary_df['Metric']=='Weighted pipeline (open)','Value'].values[0]):,.0f}</b></div>
    </div>

    <div class="section"><h2>Quick Insight</h2><p>{insight_line or "No obvious bottleneck; consider segmenting by owner or source."}</p></div>
    <div class="section"><h2>Funnel Metrics by Stage</h2>{funnel_df.to_html(index=False, classes="table", border=0, justify="left")}<p class="hint">Conv from Prev uses a point-in-time view.</p></div>
    <div class="section"><h2>Summary KPIs</h2>{summary_df.to_html(index=False, classes="table", border=0, justify="left")}</div>
    <div class="section"><h2>Charts</h2><h3>Deals by Stage</h3><img src="data:image/png;base64,{counts_b64}"/><h3 style="margin-top:12px">Pipeline Amount by Stage</h3><img src="data:image/png;base64,{amounts_b64}"/></div>
    <div class="section">{methodology}</div>
    <p class="hint">Exports: <a href="{(out_dir / 'sales_funnel_dashboard.csv').as_posix()}" download>Funnel metrics (CSV)</a> · <a href="{(out_dir / 'sales_summary_metrics.csv').as_posix()}" download>Summary KPIs (CSV)</a></p>
    </div></body></html>
    """
    html_path = out_dir / "sales_funnel_dashboard.html"
    html_path.write_text(html, encoding="utf-8")

    return {"funnel_csv": str(funnel_csv), "summary_csv": str(summary_csv), "html": str(html_path),
            "chart_counts": str(chart_counts), "chart_amounts": str(chart_amounts)}

def main():
    parser = argparse.ArgumentParser(description="Build the Sales Funnel dashboard from an Excel file.")
    parser.add_argument("--input", required=True, help="Path to the Excel workbook (expects a 'deals' sheet).")
    parser.add_argument("--out", default="output", help="Output directory (default: ./output)")
    parser.add_argument("--watch", action="store_true", help="Watch the input file and rebuild on changes.")
    args = parser.parse_args()

    input_xlsx = Path(args.input).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()

    if not input_xlsx.exists():
        raise SystemExit(f"Input file not found: {input_xlsx}")

    print("Building dashboard...")
    result = build_dashboard(input_xlsx, out_dir)
    print("Done:", result["html"])

    if not args.watch:
        return

    print("Watching for changes. Press Ctrl+C to stop.")
    last_mtime = input_xlsx.stat().st_mtime
    try:
        while True:
            import time as _t; _t.sleep(1.0)
            try:
                mtime = input_xlsx.stat().st_mtime
            except FileNotFoundError:
                continue
            if mtime != last_mtime:
                last_mtime = mtime
                try:
                    print("Change detected; rebuilding...")
                    result = build_dashboard(input_xlsx, out_dir)
                    print("Rebuilt:", result["html"])
                except Exception as e:
                    print("Error rebuilding:", e)
    except KeyboardInterrupt:
        print("Stopped watcher.")

if __name__ == "__main__":
    main()
