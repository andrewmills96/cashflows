"""
cdi_visualisations.py
=====================
Reporting plots for the Fox CDI Mandate simulation results.

Usage (in a Jupyter notebook):
    from cdi_visualisations import CDIVisualiser
    vis = CDIVisualiser(df)          # df = cdi_fox.run(val_date, rates, spread_map, sim_results)
    vis.plot_all()                   # render every figure
    vis.plot_obligation_coverage()   # or call individually
    vis.plot_assets_and_funding()
    vis.plot_underperformance_risk()
    vis.plot_cashflow_distribution()
    vis.plot_return_comparison()
    vis.summary_tables()             # prints all summary DataFrames

The class expects the DataFrame produced by CDIMandate_Fox.run(), which has
a MultiIndex (scenario, date) and columns including:
    assets, cash, fee, liab_pv_gaap, liab_pv_ifrs,
    funding_level_gaap, funding_level_ifrs,
    cdi_return, net_cdi_return, bt, net_bt_return,
    hgb_payment, additional_payment, performance_payment,
    asset_cashflow, expected_cdi_cashflow, bund_comparator
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns
from scipy import stats

# ── Style ─────────────────────────────────────────────────────────────────────
PALETTE = {
    "bg":        "#0F1923",
    "panel":     "#152030",
    "border":    "#1E3048",
    "text":      "#D8E4F0",
    "muted":     "#6B85A3",
    "blue":      "#3D9CF0",
    "amber":     "#F0A23D",
    "teal":      "#3DCFB0",
    "red":       "#F0504A",
    "purple":    "#9B6EF0",
    "green":     "#5DD87A",
}

def _apply_style():
    plt.rcParams.update({
        "figure.facecolor":    PALETTE["bg"],
        "axes.facecolor":      PALETTE["panel"],
        "axes.edgecolor":      PALETTE["border"],
        "axes.labelcolor":     PALETTE["muted"],
        "axes.titlecolor":     PALETTE["text"],
        "axes.titlesize":      12,
        "axes.labelsize":      10,
        "axes.titlepad":       12,
        "axes.spines.top":     False,
        "axes.spines.right":   False,
        "axes.spines.left":    True,
        "axes.spines.bottom":  True,
        "axes.grid":           True,
        "grid.color":          PALETTE["border"],
        "grid.linewidth":      0.6,
        "grid.alpha":          0.8,
        "xtick.color":         PALETTE["muted"],
        "ytick.color":         PALETTE["muted"],
        "xtick.labelsize":     9,
        "ytick.labelsize":     9,
        "legend.facecolor":    PALETTE["panel"],
        "legend.edgecolor":    PALETTE["border"],
        "legend.labelcolor":   PALETTE["muted"],
        "legend.fontsize":     9,
        "lines.linewidth":     1.8,
        "figure.dpi":          130,
        "savefig.dpi":         200,
        "savefig.facecolor":   PALETTE["bg"],
        "savefig.bbox":        "tight",
        "font.family":         "sans-serif",
        "font.sans-serif":     ["IBM Plex Sans", "Helvetica Neue", "Arial", "DejaVu Sans"],
    })

# ── Helper functions ──────────────────────────────────────────────────────────

def _fmt_m(x, pos=None):
    """Tick formatter: millions with €."""
    return f"€{x/1e6:.0f}m"

def _fmt_pct(x, pos=None):
    return f"{x*100:.1f}%"

def _fmt_pct_int(x, pos=None):
    return f"{x*100:.0f}%"

def _fan(ax, years, p5, p25, p50, p75, p95, mean_=None,
         color=PALETTE["blue"], label_suffix=""):
    """Draw a percentile fan (bands + median line) onto ax."""
    ax.fill_between(years, p5,  p95, alpha=0.10, color=color, linewidth=0)
    ax.fill_between(years, p25, p75, alpha=0.20, color=color, linewidth=0)
    ax.plot(years, p50,   color=color,          linewidth=2.0, label=f"Median{label_suffix}")
    ax.plot(years, p5,    color=color,          linewidth=0.8, linestyle="--", alpha=0.5)
    ax.plot(years, p95,   color=color,          linewidth=0.8, linestyle="--", alpha=0.5,
            label=f"P5/P95{label_suffix}")
    if mean_ is not None:
        ax.plot(years, mean_, color=PALETTE["muted"], linewidth=1.2,
                linestyle=":", label=f"Mean{label_suffix}")

def _section_title(fig, text, y=0.98):
    fig.text(0.5, y, text, ha="center", va="top", fontsize=15, fontweight="bold",
             color=PALETTE["text"], fontfamily="serif")

def _fig_label(ax, text):
    ax.set_title(text, pad=10, fontsize=11, color=PALETTE["text"], fontweight="600")

def _despine(ax):
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(PALETTE["border"])
    ax.spines["bottom"].set_color(PALETTE["border"])


# ── Main class ────────────────────────────────────────────────────────────────

class CDIVisualiser:
    """
    Encapsulates all CDI reporting plots.

    Parameters
    ----------
    df : pd.DataFrame
        Output of CDIMandate_Fox.run(). Must contain columns described at top of file.
    discount_rate : float
        Rate used to compute present values of obligations (default: IFRS rate 3.9%).
    """

    def __init__(self, df: pd.DataFrame, discount_rate: float = 0.039):
        _apply_style()
        self.df = df.copy()
        self.r  = discount_rate

        # ── Derived per-scenario per-year pivot tables ────────────────────────
        # Each is a DataFrame: index=scenario, columns=year (datetime)
        self._scenarios  = sorted(df["scenario"].unique())
        self._dates      = sorted(df["date"].unique())
        self._n_sim      = len(self._scenarios)
        self._n_years    = len(self._dates)
        self._years_int  = [d.year for d in self._dates]

        # Wide pivot helper
        def _wide(col):
            return df.pivot(index="scenario", columns="date", values=col).values  # (n_sim, T)

        self._assets       = _wide("assets")
        self._fl_gaap      = _wide("funding_level_gaap")
        self._fl_ifrs      = _wide("funding_level_ifrs")
        self._cdi_ret      = _wide("net_cdi_return")
        self._bt_ret       = _wide("net_bt_return")
        self._hgb_pay      = _wide("hgb_payment")
        self._add_pay      = _wide("additional_payment")
        self._perf_pay     = _wide("performance_payment")
        self._fee          = _wide("fee")
        self._bund_comp    = _wide("bund_comparator")
        self._asset_cf     = _wide("asset_cashflow")
        self._exp_cdi_cf   = _wide("expected_cdi_cashflow")

        # Time fractions for PV discounting (year index 0 = year 1)
        self._t = np.arange(1, self._n_years + 1, dtype=float)

        # PV of each obligation stream per scenario: (n_sim,)
        df_factors = (1 + self.r) ** self._t          # (T,)
        self._pv_hgb  = (self._hgb_pay  / df_factors).sum(axis=1)
        self._pv_add  = (self._add_pay  / df_factors).sum(axis=1)
        self._pv_perf = (self._perf_pay / df_factors).sum(axis=1)
        self._pv_fee  = (self._fee       / df_factors).sum(axis=1)
        self._pv_total = self._pv_hgb + self._pv_add + self._pv_perf

        # Underperformance flag: CMBP comparator > assets at each year
        self._underperf = (self._bund_comp > self._assets).astype(float)

    # ── Percentile helpers ────────────────────────────────────────────────────
    def _pctiles(self, arr):
        """Returns (p5, p25, p50, p75, p95, mean) each shape (T,)."""
        return (
            np.percentile(arr, 5,  axis=0),
            np.percentile(arr, 25, axis=0),
            np.percentile(arr, 50, axis=0),
            np.percentile(arr, 75, axis=0),
            np.percentile(arr, 95, axis=0),
            arr.mean(axis=0),
        )

    # ── 1. OBLIGATION COVERAGE ────────────────────────────────────────────────
    def plot_obligation_coverage(self):
        """
        Figure 1 — Obligation Coverage
        ┌────────────────────┬────────────────────┐
        │ PV bar chart       │ Fee vs Obligations  │
        │ (breakdown)        │ scatter             │
        ├────────────────────┼────────────────────┤
        │ HGB payment dist.  │ Perf. payment dist. │
        └────────────────────┴────────────────────┘
        """
        fig = plt.figure(figsize=(14, 10))
        _section_title(fig, "1 · Obligation Coverage", y=0.995)
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32,
                               top=0.95, bottom=0.07, left=0.08, right=0.97)

        # ── 1a: Stacked mean PV bar chart ─────────────────────────────────
        ax1 = fig.add_subplot(gs[0, 0])
        labels    = ["HGB Gap\nPayments", "Year-10\nAdditional", "Perf.\nGuarantee",
                     "Total\nObligations", "Fee\nIncome (PV)"]
        means     = [self._pv_hgb.mean(), self._pv_add.mean(), self._pv_perf.mean(),
                     self._pv_total.mean(), self._pv_fee.mean()]
        p5s       = [np.percentile(self._pv_hgb, 5),  np.percentile(self._pv_add, 5),
                     np.percentile(self._pv_perf, 5),  np.percentile(self._pv_total, 5),
                     np.percentile(self._pv_fee, 5)]
        p95s      = [np.percentile(self._pv_hgb, 95), np.percentile(self._pv_add, 95),
                     np.percentile(self._pv_perf, 95), np.percentile(self._pv_total, 95),
                     np.percentile(self._pv_fee, 95)]
        colors    = [PALETTE["amber"], PALETTE["teal"], PALETTE["red"],
                     PALETTE["blue"], PALETTE["green"]]
        x         = np.arange(len(labels))
        bars      = ax1.bar(x, means, color=colors, alpha=0.85, width=0.6, zorder=3)
        ax1.errorbar(x, means,
                     yerr=[np.array(means) - np.array(p5s),
                            np.array(p95s) - np.array(means)],
                     fmt="none", color=PALETTE["text"], capsize=5, linewidth=1.2, zorder=4)
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, fontsize=8.5)
        ax1.yaxis.set_major_formatter(mtick.FuncFormatter(_fmt_m))
        ax1.axhline(self._pv_fee.mean(), color=PALETTE["green"],
                    linewidth=1.5, linestyle="--", alpha=0.7, label="Mean fee PV")
        _fig_label(ax1, "PV of Obligations vs Fee Income")
        ax1.set_ylabel("Present Value (€)", color=PALETTE["muted"])
        # annotate coverage
        coverage = self._pv_fee.mean() / self._pv_total.mean()
        ax1.text(0.97, 0.96, f"Coverage ratio: {coverage:.1%}",
                 transform=ax1.transAxes, ha="right", va="top",
                 fontsize=9, color=PALETTE["green"] if coverage >= 1 else PALETTE["red"],
                 fontweight="bold",
                 bbox=dict(facecolor=PALETTE["bg"], edgecolor=PALETTE["border"],
                           boxstyle="round,pad=0.4"))
        _despine(ax1)

        # ── 1b: Fee PV vs Total Obligation PV scatter ─────────────────────
        ax2 = fig.add_subplot(gs[0, 1])
        sc  = ax2.scatter(self._pv_total / 1e6, self._pv_fee / 1e6,
                          c=self._pv_fee / np.maximum(self._pv_total, 1),
                          cmap="RdYlGn", alpha=0.35, s=6, linewidths=0)
        plt.colorbar(sc, ax=ax2, label="Coverage ratio", format=mtick.FuncFormatter(_fmt_pct_int))
        lim = max(self._pv_total.max(), self._pv_fee.max()) / 1e6
        ax2.plot([0, lim], [0, lim], color=PALETTE["muted"],
                 linewidth=1.2, linestyle="--", label="Break-even")
        ax2.set_xlabel("Total Obligation PV (€m)")
        ax2.set_ylabel("Fee Income PV (€m)")
        _fig_label(ax2, "Fee PV vs Obligation PV (per scenario)")
        ax2.legend(fontsize=8)
        _despine(ax2)

        # ── 1c: HGB payment distribution ──────────────────────────────────
        ax3 = fig.add_subplot(gs[1, 0])
        total_hgb = self._hgb_pay.sum(axis=1)
        ax3.hist(total_hgb / 1e6, bins=50, color=PALETTE["amber"],
                 alpha=0.80, edgecolor="none", density=False)
        ax3.axvline(total_hgb.mean() / 1e6, color=PALETTE["text"],
                    linewidth=1.5, linestyle="--", label=f"Mean: {_fmt_m(total_hgb.mean())}")
        ax3.axvline(np.percentile(total_hgb, 95) / 1e6,
                    color=PALETTE["red"], linewidth=1.2, linestyle=":",
                    label=f"P95: {_fmt_m(np.percentile(total_hgb, 95))}")
        ax3.set_xlabel("Total HGB Gap Payments over 25yr (€m)")
        ax3.set_ylabel("Frequency")
        _fig_label(ax3, "Distribution of Cumulative HGB Gap Payments")
        ax3.legend(fontsize=8)
        _despine(ax3)

        # ── 1d: Performance payment distribution ──────────────────────────
        ax4 = fig.add_subplot(gs[1, 1])
        perf_25 = self._perf_pay[:, -1]  # Year-25 payment
        zero_pct = (perf_25 == 0).mean()
        ax4.hist(perf_25[perf_25 > 0] / 1e6, bins=50, color=PALETTE["red"],
                 alpha=0.80, edgecolor="none", density=False,
                 label=f"Non-zero ({1 - zero_pct:.1%} of scenarios)")
        ax4.axvline(perf_25.mean() / 1e6, color=PALETTE["text"],
                    linewidth=1.5, linestyle="--",
                    label=f"Mean (all): {_fmt_m(perf_25.mean())}")
        ax4.set_xlabel("Year-25 Performance Payment (€m)")
        ax4.set_ylabel("Frequency")
        _fig_label(ax4, "Year-25 Performance Guarantee Payment")
        ax4.legend(fontsize=8)
        ax4.text(0.97, 0.96, f"{zero_pct:.1%} scenarios: no payment",
                 transform=ax4.transAxes, ha="right", va="top",
                 fontsize=8.5, color=PALETTE["muted"])
        _despine(ax4)

        fig.suptitle("")
        plt.show()
        return fig

    # ── 2. ASSETS & FUNDING LEVEL ─────────────────────────────────────────────
    def plot_assets_and_funding(self):
        """
        Figure 2 — Asset value and funding level distributions over time.
        ┌──────────────────┬──────────────────┐
        │ Assets fan       │ FL GAAP fan       │
        ├──────────────────┼──────────────────┤
        │ FL IFRS fan      │ Year-25 FL dist.  │
        └──────────────────┴──────────────────┘
        """
        fig = plt.figure(figsize=(14, 10))
        _section_title(fig, "2 · Asset Value & Funding Level", y=0.995)
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.30,
                               top=0.95, bottom=0.07, left=0.09, right=0.97)
        yrs = self._years_int

        # ── 2a: Total assets fan ──────────────────────────────────────────
        ax1 = fig.add_subplot(gs[0, 0])
        p5, p25, p50, p75, p95, mu = self._pctiles(self._assets)
        _fan(ax1, yrs, p5, p25, p50, p75, p95, mean_=mu, color=PALETTE["blue"])
        ax1.yaxis.set_major_formatter(mtick.FuncFormatter(_fmt_m))
        ax1.set_xlabel("Year")
        ax1.set_ylabel("Total Assets (€)")
        _fig_label(ax1, "Total Asset Value — Simulation Distribution")
        ax1.legend(fontsize=8, loc="upper left")
        _despine(ax1)

        # ── 2b: Funding level GAAP fan ────────────────────────────────────
        ax2 = fig.add_subplot(gs[0, 1])
        p5, p25, p50, p75, p95, mu = self._pctiles(self._fl_gaap)
        _fan(ax2, yrs, p5, p25, p50, p75, p95, mean_=mu, color=PALETTE["teal"])
        ax2.axhline(1.0, color=PALETTE["red"], linewidth=1.5, linestyle="--",
                    alpha=0.8, label="100% (fully funded)")
        ax2.yaxis.set_major_formatter(mtick.FuncFormatter(_fmt_pct_int))
        ax2.set_xlabel("Year")
        ax2.set_ylabel("Funding Level (GAAP / HGB)")
        _fig_label(ax2, "Funding Level (GAAP) Over Time")
        ax2.legend(fontsize=8, loc="lower right")
        _despine(ax2)

        # ── 2c: Funding level IFRS fan ────────────────────────────────────
        ax3 = fig.add_subplot(gs[1, 0])
        p5, p25, p50, p75, p95, mu = self._pctiles(self._fl_ifrs)
        _fan(ax3, yrs, p5, p25, p50, p75, p95, mean_=mu, color=PALETTE["purple"])
        ax3.axhline(1.0, color=PALETTE["red"], linewidth=1.5, linestyle="--",
                    alpha=0.8, label="100% (fully funded)")
        ax3.yaxis.set_major_formatter(mtick.FuncFormatter(_fmt_pct_int))
        ax3.set_xlabel("Year")
        ax3.set_ylabel("Funding Level (IFRS)")
        _fig_label(ax3, "Funding Level (IFRS) Over Time")
        ax3.legend(fontsize=8, loc="lower right")
        _despine(ax3)

        # ── 2d: Year-25 funding level distribution ────────────────────────
        ax4 = fig.add_subplot(gs[1, 1])
        fl_final_gaap = self._fl_gaap[:, -1]
        fl_final_ifrs = self._fl_ifrs[:, -1]
        bins = np.linspace(
            min(fl_final_gaap.min(), fl_final_ifrs.min()),
            max(fl_final_gaap.max(), fl_final_ifrs.max()),
            55
        )
        ax4.hist(fl_final_gaap, bins=bins, color=PALETTE["teal"],   alpha=0.65,
                 edgecolor="none", label="GAAP", density=True)
        ax4.hist(fl_final_ifrs, bins=bins, color=PALETTE["purple"], alpha=0.55,
                 edgecolor="none", label="IFRS", density=True)
        ax4.axvline(1.0, color=PALETTE["red"], linewidth=1.8, linestyle="--",
                    label="100% funded")
        ax4.axvline(fl_final_gaap.mean(), color=PALETTE["teal"],  linewidth=1.2,
                    linestyle=":", alpha=0.9,
                    label=f"Mean GAAP: {fl_final_gaap.mean():.1%}")
        ax4.axvline(fl_final_ifrs.mean(), color=PALETTE["purple"], linewidth=1.2,
                    linestyle=":", alpha=0.9,
                    label=f"Mean IFRS: {fl_final_ifrs.mean():.1%}")
        ax4.xaxis.set_major_formatter(mtick.FuncFormatter(_fmt_pct_int))
        ax4.set_xlabel("Year-25 Funding Level")
        ax4.set_ylabel("Density")
        _fig_label(ax4, "Year-25 Funding Level Distribution")
        # annotate probabilities
        p_under_gaap = (fl_final_gaap < 1.0).mean()
        p_under_ifrs = (fl_final_ifrs < 1.0).mean()
        ax4.text(0.03, 0.96,
                 f"P(underfunded) GAAP: {p_under_gaap:.1%}\nP(underfunded) IFRS: {p_under_ifrs:.1%}",
                 transform=ax4.transAxes, va="top", fontsize=8.5,
                 color=PALETTE["red"],
                 bbox=dict(facecolor=PALETTE["bg"], edgecolor=PALETTE["border"],
                           boxstyle="round,pad=0.4"))
        ax4.legend(fontsize=8)
        _despine(ax4)

        plt.show()
        return fig

    # ── 3. UNDERPERFORMANCE RISK ──────────────────────────────────────────────
    def plot_underperformance_risk(self):
        """
        Figure 3 — Likelihood of trailing the CMBP bund comparator.
        ┌──────────────────┬──────────────────┐
        │ P(underperf)     │ Expected shortfall│
        │ over time        │ over time         │
        ├──────────────────┼──────────────────┤
        │ Year-25 CDI vs   │ Perf guarantee   │
        │ CMBP scatter     │ waterfall         │
        └──────────────────┴──────────────────┘
        """
        fig = plt.figure(figsize=(14, 10))
        _section_title(fig, "3 · Underperformance Risk vs CMBP Comparator", y=0.995)
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.44, wspace=0.30,
                               top=0.95, bottom=0.07, left=0.09, right=0.97)
        yrs = self._years_int

        # ── 3a: P(underperformance) over time ─────────────────────────────
        ax1 = fig.add_subplot(gs[0, 0])
        prob_under = self._underperf.mean(axis=0)  # (T,)
        ax1.fill_between(yrs, 0, prob_under, color=PALETTE["red"], alpha=0.3)
        ax1.plot(yrs, prob_under, color=PALETTE["red"], linewidth=2.2)
        ax1.axhline(0.25, color=PALETTE["amber"], linewidth=1.2,
                    linestyle="--", label="25% threshold")
        ax1.yaxis.set_major_formatter(mtick.FuncFormatter(_fmt_pct_int))
        ax1.set_xlabel("Year")
        ax1.set_ylabel("P(CMBP > CDI Assets)")
        ax1.set_ylim(0, 1)
        _fig_label(ax1, "Probability of Trailing CMBP Comparator")
        ax1.legend(fontsize=8)
        _despine(ax1)

        # ── 3b: Expected shortfall (conditional on underperformance) ──────
        ax2 = fig.add_subplot(gs[0, 1])
        gap = np.maximum(self._bund_comp - self._assets, 0)  # (n_sim, T)
        mean_gap    = gap.mean(axis=0)
        cond_gap    = np.where(self._underperf > 0, gap, np.nan)
        cond_mean   = np.nanmean(cond_gap, axis=0)
        ax2.fill_between(yrs, 0, mean_gap / 1e6, color=PALETTE["amber"], alpha=0.25,
                         label="Expected gap (all scenarios)")
        ax2.plot(yrs, mean_gap / 1e6, color=PALETTE["amber"], linewidth=1.8)
        ax2.plot(yrs, cond_mean / 1e6, color=PALETTE["red"],  linewidth=2.0,
                 linestyle="--", label="Conditional expected shortfall")
        ax2.yaxis.set_major_formatter(mtick.FuncFormatter(_fmt_m))
        ax2.set_xlabel("Year")
        ax2.set_ylabel("Gap (€)")
        _fig_label(ax2, "Expected Shortfall vs CMBP")
        ax2.legend(fontsize=8)
        _despine(ax2)

        # ── 3c: Year-25 CDI Assets vs CMBP Comparator scatter ────────────
        ax3 = fig.add_subplot(gs[1, 0])
        a25   = self._assets[:, -1] / 1e6
        b25   = self._bund_comp[:, -1] / 1e6
        under = b25 > a25
        ax3.scatter(b25[~under], a25[~under], alpha=0.25, s=5,
                    color=PALETTE["teal"],  linewidths=0, label="CDI ≥ CMBP (outperform)")
        ax3.scatter(b25[under],  a25[under],  alpha=0.35, s=5,
                    color=PALETTE["red"],   linewidths=0, label="CDI < CMBP (underperform)")
        lim = max(a25.max(), b25.max()) * 1.02
        lo  = min(a25.min(), b25.min()) * 0.98
        ax3.plot([lo, lim], [lo, lim], color=PALETTE["muted"],
                 linewidth=1.2, linestyle="--", alpha=0.7, label="Break-even")
        ax3.set_xlabel("CMBP Comparator (€m)")
        ax3.set_ylabel("CDI Assets (€m)")
        _fig_label(ax3, "Year-25: CDI Assets vs CMBP Comparator")
        ax3.legend(fontsize=8, markerscale=3)
        ax3.text(0.03, 0.97,
                 f"Underperf. rate: {under.mean():.1%}",
                 transform=ax3.transAxes, va="top", fontsize=9,
                 color=PALETTE["red"], fontweight="bold",
                 bbox=dict(facecolor=PALETTE["bg"], edgecolor=PALETTE["border"],
                           boxstyle="round,pad=0.4"))
        _despine(ax3)

        # ── 3d: CMBP comparator evolution — percentile fans ───────────────
        ax4 = fig.add_subplot(gs[1, 1])
        p5b, p25b, p50b, p75b, p95b, mub = self._pctiles(self._bund_comp)
        p5a, p25a, p50a, p75a, p95a, mua = self._pctiles(self._assets)
        _fan(ax4, yrs, p5b, p25b, p50b, p75b, p95b, color=PALETTE["amber"],
             label_suffix=" — CMBP")
        _fan(ax4, yrs, p5a, p25a, p50a, p75a, p95a, color=PALETTE["blue"],
             label_suffix=" — CDI")
        ax4.yaxis.set_major_formatter(mtick.FuncFormatter(_fmt_m))
        ax4.set_xlabel("Year")
        ax4.set_ylabel("Value (€)")
        _fig_label(ax4, "CDI Assets vs CMBP — Percentile Fans")
        ax4.legend(fontsize=8, ncol=2)
        _despine(ax4)

        plt.show()
        return fig

    # ── 4. CDI CASHFLOW DISTRIBUTION ──────────────────────────────────────────
    def plot_cashflow_distribution(self):
        """
        Figure 4 — Realised vs expected CDI cashflows.
        ┌──────────────────────────────────────┐
        │  Fan chart: realised vs expected CF   │
        ├──────────────────┬───────────────────┤
        │  Y5 CF dist.     │  Y10 CF dist.     │
        ├──────────────────┼───────────────────┤
        │  Y15 CF dist.    │  Y25 CF dist.     │
        └──────────────────┴───────────────────┘
        """
        fig = plt.figure(figsize=(14, 12))
        _section_title(fig, "4 · Realised vs Expected CDI Cashflows", y=0.995)
        gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.46, wspace=0.30,
                               top=0.95, bottom=0.05, left=0.09, right=0.97)
        yrs = self._years_int

        # ── 4a: Fan chart spanning full width ─────────────────────────────
        ax_top = fig.add_subplot(gs[0, :])
        p5r, p25r, p50r, p75r, p95r, mur = self._pctiles(self._asset_cf)
        p50e = self._exp_cdi_cf.mean(axis=0)     # expected is deterministic given ratings
        _fan(ax_top, yrs, p5r, p25r, p50r, p75r, p95r, color=PALETTE["blue"],
             label_suffix=" (realised)")
        ax_top.plot(yrs, p50e, color=PALETTE["amber"], linewidth=2.2,
                    linestyle="--", label="Expected (no default)")
        ax_top.yaxis.set_major_formatter(mtick.FuncFormatter(_fmt_m))
        ax_top.set_xlabel("Year")
        ax_top.set_ylabel("Annual CDI Cashflow (€)")
        _fig_label(ax_top, "CDI Portfolio Cashflows — Realised vs Expected")
        ax_top.legend(fontsize=8, loc="upper right")
        _despine(ax_top)

        # ── 4b–4e: Cross-sections at years 5, 10, 15, 25 ─────────────────
        snap_years = [4, 9, 14, -1]
        snap_labels = ["Year 5", "Year 10", "Year 15", "Year 25"]
        axes = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]),
                fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])]

        for ax, t_idx, lbl in zip(axes, snap_years, snap_labels):
            realised = self._asset_cf[:, t_idx]
            expected = self._exp_cdi_cf[:, t_idx].mean()
            ax.hist(realised / 1e6, bins=50, color=PALETTE["blue"],
                    alpha=0.75, edgecolor="none", density=True)
            ax.axvline(realised.mean() / 1e6, color=PALETTE["teal"],
                       linewidth=1.8, linestyle="-",
                       label=f"Realised mean: {_fmt_m(realised.mean())}")
            ax.axvline(expected / 1e6, color=PALETTE["amber"],
                       linewidth=1.8, linestyle="--",
                       label=f"Expected: {_fmt_m(expected)}")
            ax.xaxis.set_major_formatter(mtick.FuncFormatter(_fmt_m))
            ax.set_xlabel(f"CDI Cashflow {lbl} (€m)")
            ax.set_ylabel("Density")
            _fig_label(ax, f"Cashflow Distribution — {lbl}")
            ax.legend(fontsize=7.5)
            # annotate shortfall probability
            p_short = (realised < expected).mean()
            ax.text(0.97, 0.96, f"P(below expected): {p_short:.1%}",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=8, color=PALETTE["amber"])
            _despine(ax)

        plt.show()
        return fig

    # ── 5. RETURN COMPARISON ──────────────────────────────────────────────────
    def plot_return_comparison(self):
        """
        Figure 5 — Net CDI return vs CMBP (bund) return.
        ┌──────────────────┬──────────────────┐
        │ CDI fan          │ CMBP fan          │
        ├──────────────────┼──────────────────┤
        │ Spread (CDI–CMBP)│ Joint dist Y10    │
        │ fan over time    │ KDE               │
        └──────────────────┴──────────────────┘
        """
        fig = plt.figure(figsize=(14, 10))
        _section_title(fig, "5 · Net CDI Return vs CMBP Return", y=0.995)
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32,
                               top=0.95, bottom=0.07, left=0.09, right=0.97)
        yrs = self._years_int

        # ── 5a: Net CDI return fan ────────────────────────────────────────
        ax1 = fig.add_subplot(gs[0, 0])
        p5, p25, p50, p75, p95, mu = self._pctiles(self._cdi_ret)
        _fan(ax1, yrs, p5, p25, p50, p75, p95, mean_=mu, color=PALETTE["blue"])
        ax1.axhline(0, color=PALETTE["muted"], linewidth=1.0, linestyle="--", alpha=0.6)
        ax1.yaxis.set_major_formatter(mtick.FuncFormatter(_fmt_pct))
        ax1.set_xlabel("Year")
        ax1.set_ylabel("Net CDI Return")
        _fig_label(ax1, "Net CDI Return (after fees)")
        ax1.legend(fontsize=8)
        _despine(ax1)

        # ── 5b: CMBP (bt) return fan ──────────────────────────────────────
        ax2 = fig.add_subplot(gs[0, 1])
        p5, p25, p50, p75, p95, mu = self._pctiles(self._bt_ret)
        _fan(ax2, yrs, p5, p25, p50, p75, p95, mean_=mu, color=PALETTE["amber"])
        ax2.axhline(0, color=PALETTE["muted"], linewidth=1.0, linestyle="--", alpha=0.6)
        ax2.yaxis.set_major_formatter(mtick.FuncFormatter(_fmt_pct))
        ax2.set_xlabel("Year")
        ax2.set_ylabel("CMBP Net Return")
        _fig_label(ax2, "CMBP Net Return (bt + margin)")
        ax2.legend(fontsize=8)
        _despine(ax2)

        # ── 5c: Return spread (CDI − CMBP) fan ───────────────────────────
        ax3 = fig.add_subplot(gs[1, 0])
        spread = self._cdi_ret - self._bt_ret
        p5, p25, p50, p75, p95, mu = self._pctiles(spread)
        _fan(ax3, yrs, p5, p25, p50, p75, p95, mean_=mu, color=PALETTE["teal"])
        ax3.axhline(0, color=PALETTE["red"], linewidth=1.4, linestyle="--",
                    alpha=0.8, label="Break-even")
        ax3.yaxis.set_major_formatter(mtick.FuncFormatter(_fmt_pct))
        ax3.set_xlabel("Year")
        ax3.set_ylabel("CDI − CMBP (bps-equivalent)")
        _fig_label(ax3, "Return Spread: Net CDI minus Net CMBP")
        ax3.legend(fontsize=8)
        _despine(ax3)

        # ── 5d: Joint return distribution at year 10 (KDE) ────────────────
        ax4 = fig.add_subplot(gs[1, 1])
        t10  = min(9, self._n_years - 1)
        cdi10 = self._cdi_ret[:, t10]
        bt10  = self._bt_ret[:, t10]

        # 2-D KDE via seaborn
        sns.kdeplot(
            x=bt10, y=cdi10,
            fill=True, levels=8,
            cmap="YlOrRd_r", alpha=0.7,
            ax=ax4
        )
        lim_lo = min(bt10.min(), cdi10.min())
        lim_hi = max(bt10.max(), cdi10.max())
        ax4.plot([lim_lo, lim_hi], [lim_lo, lim_hi], color=PALETTE["muted"],
                 linewidth=1.2, linestyle="--", label="CDI = CMBP")
        ax4.scatter(bt10.mean(), cdi10.mean(), color=PALETTE["text"],
                    s=60, zorder=5, label=f"Mean (CDI={cdi10.mean():.2%}, CMBP={bt10.mean():.2%})")
        ax4.xaxis.set_major_formatter(mtick.FuncFormatter(_fmt_pct))
        ax4.yaxis.set_major_formatter(mtick.FuncFormatter(_fmt_pct))
        ax4.set_xlabel("CMBP Net Return")
        ax4.set_ylabel("Net CDI Return")
        _fig_label(ax4, "Joint Return Distribution — Year 10")
        ax4.legend(fontsize=7.5)
        _despine(ax4)

        plt.show()
        return fig

    # ── 6. SUMMARY TABLES ─────────────────────────────────────────────────────
    def summary_tables(self) -> dict:
        """
        Print and return a dict of summary DataFrames:
          - obligation_pv      : PV statistics for each obligation + fee
          - asset_snapshot     : asset / funding / underperf at key years
          - return_snapshot    : CDI vs CMBP return at key years
          - cashflow_snapshot  : realised vs expected CF at key years
        """
        pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
        pd.set_option("display.max_columns", 20)
        pd.set_option("display.width", 120)

        # ── Table 1: Obligation PV summary ────────────────────────────────
        def _stats_row(arr, label):
            return {
                "Component":  label,
                "Mean (€m)":  arr.mean() / 1e6,
                "P5 (€m)":    np.percentile(arr, 5) / 1e6,
                "P25 (€m)":   np.percentile(arr, 25) / 1e6,
                "Median (€m)":np.percentile(arr, 50) / 1e6,
                "P75 (€m)":   np.percentile(arr, 75) / 1e6,
                "P95 (€m)":   np.percentile(arr, 95) / 1e6,
                "P(>0)":      f"{(arr > 0).mean():.1%}",
            }
        obl_rows = [
            _stats_row(self._pv_hgb,   "HGB Gap Payments (PV)"),
            _stats_row(self._pv_add,   "Year-10 Additional Payment (PV)"),
            _stats_row(self._pv_perf,  "Perf. Guarantee — Year 25 (PV)"),
            _stats_row(self._pv_total, "Total Obligations (PV)"),
            _stats_row(self._pv_fee,   "Fee Income (PV)"),
        ]
        t1 = pd.DataFrame(obl_rows).set_index("Component")
        # add coverage ratio row
        cover = self._pv_fee / np.maximum(self._pv_total, 1)
        t1.loc["Coverage Ratio (Fee / Obligations)"] = {
            "Mean (€m)":   f"{cover.mean():.1%}",
            "P5 (€m)":     f"{np.percentile(cover, 5):.1%}",
            "P25 (€m)":    f"{np.percentile(cover, 25):.1%}",
            "Median (€m)": f"{np.percentile(cover, 50):.1%}",
            "P75 (€m)":    f"{np.percentile(cover, 75):.1%}",
            "P95 (€m)":    f"{np.percentile(cover, 95):.1%}",
            "P(>0)":       "—",
        }
        print("\n" + "═"*80)
        print("TABLE 1: Obligation PV Summary  (discounted at {:.1%})".format(self.r))
        print("═"*80)
        print(t1.to_string())

        # ── Table 2: Asset & funding snapshot ─────────────────────────────
        snap_tidx = {y: min(y - 1, self._n_years - 1) for y in [5, 10, 15, 20, 25]}
        snap_rows = []
        for yr, t in snap_tidx.items():
            a = self._assets[:, t]
            f = self._fl_gaap[:, t]
            u = self._underperf[:, t]
            snap_rows.append({
                "Year":                 self._years_int[t],
                "Mean Assets (€m)":     a.mean() / 1e6,
                "P5 Assets (€m)":       np.percentile(a, 5) / 1e6,
                "P95 Assets (€m)":      np.percentile(a, 95) / 1e6,
                "Mean FL (GAAP)":       f"{f.mean():.1%}",
                "P5 FL (GAAP)":         f"{np.percentile(f, 5):.1%}",
                "P(underfunded GAAP)":  f"{(f < 1).mean():.1%}",
                "P(CMBP > CDI)":        f"{u.mean():.1%}",
            })
        t2 = pd.DataFrame(snap_rows).set_index("Year")
        print("\n" + "═"*80)
        print("TABLE 2: Asset & Funding Level Snapshot")
        print("═"*80)
        print(t2.to_string())

        # ── Table 3: Return snapshot ───────────────────────────────────────
        ret_rows = []
        for yr, t in snap_tidx.items():
            c = self._cdi_ret[:, t]
            b = self._bt_ret[:, t]
            sp = c - b
            ret_rows.append({
                "Year":             self._years_int[t],
                "CDI Mean":         f"{c.mean():.2%}",
                "CDI P5":           f"{np.percentile(c, 5):.2%}",
                "CDI P95":          f"{np.percentile(c, 95):.2%}",
                "CMBP Mean":        f"{b.mean():.2%}",
                "CMBP P5":          f"{np.percentile(b, 5):.2%}",
                "CMBP P95":         f"{np.percentile(b, 95):.2%}",
                "Mean Spread":      f"{sp.mean():.2%}",
                "P(CDI > CMBP)":    f"{(sp > 0).mean():.1%}",
            })
        t3 = pd.DataFrame(ret_rows).set_index("Year")
        print("\n" + "═"*80)
        print("TABLE 3: Return Statistics Snapshot")
        print("═"*80)
        print(t3.to_string())

        # ── Table 4: Cashflow snapshot ─────────────────────────────────────
        cf_rows = []
        for yr, t in snap_tidx.items():
            r = self._asset_cf[:, t]
            e = self._exp_cdi_cf[:, t].mean()
            cf_rows.append({
                "Year":             self._years_int[t],
                "Realised Mean (€m)": r.mean() / 1e6,
                "Realised P5 (€m)":   np.percentile(r, 5) / 1e6,
                "Realised P95 (€m)":  np.percentile(r, 95) / 1e6,
                "Expected (€m)":      e / 1e6,
                "P(below expected)":  f"{(r < e).mean():.1%}",
                "Mean shortfall (€m)": np.maximum(e - r, 0).mean() / 1e6,
            })
        t4 = pd.DataFrame(cf_rows).set_index("Year")
        print("\n" + "═"*80)
        print("TABLE 4: Cashflow Snapshot — Realised vs Expected")
        print("═"*80)
        print(t4.to_string())

        return dict(obligation_pv=t1, asset_snapshot=t2,
                    return_snapshot=t3, cashflow_snapshot=t4)

    # ── Convenience: run everything ───────────────────────────────────────────
    def plot_all(self):
        """Render all five figures and print all tables."""
        self.plot_obligation_coverage()
        self.plot_assets_and_funding()
        self.plot_underperformance_risk()
        self.plot_cashflow_distribution()
        self.plot_return_comparison()
        return self.summary_tables()
