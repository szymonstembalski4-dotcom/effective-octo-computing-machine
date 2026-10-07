"""
Does money buy a longer life? The Preston curve.

Analysis of Gapminder data for MAT2007.

For every country we take life expectancy at birth and GDP per capita in
2019 (the last year before COVID-19) and fit

    life_expectancy = a + b * log2(GDP per capita)

so that the slope b is the number of years of life expectancy that go
together with a doubling of income. To check whether the gain flattens out
at high income, the same fit is repeated separately for the poorer and the
richer half of the countries.

Usage:
    python3 analysis.py
"""

import os
import urllib.request

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# --- Configuration -----------------------------------------------------------

# Pinned to a fixed commit of the Gapminder repository so results are
# reproducible.
DATA_COMMIT = "85af9e1b82e5f5de64b2ad52c5fe2bd1c47b074b"
BASE_URL = (
    "https://raw.githubusercontent.com/open-numbers/ddf--gapminder--fasttrack/"
    f"{DATA_COMMIT}/"
)
FILES = {
    "lex": "countries_etc_datapoints/ddf--datapoints--lex--by--country--time.csv",
    "gdp_pcap": "countries_etc_datapoints/ddf--datapoints--gdp_pcap--by--country--time.csv",
    "pop": "countries_etc_datapoints/ddf--datapoints--pop--by--country--time.csv",
    "countries": "ddf--entities--geo--country.csv",
}
DATA_DIR = "data"
OUTPUT_DIR = "figures"

YEAR = 2019  # last year before the COVID-19 pandemic
MIN_POPULATION = 1_000_000  # skip micro-states


# --- Data loading and cleaning -----------------------------------------------

def load_data():
    """Download the data files (if not present) and return them."""
    os.makedirs(DATA_DIR, exist_ok=True)
    tables = {}
    for name, path in FILES.items():
        local = os.path.join(DATA_DIR, os.path.basename(path))
        if not os.path.exists(local):
            print(f"Downloading {name} ...")
            urllib.request.urlretrieve(BASE_URL + path, local)
        tables[name] = pd.read_csv(local)
    return tables


def clean_data(tables):
    """Build one row per country for YEAR with all variables present."""
    df = None
    for name in ["lex", "gdp_pcap", "pop"]:
        t = tables[name]
        t = t[t["time"] == YEAR][["country", name]]
        df = t if df is None else df.merge(t, on="country")

    countries = tables["countries"][["country", "name", "world_4region"]]
    df = df.merge(countries, on="country")

    df = df.dropna(subset=["lex", "gdp_pcap", "pop"])
    df = df[df["pop"] >= MIN_POPULATION].copy()
    df["log2_gdp"] = np.log2(df["gdp_pcap"])
    return df


# --- Analysis ----------------------------------------------------------------

def fit(df):
    """Linear fit of life expectancy against log2(GDP per capita)."""
    return stats.linregress(df["log2_gdp"], df["lex"])


def analyse(df):
    """Print the main quantitative results and return the fits."""
    print(f"\nCountries analysed: {len(df)} (year {YEAR}, "
          f"population >= {MIN_POPULATION:,})")
    print(f"Life expectancy: {df['lex'].min():.1f} to {df['lex'].max():.1f} years")
    print(f"GDP per capita:  ${df['gdp_pcap'].min():,.0f} to "
          f"${df['gdp_pcap'].max():,.0f}")

    full = fit(df)
    print("\nAll countries: life expectancy = a + b * log2(GDP per capita)")
    print(f"  b = {full.slope:.2f} +/- {full.stderr:.2f} years per doubling of income")
    print(f"  a = {full.intercept:.1f} +/- {full.intercept_stderr:.1f} years")
    print(f"  r = {full.rvalue:.2f}, R^2 = {full.rvalue ** 2:.2f}, "
          f"p = {full.pvalue:.1e}")

    # Does the gain flatten out? Fit the poorer and richer halves separately.
    median = df["gdp_pcap"].median()
    poor = fit(df[df["gdp_pcap"] < median])
    rich = fit(df[df["gdp_pcap"] >= median])
    diff = poor.slope - rich.slope
    diff_err = np.hypot(poor.stderr, rich.stderr)
    print(f"\nSplit at the median GDP per capita (${median:,.0f}):")
    print(f"  poorer half: b = {poor.slope:.2f} +/- {poor.stderr:.2f} years per doubling")
    print(f"  richer half: b = {rich.slope:.2f} +/- {rich.stderr:.2f} years per doubling")
    print(f"  difference:  {diff:.2f} +/- {diff_err:.2f} ({diff / diff_err:.1f} sigma)")

    # Which countries live much longer or shorter than their income predicts?
    df["predicted"] = full.intercept + full.slope * df["log2_gdp"]
    df["residual"] = df["lex"] - df["predicted"]
    cols = ["name", "gdp_pcap", "lex", "residual"]
    print("\nLonger life than income predicts:")
    print(df.nlargest(5, "residual")[cols].to_string(index=False, float_format="%.1f"))
    print("\nShorter life than income predicts:")
    print(df.nsmallest(5, "residual")[cols].to_string(index=False, float_format="%.1f"))
    print(f"\nScatter around the fit: {df['residual'].std(ddof=2):.1f} years (1 sigma)")

    df.sort_values("residual").to_csv(
        os.path.join(OUTPUT_DIR, "country_data.csv"), index=False)
    return full


# --- Plot --------------------------------------------------------------------

REGIONS = {"africa": ("Africa", "#1b9e77"), "americas": ("Americas", "#d95f02"),
           "asia": ("Asia", "#7570b3"), "europe": ("Europe", "#e7298a")}


def plot_preston(df, full):
    """Scatter of countries (log income axis) with the fitted straight line."""
    fig, ax = plt.subplots(figsize=(8, 4.6))

    for key, (label, color) in REGIONS.items():
        sub = df[df["world_4region"] == key]
        ax.scatter(sub["gdp_pcap"] / 1000, sub["lex"], s=25, color=color,
                   alpha=0.8, label=label)

    x = np.logspace(np.log10(df["gdp_pcap"].min()),
                    np.log10(df["gdp_pcap"].max()), 200)
    ax.plot(x / 1000, full.intercept + full.slope * np.log2(x),
            color="black", lw=2,
            label=f"Fit: {full.slope:.2f} $\\pm$ {full.stderr:.2f} years per doubling")

    for name in ["Nigeria", "India", "China", "USA", "Japan", "South Africa"]:
        row = df[df["name"] == name]
        if not row.empty:
            ax.annotate(name, (row["gdp_pcap"].iloc[0] / 1000, row["lex"].iloc[0]),
                        fontsize=8, xytext=(5, -10), textcoords="offset points")

    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100])
    ax.set_xticklabels(["1", "2", "5", "10", "20", "50", "100"])
    ax.set_xlabel(f"GDP per capita in {YEAR} (thousand dollars, log scale)")
    ax.set_ylabel("Life expectancy at birth (years)")
    ax.set_title(f"Life expectancy vs income: {len(df)} countries in {YEAR}")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "preston_curve.png"), dpi=200)
    plt.close(fig)


# --- Main --------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = clean_data(load_data())
    full = analyse(df)
    plot_preston(df, full)
    print(f"\nFigure and table written to '{OUTPUT_DIR}/'.")


if __name__ == "__main__":
    main()
