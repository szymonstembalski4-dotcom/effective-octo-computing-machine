"""
Is economic growth decoupling from CO2 emissions?

Analysis of the Our World in Data (OWID) CO2 dataset for MAT2007.

For every country we fit log-linear trends to GDP and to CO2 emissions over
2005-2022 to get average annual growth rates (with standard errors). Each
country is then classified as
  - absolute decoupling: GDP grows, CO2 falls (significantly, at 95% CL)
  - relative decoupling: both grow, but CO2 grows slower than GDP
  - no decoupling:       CO2 grows at least as fast as GDP
  - GDP decline:         the economy shrank, so decoupling is not defined
The classification is done for territorial (production-based) emissions and
for consumption-based emissions (which account for emissions embedded in
imported goods), and is compared across income groups.

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

# Pinned to a fixed commit of the OWID repository so results are reproducible.
DATA_COMMIT = "382ee6c662b0ece26e111f263b44c029afad7787"
DATA_URL = (
    "https://raw.githubusercontent.com/owid/co2-data/"
    f"{DATA_COMMIT}/owid-co2-data.csv"
)
DATA_FILE = os.path.join("data", "owid-co2-data.csv")
OUTPUT_DIR = "figures"

START_YEAR = 2005
END_YEAR = 2022  # last year with GDP data in the dataset
MIN_POPULATION = 1_000_000  # skip micro-states, whose data are noisy
Z95 = 1.96  # two-sided 95% confidence level


# --- Data loading and cleaning -----------------------------------------------

def load_data():
    """Download the dataset (if not present) and return it as a DataFrame."""
    if not os.path.exists(DATA_FILE):
        os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
        print(f"Downloading dataset from {DATA_URL} ...")
        urllib.request.urlretrieve(DATA_URL, DATA_FILE)
    return pd.read_csv(DATA_FILE)


def clean_data(df):
    """Keep real countries in the study period with complete GDP/CO2 data."""
    # Aggregates such as "World" or "Europe" have no ISO code (or start with
    # "OWID_"); we only want individual countries.
    df = df[df["iso_code"].notna() & ~df["iso_code"].str.startswith("OWID")]
    df = df[(df["year"] >= START_YEAR) & (df["year"] <= END_YEAR)]
    df = df[["country", "iso_code", "year", "population", "gdp",
             "co2", "consumption_co2"]].copy()

    n_years = END_YEAR - START_YEAR + 1
    complete = df.dropna(subset=["gdp", "co2", "population"])
    counts = complete.groupby("country")["year"].count()
    keep = counts[counts == n_years].index
    df = df[df["country"].isin(keep)]

    pop_end = df[df["year"] == END_YEAR].set_index("country")["population"]
    df = df[df["country"].isin(pop_end[pop_end >= MIN_POPULATION].index)]
    return df


# --- Analysis ----------------------------------------------------------------

def growth_rate(years, values):
    """
    Fit log(values) = a + b * year and return the average annual growth rate
    (in %) and its standard error. Returns (nan, nan) if data are missing.
    """
    mask = np.isfinite(values) & (values > 0)
    if mask.sum() != len(values):
        return np.nan, np.nan
    fit = stats.linregress(years, np.log(values))
    rate = 100 * (np.exp(fit.slope) - 1)
    err = 100 * np.exp(fit.slope) * fit.stderr  # error propagation
    return rate, err


def classify(gdp_rate, co2_rate, co2_err):
    """Classify a country's decoupling status."""
    if np.isnan(co2_rate):
        return np.nan
    if gdp_rate <= 0:
        return "GDP decline"
    if co2_rate + Z95 * co2_err < 0:
        return "absolute"
    if co2_rate < gdp_rate:
        return "relative"
    return "none"


def per_country_trends(df):
    """Compute growth rates and decoupling class for each country."""
    rows = []
    for country, g in df.groupby("country"):
        g = g.sort_values("year")
        years = g["year"].to_numpy()
        gdp_pc = (g["gdp"] / g["population"]).to_numpy()

        gdp_rate, gdp_err = growth_rate(years, g["gdp"].to_numpy())
        co2_rate, co2_err = growth_rate(years, g["co2"].to_numpy())
        cons_rate, cons_err = growth_rate(years, g["consumption_co2"].to_numpy())

        rows.append({
            "country": country,
            "gdp_per_capita_start": gdp_pc[0],
            "gdp_rate": gdp_rate, "gdp_err": gdp_err,
            "co2_rate": co2_rate, "co2_err": co2_err,
            "cons_rate": cons_rate, "cons_err": cons_err,
            "class_territorial": classify(gdp_rate, co2_rate, co2_err),
            "class_consumption": classify(gdp_rate, cons_rate, cons_err),
        })
    res = pd.DataFrame(rows)

    # Income groups: quartiles of GDP per capita in the starting year.
    res["income_group"] = pd.qcut(
        res["gdp_per_capita_start"], 4,
        labels=["Low", "Lower-middle", "Upper-middle", "High"],
    )
    return res


def fraction_with_error(k, n):
    """Fraction k/n with its binomial standard error."""
    p = k / n
    return p, np.sqrt(p * (1 - p) / n)


def summarise(res):
    """Print the main quantitative results and return them as a table."""
    print(f"\nCountries analysed: {len(res)} "
          f"(period {START_YEAR}-{END_YEAR}, population >= {MIN_POPULATION:,})")

    summary = []
    for col, label in [("class_territorial", "territorial"),
                       ("class_consumption", "consumption-based")]:
        sub = res.dropna(subset=[col])
        n = len(sub)
        print(f"\nDecoupling using {label} CO2 (N = {n}):")
        for cls in ["absolute", "relative", "none", "GDP decline"]:
            k = (sub[col] == cls).sum()
            p, e = fraction_with_error(k, n)
            print(f"  {cls:11s}: {k:3d}  ({100 * p:5.1f} +/- {100 * e:4.1f} %)")

        for grp, gsub in sub.groupby("income_group", observed=True):
            k = (gsub[col] == "absolute").sum()
            p, e = fraction_with_error(k, len(gsub))
            summary.append({"emissions": label, "income_group": grp,
                            "n": len(gsub), "n_absolute": k,
                            "frac": p, "frac_err": e})

    table = pd.DataFrame(summary)
    print("\nFraction of countries with absolute decoupling, by income group:")
    print(table.to_string(index=False, float_format="%.3f"))

    # Is the CO2 growth rate correlated with initial income?
    rho, pval = stats.spearmanr(np.log(res["gdp_per_capita_start"]),
                                res["co2_rate"])
    print(f"\nSpearman correlation, log(GDP per capita {START_YEAR}) vs "
          f"CO2 growth rate: rho = {rho:.2f}, p = {pval:.1e}")

    # Difference between high- and low-income groups (territorial).
    terr = table[table["emissions"] == "territorial"].set_index("income_group")
    diff = terr.loc["High", "frac"] - terr.loc["Low", "frac"]
    diff_err = np.hypot(terr.loc["High", "frac_err"],
                        terr.loc["Low", "frac_err"])
    print(f"High minus low income (territorial): {diff:.2f} +/- {diff_err:.2f} "
          f"({diff / diff_err:.1f} sigma)")

    res.sort_values("co2_rate").to_csv(
        os.path.join(OUTPUT_DIR, "country_trends.csv"), index=False)
    table.to_csv(os.path.join(OUTPUT_DIR, "decoupling_by_income.csv"),
                 index=False)
    return table


# --- Plots -------------------------------------------------------------------

COLORS = {"Low": "#d95f02", "Lower-middle": "#e6ab02",
          "Upper-middle": "#66a61e", "High": "#1b6ca8"}


def plot_scatter(res):
    """GDP growth vs CO2 growth for every country, with decoupling regions."""
    fig, ax = plt.subplots(figsize=(7, 5))
    lim_x = (min(-2, res["gdp_rate"].min() - 0.5), res["gdp_rate"].max() + 0.5)
    lim_y = (res["co2_rate"].min() - 0.5, res["co2_rate"].max() + 0.5)

    x = np.linspace(0, lim_x[1], 2)
    ax.fill_between(x, lim_y[0], 0, color="#1b9e77", alpha=0.12,
                    label="Absolute decoupling region")
    ax.fill_between(x, 0, x, color="#7570b3", alpha=0.12,
                    label="Relative decoupling region")
    ax.plot([lim_x[0], lim_x[1]], [lim_x[0], lim_x[1]], "k--", lw=0.8,
            label="CO$_2$ growth = GDP growth")
    ax.axhline(0, color="grey", lw=0.6)
    ax.axvline(0, color="grey", lw=0.6)

    for grp, sub in res.groupby("income_group", observed=True):
        ax.errorbar(sub["gdp_rate"], sub["co2_rate"], yerr=sub["co2_err"],
                    fmt="o", ms=4, lw=0.6, alpha=0.8, color=COLORS[grp],
                    label=f"{grp} income")

    # Label a few well-known countries (offsets chosen to avoid overlaps).
    labels = {"United States": (8, 10), "China": (6, -10), "India": (6, 4),
              "Germany": (-55, -12), "United Kingdom": (6, -8),
              "Brazil": (6, 2)}
    for name, offset in labels.items():
        row = res[res["country"] == name]
        if not row.empty:
            ax.annotate(name, (row["gdp_rate"].iloc[0], row["co2_rate"].iloc[0]),
                        fontsize=7, xytext=offset, textcoords="offset points",
                        arrowprops=dict(arrowstyle="-", lw=0.4, color="grey"))

    ax.set_xlim(lim_x)
    ax.set_ylim(lim_y)
    ax.set_xlabel(f"Average annual GDP growth {START_YEAR}-{END_YEAR} (%)")
    ax.set_ylabel(f"Average annual CO$_2$ growth {START_YEAR}-{END_YEAR} (%)")
    ax.set_title("Economic growth vs territorial CO$_2$ emission growth")
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "growth_scatter.png"), dpi=200)
    plt.close(fig)


def plot_fractions(table):
    """Fraction of countries with absolute decoupling per income group."""
    fig, ax = plt.subplots(figsize=(7, 3.8))
    groups = ["Low", "Lower-middle", "Upper-middle", "High"]
    width = 0.38
    pos = np.arange(len(groups))
    for i, (label, color) in enumerate([("territorial", "#1b6ca8"),
                                        ("consumption-based", "#a6cee3")]):
        sub = table[table["emissions"] == label].set_index("income_group")
        sub = sub.loc[groups]
        x = pos + (i - 0.5) * width
        ax.bar(x, 100 * sub["frac"], width,
               yerr=100 * sub["frac_err"], capsize=3, color=color,
               label=f"{label.capitalize()} CO$_2$")
        # Show the raw counts k/n above each bar.
        for xi, f, e, k, n in zip(x, sub["frac"], sub["frac_err"],
                                  sub["n_absolute"], sub["n"]):
            ax.text(xi, 100 * (f + e) + 1.5, f"{k}/{n}", ha="center",
                    fontsize=7)
    ax.set_xticks(pos)
    ax.set_xticklabels([f"{g}\nincome" for g in groups])
    ax.set_ylabel("Countries with absolute\ndecoupling (%)")
    ax.set_ylim(0, 80)
    ax.set_title(f"Absolute decoupling by income group ({START_YEAR}-{END_YEAR})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "decoupling_by_income.png"), dpi=200)
    plt.close(fig)


# --- Main --------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = clean_data(load_data())
    res = per_country_trends(df)
    table = summarise(res)
    plot_scatter(res)
    plot_fractions(table)
    print(f"\nFigures and tables written to '{OUTPUT_DIR}/'.")


if __name__ == "__main__":
    main()
