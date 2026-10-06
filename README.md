# Is Economic Growth Decoupling from CO₂ Emissions?

**Author:** Szymon Stembalski (i6389327)

This is the data-analysis project for **MAT2007 Introduction to Programming** at Maastricht University.

## Research question

**How many countries achieved absolute decoupling between 2005 and 2022, meaning their GDP grew while their CO₂ emissions fell? How does this depend on a country's income level? Does the result still hold when emissions are counted by consumption rather than production?**

## Dataset

- **Source:** [Our World in Data, CO₂ and Greenhouse Gas Emissions dataset](https://github.com/owid/co2-data)
- **File used:** `owid-co2-data.csv` (about 14 MB, 79 columns, one row per country per year)
- **Version:** pinned to commit [`382ee6c`](https://github.com/owid/co2-data/tree/382ee6c662b0ece26e111f263b44c029afad7787) so the results are reproducible
- **Columns used:** `country`, `iso_code`, `year`, `population`, `gdp`, `co2`, `consumption_co2`

You do **not** need to download the dataset yourself. On its first run, `analysis.py` downloads it into `data/`. The script also does all the cleaning, so no manual preparation is needed. It:
1. removes regional aggregates such as "World" and "Europe" (rows with no ISO code or with an `OWID_` code);
2. keeps only countries with complete GDP, CO₂ and population data for every year from 2005 to 2022;
3. drops countries with fewer than 1 million inhabitants.

That leaves 153 countries. Consumption-based CO₂ data exist for 116 of them.

## What the script does

1. For each country, it fits a log-linear trend `ln X = a + b·year` to GDP, territorial CO₂ and consumption-based CO₂. From each fit it gets the average annual growth rate and its uncertainty.
2. It classifies each country as one of the following:
   - **absolute decoupling:** GDP grows and CO₂ falls at 95% confidence;
   - **relative decoupling:** CO₂ grows, but more slowly than GDP;
   - **no decoupling:** CO₂ grows at least as fast as GDP;
   - **GDP decline:** the economy shrank.
3. It compares the fraction of decoupling countries across four income groups (quartiles of GDP per capita in 2005), with binomial uncertainties.
4. It prints the quantitative results and writes plots and tables to `figures/`.

## Main result

| Emissions measure | Absolute decoupling, all countries | Low income | High income |
|---|---|---|---|
| Territorial | 22.2 ± 3.4 % (34/153) | 0/39 | 60.5 ± 7.9 % (23/38) |
| Consumption-based | 19.8 ± 3.7 % (23/116) | 0/22 | 48.6 ± 8.2 % (18/37) |

Absolute decoupling is real but mostly limited to rich countries. The gap between the high- and low-income groups is significant at 7.6σ. Counting emissions by consumption reduces the effect only slightly, so offshoring of production does not explain most of it. The full discussion is in [`report/report.pdf`](report/report.pdf).

## Repository structure

```
analysis.py               # the analysis code
requirements.txt          # Python dependencies
figures/                  # output plots and tables (created by analysis.py)
  growth_scatter.png
  decoupling_by_income.png
  country_trends.csv      # growth rates and classification for every country
  decoupling_by_income.csv
report/
  report.tex              # 2-page LaTeX report
  report.pdf              # compiled report
data/                     # dataset (downloaded automatically, not committed)
```

## How to run

### Requirements
- Python 3.9 or newer (tested with Python 3.13)
- The packages in `requirements.txt`: pandas, numpy, scipy, matplotlib
- An internet connection on the first run, to download the dataset

### Installation
```bash
git clone <this-repository-url>
cd <repository-folder>
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Execution
```bash
python3 analysis.py
```
To run it in **VS Code**, open the folder, select the `.venv` interpreter (Ctrl+Shift+P → *Python: Select Interpreter*), open `analysis.py` and press ▶ *Run Python File*.

The script prints a summary to the terminal and saves the figures and CSV tables in `figures/`. It takes a few seconds once the dataset has been downloaded.

If you are offline, download the CSV manually from the pinned link above and save it as `data/owid-co2-data.csv`.

### Building the report
You can upload `report/report.tex` and the two PNG files to Overleaf, or compile it locally:
```bash
cd report
pdflatex report.tex
```
The report looks for the figures at `../figures/`. On Overleaf, upload them into a `figures/` folder and change the paths to `figures/...`.
