# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "altair==6.3.0",
#     "marimo",
#     "pandas",
# ]
# ///

import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import io

    return io, mo, pd


@app.cell
def _(mo):
    mo.md(r"""
    # From Raw Eurostat Data to a Clean Dataframe

    ## What is Eurostat?

    [Eurostat](https://ec.europa.eu/eurostat) is the statistical office of the
    European Union. It collects, harmonises, and publishes comparable statistics
    for all EU member states; covering topics from GDP and inflation to health,
    education, and the environment. Because each country uses the same definitions
    and measurement methods, Eurostat data allows meaningful cross-country
    comparisons that are difficult to make from national statistics alone.

    You can browse available datasets at the
    [Eurostat data browser](https://ec.europa.eu/eurostat/databrowser).
    Every dataset has a short code visible in the browser (for example
    [`hlth_cd_apr`](https://ec.europa.eu/eurostat/databrowser/view/hlth_cd_apr/default/table?lang=en)). This code is all you need to download the data in Python.

    ## Accessing Eurostat data via an API

    An **API** (Application Programming Interface) is a standardised way for
    programs to request data from an external server. Instead of clicking through
    a website to download a file, you send a request from your Python script and
    receive the data directly.

    The `eurostat` Python package wraps Eurostat's REST API into a single
    function call:

    ```python
    import eurostat
    df = eurostat.get_data_df('hlth_cd_apr')
    ```

    This fetches the full dataset with code `hlth_cd_apr` and returns it as a
    pandas DataFrame. Using the API rather than manual downloads has three
    advantages: the code is **reproducible** (anyone can rerun it), it
    **scales** to many datasets, and it automatically picks up **revisions**
    when Eurostat updates its numbers.

    ## Treatable and preventable mortality

    The dataset `hlth_cd_apr` distinguishes two types of premature death:

    **Preventable mortality** (`PRVT`) counts deaths that could have been avoided
    through effective public-health interventions *before* a disease develops —
    for example:

    - deaths from lung cancer caused by smoking
    - deaths from alcohol-related liver disease
    - deaths from road traffic accidents
    - deaths from obesity-related conditions

    These deaths reflect "upstream" factors like lifestyle, environment, and public
    health policy.

    **Treatable mortality** (`TRT`) counts deaths that could have been avoided
    if patients had received timely and effective medical treatment *after*
    a disease was diagnosed — for example:

    - deaths from appendicitis
    - deaths from breast cancer detected at an early stage
    - deaths from bacterial infections treatable with antibiotics
    - deaths from a heart attack where rapid intervention saves lives

    These deaths reflect the quality and accessibility of the healthcare system.

    The values in the dataset are **rates per 100 000 population** (unit code
    `RT`), standardised for age, so they are comparable across countries and
    years without being distorted by differences in population age structure.

    ## Reshaping the data

    Eurostat delivers data in **wide format**: one row per combination of
    categorical dimensions and one **column per year**. Before we can analyse
    or visualise the data, we need to reshape it into **long format**.

    To illustrate, the raw data we get from Eurostat is of the form:


    | freq | mortalit | sex | icd10 | unit | geo\TIME_PERIOD | 2011 | 2012 | 2013 | 2014 | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 |
    |------|----------|-----|-------|------|-----------------|------|------|------|------|------|------|------|------|------|------|
    | A | PRVT | F | A00-A09 | NR | AT | 7.0 | 12.0 | 9.0 | 6.0 | 5.0 | 15.0 | 10.0 | 8.0 | 6.0 | 13.0 |
    | A | PRVT | F | A00-A09 | NR | BE | 25.0 | 11.0 | 20.0 | 26.0 | 26.0 | 27.0 | 45.0 | 26.0 | 44.0 | 39.0 |
    | A | PRVT | F | A00-A09 | NR | BG | 5.0 | 4.0 | 10.0 | 11.0 | 13.0 | 7.0 | 4.0 | 3.0 | 4.0 | 5.0 |
    | A | PRVT | F | A00-A09 | NR | CH | 14.0 | 14.0 | 9.0 | 9.0 | 5.0 | 13.0 | 13.0 | 11.0 | 14.0 | 10.0 |
    | A | PRVT | F | A00-A09 | NR | CY | 6.0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 2.0 | 2.0 | 3.0 | 3.0 |

    And we want to get it into the following format:

    | country | year | sex | Preventable mortality | Treatable mortality |
    |---------|------|-----|-----------------------|---------------------|
    | Austria | 2011 | F | 95.29 | 67.41 |
    | Austria | 2011 | M | 248.50 | 96.86 |
    | Austria | 2012 | F | 96.16 | 69.72 |
    | Austria | 2012 | M | 252.28 | 91.45 |
    | Austria | 2013 | F | 93.22 | 66.84 |


    This app walks you through the four steps used for the mortality dataset, using a shortened version of this data.
    The column `mortalit` in the raw data contains the codes `PRVT`
    (preventable) and `TRT` (treatable) explained above.

    | Step | Operation | Shape change |
    |---|---|---|
    | 1 | Raw data from Eurostat | 16 rows — one per (country, sex, mortalit) |
    | 2 | `pd.melt` | 64 rows — one per (country, sex, mortalit, year) |
    | 3 | `set_index` + `unstack` | 32 rows — PRVT and TRT become separate columns |
    | 4 | `.rename` | 32 rows — column names become readable |

    Use the dropdown below to step through each transformation.
    """)
    return


@app.cell
def _(io, pd):
    # Small realistic slice of hlth_cd_apr (4 countries, 4 years, after basic
    # filtering to mortalit ∈ {PRVT, TRT}, unit == RT, icd10 == TOTAL)
    _raw_csv = """\
    mortalit,sex,country,2014,2015,2016,2017
    PRVT,F,Austria,93.95,97.52,96.39,92.94
    PRVT,M,Austria,249.58,258.21,249.85,245.72
    TRT,F,Austria,68.40,70.15,69.83,68.01
    TRT,M,Austria,90.55,89.53,86.77,84.90
    PRVT,F,Belgium,95.40,99.71,94.13,93.42
    PRVT,M,Belgium,249.58,258.87,247.96,244.83
    TRT,F,Belgium,73.24,74.55,72.13,71.05
    TRT,M,Belgium,102.78,103.12,98.45,97.33
    PRVT,F,Bulgaria,129.04,128.32,120.14,121.87
    PRVT,M,Bulgaria,385.92,387.21,362.45,368.23
    TRT,F,Bulgaria,118.41,120.95,115.62,113.45
    TRT,M,Bulgaria,271.34,285.44,273.28,265.98
    PRVT,F,Germany,91.27,93.15,89.94,87.62
    PRVT,M,Germany,214.83,219.42,211.35,206.74
    TRT,F,Germany,65.11,67.28,64.93,63.45
    TRT,M,Germany,87.46,89.23,85.61,83.72
    """
    df1 = pd.read_csv(io.StringIO(_raw_csv))

    # ── Step 2: melt ──────────────────────────────────────────────────────────
    df2 = pd.melt(
        df1,
        id_vars=["country", "sex", "mortalit"],
        value_vars=["2014", "2015", "2016", "2017"],
        var_name="year",
        value_name="rate",
    )
    df2["year"] = df2["year"].astype(int)

    # ── Step 3: set_index → unstack → reset_index → flatten column names ──────
    _tmp = df2.set_index(["country", "year", "sex", "mortalit"])
    _tmp = _tmp.unstack()
    _tmp = _tmp.reset_index()
    df3  = _tmp.copy()
    df3.columns = [" ".join(col).strip() for col in df3.columns.values]

    # ── Step 4: rename ────────────────────────────────────────────────────────
    df4 = df3.rename(
        {"rate PRVT": "Preventable mortality", "rate TRT": "Treatable mortality"},
        axis=1,
    )
    return df1, df2, df3, df4


@app.cell
def _(mo):
    step_dd = mo.ui.dropdown(
        options=[
            "1 — Wide format (raw data from Eurostat)",
            "2 — Long format (after melt)",
            "3 — Unstacked (PRVT and TRT as separate columns)",
            "4 — Final clean dataframe (after rename)",
        ],
        value="1 — Wide format (raw data from Eurostat)",
        label="Transformation step",
    )
    step_dd
    return (step_dd,)


@app.cell
def _(df1, df2, df3, df4, mo, step_dd):
    _dfs = {
        "1 — Wide format (raw data from Eurostat)": df1,
        "2 — Long format (after melt)": df2,
        "3 — Unstacked (PRVT and TRT as separate columns)": df3,
        "4 — Final clean dataframe (after rename)": df4,
    }

    _codes = {
        "1 — Wide format (raw data from Eurostat)":
            "# Data as returned by eurostat.get_data_df('hlth_cd_apr')\n"
            "# after filtering to EU countries, unit == RT, icd10 == TOTAL.\n"
            "# Each year is a separate column — this is wide format.",

        "2 — Long format (after melt)":
            "df = pd.melt(\n"
            "    df,\n"
            "    id_vars=[\"country\", \"sex\", \"mortalit\"],\n"
            "    value_vars=[\"2014\", \"2015\", \"2016\", \"2017\"],\n"
            "    var_name=\"year\",\n"
            "    value_name=\"rate\",\n"
            ")\n"
            "df[\"year\"] = df[\"year\"].astype(int)",

        "3 — Unstacked (PRVT and TRT as separate columns)":
            "df.set_index([\"country\", \"year\", \"sex\", \"mortalit\"], inplace=True)\n"
            "df = df.unstack()        # pivots last index level (mortalit) into columns\n"
            "df.reset_index(inplace=True)\n"
            "# Flatten MultiIndex columns: ('rate', 'PRVT') → 'rate PRVT'\n"
            "df.columns = [\" \".join(col).strip() for col in df.columns.values]",

        "4 — Final clean dataframe (after rename)":
            "df.rename(\n"
            "    {\"rate PRVT\": \"Preventable mortality\",\n"
            "     \"rate TRT\":  \"Treatable mortality\"},\n"
            "    inplace=True, axis=1,\n"
            ")",
    }

    _explanations = {
        "1 — Wide format (raw data from Eurostat)":
            "**Wide format** — years are columns.\n\n"
            "Each of the 16 rows represents one combination of `mortalit` "
            "(`PRVT` = preventable, `TRT` = treatable), `sex`, and `country`. "
            "The values for 2014–2017 are rates per 100 000 population, spread across "
            "four separate year columns.\n\n"
            "This format is natural for a spreadsheet but awkward in Python: "
            "you cannot filter by year, use year as a plot axis, "
            "or group observations by year without extra work.",

        "2 — Long format (after melt)":
            "**Long format** — years are rows.\n\n"
            "`pd.melt` turns the four year columns into two new columns: "
            "`year` (which year) and `rate` (the value). "
            "The row count goes from 16 to 64 (16 rows × 4 years).\n\n"
            "Now `year` is a proper variable. You can write "
            "`df[df.year == 2016]` to filter, or put `year` on a plot axis.",

        "3 — Unstacked (PRVT and TRT as separate columns)":
            "**After unstack** — PRVT and TRT become two separate columns.\n\n"
            "`mortalit` had two values: `PRVT` (preventable) and `TRT` (treatable). "
            "These are really two *different variables*, not two values of the same variable. "
            "`unstack()` pivots the last index level (`mortalit`) into columns.\n\n"
            "The row count halves: from 64 to 32 "
            "(4 countries × 4 years × 2 sexes). "
            "The column names are still the raw joined form `rate PRVT` and `rate TRT`.",

        "4 — Final clean dataframe (after rename)":
            "**Final shape** — human-readable column names.\n\n"
            "`.rename` replaces `rate PRVT` and `rate TRT` with descriptive labels. "
            "The dataframe now has one row per (country, year, sex) "
            "with preventable and treatable mortality as separate, clearly named columns. "
            "This is the shape we want to use for further analysis.",
    }

    _key  = step_dd.value
    _df   = _dfs[_key]

    _shape_note = f"{_df.shape[0]} rows × {_df.shape[1]} columns"

    mo.vstack([
        mo.callout(mo.md(_explanations[_key]), kind="info"),
        mo.md(f"```python\n{_codes[_key]}\n```"),
        mo.md(f"**Shape:** {_shape_note}"),
        mo.ui.table(_df, selection=None),
    ])
    return


if __name__ == "__main__":
    app.run()
