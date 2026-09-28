# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy",
#     "matplotlib",
# ]
# ///

import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    return mo, np, plt


@app.cell
def _(mo):
    mo.md(r"""
    # Significance and Variable Selection

    A common mistake is to select variables for a regression based on whether they are
    "significant". Consider the following thought experiment:

    - Generate $k$ variables $X_1, \ldots, X_k$, all **independent** of the outcome $Y$.
    - Run **one multiple regression** of $Y$ on all $k$ variables simultaneously.
    - For each coefficient $\hat{\beta}_i$, record the t-statistic testing $H_0: \beta_i = 0$.
    - Report every $X_i$ where $|t| > t_\alpha$ as an "important predictor of $Y$".

    There is no true effect of any $X_i$ on $Y$. Nevertheless, a fraction $\alpha$ of the $k$
    individual t-tests will reject the null hypothesis just by chance — that is exactly what a
    significance level of $\alpha$ means. With $k = 100$ variables and $\alpha = 5\%$, you
    expect 5 false discoveries purely from noise.
    """)
    return


@app.cell
def _(mo):
    k_slider = mo.ui.slider(
        5, 200, value=20, step=5, label="Number of candidate variables ($k$)"
    )
    n_slider = mo.ui.slider(
        250, 1000, value=300, step=50, label="Sample size ($n$)"
    )
    alpha_slider = mo.ui.slider(
        0.01, 0.10, value=0.05, step=0.01, label="Significance level (α)"
    )
    mo.vstack([k_slider, n_slider, alpha_slider])
    return alpha_slider, k_slider, n_slider


@app.cell
def _(alpha_slider, k_slider, n_slider, np):
    k = k_slider.value
    n = n_slider.value
    alpha = alpha_slider.value

    # Multiple regression requires n > k + 1 degrees of freedom
    k_eff = min(k, n - 2)

    rng = np.random.default_rng()
    Y = rng.normal(0, 1, n)
    X = rng.normal(0, 1, (n, k_eff))

    # One multiple regression: Y ~ X_1 + ... + X_{k_eff}
    Xmat = np.column_stack([np.ones(n), X])             # (n, k_eff+1)
    beta = np.linalg.solve(Xmat.T @ Xmat, Xmat.T @ Y)  # (k_eff+1,)
    resid = Y - Xmat @ beta
    sigma2 = np.sum(resid ** 2) / (n - k_eff - 1)
    cov_beta = sigma2 * np.linalg.inv(Xmat.T @ Xmat)
    se = np.sqrt(np.diag(cov_beta))   # (k_eff+1,)
    t_stats = beta[1:] / se[1:]       # (k_eff,) — slopes only, exclude intercept

    # Critical value (normal approximation — accurate for n > 30)
    crit_lookup = {
        0.01: 2.576, 0.02: 2.326, 0.03: 2.170, 0.04: 2.054,
        0.05: 1.960, 0.06: 1.881, 0.07: 1.812, 0.08: 1.750,
        0.09: 1.695, 0.10: 1.645,
    }
    crit_val = crit_lookup.get(round(alpha, 2), 1.960)

    n_significant = int(np.sum(np.abs(t_stats) > crit_val))
    expected_significant = alpha * k_eff
    return (
        alpha,
        crit_val,
        expected_significant,
        k,
        k_eff,
        n,
        n_significant,
        t_stats,
    )


@app.cell
def _(
    alpha,
    crit_val,
    expected_significant,
    k_eff,
    n_significant,
    np,
    plt,
    t_stats,
):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: histogram of t-statistics
    t_lo = min(-4.5, t_stats.min() - 0.3)
    t_hi = max(4.5, t_stats.max() + 0.3)
    bins = np.linspace(t_lo, t_hi, 40)
    axes[0].hist(t_stats, bins=bins, color="steelblue", alpha=0.75, edgecolor="white")
    axes[0].axvline(crit_val, color="red", lw=2, linestyle="--",
                    label=f"±{crit_val:.3f}  (α = {alpha:.2f})")
    axes[0].axvline(-crit_val, color="red", lw=2, linestyle="--")
    axes[0].set_xlabel("t-statistic")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"t-statistics from one multiple regression\n({k_eff} predictors, none with a true effect!)")
    axes[0].legend(fontsize=9)

    # Right: expected vs actual significant
    labels = [f"Expected\n(α × k = {expected_significant:.1f})", f"Actual\n(observed = {n_significant})"]
    heights = [expected_significant, float(n_significant)]
    bar_colors = ["#2c7a7b", "#e05c5c" if n_significant > expected_significant else "#2c7a7b"]
    bars = axes[1].bar(labels, heights, color=bar_colors, alpha=0.85, edgecolor="white")
    for bar, h in zip(bars, heights):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.15,
            f"{h:.1f}",
            ha="center",
            fontsize=13,
            fontweight="bold",
        )
    axes[1].set_ylabel("Number of significant results")
    axes[1].set_title(f"False discoveries at α = {alpha:.2f}\n({k_eff} coefficients tested, all pure noise!)")
    axes[1].set_ylim(0, max(heights) + 2.5)

    plt.tight_layout()
    fig
    return


@app.cell
def _(alpha, expected_significant, k, k_eff, mo, n, n_significant):
    capped_note = (
        f"  *(k capped from {k} to {k_eff} because multiple regression requires n > k + 1)*"
        if k_eff < k else ""
    )
    kind = "warn" if n_significant > 0 else "success"
    mo.callout(
        mo.md(
            f"""
            **Results with k = {k_eff} predictors and n = {n} observations:**{capped_note}

            - Significance level: α = {alpha:.0%}
            - Expected false discoveries: α × k = **{expected_significant:.1f}**
            - Actual significant results: **{n_significant}** — none has a true effect on Y!

            If you selected these "significant" predictors from the multiple regression and reported
            them as important, you would be reporting noise. The expected false-discovery rate
            equals α by construction.

            Try increasing k to 100 or 200 to see how many false discoveries accumulate.
            """
        ),
        kind=kind,
    )
    return


if __name__ == "__main__":
    app.run()
