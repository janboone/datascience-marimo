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
    # Approximating Functions with ReLU

    The **ReLU** (Rectified Linear Unit) activation function is:

    $$\text{relu}(z) = \max(0,\, z)$$

    It is zero for $z \le 0$ and grows linearly for $z > 0$ and therefore has a "kink" at the origin.
    Simple as it looks, ReLU is the most widely used activation in modern deep learning.

    ## Why ReLU is so useful for function approximation

    A single *shifted* ReLU,

    $$\phi_j(x) = \text{relu}(x - t_j) = \max(0,\, x - t_j),$$

    contributes nothing to the left of knot $t_j$ and adds a slope of 1 to the right.
    A *scaled* and shifted ReLU $\beta_j \phi_j(x)$ can also decrease (if $\beta_j < 0$).

    By stacking $k$ such terms with different knot positions $t_j$ and learned amplitudes $\beta_j$,

    $$\hat{f}(x) = \beta_0 + \sum_{j=1}^{k} \beta_j\,\text{relu}(x - t_j),$$

    we can change the slope at each knot and approximate any continuous function to arbitrary
    precision — this is a consequence of the [universal approximation theorem](https://en.wikipedia.org/wiki/Universal_approximation_theorem).

    In a trained neural network the knot positions are also learned from data. Here we fix them evenly and learn only the amplitudes $\beta_j$ which is enough to demonstrate the core idea.

    We approximate the target $f(x) = x^3 + x^2 - x - 1$ on $[-2,\; 1.5]$.
    """)
    return


@app.cell
def _(mo):
    k_slider = mo.ui.slider(1, 20, value=3, step=1,
                             label="Number of ReLU basis functions (k)")
    k_slider
    return (k_slider,)


@app.cell
def _(k_slider, np):
    _x_min, _x_max = -2.0, 1.5
    x_plot = np.linspace(_x_min, _x_max, 500)
    y_true = x_plot**3 + x_plot**2 - x_plot - 1

    k = k_slider.value
    # k interior knots, evenly spaced (excluding endpoints)
    knots = np.linspace(_x_min, _x_max, k + 2)[1:-1]

    # Design matrix: [1, relu(x-t_1), ..., relu(x-t_k)]
    Phi = np.column_stack(
        [np.ones(len(x_plot))] + [np.maximum(0.0, x_plot - t) for t in knots]
    )

    # Ordinary least squares: fit amplitudes beta
    beta_ols, _, _, _ = np.linalg.lstsq(Phi, y_true, rcond=None)
    y_hat    = Phi @ beta_ols
    approx_mse = float(np.mean((y_true - y_hat) ** 2))

    # Individual scaled contributions: beta_j * relu(x - t_j)
    contributions = [beta_ols[0] * np.ones(len(x_plot))]
    for _j, _t in enumerate(knots):
        contributions.append(beta_ols[_j + 1] * np.maximum(0.0, x_plot - _t))
    return approx_mse, contributions, k, knots, x_plot, y_hat, y_true


@app.cell
def _(approx_mse, contributions, k, knots, np, plt, x_plot, y_hat, y_true):
    _colors = plt.cm.tab10(np.linspace(0, 0.9, k + 1))

    _fig, (_ax1, _ax2) = plt.subplots(
        2, 1, figsize=(10, 10),
        gridspec_kw={"height_ratios": [2.5, 2.3]}
    )

    # ── Top: target + approximation ───────────────────────────────────────────
    _ax1.plot(x_plot, y_true, color="black", lw=2.5,
              label=r"Target: $f(x) = x^3 + x^2 - x - 1$")
    _ax1.plot(x_plot, y_hat, color="steelblue", lw=2.2, ls="--",
              label=f"ReLU approximation  (k = {k},  MSE = {approx_mse:.4f})")
    for _t in knots:
        _ax1.axvline(_t, color="grey", lw=0.8, ls=":", alpha=0.55)
    _ax1.axhline(0, color="grey", lw=0.4)
    _ax1.set_xlim(-2.0, 1.5)
    _ax1.set_ylabel("$f(x)$")
    _ax1.legend(fontsize=9)
    _ax1.set_title(
        f"Approximating $f(x)$ with {k} ReLU basis function{'s' if k > 1 else ''}  "
        f"(knot positions shown as dotted verticals)"
    )

    # ── Bottom: individual scaled contributions ───────────────────────────────
    for _j, _c in enumerate(contributions):
        #if _j == 0:
        #    _lbl = f"Intercept  $\\beta_0 = {beta_ols[0]:.2f}$"
        #else:
        #    _lbl = (f"$\\beta_{{{_j}}}\\,\\mathrm{{relu}}(x - {knots[_j-1]:.2f})$"
        #            f",  $\\beta_{{{_j}}} = {beta_ols[_j]:.2f}$")
        _ax2.plot(x_plot, _c, color=_colors[_j], lw=1.6, alpha=0.85)#, label=_lbl)
    _ax2.axhline(0, color="grey", lw=0.4)
    _ax2.set_xlim(-2.0, 1.5)
    _ax2.set_xlabel("$x$")
    _ax2.set_ylabel("Component")
    _ax2.set_title("Individual scaled ReLU contributions  (sum of all gives the approximation above)")
    #_ax2.legend(fontsize=7, ncol=min(k + 1, 5), loc="upper left")

    plt.tight_layout()
    _fig
    return


@app.cell
def _(approx_mse, k, mo):
    if approx_mse > 0.5:
        _kind = "info"
        _msg = f"""
            **Coarse approximation** — {k} ReLU function{'s' if k > 1 else ''},  MSE = {approx_mse:.4f}

            With only {k} basis function{'s' if k > 1 else ''} the piecewise-linear model can
            capture only the broadest shape of $f$.  Increase $k$ to add more kinks and reduce
            the approximation error.
            """
    elif approx_mse < 0.02:
        _kind = "success"
        _msg = f"""
            **Excellent approximation** — {k} ReLU functions,  MSE = {approx_mse:.4f}

            The piecewise-linear model with {k} basis functions tracks the cubic target closely.
            In a trained neural network the knot positions are also learned from data — making
            the same number of ReLUs even more powerful.
            """
    else:
        _kind = "neutral"
        _msg = f"""
            **Moderate approximation** — {k} ReLU functions,  MSE = {approx_mse:.4f}

            The model captures the main shape of $f$ but smooths over some local curvature.
            Add more ReLU units to reduce the MSE further.  The bottom panel shows how the
            individual contributions combine to produce the blue curve above.
            """
    mo.callout(mo.md(_msg), kind=_kind)
    return


if __name__ == "__main__":
    app.run()
