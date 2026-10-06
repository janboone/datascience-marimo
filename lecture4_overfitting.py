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
    # Overfitting and Underfitting: Polynomial Regression Example

    We generate 30 observations from the **true model**

    $$y = 1 + 2x^2 + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0,\, 1.5)$$

    The first 20 observations form the **train set** (blue); the last 10 form the **test set** (orange).
    We fit a polynomial of degree $d$ to the *training data only*, then evaluate on both sets using mean squared error (MSE).

    Use the slider to vary $d$:
    - **Train MSE** decreases monotonically as $d$ grows — adding parameters can only improve the fit
      on the data used for estimation.
    - **Test MSE** first falls (the model captures the true quadratic shape), then **rises**: the polynomial
      starts fitting the noise in the training sample instead of the underlying signal.
    """)
    return


@app.cell
def _(mo):
    degree_slider = mo.ui.slider(1, 9, value=2, step=1, label="Polynomial degree (d)")
    degree_slider
    return (degree_slider,)


@app.cell
def _(np):
    _rng = np.random.default_rng(42)
    _N, _n_train = 30, 20
    x_all   = np.sort(_rng.uniform(-2.0, 2.0, _N))
    y_all   = 1.0 + 2.0 * x_all ** 2 + _rng.normal(0, 1.5, _N)
    x_train = x_all[:_n_train]
    y_train = y_all[:_n_train]
    x_test  = x_all[_n_train:]
    y_test  = y_all[_n_train:]
    x_curve = np.linspace(-2.2, 2.2, 400)
    return x_all, x_curve, x_test, x_train, y_all, y_test, y_train


@app.cell
def _(degree_slider, np, x_curve, x_test, x_train, y_test, y_train):
    d = degree_slider.value
    coeffs = np.polyfit(x_train, y_train, d)
    poly   = np.poly1d(coeffs)
    y_fit  = poly(x_curve)
    train_mse = float(np.mean((y_train - poly(x_train)) ** 2))
    test_mse  = float(np.mean((y_test  - poly(x_test))  ** 2))
    return d, test_mse, train_mse, y_fit


@app.cell
def _(d, plt, x_all, x_curve, y_all, y_fit):
    fig_all, ax_all = plt.subplots(figsize=(9, 4))
    ax_all.scatter(x_all[:20], y_all[:20], color="steelblue", s=55, alpha=0.85,
                   label="Train data (n=20)")
    ax_all.scatter(x_all[20:], y_all[20:], color="darkorange", s=55, alpha=0.85,
                   label="Test data (n=10)")
    ax_all.plot(x_curve, y_fit, color="crimson", lw=2.2, label=f"Degree-{d} fit")
    ax_all.set_xlim(-2.3, 2.3)
    ax_all.set_ylim(-2, 16)
    ax_all.set_xlabel("x")
    ax_all.set_ylabel("y")
    ax_all.set_title(f"All 30 observations with degree-{d} polynomial fitted on training data only")
    ax_all.legend(fontsize=9)
    plt.tight_layout()
    fig_all
    return


@app.cell
def _(
    d,
    plt,
    test_mse,
    train_mse,
    x_curve,
    x_test,
    x_train,
    y_fit,
    y_test,
    y_train,
):
    fig_tt, (ax_tr, ax_te) = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    ax_tr.scatter(x_train, y_train, color="steelblue", s=60, alpha=0.85)
    ax_tr.plot(x_curve, y_fit, color="crimson", lw=2.2)
    ax_tr.set_xlim(-2.3, 2.3)
    ax_tr.set_ylim(-2, 16)
    ax_tr.set_xlabel("x")
    ax_tr.set_ylabel("y")
    ax_tr.set_title(f"Train set — MSE = {train_mse:.2f}")

    ax_te.scatter(x_test, y_test, color="darkorange", s=60, alpha=0.85)
    ax_te.plot(x_curve, y_fit, color="crimson", lw=2.2)
    ax_te.set_xlim(-2.3, 2.3)
    ax_te.set_ylim(-2, 16)
    ax_te.set_xlabel("x")
    ax_te.set_title(f"Test set — MSE = {test_mse:.2f}")

    plt.suptitle(
        f"Degree {d}: as degree increases, train MSE ↓  but  test MSE ↑",
        fontsize=11, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    fig_tt
    return


@app.cell
def _(d, mo, test_mse, train_mse):
    if d <= 1:
        kind = "info"
        body = f"""
            **Underfitting** (degree {d}) — Train MSE = {train_mse:.2f},  Test MSE = {test_mse:.2f}

            A degree-1 line cannot capture the true quadratic signal.  Both errors are high.
            Increase the degree toward 2 to see both errors fall.
            """
    elif test_mse > 6.0:
        kind = "warn"
        body = f"""
            **Overfitting** (degree {d}) — Train MSE = {train_mse:.2f},  Test MSE = {test_mse:.2f}

            The polynomial fits the training noise closely but predicts the test set poorly.
            The model has memorised the 20 training points rather than the underlying $y = 1 + 2x^2$ pattern.
            """
    else:
        kind = "success"
        body = f"""
            **Reasonable fit** (degree {d}) — Train MSE = {train_mse:.2f},  Test MSE = {test_mse:.2f}

            The polynomial captures the true quadratic shape without chasing too much noise.
            The irreducible noise variance is $\\sigma^2 = 2.25$, so the best achievable test MSE
            is around 2.25.
            """
    mo.callout(mo.md(body), kind=kind)
    return


if __name__ == "__main__":
    app.run()
