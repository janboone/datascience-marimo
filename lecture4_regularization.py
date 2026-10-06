# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy",
#     "matplotlib",
#     "scipy==1.18.1",
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
    # Ridge and Lasso Regularization

    We use the same data as before: $y = 1 + 2x^2 + \varepsilon$, 20 training and 10 test observations.
    We fit a **degree-9 polynomial** — 10 parameters on 20 training points — which badly overfits without
    regularization.

    Two methods shrink coefficients toward zero to prevent overfitting:

    - **Ridge** adds penalty $\lambda \sum_{j=1}^{9} \beta_j^2$.  In this case there is a closed-form solution:
      $\hat\beta_{\text{ridge}} = (X^\top X/n + \lambda \Lambda)^{-1} X^\top y/n$,
      where $\Lambda = \operatorname{diag}(0,1,\ldots,1)$ leaves the intercept unpenalised.
      Ridge shrinks all coefficients but never to exactly zero.
    - **Lasso** adds penalty $\lambda \sum_{j=1}^{9} |\beta_j|$, solved numerically. Lasso sets some coefficients *exactly* to zero: automatic variable selection.

    Use the slider to increase $\lambda$:
    - Train MSE rises (more shrinkage = more bias).
    - Test MSE first **falls** (overfitting reduced), then **rises** (underfitting).
    """)
    return


@app.cell
def _(mo):
    lam_slider = mo.ui.slider(0.0, 3.0, value=0.0, step=0.05,
                               label="Regularisation strength (λ)")
    lam_slider
    return (lam_slider,)


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

    # Doing our own np.vander: columns [1, x, x^2, ..., x^9]
    _deg = 9
    def _vander(xv):
        return np.column_stack([xv ** j for j in range(_deg + 1)])

    Xtr_raw = _vander(x_train)
    Xte_raw = _vander(x_test)
    Xcu_raw = _vander(x_curve)

    # For lasso/ridge standardise columns 1..9 using training mean/std (intercept column untouched)
    feat_mean = Xtr_raw[:, 1:].mean(axis=0)
    feat_std  = Xtr_raw[:, 1:].std(axis=0) + 1e-8

    def _std(Xraw):
        Xs = Xraw.copy()
        Xs[:, 1:] = (Xs[:, 1:] - feat_mean) / feat_std
        return Xs

    X_tr = _std(Xtr_raw)
    X_te = _std(Xte_raw)
    X_cu = _std(Xcu_raw)
    return X_cu, X_te, X_tr, x_curve, x_test, x_train, y_test, y_train


@app.cell
def _(X_cu, X_te, X_tr, lam_slider, np, y_test, y_train):
    lam = lam_slider.value
    p   = X_tr.shape[1]          # 10 (intercept + 9 polynomial terms)
    n_tr = X_tr.shape[0]         # 20

    # Ridge: closed-form with 1/n normalisation
    Lambda_diag = np.array([0.0] + [1.0] * (p - 1))
    A_ridge = X_tr.T @ X_tr / n_tr + lam * np.diag(Lambda_diag)
    b_rhs   = X_tr.T @ y_train  / n_tr
    beta_ridge = np.linalg.solve(A_ridge, b_rhs)

    # Lasso version that makes lambda comparable to Ridge
    def _lasso_cd(X, y, lam_l, max_iter=3000, tol=1e-8):
        nn, pp = X.shape
        beta = np.zeros(pp)
        col_norms = np.sum(X ** 2, axis=0) / nn
        for _ in range(max_iter):
            beta_old = beta.copy()
            for j in range(pp):
                r_j = y - X @ beta + X[:, j] * beta[j]
                z_j = X[:, j] @ r_j / nn
                if j == 0 or lam_l == 0.0:          # intercept: no penalty
                    beta[j] = z_j / col_norms[j]
                else:
                    sign_z = 1.0 if z_j > 0 else -1.0
                    beta[j] = sign_z * max(abs(z_j) - lam_l, 0.0) / col_norms[j]
            if np.max(np.abs(beta - beta_old)) < tol:
                break
        return beta

    beta_lasso = _lasso_cd(X_tr, y_train, lam)

    y_ridge_curve = X_cu @ beta_ridge
    y_lasso_curve = X_cu @ beta_lasso

    train_mse_ridge = float(np.mean((y_train - X_tr @ beta_ridge) ** 2))
    test_mse_ridge  = float(np.mean((y_test  - X_te @ beta_ridge) ** 2))
    train_mse_lasso = float(np.mean((y_train - X_tr @ beta_lasso) ** 2))
    test_mse_lasso  = float(np.mean((y_test  - X_te @ beta_lasso) ** 2))
    n_zeros_lasso   = int(np.sum(np.abs(beta_lasso[1:]) < 1e-6))
    return (
        lam,
        n_zeros_lasso,
        test_mse_lasso,
        test_mse_ridge,
        train_mse_lasso,
        train_mse_ridge,
        y_lasso_curve,
        y_ridge_curve,
    )


@app.cell
def _(
    lam,
    plt,
    test_mse_lasso,
    test_mse_ridge,
    train_mse_lasso,
    train_mse_ridge,
    x_curve,
    x_test,
    x_train,
    y_lasso_curve,
    y_ridge_curve,
    y_test,
    y_train,
):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharey=True)
    _xlim = (-2.3, 2.3)
    _ylim = (-3, 18)

    # Define separately because we need to use each twice
    _scatter_kw_tr = dict(color="steelblue",   s=55, alpha=0.85)
    _scatter_kw_te = dict(color="darkorange",  s=55, alpha=0.85)

    # Row 0: Ridge
    axes[0, 0].scatter(x_train, y_train, **_scatter_kw_tr)
    axes[0, 0].plot(x_curve, y_ridge_curve, color="royalblue", lw=2.2)
    axes[0, 0].set_title(f"Ridge — Train MSE = {train_mse_ridge:.2f}")

    axes[0, 1].scatter(x_test, y_test, **_scatter_kw_te)
    axes[0, 1].plot(x_curve, y_ridge_curve, color="royalblue", lw=2.2)
    axes[0, 1].set_title(f"Ridge — Test MSE = {test_mse_ridge:.2f}")

    # Row 1: Lasso
    axes[1, 0].scatter(x_train, y_train, **_scatter_kw_tr)
    axes[1, 0].plot(x_curve, y_lasso_curve, color="forestgreen", lw=2.2)
    axes[1, 0].set_title(f"Lasso — Train MSE = {train_mse_lasso:.2f}")

    axes[1, 1].scatter(x_test, y_test, **_scatter_kw_te)
    axes[1, 1].plot(x_curve, y_lasso_curve, color="forestgreen", lw=2.2)
    axes[1, 1].set_title(f"Lasso — Test MSE = {test_mse_lasso:.2f}")

    for ax in axes.flat:
        ax.set_xlim(*_xlim)
        ax.set_ylim(*_ylim)
        ax.set_xlabel("x")
    axes[0, 0].set_ylabel("y")
    axes[1, 0].set_ylabel("y")

    plt.suptitle(
        f"Degree-9 polynomial with λ = {lam:.2f}  "
        f"(train fit worsens ↑λ, test fit first improves then worsens)",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    fig
    return


@app.cell
def _(
    lam,
    mo,
    n_zeros_lasso,
    test_mse_lasso,
    test_mse_ridge,
    train_mse_lasso,
    train_mse_ridge,
):
    if lam < 0.01:
        kind = "warn"
        body = f"""
            **λ = {lam:.2f} — no regularisation (OLS)**

            Ridge: Train MSE = {train_mse_ridge:.2f},  Test MSE = {test_mse_ridge:.2f}
            Lasso: Train MSE = {train_mse_lasso:.2f},  Test MSE = {test_mse_lasso:.2f}

            The degree-9 polynomial with no regularisation **overfits**: it captures noise in the
            20 training points and predicts the 10 test points poorly.
            Increase λ to see regularisation improve the test-set performance.
            """
    elif min(test_mse_ridge, test_mse_lasso) > 7.0:
        kind = "info"
        body = f"""
            **λ = {lam:.2f} — heavy regularisation (underfitting)**

            Ridge: Train MSE = {train_mse_ridge:.2f},  Test MSE = {test_mse_ridge:.2f}
            Lasso: Train MSE = {train_mse_lasso:.2f},  Test MSE = {test_mse_lasso:.2f}
            Lasso zeroed {n_zeros_lasso} of 9 slope coefficients.

            The penalty is now so strong that the model underfits: it has too little flexibility
            to capture even the true quadratic signal.  Try reducing λ.
            """
    else:
        kind = "success"
        body = f"""
            **λ = {lam:.2f} — moderate regularisation**

            | Method | Train MSE | Test MSE |
            |--------|-----------|----------|
            | Ridge  | {train_mse_ridge:.2f} | {test_mse_ridge:.2f} |
            | Lasso  | {train_mse_lasso:.2f} | {test_mse_lasso:.2f} |

            Lasso has set **{n_zeros_lasso} of 9** slope coefficients exactly to zero.
            Ridge keeps all coefficients non-zero but small.

            The irreducible noise variance is $\\sigma^2 = 2.25$; test MSE near this value indicates
            a well-calibrated model.
            """
    mo.callout(mo.md(body), kind=kind)
    return


if __name__ == "__main__":
    app.run()
