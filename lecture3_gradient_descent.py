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
    # Gradient Descent for Linear Regression

    We observe data $(x_i, y_i)$ and fit the line $\hat y = m x + b$ with slope $m$ and intercept $b$.
    Instead of computing the OLS solution analytically using matrix algebra, we minimise the loss function which equals the
    mean squared error:

    $$
    L(m,\, b) = \frac{1}{n}\sum_{i=1}^{n}(y_i - m x_i - b)^2
    $$

    by **gradient descent**. We start at some initial guess for the parameters, say $(m_0, b_0) = (0, 0)$, and at each step
    move a small distance in the direction of steepest descent,

    $$
    m_{t+1} = m_t - \eta\,\frac{\partial L}{\partial m}(m_t,b_t),
    \qquad
    b_{t+1} = b_t - \eta\,\frac{\partial L}{\partial b}(m_t,b_t)
    $$

    where $\eta$ is the **learning rate**. Note that when $\partial L/\partial m = 2.0$ (loss function is increasing at $m_t,b_t$) we update $m_{t+1}$ by moving $2 \eta$ to the left from $m_t$. This reduces the value for $L$ and we move more to the left the higher $\eta$ and/or the derivative is.



    The OLS formula gives the exact
    minimum of ${L}$; gradient descent should converge to it if
    $\eta$ is not too large.  When $\eta$ is too large the updates overshoot
    and the algorithm diverges leading to astronomical values for the loss function and "distorted" figures below.

    The left panel shows the loss surface $L(m, b)$ as a contour
    plot (showing iso-loss lines with darker hues indicating lower values of $L$), with the gradient-descent path overlaid. The right panel shows the
    loss at each iteration.
    """)
    return


@app.cell
def _(mo):
    lr_slider = mo.ui.slider(
        0.01, 1.8, value=0.3, step=0.01,
        label="Learning rate (η)",
    )
    steps_slider = mo.ui.slider(
        10, 500, value=100, step=10,
        label="Number of gradient descent steps",
    )
    mo.vstack([lr_slider, steps_slider])
    return lr_slider, steps_slider


@app.cell
def _(np):
    # Fixed dataset — seeded so OLS values are reproducible
    _rng = np.random.default_rng(42)
    _n = 60
    x_data = np.sort(_rng.uniform(0, 2, _n))
    y_data = 0.5 * x_data + 0.3 + _rng.normal(0, 0.2, _n)

    # OLS reference: Y = b + m*x  →  [b_ols, m_ols]
    _Xmat = np.column_stack([np.ones(_n), x_data])
    _beta = np.linalg.solve(_Xmat.T @ _Xmat, _Xmat.T @ y_data)
    b_ols   = float(_beta[0])
    m_ols   = float(_beta[1])
    loss_ols = float(np.mean((y_data - m_ols * x_data - b_ols) ** 2))
    n_data   = _n
    return b_ols, loss_ols, m_ols, x_data, y_data


@app.cell
def _(lr_slider, np, steps_slider, x_data, y_data):
    lr      = lr_slider.value
    n_steps = steps_slider.value

    m_path   = [0.0]
    b_path   = [0.0]
    loss_path = []
    diverged  = False

    for _ in range(n_steps):
        m_t     = m_path[-1]
        b_t     = b_path[-1]
        y_pred  = m_t * x_data + b_t
        resid   = y_pred - y_data
        loss_val = float(np.mean(resid ** 2))
        dm = 2.0 * float(np.mean(resid * x_data))
        db = 2.0 * float(np.mean(resid))
        m_new = m_t - lr * dm
        b_new = b_t - lr * db
        if ((np.abs(m_new) > 1000) and (np.abs(b_new) > 1000) and (loss_val > 1000)):
            diverged = True
            break
        loss_path.append(loss_val)
        m_path.append(m_new)
        b_path.append(b_new)

    m_final = m_path[-1]
    b_final = b_path[-1]
    return b_final, b_path, diverged, loss_path, lr, m_final, m_path, n_steps


@app.cell
def _(
    b_final,
    b_ols,
    b_path,
    diverged,
    loss_ols,
    loss_path,
    lr,
    m_final,
    m_ols,
    m_path,
    n_steps,
    np,
    plt,
    x_data,
    y_data,
):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # ── Left: contour of loss surface + GD path ──────────────────────────────
    _margin_m = 1.5
    _margin_b = 1.0
    m_lo = m_ols - _margin_m
    m_hi = m_ols + _margin_m
    b_lo = b_ols - _margin_b
    b_hi = b_ols + _margin_b

    _m_grid = np.linspace(m_lo, m_hi, 80)
    _b_grid = np.linspace(b_lo, b_hi, 80)
    _M, _B  = np.meshgrid(_m_grid, _b_grid)
    # Broadcasting: x_data (n,) → (1,1,n); _M and _B → (80,80,1)
    _L = np.mean(
        (y_data[None, None, :] - _M[:, :, None] * x_data[None, None, :] - _B[:, :, None]) ** 2,
        axis=2,
    )
    axes[0].contourf(_M, _B, _L, levels=30, cmap="Blues_r", alpha=0.75)
    axes[0].contour(_M, _B, _L, levels=15, colors="white",
                    linewidths=0.4, alpha=0.5)

    # GD path — clamp to plot range in case of near-divergence
    _mp = np.clip(m_path, m_lo, m_hi)
    _bp = np.clip(b_path, b_lo, b_hi)
    axes[0].plot(_mp, _bp, "o-", color="coral", ms=3, lw=1.5,
                 label="GD path", zorder=3)
    axes[0].plot(_mp[0], _bp[0], "^", color="gold", ms=10,
                 label="start  (0, 0)", zorder=4)
    axes[0].plot(m_ols, b_ols, "*", color="limegreen", ms=14,
                 label=f"OLS  ({m_ols:.3f}, {b_ols:.3f})", zorder=4)
    if not diverged:
        axes[0].plot(m_final, b_final, "s", color="red", ms=8,
                     label=f"GD final  ({m_final:.3f}, {b_final:.3f})", zorder=4)
    axes[0].set_xlabel("slope  m")
    axes[0].set_ylabel("intercept  b")
    axes[0].set_title(f"Loss surface  (η = {lr:.2f},  {n_steps} steps)")
    axes[0].legend(fontsize=8, loc="upper right")

    # ── Right: loss over iterations ──────────────────────────────────────────
    if loss_path:
        axes[1].plot(range(len(loss_path)), loss_path, color="steelblue", lw=2)
        axes[1].axhline(loss_ols, color="limegreen", lw=2, linestyle="--",
                        label=f"OLS minimum  ({loss_ols:.4f})")
        axes[1].set_xlabel("Gradient descent step")
        axes[1].set_ylabel("MSE loss")
        axes[1].set_title("Loss per iteration")
        axes[1].legend(fontsize=9)
        if max(loss_path) > 10 * loss_ols:
            axes[1].set_yscale("log")
    else:
        axes[1].text(0.5, 0.5, "Diverged on step 1\n(η too large)",
                     ha="center", va="center", transform=axes[1].transAxes,
                     fontsize=13, color="red")
        axes[1].axis("off")

    plt.tight_layout()
    fig
    return


@app.cell
def _(
    b_final,
    b_ols,
    diverged,
    loss_ols,
    loss_path,
    lr,
    m_final,
    m_ols,
    mo,
    n_steps,
):
    if diverged:
        kind = "warn"
        body = f"""
            **Diverged!**  Learning rate η = {lr:.2f} is too large.

            - The gradient updates overshoot the minimum and the loss explodes.
            - Each step takes the parameters *past* the bottom of the bowl and
              further away on the other side.
            - Try reducing η below 0.5 to see stable convergence.
            """
    else:
        final_loss = loss_path[-1] if loss_path else float("nan")
        converged  = (final_loss - loss_ols) < 5e-4
        kind = "success" if converged else "info"
        body = f"""
            **After {n_steps} steps with η = {lr:.2f}:**

            - OLS (exact minimum): slope = **{m_ols:.4f}**, intercept = **{b_ols:.4f}**
            - Gradient descent: slope = **{m_final:.4f}**, intercept = **{b_final:.4f}**
            - Final loss: **{final_loss:.5f}** (OLS minimum: {loss_ols:.5f})

            {"Converged to the OLS solution." if converged else "Not yet converged — try more steps or a larger η."}

            Increase η for fast convergence but push it too far and the algorithm diverges.
            """

    mo.callout(mo.md(body), kind=kind)
    return


if __name__ == "__main__":
    app.run()
