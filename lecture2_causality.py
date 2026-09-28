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
    # Fork, Pipe, and Collider

    Multiple regression measures correlations, not causal effects. When you interpret
    regression coefficients as causal, the answer depends critically on the **causal
    structure** (the DAG — directed acyclic graph) among the variables.

    We study three structures involving variables $X$ (predictor), $Y$ (outcome), and
    a third variable $Z$. The central question in each case is: **should you control for $Z$?**

    We use a simple set-up here:
    - if there is no effect, the effect is (obviously) 0;
    - if there is an effect, the effect equals $\gamma$ (i.e. to simplify notation we do not subscript like $\gamma\_{Y}$ etc.)

    | DAG type | Structure | Control for Z? |
    |----------|-----------|----------------|
    | **Fork** | $X \leftarrow Z \rightarrow Y$ | Yes — Z is a common cause |
    | **Pipe** | $X \rightarrow Z \rightarrow Y$ | No — Z mediates the causal path |
    | **Collider** | $X \rightarrow Z \leftarrow Y$ | No — conditioning on Z creates bias |

    Select a DAG type and observe how the coefficient of $X$ changes when $Z$ is included.
    """)
    return


@app.cell
def _(mo):
    dag_dropdown = mo.ui.dropdown(
        options=["Fork", "Pipe", "Collider"],
        value="Fork",
        label="DAG type",
    )
    n_slider = mo.ui.slider(
        200, 5000, value=1000, step=100, label="Sample size ($n$)"
    )
    effect_slider = mo.ui.slider(
        0.3, 2.0, value=1.0, step=0.1, label="Effect strength (γ)"
    )
    mo.vstack([dag_dropdown, n_slider, effect_slider])
    return dag_dropdown, effect_slider, n_slider


@app.cell
def _(dag_dropdown, effect_slider, n_slider, np):
    dag_type = dag_dropdown.value
    n = n_slider.value
    gamma = effect_slider.value

    rng = np.random.default_rng()

    if dag_type == "Fork":
        # Z causes both X and Y; no direct X → Y
        Z = rng.normal(0, 1, n)
        X = gamma * Z + rng.normal(0, 1, n)
        Y = gamma * Z + rng.normal(0, 1, n)
        true_x_effect = 0.0
        correct_model = "without Z"

    elif dag_type == "Pipe":
        # X → Z → Y; causal effect of X on Y is fully mediated by Z
        X = rng.normal(0, 1, n)
        Z = gamma * X + rng.normal(0, 1, n)
        Y = gamma * Z + rng.normal(0, 1, n)
        true_x_effect = gamma ** 2
        correct_model = "without Z"

    else:  # Collider
        # X and Y are independent; both cause Z
        X = rng.normal(0, 1, n)
        Y = rng.normal(0, 1, n)          # truly independent of X
        Z = gamma * X + gamma * Y + rng.normal(0, 1, n)
        true_x_effect = 0.0
        correct_model = "without Z"

    # OLS: Y ~ X  (bivariate)
    M1 = np.column_stack([np.ones(n), X])
    b1 = np.linalg.solve(M1.T @ M1, M1.T @ Y)
    r1 = Y - M1 @ b1
    s1 = np.sqrt(np.sum(r1 ** 2) / (n - 2) * np.linalg.inv(M1.T @ M1).diagonal())
    coef_x_no_z = float(b1[1])
    t_x_no_z = float(b1[1] / s1[1])

    # OLS: Y ~ X + Z  (multivariate)
    M2 = np.column_stack([np.ones(n), X, Z])
    b2 = np.linalg.solve(M2.T @ M2, M2.T @ Y)
    r2 = Y - M2 @ b2
    s2 = np.sqrt(np.sum(r2 ** 2) / (n - 3) * np.linalg.inv(M2.T @ M2).diagonal())
    coef_x_with_z = float(b2[1])
    t_x_with_z = float(b2[1] / s2[1])
    return (
        coef_x_no_z,
        coef_x_with_z,
        dag_type,
        gamma,
        n,
        t_x_no_z,
        t_x_with_z,
        true_x_effect,
    )


@app.cell
def _(
    coef_x_no_z,
    coef_x_with_z,
    dag_type,
    gamma,
    n,
    plt,
    t_x_no_z,
    t_x_with_z,
):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: DAG diagram
    ax = axes[0]
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 3)
    ax.axis("off")

    node_box = dict(
        boxstyle="round,pad=0.5",
        facecolor="#1f3c88",
        alpha=0.9,
        edgecolor="white",
        linewidth=2,
    )
    nkw = dict(ha="center", va="center", color="white",
               fontsize=18, fontweight="bold", bbox=node_box)
    akw = dict(arrowstyle="->", lw=2.5, color="black")

    if dag_type == "Fork":
        ax.text(2.0, 2.6, "Z", **nkw)
        ax.text(0.7, 0.5, "X", **nkw)
        ax.text(3.3, 0.5, "Y", **nkw)
        ax.annotate("", xy=(0.9, 0.85), xytext=(1.8, 2.3), arrowprops=akw)
        ax.annotate("", xy=(3.1, 0.85), xytext=(2.2, 2.3), arrowprops=akw)
        ax.set_title(
            "Fork: Z → X  and  Z → Y\n(no direct X → Y effect)",
            fontsize=11, pad=8,
        )

    elif dag_type == "Pipe":
        ax.text(0.5, 1.5, "X", **nkw)
        ax.text(2.0, 1.5, "Z", **nkw)
        ax.text(3.5, 1.5, "Y", **nkw)
        ax.annotate("", xy=(1.65, 1.5), xytext=(0.85, 1.5), arrowprops=akw)
        ax.annotate("", xy=(3.15, 1.5), xytext=(2.35, 1.5), arrowprops=akw)
        ax.set_title(
            "Pipe: X → Z → Y\n(causal effect fully mediated by Z)",
            fontsize=11, pad=8,
        )

    else:  # Collider
        ax.text(0.7, 2.5, "X", **nkw)
        ax.text(3.3, 2.5, "Y", **nkw)
        ax.text(2.0, 0.5, "Z", **nkw)
        ax.annotate("", xy=(1.8, 0.85), xytext=(0.9, 2.15), arrowprops=akw)
        ax.annotate("", xy=(2.2, 0.85), xytext=(3.1, 2.15), arrowprops=akw)
        ax.set_title(
            "Collider: X → Z ← Y\n(X and Y are independent; both cause Z)",
            fontsize=11, pad=8,
        )

    # Right: bar chart of coefficient of X in two regressions
    ax2 = axes[1]
    values = [coef_x_no_z, coef_x_with_z]
    labels = ["Y ~ X\n(without Z)", "Y ~ X + Z\n(with Z)"]
    colors = ["steelblue", "coral"]
    bars = ax2.bar(labels, values, color=colors, alpha=0.85,
                   edgecolor="white", width=0.45)
    ax2.axhline(0, color="black", lw=1.2)
    ax2.set_ylabel("Coefficient of X")
    ax2.set_title(
        f"Coefficient of X  (n = {n}, γ = {gamma:.1f})\n"
        f"t-stats: {t_x_no_z:.2f}  |  {t_x_with_z:.2f}"
    )
    for bar, v in zip(bars, values):
        y_offset = 0.02 if v >= 0 else -0.06
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            v + y_offset,
            f"{v:.3f}",
            ha="center",
            fontsize=13,
            fontweight="bold",
        )

    plt.tight_layout()
    fig
    return


@app.cell
def _(
    coef_x_no_z,
    coef_x_with_z,
    dag_type,
    mo,
    t_x_no_z,
    t_x_with_z,
    true_x_effect,
):
    if dag_type == "Fork":
        text = f"""
    **Fork — Z causes both X and Y:**

    - **Y ~ X (without Z):** coef = {coef_x_no_z:.3f}  (t = {t_x_no_z:.2f}) — **spurious**: Z induces a correlation between X and Y.
    - **Y ~ X + Z (with Z):** coef = {coef_x_with_z:.3f}  (t = {t_x_with_z:.2f}) — **correct**: the X coefficient is close to zero.
    - True direct effect of X on Y: **{true_x_effect}**

    **Lesson:** you *should* control for Z. Without it you would wrongly conclude that X causes Y.
    """
        kind = "info"

    elif dag_type == "Pipe":
        text = f"""
    **Pipe — X → Z → Y:**

    - **Y ~ X (without Z):** coef = {coef_x_no_z:.3f}  (t = {t_x_no_z:.2f}) — **correct**: captures the total causal effect γ² ≈ {true_x_effect:.2f}.
    - **Y ~ X + Z (with Z):** coef = {coef_x_with_z:.3f}  (t = {t_x_with_z:.2f}) — **incorrect**: Z is the mechanism, so controlling for it removes the effect.
    - True total effect of X on Y: **γ² ≈ {true_x_effect:.2f}**

    **Lesson:** you *should not* control for Z. It sits on the causal path from X to Y, so conditioning on it blocks the effect you want to measure.
    """
        kind = "warn"

    else:  # Collider
        text = f"""
    **Collider — X and Y both cause Z:**

    - **Y ~ X (without Z):** coef = {coef_x_no_z:.3f}  (t = {t_x_no_z:.2f}) — **correct**: X and Y are truly independent; coefficient near zero.
    - **Y ~ X + Z (with Z):** coef = {coef_x_with_z:.3f}  (t = {t_x_with_z:.2f}) — **incorrect**: conditioning on Z "opens" a backdoor path and creates a spurious (negative) X-Y correlation.
    - True direct effect of X on Y: **{true_x_effect}**

    **Intuition:** if Z = γX + γY + noise is held fixed, then a higher X implies a lower Y (to keep Z the same).
    So within any slice of Z, X and Y appear negatively correlated — even though they are truly independent.

    **Lesson:** you *should not* control for Z. Adding it introduces bias where none existed before.
    """
        kind = "warn"

    mo.callout(mo.md(text), kind=kind)
    return


if __name__ == "__main__":
    app.run()
