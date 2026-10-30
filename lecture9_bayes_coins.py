# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "matplotlib",
#     "numpy",
#     "scipy",
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
    from scipy.stats import binom, beta as beta_dist

    return beta_dist, binom, mo, np, plt


@app.cell
def _(mo):
    mo.md(r"""
    # Bayesian Updating: Coin Flipping

    In Bayesian statistics an estimate of a parameter is not a number; it is actually a distribution. Suppose the parameter of interest is $p$ and we have observed data. Then we want to find the so called posterior distribution:

    $$
    Prob(p|data) = \frac{P(p \text{ and data})}{P(data)}
    $$

    With this distribution, we can calculate the probability that $p$ exceeds 0.8 given our observed data: $Prob(p > 0.8 | data)$.

    The notation \(P(p \text{ and data})\) is sometimes written as the joint probability \(P(p, data)\). Bayes' theorem then rewrites it as \(P(p, data) =P(data|p)P(p) \). Hence, we get

    $$
    Prob(p|data) = \frac{P(data|p)P(p)}{P(data)}
    $$

    In words:

    - \(P(p)\) is the **prior**: what we believed about \(p\) before observing data.
    - \(P(data|p)\) is the **likelihood**: how probable the observed data are for a given value of \(p\).
    - \(P(data)\) is the **evidence** or normalising constant.
    - \(P(p|data)\) is the **posterior**: our updated belief about \(p\) after seeing the data.


    ## Bayes' theorem

    You want to estimate $p$, the probability a coin lands heads.
    Before seeing any data you hold a **prior** belief $P(p)$.
    After observing $y$ heads in $n$ throws, Bayes' theorem gives the **posterior**:

    $$
    P(p \mid y, n)
    = \frac{P(y \mid p, n)\; P(p)}{\int_0^1 P(y \mid \pi, n)\,P(\pi)\,d\pi}
    \;\propto\;
    \underbrace{P(y \mid p, n)}_{\text{likelihood}}\;
    \underbrace{P(p)}_{\text{prior}}.
    $$

    The **likelihood** $P(y \mid p, n) = \binom{n}{y}p^y(1-p)^{n-y}$ is the probability of
    observing exactly $y$ heads in $n$ independent throws when the true probability is $p$.
    The denominator is a normalising constant that makes the posterior integrate to 1;
    it does not depend on $p$ and can be ignored when comparing posteriors.

    If we observe 14 heads in 20 throws, we can write the posterior as follows:
    $$
    Prob(p|data) =  \frac{\binom{20}{14}p^{14}(1-p)^{20-14}P(p)}{\int_{0}^1 \binom{20}{14}\pi^{14}(1-\pi)^{20-14}P(\pi)d\pi}
    $$

    ## Grid approximation

    We evaluate the right-hand side on a fine grid of 600 values of $p \in (0,1)$,
    multiply prior by likelihood at each grid point, and divide by the sum to normalise.
    The result is a discrete approximation to the posterior density.

    ## Sequential vs.\ batch updating

    Suppose you throw the coin $n$ times, recording outcomes $x_1, x_2, \ldots, x_n \in \{heads,tails\}$.

    **Step by step**: use each flip to update the current posterior, which then becomes the
    prior for the next flip:
    $$
    P(p \mid x_1) \propto P(x_1 \mid p)\,P(p),
    \quad
    P(p \mid x_1, x_2) \propto P(x_2 \mid p)\underbrace{P(p \mid x_1)}_{\text{new prior}}
    \propto P(x_2 \mid p)\,P(x_1 \mid p)\,P(p).
    $$

    **All at once**: feed all $n$ flips to the original prior simultaneously:
    $$
    P(p \mid y, n) \propto P(y \mid p, n)\,P(p)
    = \prod_{i=1}^n P(x_i \mid p)\cdot P(p).
    $$
    where $y$ equals the number of heads in $n$ flips.

    The two expressions are identical because the flips are independent.
    It does not matter whether you update after every throw or after all $n$ throws —
    the final posterior is the same.

    ---

    Use the controls below to set the true probability and choose a prior.
    The six panels show how the posterior (solid blue) concentrates around the
    true value as more data arrive.  The dashed black line is the prior.
    """)
    return


@app.cell
def _(mo):
    p_slider = mo.ui.slider(
        0.1, 0.9, value=0.7, step=0.05,
        label=r"True probability of head $p$",
    )
    prior_dd = mo.ui.dropdown(
        options={
            "Uniform  —  Beta(1, 1)": (1, 1),
            "Sceptical  —  Beta(2, 5),  mode = 0.2": (2, 5),
            "Optimistic  —  Beta(5, 2),  mode = 0.8": (5, 2),
        },
        value="Uniform  —  Beta(1, 1)",
        label="Prior for p",
    )
    mo.vstack([p_slider, prior_dd])
    return p_slider, prior_dd


@app.cell
def _(beta_dist, binom, np, p_slider, plt, prior_dd):
    # Fixed seed so the same p always produces the same coin sequence.
    _rng = np.random.default_rng(42)
    _flips = _rng.binomial(1, p_slider.value, 20)

    # Grid
    _p = np.linspace(0.001, 0.999, 600)
    _dp = _p[1] - _p[0]
    _a, _b = prior_dd.value

    def _normalize(arr):
        return arr / (arr.sum() * _dp)

    def _update(current_pdf, y_new, n_new):
        return _normalize(binom.pmf(y_new, n_new, _p) * current_pdf)

    # Build sequential posteriors.
    # n=1..5: update one flip at a time.
    # n=20: update from the n=5 posterior with the 15 new flips at once
    #       (valid because sequential = batch).
    _current = _normalize(beta_dist.pdf(_p, _a, _b))
    _priors_to_plot = []
    _posteriors = []
    _y_cumul = []

    for _i in range(5):
        _priors_to_plot.append(_current)
        _current = _update(_current, int(_flips[_i]), 1)
        _posteriors.append(_current)
        _y_cumul.append(int(_flips[: _i + 1].sum()))

    _priors_to_plot.append(_current)          # n=5 posterior becomes prior for n=20
    _current = _update(_current, int(_flips[5:].sum()), 15)
    _posteriors.append(_current)
    _y_cumul.append(int(_flips.sum()))

    _n_vals = [1, 2, 3, 4, 5, 20]
    _fig, _axes = plt.subplots(2, 3, figsize=(12, 7))

    for _idx, (_ax, _n, _y, _pri, _post) in enumerate(
        zip(_axes.flat, _n_vals, _y_cumul, _priors_to_plot, _posteriors)
    ):
        _label_prior = "Prior  Beta({}, {})".format(_a, _b) if _idx == 0 else "Previous posterior"
        _ax.plot(_p, _pri, color="black", lw=1.5, ls="--", label=_label_prior)
        _ax.plot(_p, _post, color="steelblue", lw=2.0, label="Posterior")
        _ax.axvline(
            p_slider.value, color="crimson", ls=":", lw=1.5,
            label="True p = {:.2f}".format(p_slider.value),
        )
        _ax.set_title("$n$ = {},   $y$ = {}".format(_n, _y))
        _ax.set_xlabel("$p$")
        _ax.set_xlim(0, 1)
        _ax.set_ylim(bottom=0)
        if _idx % 3 == 0:
            _ax.set_ylabel("Density")
        if _idx == 0:
            _ax.legend(fontsize=8, loc="upper left")

    _fig.suptitle(
        "Sequential updating — Prior: Beta({}, {})   |   True p = {:.2f}".format(
            _a, _b, p_slider.value
        ),
        fontsize=11,
    )
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(np, p_slider, prior_dd):
    # Verify sequential = batch using the exact conjugate Beta-Binomial formula.
    _rng = np.random.default_rng(42)
    _flips = _rng.binomial(1, p_slider.value, 20)
    _a, _b = prior_dd.value

    # Sequential: update one flip at a time through all 20 flips
    _aa, _bb = _a, _b
    for _x in _flips:
        _aa += int(_x)
        _bb += 1 - int(_x)

    # Batch: original prior + all 20 flips at once
    _y20 = int(_flips.sum())
    _a_batch = _a + _y20
    _b_batch = _b + (20 - _y20)

    _match = "✓ identical" if (_aa == _a_batch and _bb == _b_batch) else "✗ mismatch"

    # Also show the n=5→n=20 shortcut used in the last panel
    _y5 = int(_flips[:5].sum())
    _y_6_20 = int(_flips[5:].sum())

    return


if __name__ == "__main__":
    app.run()
