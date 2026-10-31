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
    from scipy.stats import beta as beta_dist

    return beta_dist, mo, np, plt


@app.cell
def _(mo):
    mo.md(r"""
    # Vampire Incarceration: Frequentist vs. Bayesian Decision Making

    Consider a world where humans and vampires live together. Most vampires are innocent peace loving beings that do not harm humans. However, some vampires are rogue and there is a risk that at some point they will attack a human. A test has been developed that can spot rogue vampires. But the test is not perfect; mistakes of the first and second type can be made. There is a probability $p_I$ that an innocent vampire produces a positive result on the test. Also there is a probability $p_{II}$ that a rogue vampire produces a negative result on the test.

    These probabilities have been estimated from experimental data from past tests. In a frequentist approach one would use these probabilities as given numbers. But in a Bayesian approach one uses the posterior distributions of $p_I,p_{II}$ in policy analysis. That is, the uncertainty surrounding these estimates feed directly into the policy decisions.

    The policy question that we analyze here is: should a vampire with a positive test result be incarcerated? There is a social damage $D_I$ if we incarcerate an innocent vampire. On the other hand, if we release a rogue vampire there is a probability that a human is attacked and this leads to expected damage $D_{II}$.

    One possible decision rule is to release the vampire if the expected damage of incarceration is higher than the expected damage of release.

    As this decision rule is not linear in $p_I,p_{II}$ taking the uncertainty of these parameters into account can lead to different decisions. Below you can adjust the relevant parameters for this problem and see for yourself when a frequentist decision maker would decide to incarcerate while a Bayesian decision maker would decide to release the vampire (and the other way around).

    ## The setting

    Summarizing the test determining whether a vampire will turn rogue and attack humans:


    | Symbol | Meaning |
    |---|---|
    | $p_I$ | P(positive test \| innocent vampire) — false positive rate |
    | $p_{II}$ | P(negative test \| rogue vampire) — false negative rate |
    | $D_I$ | damage from incarcerating an innocent vampire |
    | $D_{II}$ | damage from releasing a rogue vampire |
    | $\pi$ | prior probability a vampire is rogue (base rate in the population) |

    ## Posterior probability of being rogue

    A vampire tests positive. By Bayes' theorem, the probability it is rogue is

    $$P(\text{rogue} \mid +) = \frac{(1-p_{II})\,\pi}{(1-p_{II})\,\pi + p_I\,(1-\pi)}.$$

    ## Decision rule

    Both approaches apply Bayes' theorem to compute the posterior probability the vampire
    is rogue given the positive test and the base rate $\pi$. Incarcerate when expected damage from incarcerating is lower than from releasing:

    $$\underbrace{D_I \cdot P(\text{innocent} \mid +)}_{\text{cost of incarcerating}} < \underbrace{D_{II} \cdot P(\text{rogue} \mid +)}_{\text{cost of releasing}}.$$

    ## Frequentist vs. Bayesian under uncertainty

    The difference between the two approaches is not Bayes' theorem — both use it.
    The difference is whether $p_I$ and $p_{II}$ are treated as **known constants** or as
    **uncertain parameters**.

    Under the **frequentist** approach $p_I$ and $p_{II}$ are given point estimates.
    $P(\text{rogue}\mid+)$ is a single number, so the expected damage is a **single number**.

    Under the **Bayesian** approach $p_I$ and $p_{II}$ are themselves uncertain which we model here as follows:

    $$p_I \sim \mathrm{Beta}(p_I\kappa_I,\,(1-p_I)\kappa_I), \qquad
      p_{II} \sim \mathrm{Beta}(p_{II}\kappa_{II},\,(1-p_{II})\kappa_{II}).$$

    A large $\kappa$ concentrates the distribution tightly around the slider value;
    a small $\kappa$ spreads it widely.  For each draw $(p_I^{(s)},\, p_{II}^{(s)})$ from the posterior distribution
    the expected damage is different, so the expected damage is a **distribution**.

    Because $P(\text{rogue}\mid+)$ is **convex** in $p_I$ and **concave** in $p_{II}$,
    Jensen's inequality pulls the Bayesian expectation in opposite directions:

    - Small $\kappa_I$ (uncertain $p_I$) → $E[P(\text{rogue}\mid+)]$ rises → Bayesian incarcerates **more**.
    - Small $\kappa_{II}$ (uncertain $p_{II}$) → $E[P(\text{rogue}\mid+)]$ falls → Bayesian incarcerates **less**.

    Use the two $\kappa$ sliders independently to isolate each effect.
    As either $\kappa \to \infty$ the corresponding distribution collapses to a point mass
    and that component converges to the frequentist value.
    """)
    return


@app.cell
def _(mo):
    pI_slider = mo.ui.slider(0.01, 0.50, value=0.10, step=0.01,
                              label=r"$p_I$ — Type I error rate (false positive)")
    pII_slider = mo.ui.slider(0.01, 0.50, value=0.20, step=0.01,
                               label=r"$p_{II}$ — Type II error rate (false negative)")
    DI_slider = mo.ui.slider(10, 300, value=40, step=10,
                              label=r"$D_I$ — damage from incarcerating innocent")
    DII_slider = mo.ui.slider(10, 300, value=120, step=10,
                               label=r"$D_{II}$ — damage from releasing rogue")
    pi_slider = mo.ui.slider(0.01, 0.50, value=0.05, step=0.01,
                              label=r"$\pi$ — prior probability vampire is rogue")
    kappa_I_slider = mo.ui.slider(5, 300, value=20, step=5,
                                   label=r"$\kappa_I$ — certainty about $p_I$ (higher = more certain)")
    kappa_II_slider = mo.ui.slider(5, 300, value=20, step=5,
                                    label=r"$\kappa_{II}$ — certainty about $p_{II}$ (higher = more certain)")
    mo.vstack([
        mo.hstack([pI_slider, pII_slider], gap=3),
        mo.hstack([DI_slider, DII_slider], gap=3),
        mo.hstack([kappa_I_slider, kappa_II_slider], gap=3),
        pi_slider,
    ])
    return (
        DII_slider,
        DI_slider,
        kappa_II_slider,
        kappa_I_slider,
        pII_slider,
        pI_slider,
        pi_slider,
    )


@app.cell
def _(
    DII_slider,
    DI_slider,
    beta_dist,
    kappa_II_slider,
    kappa_I_slider,
    mo,
    np,
    pII_slider,
    pI_slider,
    pi_slider,
    plt,
):
    _pI  = pI_slider.value
    _pII = pII_slider.value
    _DI  = DI_slider.value
    _DII = DII_slider.value
    _pi  = pi_slider.value
    _kI  = kappa_I_slider.value
    _kII = kappa_II_slider.value

    def _post_rogue(pI, pII, pi):
        return (1.0 - pII) * pi / ((1.0 - pII) * pi + pI * (1.0 - pi))

    # ── Frequentist: Bayes at fixed p_I, p_II (no uncertainty in error rates) ──
    _pr_rogue_freq = _post_rogue(_pI, _pII, _pi)
    _dmg_inc_freq  = _DI  * (1.0 - _pr_rogue_freq)
    _dmg_rel_freq  = _DII * _pr_rogue_freq

    # ── Bayesian: sample from Beta posteriors ─────────────────────────────────
    _N   = 10_000
    _rng = np.random.default_rng(42)
    _pI_s  = _rng.beta(_pI  * _kI,  (1.0 - _pI)  * _kI,  _N)
    _pII_s = _rng.beta(_pII * _kII, (1.0 - _pII) * _kII, _N)

    _pr_s       = _post_rogue(_pI_s, _pII_s, _pi)
    _dmg_inc_s  = _DI  * (1.0 - _pr_s)
    _dmg_rel_s  = _DII * _pr_s
    _mean_inc   = _dmg_inc_s.mean()
    _mean_rel   = _dmg_rel_s.mean()

    # ── Figure ────────────────────────────────────────────────────────────────
    _fig, _ax = plt.subplots(1, 4, figsize=(16, 4))

    # Panel 1: Beta distributions for pI and pII
    _pg = np.linspace(0.001, 0.999, 300)
    _ax[0].plot(_pg, beta_dist.pdf(_pg, _pI * _kI, (1 - _pI) * _kI),
                color="steelblue", lw=2, label="$p_I$")
    _ax[0].plot(_pg, beta_dist.pdf(_pg, _pII * _kII, (1 - _pII) * _kII),
                color="firebrick", lw=2, ls="--", label="$p_{II}$")
    _ax[0].set(xlabel="Error rate", ylabel="Density",
               title="Uncertainty about test accuracy", xlim=(0, 1))
    _ax[0].legend()

    # Panel 2: distribution of E[damage | incarcerate]
    _bins = 60
    _ax[1].hist(_dmg_inc_s, bins=_bins, density=True, alpha=0.5, color="steelblue")
    _ax[1].axvline(_dmg_inc_freq, color="black",     lw=2, ls="--", label="Frequentist")
    _ax[1].axvline(_mean_inc,     color="steelblue", lw=2,           label="Bayesian mean")
    _ax[1].set(xlabel="Expected damage", title="E[damage | incarcerate]")
    _ax[1].legend(fontsize=8)

    # Panel 3: distribution of E[damage | release]
    _ax[2].hist(_dmg_rel_s, bins=_bins, density=True, alpha=0.5, color="firebrick")
    _ax[2].axvline(_dmg_rel_freq, color="black",     lw=2, ls="--", label="Frequentist")
    _ax[2].axvline(_mean_rel,     color="firebrick", lw=2,           label="Bayesian mean")
    _ax[2].set(xlabel="Expected damage", title="E[damage | release]")
    _ax[2].legend(fontsize=8)

    # Panel 4: E[damage|incarcerate] - E[damage|release]
    _diff_s    = _dmg_inc_s - _dmg_rel_s
    _diff_freq = _dmg_inc_freq - _dmg_rel_freq
    _ax[3].hist(_diff_s, bins=_bins, density=True, alpha=0.5, color="mediumpurple")
    _ax[3].axvline(_diff_freq,    color="black",        lw=2, ls="--", label="Frequentist")
    _ax[3].axvline(_diff_s.mean(), color="mediumpurple", lw=2,          label="Bayesian mean")
    _ax[3].axvline(0, color="gray", lw=1.2, ls=":",     label="Break-even")
    _ax[3].set(xlabel="Damage difference", title="E[damage|incarcerate] − E[damage|release]")
    _ax[3].legend(fontsize=8)

    _fig.tight_layout()

    # ── Callout ───────────────────────────────────────────────────────────────
    _freq_dec  = "incarcerate" if _dmg_inc_freq < _dmg_rel_freq else "release"
    _bayes_dec = "incarcerate" if _mean_inc     < _mean_rel     else "release"
    _p_inc     = float((_dmg_inc_s < _dmg_rel_s).mean())

    _msg = (
        "| | Frequentist | Bayesian mean |\n"
        + "|---|---|---|\n"
        + "| E[damage, incarcerate] | {:.1f} | {:.1f} |\n".format(_dmg_inc_freq, _mean_inc)
        + "| E[damage, release]     | {:.1f} | {:.1f} |\n\n".format(_dmg_rel_freq, _mean_rel)
        + "**Frequentist decision:** {}  |  ".format(_freq_dec)
        + "**Bayesian decision:** {}\n\n".format(_bayes_dec)
        + "Both approaches use Bayes' theorem with $\\pi$. "
        + "The difference is that the frequentist treats $p_I$, $p_{II}$ as known constants, "
        + "while the Bayesian accounts for uncertainty in those error rates via $\\kappa$. "
        + "Increase $\\kappa$ to concentrate the Beta distributions and watch the Bayesian "
        + "converge to the frequentist result.\n\n"
        + "Given parameter uncertainty, incarcerating is optimal in "
        + "**{:.0%}** of draws from the Beta posteriors.".format(_p_inc)
    )
    _kind = "success" if _freq_dec == _bayes_dec else "warn"

    mo.vstack([_fig, mo.callout(mo.md(_msg), kind=_kind)])
    return


if __name__ == "__main__":
    app.run()
