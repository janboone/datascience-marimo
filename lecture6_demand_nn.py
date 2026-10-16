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
    # Demand Estimation with Neural Networks

    **The economic setting.** A firm has offered 1 500 consumers a product with different quality $q$ and
    price $p$ combinations in a small randomized experiment. A consumer buys at max. one unit of the product. Each consumer has 10 characteristics (age, gender, income, health consciousness,
    tech savviness, outdoor hobby, children, education, brand loyalty, and time scarcity) that
    shape their valuation of quality and their sensitivity to price.  A consumer buys if

    $$U(q,p \mid x_i) = \alpha(x_i)\,q + \phi(x_i)\,q^2 - \beta(x_i)\,p + \delta(x_i)\,q\,p + \varepsilon_i > 0$$

    where $\varepsilon_i \sim \text{Logistic}(0,1)$ is idiosyncratic taste noise.  The
    coefficients $\alpha, \phi, \beta, \delta$ depend on consumer characteristics through
    non-linear interactions that a simple logistic regression cannot fully capture.

    **What we estimate.** A neural network with 12 inputs (10 characteristics + quality + price)
    and a sigmoid output predicts purchase probability.  Comparing the neural network to a linear logistic
    regression shows how much the non-linear interactions matter.

    We measure the fit of the predicted purchase probabilities using binary cross-entropy (log loss), which is defined by
    $$
    \text{BCE}
    =
    -\frac{1}{n}\sum_{i=1}^{n}
    \left[
    y_i \log(\hat p_i) + (1-y_i)\log(1-\hat p_i)
    \right],
    $$
    where $y_i \in \{0,1\}$ is the observed purchase decision and $\hat p_i$ is the model-predicted purchase probability. Lower values mean better calibrated predictions.

    **Reading the training curves.** After clicking the button *Train models* below, the graph shows
    binary cross-entropy loss on the training and validation sets epoch by epoch:

    - **Both losses high and flat or falling**: underfitting; try more layers, more nodes, or more epochs.
    - **Validation loss platteaus or increases while training loss keeps falling**: overfitting; reduce complexity or train for fewer epochs.
    - **Optimal number of epochs**: around the lowest validation loss or when validation loss hardly decreases anymore.

    Use the controls to explore over- and under-fitting.
    """)
    return


@app.cell
def _(mo, np):
    # ── Simulate 1 500 consumers with the true DGP ────────────────────────────
    _N   = 1500
    _rng = np.random.default_rng(0)

    _X = np.zeros((_N, 10))
    _X[:, 0] = _rng.uniform(0, 1, _N)       # age
    _X[:, 1] = _rng.binomial(1, 0.5, _N)    # gender
    _X[:, 2] = _rng.beta(2, 5, _N)           # income
    _X[:, 3] = _rng.uniform(0, 1, _N)        # health
    _X[:, 4] = _rng.uniform(0, 1, _N)        # tech
    _X[:, 5] = _rng.binomial(1, 0.4, _N)    # outdoor
    _X[:, 6] = _rng.binomial(1, 0.35, _N)   # children
    _X[:, 7] = _rng.beta(3, 2, _N)           # education
    _X[:, 8] = _rng.uniform(0, 1, _N)        # brand_loyalty
    _X[:, 9] = _rng.uniform(0, 1, _N)        # time_scarcity

    _q = _rng.uniform(1, 5, _N)
    _p = _rng.uniform(1, 10, _N)

    _age, _gender, _income, _health, _tech, _outdoor, _children, _educ, _loyal, _time_s = _X.T

    _alpha  = (1.5 + 2.0 * _health * (_age > 0.6).astype(float)
               + 1.5 * _outdoor * _children + 0.8 * _tech * _educ
               + 0.5 * _gender * _loyal + 1.0 * _time_s * _income + 0.3 * _educ)
    _phi    = (0.5 * _tech - 0.4 * _health * (_age > 0.6).astype(float)
               + 0.3 * _outdoor * _children - 0.2 * (1 - _income))
    _beta_u = (2.0 - 1.2 * _income * _tech - 0.8 * _income * _educ
               + 0.6 * (1-_income)*(1-_educ) + 0.3 * _children*(1-_income) - 0.2 * _loyal)
    _delta  = (-0.3 + 0.4 * _income * _loyal
               - 0.5 * (1-_income)*(1-_tech) + 0.2 * _educ * _tech)

    _det_u  = _alpha*_q + _phi*_q**2 - _beta_u*_p + _delta*_q*_p
    _bought = ((_det_u + _rng.logistic(0, 1, _N)) > 0).astype(float)

    # Full feature matrix: 10 characteristics + quality + price
    X_full = np.column_stack([_X, _q, _p])
    y_all   = _bought

    # ── 80 / 20 train / validation split ──────────────────────────────────────
    _idx  = np.random.default_rng(1).permutation(_N)
    _n_tr = int(0.8 * _N)                    # 1 200 train, 300 val
    X_tr_raw = X_full[_idx[:_n_tr]]
    X_va_raw = X_full[_idx[_n_tr:]]
    y_tr     = y_all[_idx[:_n_tr]]
    y_va     = y_all[_idx[_n_tr:]]

    # Z-score normalise using training statistics
    _mu  = X_tr_raw.mean(0)
    _sig = X_tr_raw.std(0) + 1e-8
    X_tr = (X_tr_raw - _mu) / _sig
    X_va = (X_va_raw - _mu) / _sig

    purchase_rate = float(y_all.mean())

    mo.callout(mo.md(f"""
        **Simulated market data** — {_N} consumers, each offered a random (quality, price) bundle.

        Purchase rate: **{purchase_rate:.1%}**.  Features: 10 consumer characteristics + quality + price
        = 12 inputs.  Training set: {_n_tr} observations.  Validation set: {_N - _n_tr} observations.
        """), kind="neutral")
    return X_tr, X_va, y_tr, y_va


@app.cell
def _(mo):
    n_layers_radio = mo.ui.radio(
        options=["1", "2", "3"], value="1",
        label="Hidden layers",
    )
    nodes_dd = mo.ui.dropdown(
        ["8", "16", "32"], value="16",
        label="Nodes per hidden layer",
    )
    epochs_slider = mo.ui.slider(
        20, 300, value=100, step=20,
        label="Training epochs",
    )
    run_btn = mo.ui.run_button(label="▶ Train models")
    mo.vstack([
        mo.hstack([n_layers_radio, nodes_dd, epochs_slider], gap=3),
        run_btn,
    ])
    return epochs_slider, n_layers_radio, nodes_dd, run_btn


@app.cell
def _(
    X_tr,
    X_va,
    epochs_slider,
    mo,
    n_layers_radio,
    nodes_dd,
    np,
    run_btn,
    y_tr,
    y_va,
):
    mo.stop(
        not run_btn.value,
        mo.callout(
            mo.md("Set the architecture above, then click **▶ Train models**."),
            kind="neutral",
        ),
    )

    # ── Neural-network helpers ────────────────────────────────────────────────
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))

    def _bce(y_hat, y):
        p = np.clip(y_hat.squeeze(), 1e-12, 1 - 1e-12)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    def _init(layer_sizes, seed=42):
        rng = np.random.default_rng(seed)
        params = []
        for i in range(len(layer_sizes) - 1):
            scale = np.sqrt(2.0 / layer_sizes[i])
            W = rng.normal(0, scale, (layer_sizes[i], layer_sizes[i + 1]))
            b = np.zeros(layer_sizes[i + 1])
            params.append([W, b])
        return params

    def _fwd(X, params):
        As, Zs = [X], []
        for k, (W, b) in enumerate(params):
            Z = As[-1] @ W + b
            Zs.append(Z)
            A = np.maximum(0.0, Z) if k < len(params) - 1 else _sigmoid(Z)
            As.append(A)
        return As, Zs

    def _bwd(As, Zs, params, y):
        n = len(y)
        grads = []
        dZ = (As[-1].reshape(-1, 1) - y.reshape(-1, 1)) / n
        for i in range(len(params) - 1, -1, -1):
            dW = As[i].T @ dZ
            db = dZ.sum(axis=0)
            grads.insert(0, (dW, db))
            if i > 0:
                dA = dZ @ params[i][0].T
                dZ = dA * (Zs[i - 1] > 0).astype(float)
        return grads

    def _train(layer_sizes, n_epochs, lr=0.1, seed=42):
        params = _init(layer_sizes, seed)
        tr_hist, va_hist = [], []
        for _ in range(n_epochs):
            As, Zs = _fwd(X_tr, params)
            tr_hist.append(_bce(As[-1], y_tr))
            for j, (dW, db) in enumerate(_bwd(As, Zs, params, y_tr)):
                params[j][0] -= lr * dW
                params[j][1] -= lr * db
            As_va, _ = _fwd(X_va, params)
            va_hist.append(_bce(As_va[-1], y_va))
        As_tr_f, _ = _fwd(X_tr, params)
        As_va_f, _ = _fwd(X_va, params)
        tr_acc = float(np.mean((As_tr_f[-1].squeeze() > 0.5) == (y_tr > 0.5)))
        va_acc = float(np.mean((As_va_f[-1].squeeze() > 0.5) == (y_va > 0.5)))
        return tr_hist, va_hist, tr_acc, va_acc

    # ── Logistic regression (0 hidden layers) — always trained as benchmark ──
    logit_tr_hist, logit_va_hist, logit_tr_acc, logit_va_acc = _train([12, 1], 200)

    # ── Neural network with chosen architecture ───────────────────────────────
    n_hidden = int(n_layers_radio.value)
    n_nodes  = int(nodes_dd.value)
    n_epochs = epochs_slider.value
    _sizes   = [12] + [n_nodes] * n_hidden + [1]

    nn_tr_hist, nn_va_hist, nn_tr_acc, nn_va_acc = _train(_sizes, n_epochs)
    return (
        logit_tr_acc,
        logit_va_acc,
        logit_va_hist,
        n_epochs,
        n_hidden,
        n_nodes,
        nn_tr_acc,
        nn_tr_hist,
        nn_va_acc,
        nn_va_hist,
    )


@app.cell
def _(logit_va_hist, n_epochs, n_hidden, n_nodes, nn_tr_hist, nn_va_hist, plt):
    _epochs = list(range(1, n_epochs + 1))

    _fig_loss, _ax = plt.subplots(figsize=(10, 4))
    _ax.plot(_epochs, nn_tr_hist, color="steelblue",  lw=2.2, label="NN — train loss")
    _ax.plot(_epochs, nn_va_hist, color="steelblue",  lw=2.2, ls="--", label="NN — val loss")
    _ax.axhline(logit_va_hist[-1], color="grey", lw=1.4, ls=":",
                label=f"Logistic regression val loss ({logit_va_hist[-1]:.3f})")
    _ax.set_xlabel("Epoch")
    _ax.set_ylabel("Binary cross-entropy loss")
    _ax.set_title(
        f"Training curves — {n_hidden} hidden layer{'s' if n_hidden > 1 else ''} "
        f"× {n_nodes} nodes"
    )
    _ax.legend(fontsize=9)
    plt.tight_layout()
    _fig_loss
    return


@app.cell
def _(
    logit_tr_acc,
    logit_va_acc,
    mo,
    n_epochs,
    n_hidden,
    n_nodes,
    nn_tr_acc,
    nn_va_acc,
):
    _gap = nn_tr_acc - nn_va_acc
    _gain = nn_va_acc - logit_va_acc

    if nn_va_acc < 0.75:
        _kind = "info"
        _msg = f"""
            **Underfitting** — NN val accuracy = {nn_va_acc:.1%}

            The model has not learned the demand patterns well.
            Try more hidden layers, more nodes per layer, or more epochs.

            | Model | Train acc | Val acc |
            |---|---|---|
            | Logistic reg. | {logit_tr_acc:.1%} | {logit_va_acc:.1%} |
            | NN ({n_hidden}L × {n_nodes}N, {n_epochs} ep) | {nn_tr_acc:.1%} | {nn_va_acc:.1%} |
            """
    elif _gap > 0.08:
        _kind = "warn"
        _msg = f"""
            **Overfitting** — train acc = {nn_tr_acc:.1%}, val acc = {nn_va_acc:.1%}
            (gap = {_gap:.1%})

            The network has memorised the training data.
            Try fewer hidden layers, fewer nodes, or fewer epochs.

            | Model | Train acc | Val acc |
            |---|---|---|
            | Logistic reg. | {logit_tr_acc:.1%} | {logit_va_acc:.1%} |
            | NN ({n_hidden}L × {n_nodes}N, {n_epochs} ep) | {nn_tr_acc:.1%} | {nn_va_acc:.1%} |
            """
    else:
        _kind = "success"
        _msg = f"""
            **Good fit** — NN val accuracy = {nn_va_acc:.1%}
            ({'+' if _gain >= 0 else ''}{_gain:.1%} vs logistic regression)

            The network generalises well and outperforms logistic regression.
            This accuracy advantage translates directly into higher profit when optimising
            price and quality — because better demand predictions lead to better pricing.

            | Model | Train acc | Val acc |
            |---|---|---|
            | Logistic reg. | {logit_tr_acc:.1%} | {logit_va_acc:.1%} |
            | NN ({n_hidden}L × {n_nodes}N, {n_epochs} ep) | {nn_tr_acc:.1%} | {nn_va_acc:.1%} |
            """
    mo.callout(mo.md(_msg), kind=_kind)
    return


if __name__ == "__main__":
    app.run()
