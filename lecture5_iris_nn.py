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
    import io as io
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    return io, mo, np, plt


@app.cell
def _(mo):
    mo.md(r"""
    # Neural Networks: Classifying Iris Flowers

    The [Iris dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set) contains 150 measurements of
    iris flowers from three species: *setosa*, *versicolor*, and *virginica*. Each observation records
    four measurements (in cm):

    | Feature | Description |
    |---------|-------------|
    | Sepal length | Length of the sepal |
    | Sepal width  | Width of the sepal  |
    | Petal length | Length of the petal |
    | Petal width  | Width of the petal  |

    We train a **feedforward neural network** to predict which species a flower belongs to from these four
    measurements. Choose the architecture and training hyperparameters below, then click
    **Train network** to fit the model.

    The scatter plots show two selected features on the axes with **filled circles** for correctly
    classified flowers and **open circles** for misclassified ones. The confusion matrices count how
    often each true species was predicted as each class. To illustrate, to see how often *setosa* is classified as *versicolor*, look at the row for *setosa* and the column for *versicolor*.
    """)
    return


@app.cell
def _(io, np):
    _IRIS_CSV = (
        "5.1,3.5,1.4,0.2,Iris-setosa\n4.9,3.0,1.4,0.2,Iris-setosa\n4.7,3.2,1.3,0.2,Iris-setosa\n"
        "4.6,3.1,1.5,0.2,Iris-setosa\n5.0,3.6,1.4,0.2,Iris-setosa\n5.4,3.9,1.7,0.4,Iris-setosa\n"
        "4.6,3.4,1.4,0.3,Iris-setosa\n5.0,3.4,1.5,0.2,Iris-setosa\n4.4,2.9,1.4,0.2,Iris-setosa\n"
        "4.9,3.1,1.5,0.1,Iris-setosa\n5.4,3.7,1.5,0.2,Iris-setosa\n4.8,3.4,1.6,0.2,Iris-setosa\n"
        "4.8,3.0,1.4,0.1,Iris-setosa\n4.3,3.0,1.1,0.1,Iris-setosa\n5.8,4.0,1.2,0.2,Iris-setosa\n"
        "5.7,4.4,1.5,0.4,Iris-setosa\n5.4,3.9,1.3,0.4,Iris-setosa\n5.1,3.5,1.4,0.3,Iris-setosa\n"
        "5.7,3.8,1.7,0.3,Iris-setosa\n5.1,3.8,1.5,0.3,Iris-setosa\n5.4,3.4,1.7,0.2,Iris-setosa\n"
        "5.1,3.7,1.5,0.4,Iris-setosa\n4.6,3.6,1.0,0.2,Iris-setosa\n5.1,3.3,1.7,0.5,Iris-setosa\n"
        "4.8,3.4,1.9,0.2,Iris-setosa\n5.0,3.0,1.6,0.2,Iris-setosa\n5.0,3.4,1.6,0.4,Iris-setosa\n"
        "5.2,3.5,1.5,0.2,Iris-setosa\n5.2,3.4,1.4,0.2,Iris-setosa\n4.7,3.2,1.6,0.2,Iris-setosa\n"
        "4.8,3.1,1.6,0.2,Iris-setosa\n5.4,3.4,1.5,0.4,Iris-setosa\n5.2,4.1,1.5,0.1,Iris-setosa\n"
        "5.5,4.2,1.4,0.2,Iris-setosa\n4.9,3.1,1.5,0.1,Iris-setosa\n5.0,3.2,1.2,0.2,Iris-setosa\n"
        "5.5,3.5,1.3,0.2,Iris-setosa\n4.9,3.1,1.5,0.1,Iris-setosa\n4.4,3.0,1.3,0.2,Iris-setosa\n"
        "5.1,3.4,1.5,0.2,Iris-setosa\n5.0,3.5,1.3,0.3,Iris-setosa\n4.5,2.3,1.3,0.3,Iris-setosa\n"
        "4.4,3.2,1.3,0.2,Iris-setosa\n5.0,3.5,1.6,0.6,Iris-setosa\n5.1,3.8,1.9,0.4,Iris-setosa\n"
        "4.8,3.0,1.4,0.3,Iris-setosa\n5.1,3.8,1.6,0.2,Iris-setosa\n4.6,3.2,1.4,0.2,Iris-setosa\n"
        "5.3,3.7,1.5,0.2,Iris-setosa\n5.0,3.3,1.4,0.2,Iris-setosa\n"
        "7.0,3.2,4.7,1.4,Iris-versicolor\n6.4,3.2,4.5,1.5,Iris-versicolor\n"
        "6.9,3.1,4.9,1.5,Iris-versicolor\n5.5,2.3,4.0,1.3,Iris-versicolor\n"
        "6.5,2.8,4.6,1.5,Iris-versicolor\n5.7,2.8,4.5,1.3,Iris-versicolor\n"
        "6.3,3.3,4.7,1.6,Iris-versicolor\n4.9,2.4,3.3,1.0,Iris-versicolor\n"
        "6.6,2.9,4.6,1.3,Iris-versicolor\n5.2,2.7,3.9,1.4,Iris-versicolor\n"
        "5.0,2.0,3.5,1.0,Iris-versicolor\n5.9,3.0,4.2,1.5,Iris-versicolor\n"
        "6.0,2.2,4.0,1.0,Iris-versicolor\n6.1,2.9,4.7,1.4,Iris-versicolor\n"
        "5.6,2.9,3.6,1.3,Iris-versicolor\n6.7,3.1,4.4,1.4,Iris-versicolor\n"
        "5.6,3.0,4.5,1.5,Iris-versicolor\n5.8,2.7,4.1,1.0,Iris-versicolor\n"
        "6.2,2.2,4.5,1.5,Iris-versicolor\n5.6,2.5,3.9,1.1,Iris-versicolor\n"
        "5.9,3.2,4.8,1.8,Iris-versicolor\n6.1,2.8,4.0,1.3,Iris-versicolor\n"
        "6.3,2.5,4.9,1.5,Iris-versicolor\n6.1,2.8,4.7,1.2,Iris-versicolor\n"
        "6.4,2.9,4.3,1.3,Iris-versicolor\n6.6,3.0,4.4,1.4,Iris-versicolor\n"
        "6.8,2.8,4.8,1.4,Iris-versicolor\n6.7,3.0,5.0,1.7,Iris-versicolor\n"
        "6.0,2.9,4.5,1.5,Iris-versicolor\n5.7,2.6,3.5,1.0,Iris-versicolor\n"
        "5.5,2.4,3.8,1.1,Iris-versicolor\n5.5,2.4,3.7,1.0,Iris-versicolor\n"
        "5.8,2.7,3.9,1.2,Iris-versicolor\n6.0,2.7,5.1,1.6,Iris-versicolor\n"
        "5.4,3.0,4.5,1.5,Iris-versicolor\n6.0,3.4,4.5,1.6,Iris-versicolor\n"
        "6.7,3.1,4.7,1.5,Iris-versicolor\n6.3,2.3,4.4,1.3,Iris-versicolor\n"
        "5.6,3.0,4.1,1.3,Iris-versicolor\n5.5,2.5,4.0,1.3,Iris-versicolor\n"
        "5.5,2.6,4.4,1.2,Iris-versicolor\n6.1,3.0,4.6,1.4,Iris-versicolor\n"
        "5.8,2.6,4.0,1.2,Iris-versicolor\n5.0,2.3,3.3,1.0,Iris-versicolor\n"
        "5.6,2.7,4.2,1.3,Iris-versicolor\n5.7,3.0,4.2,1.2,Iris-versicolor\n"
        "5.7,2.9,4.2,1.3,Iris-versicolor\n6.2,2.9,4.3,1.3,Iris-versicolor\n"
        "5.1,2.5,3.0,1.1,Iris-versicolor\n5.7,2.8,4.1,1.3,Iris-versicolor\n"
        "6.3,3.3,6.0,2.5,Iris-virginica\n5.8,2.7,5.1,1.9,Iris-virginica\n"
        "7.1,3.0,5.9,2.1,Iris-virginica\n6.3,2.9,5.6,1.8,Iris-virginica\n"
        "6.5,3.0,5.8,2.2,Iris-virginica\n7.6,3.0,6.6,2.1,Iris-virginica\n"
        "4.9,2.5,4.5,1.7,Iris-virginica\n7.3,2.9,6.3,1.8,Iris-virginica\n"
        "6.7,2.5,5.8,1.8,Iris-virginica\n7.2,3.6,6.1,2.5,Iris-virginica\n"
        "6.5,3.2,5.1,2.0,Iris-virginica\n6.4,2.7,5.3,1.9,Iris-virginica\n"
        "6.8,3.0,5.5,2.1,Iris-virginica\n5.7,2.5,5.0,2.0,Iris-virginica\n"
        "5.8,2.8,5.1,2.4,Iris-virginica\n6.4,3.2,5.3,2.3,Iris-virginica\n"
        "6.5,3.0,5.5,1.8,Iris-virginica\n7.7,3.8,6.7,2.2,Iris-virginica\n"
        "7.7,2.6,6.9,2.3,Iris-virginica\n6.0,2.2,5.0,1.5,Iris-virginica\n"
        "6.9,3.2,5.7,2.3,Iris-virginica\n5.6,2.8,4.9,2.0,Iris-virginica\n"
        "7.7,2.8,6.7,2.0,Iris-virginica\n6.3,2.7,4.9,1.8,Iris-virginica\n"
        "6.7,3.3,5.7,2.1,Iris-virginica\n7.2,3.2,6.0,1.8,Iris-virginica\n"
        "6.2,2.8,4.8,1.8,Iris-virginica\n6.1,3.0,4.9,1.8,Iris-virginica\n"
        "6.4,2.8,5.6,2.1,Iris-virginica\n7.2,3.0,5.8,1.6,Iris-virginica\n"
        "7.4,2.8,6.1,1.9,Iris-virginica\n7.9,3.8,6.4,2.0,Iris-virginica\n"
        "6.4,2.8,5.6,2.2,Iris-virginica\n6.3,2.8,5.1,1.5,Iris-virginica\n"
        "6.1,2.6,5.6,1.4,Iris-virginica\n7.7,3.0,6.1,2.3,Iris-virginica\n"
        "6.3,3.4,5.6,2.4,Iris-virginica\n6.4,3.1,5.5,1.8,Iris-virginica\n"
        "6.0,3.0,4.8,1.8,Iris-virginica\n6.9,3.1,5.4,2.1,Iris-virginica\n"
        "6.7,3.1,5.6,2.4,Iris-virginica\n6.9,3.1,5.1,2.3,Iris-virginica\n"
        "5.8,2.7,5.1,1.9,Iris-virginica\n6.8,3.2,5.9,2.3,Iris-virginica\n"
        "6.7,3.3,5.7,2.5,Iris-virginica\n6.7,3.0,5.2,2.3,Iris-virginica\n"
        "6.3,2.5,5.0,1.9,Iris-virginica\n6.5,3.0,5.2,2.0,Iris-virginica\n"
        "6.2,3.4,5.4,2.3,Iris-virginica\n5.9,3.0,5.1,1.8,Iris-virginica\n"
    )
    _raw = np.genfromtxt(io.StringIO(_IRIS_CSV), delimiter=",", dtype=str)
    X_raw   = _raw[:, :4].astype(float)
    _smap   = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
    y_all   = np.array([_smap[s] for s in _raw[:, 4]])

    # Shuffle then split 80/20 (120 train, 30 test)
    _rng    = np.random.default_rng(0)
    _idx    = _rng.permutation(150)
    _n_tr   = 120
    idx_tr  = _idx[:_n_tr]
    idx_te  = _idx[_n_tr:]

    X_tr_raw = X_raw[idx_tr]          # original scale (for scatter display)
    X_te_raw = X_raw[idx_te]
    y_train  = y_all[idx_tr]
    y_test   = y_all[idx_te]

    # Z-score normalise features using training statistics only
    _mu  = X_tr_raw.mean(axis=0)
    _sig = X_tr_raw.std(axis=0) + 1e-8
    X_train = (X_tr_raw - _mu) / _sig   # normalised (for NN input)
    X_test  = (X_te_raw - _mu) / _sig

    feat_names    = ["Sepal length", "Sepal width", "Petal length", "Petal width"]
    species_names = ["Setosa", "Versicolor", "Virginica"]
    species_colors = ["steelblue", "darkorange", "forestgreen"]
    return (
        X_te_raw,
        X_test,
        X_tr_raw,
        X_train,
        feat_names,
        species_colors,
        species_names,
        y_test,
        y_train,
    )


@app.cell
def _(feat_names, mo):
    x_feat_dd = mo.ui.dropdown(feat_names, value="Petal length", label="X-axis feature")
    y_feat_dd = mo.ui.dropdown(feat_names, value="Petal width",  label="Y-axis feature")
    mo.hstack([x_feat_dd, y_feat_dd], gap=2)
    return x_feat_dd, y_feat_dd


@app.cell
def _(mo):
    n_layers_radio  = mo.ui.radio(
        options=["1", "2", "3"], value="1",
        label="Hidden layers (5 nodes each)",
    )
    activation_dd = mo.ui.dropdown(
        ["relu", "sigmoid", "tanh"], value="relu",
        label="Hidden-layer activation",
    )
    epochs_slider = mo.ui.slider(
        10, 500, value=100, step=10,
        label="Training epochs",
    )
    lr_dd = mo.ui.dropdown(
        ["0.001", "0.01", "0.05", "0.1", "0.3", "1.0"], value="0.1",
        label="Learning rate",
    )
    run_btn = mo.ui.run_button(label="▶ Train network")
    mo.vstack([
        mo.hstack([n_layers_radio, activation_dd], gap=3),
        mo.hstack([epochs_slider, lr_dd],          gap=3),
        run_btn,
    ])
    return activation_dd, epochs_slider, lr_dd, n_layers_radio, run_btn


@app.cell
def _(
    X_test,
    X_train,
    activation_dd,
    epochs_slider,
    lr_dd,
    mo,
    n_layers_radio,
    np,
    run_btn,
    y_test,
    y_train,
):
    mo.stop(
        not run_btn.value,
        mo.callout(
            mo.md("Set the architecture and hyperparameters above, then click **▶ Train network**."),
            kind="neutral",
        ),
    )

    # ── Neural-network helpers (all pure numpy) ───────────────────────────────
    def _act(Z, fn):
        if fn == "relu":
            return np.maximum(0.0, Z)
        if fn == "sigmoid":
            return 1.0 / (1.0 + np.exp(-np.clip(Z, -40, 40)))
        return np.tanh(Z)

    def _act_d(Z, A, fn):
        if fn == "relu":
            return (Z > 0).astype(float)
        if fn == "sigmoid":
            return A * (1.0 - A)
        return 1.0 - A ** 2

    def _softmax(Z):
        E = np.exp(Z - Z.max(axis=1, keepdims=True))
        return E / E.sum(axis=1, keepdims=True)

    def _init(n_hidden, fn, seed=42):
        rng = np.random.default_rng(seed)
        sizes = [4] + [5] * n_hidden + [3]
        params = []
        for i in range(len(sizes) - 1):
            scale = np.sqrt(2.0 / sizes[i]) if fn == "relu" else np.sqrt(1.0 / sizes[i])
            params.append([rng.normal(0, scale, (sizes[i], sizes[i + 1])), np.zeros(sizes[i + 1])])
        return params

    def _forward(X, params, fn):
        As, Zs = [X], []
        for k, (W, b) in enumerate(params):
            Z = As[-1] @ W + b
            Zs.append(Z)
            As.append(_act(Z, fn) if k < len(params) - 1 else _softmax(Z))
        return As, Zs

    def _backward(As, Zs, params, Y_oh, fn):
        n = len(Y_oh)
        grads = []
        dZ = (As[-1] - Y_oh) / n
        for i in range(len(params) - 1, -1, -1):
            grads.insert(0, (As[i].T @ dZ, dZ.sum(axis=0)))
            if i > 0:
                dZ = (dZ @ params[i][0].T) * _act_d(Zs[i - 1], As[i], fn)
        return grads

    # ── Training ──────────────────────────────────────────────────────────────
    n_hidden   = int(n_layers_radio.value)
    fn_name    = activation_dd.value
    n_epochs   = epochs_slider.value
    lr         = float(lr_dd.value)
    Y_oh_train = np.eye(3)[y_train]

    params_nn = _init(n_hidden, fn_name)
    loss_hist = []
    for _ in range(n_epochs):
        As_t, Zs_t = _forward(X_train, params_nn, fn_name)
        loss_hist.append(float(-np.mean(np.sum(Y_oh_train * np.log(As_t[-1] + 1e-12), axis=1))))
        grads = _backward(As_t, Zs_t, params_nn, Y_oh_train, fn_name)
        for j, (dW, db) in enumerate(grads):
            params_nn[j][0] -= lr * dW
            params_nn[j][1] -= lr * db

    # ── Predictions ───────────────────────────────────────────────────────────
    def _pred(X):
        As, _ = _forward(X, params_nn, fn_name)
        return np.argmax(As[-1], axis=1)

    train_preds = _pred(X_train)
    test_preds  = _pred(X_test)
    train_acc   = float(np.mean(train_preds == y_train))
    test_acc    = float(np.mean(test_preds  == y_test))
    return test_acc, test_preds, train_acc, train_preds


@app.cell
def _(
    X_te_raw,
    X_tr_raw,
    feat_names,
    plt,
    species_colors,
    species_names,
    test_acc,
    test_preds,
    train_acc,
    train_preds,
    x_feat_dd,
    y_feat_dd,
    y_test,
    y_train,
):
    xi = feat_names.index(x_feat_dd.value)
    yi = feat_names.index(y_feat_dd.value)

    fig_sc, (ax_tr, ax_te) = plt.subplots(1, 2, figsize=(13, 5))

    for ax, Xd, yt, preds, title in [
        (ax_tr, X_tr_raw, y_train, train_preds, f"Train set  (acc = {train_acc:.1%})"),
        (ax_te, X_te_raw, y_test,  test_preds,  f"Test set   (acc = {test_acc:.1%})"),
    ]:
        for cls, (color, name) in enumerate(zip(species_colors, species_names)):
            correct   = (yt == cls) & (preds == cls)
            incorrect = (yt == cls) & (preds != cls)
            if correct.any():
                ax.scatter(Xd[correct,   xi], Xd[correct,   yi],
                           color=color, s=65, alpha=0.85, zorder=3, label=name)
            if incorrect.any():
                ax.scatter(Xd[incorrect, xi], Xd[incorrect, yi],
                           facecolors="none", edgecolors=color,
                           linewidths=1.8, s=100, zorder=3)
        ax.set_xlabel(x_feat_dd.value + " (cm)")
        ax.set_ylabel(y_feat_dd.value + " (cm)")
        ax.set_title(title)

    # Shared legend: species colours + correct/incorrect style
    ax_tr.legend(title="Species (filled=correct,\nopen=wrong)", fontsize=8, title_fontsize=8)
    plt.suptitle("Classification results — scatter view", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig_sc
    return


@app.cell
def _(
    np,
    plt,
    species_names,
    test_acc,
    test_preds,
    train_acc,
    train_preds,
    y_test,
    y_train,
):
    def _conf_matrix(y_true, y_pred):
        cm = np.zeros((3, 3), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1
        return cm

    def _plot_cm(ax, cm, title):
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=max(cm.max(), 1))
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels(species_names, rotation=30, ha="right", fontsize=9)
        ax.set_yticklabels(species_names, fontsize=9)
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("True", fontsize=9)
        ax.set_title(title, fontsize=10)
        thresh = cm.max() / 2.0
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=12, fontweight="bold")
        return im

    cm_tr = _conf_matrix(y_train, train_preds)
    cm_te = _conf_matrix(y_test,  test_preds)

    fig_cm, (ax_cm_tr, ax_cm_te) = plt.subplots(1, 2, figsize=(11, 4))
    _plot_cm(ax_cm_tr, cm_tr, f"Train confusion matrix  (acc = {train_acc:.1%})")
    _plot_cm(ax_cm_te, cm_te, f"Test confusion matrix   (acc = {test_acc:.1%})")
    plt.suptitle("Confusion matrices", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig_cm
    return


@app.cell
def _(mo, test_acc, train_acc):
    _gap = train_acc - test_acc
    if train_acc < 0.7:
        _kind = "info"
        _msg  = f"""
            **Underfitting** — Train accuracy = {train_acc:.1%},  Test accuracy = {test_acc:.1%}

            The model is not complex enough (or not trained long enough) to learn the iris patterns.
            Try more epochs, more hidden layers, or a higher learning rate.
            """
    elif _gap > 0.12:
        _kind = "warn"
        _msg  = f"""
            **Overfitting** — Train accuracy = {train_acc:.1%},  Test accuracy = {test_acc:.1%}

            The model fits the training set well but generalises poorly.
            Try fewer hidden layers, fewer epochs, or a smaller learning rate.
            """
    else:
        _kind = "success"
        _msg  = f"""
            **Good fit** — Train accuracy = {train_acc:.1%},  Test accuracy = {test_acc:.1%}

            The model performs consistently on both sets.
            Iris is a relatively easy classification problem; test accuracy above 90% is typical
            for a well-tuned network.
            """
    mo.callout(mo.md(_msg), kind=_kind)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    It is easy to under-fit the network when you use too few hidden layers, low learning rate or train for too few epochs. As the Iris dataset is quite small and linearly separable it is hard to over-fit a model on this data.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
