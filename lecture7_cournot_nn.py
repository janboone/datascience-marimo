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
    # Cournot Duopoly: Neural Network Equilibrium Solver

    Up till now we have used neural networks as function approximators in supervised-learning settings.  Here we use them in a different way: the economic model itself supplies the "training signal".  The network is trained to output quantities that satisfy the equilibrium first-order conditions, rather than to match "$y$ values" or target labels.

    ## Economic model

    To illustrate this idea, consider two firms producing a homogeneous good. The inverse demand function is

    $$p(x_1, x_2) = 1 - x_1 - x_2,$$

    and firm $i$ has quadratic cost $C_i(x_i) = \tfrac{1}{2} c_i x_i^2$.  Firm $i$ maximises profit

    $$\pi_i = p(x_1, x_2)\, x_i - \tfrac{1}{2} c_i x_i^2.$$

    At the Nash equilibrium each firm's first-order condition (FOC) holds simultaneously:

    $$r_i(x_i,x_j) \equiv \frac{\partial \pi_i}{\partial x_i} = 1 - 2x_i - x_j - c_i x_i = 0.$$

    In this simple example we can actually solve the two FOCs gives the closed-form equilibrium

    $$x_1^* = \frac{1 + c_2}{(2+c_1)(2+c_2)-1}, \qquad x_2^* = \frac{1 + c_1}{(2+c_1)(2+c_2)-1}.$$

    This allows us to check the fit of the network against the exact analytical solution.For example, if $c_1=c_2=1$, the analytical solution is

    $$x_1^*=x_2^*=\frac{2}{(3)(3)-1}=\frac{1}{4}.$$

    At this point the FOC residuals are zero:

    $$1-2(1/4)-(1/4)-1(1/4)=0.$$

    The neural network looks for the coefficients (weights) that make the two residuals equal to zero for any given pair of cost parameters $(c_1,c_2)$.

    To see the idea clearly, start from the residual equations themselves. Given a candidate output
    $\hat x_1, \hat x_2$ from the network, define

    $$
    r_1 = 1 - 2\hat x_1 - \hat x_2 - c_1 \hat x_1,
    $$

    and

    $$
    r_2 = 1 - \hat x_1 - 2\hat x_2 - c_2 \hat x_2.
    $$

    The network is trained to make these residuals small across many randomly drawn values of
    $(c_1,c_2)$.

    An advantage of using neural networks to solve for equilibrium is that TensorFlow (or PyTorch, JAX, or another automatic-differentiation framework) can derive the first order conditions analytically. This makes it possible to write the profit functions directly and let TensorFlow compute the residuals rather than coding them yourself.

    ## What the neural network does

    Instead of solving the linear system analytically, we train a small network
    $(c_1, c_2) \mapsto (x_1, x_2)$ so that **both FOCs are as close to zero as possible**. In words, the input layer is the pair of cost parameters $(c_1,c_2)$ and the output layer is the pair of equilibrium quantities $(\hat x_1,\hat x_2)$. The loss function is the mean squared deviation from the optimality conditions:

    $$\mathcal{L} = \frac{1}{n}\sum_{i=1}^{n}\bigl[r_1^{(i)2} + r_2^{(i)2}\bigr], \quad
    r_k^{(i)} = \frac{\partial \pi_k}{\partial x_k}\Bigg|_{x = \hat x^{(i)}}.$$

    To be clear, this is not a supervised learning problem. That is, we do **not** simulate training examples of the form `cost parameters -> known equilibrium quantities` and train the network on these simulated data points. Indeed, in a real application we do not know the exact equilibrium quantities in advance. Instead, the network generates candidate quantities and is penalised when those quantities do not satisfy the firms' optimality conditions.
    This is the main difference: the network learns equilibrium relations, not patterns in observed data.

    But the general idea is the same as before: we are looking for the network weights to minimize a loss function.

    As we are running this code in a browser, it is hard to get a good fit. In the lecture we will use TensorFlow and more elaborate networks. But the example does illustrate the principle of solving for an equilibrium using a neural network.
    """)
    return


@app.cell
def _(mo):
    c2_slider = mo.ui.slider(0.1, 3.0, value=1.5, step=0.1,
                              label="Firm 2 cost $c_2$ (fixes the plot's baseline)")
    epochs_slider = mo.ui.slider(200, 1000, value=500, step=100,
                                  label="Training iterations")
    run_btn = mo.ui.run_button(label="▶ Train neural network")
    mo.vstack([
        mo.hstack([c2_slider, epochs_slider], gap=3),
        run_btn,
    ])
    return c2_slider, epochs_slider, run_btn


@app.cell
def _(epochs_slider, mo, np, run_btn):
    mo.stop(
        not run_btn.value,
        mo.callout(
            mo.md("Set the sliders above, then click **▶ Train neural network**."),
            kind="neutral",
        ),
    )

    _H = 16   # nodes per hidden layer
    _rng = np.random.default_rng(42)

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))

    def _softplus(z):
        return np.log1p(np.exp(np.clip(z, -30, 30)))

    def _fwd(cc, W1, b1, W2, b2, W3, b3):
        Z1 = cc @ W1 + b1
        A1 = np.tanh(Z1)
        Z2 = A1 @ W2 + b2
        A2 = np.tanh(Z2)
        Z3 = A2 @ W3 + b3
        out = _softplus(Z3)
        return out, (Z1, A1, Z2, A2, Z3)

    def _step(cc, W1, b1, W2, b2, W3, b3):
        n = len(cc)
        c1, c2_col = cc[:, 0], cc[:, 1]
        out, (Z1, A1, Z2, A2, Z3) = _fwd(cc, W1, b1, W2, b2, W3, b3)
        x1, x2 = out[:, 0], out[:, 1]

        # Analytical FOC residuals: ∂πi/∂xi = 1 - (2+ci)*xi - xj
        r1 = 1.0 - (2.0 + c1) * x1 - x2
        r2 = 1.0 - x1 - (2.0 + c2_col) * x2
        loss = float(np.mean(r1**2 + r2**2))

        # ∂L/∂(x1,x2) per sample
        dx1 = (2.0 / n) * (r1 * (-(2.0 + c1)) + r2 * (-1.0))
        dx2 = (2.0 / n) * (r1 * (-1.0)         + r2 * (-(2.0 + c2_col)))
        dout = np.column_stack([dx1, dx2])

        # Backprop through softplus → tanh → tanh
        dZ3 = dout * _sigmoid(Z3)
        gW3 = A2.T @ dZ3
        gb3 = dZ3.sum(0)
        dA2 = dZ3 @ W3.T

        dZ2 = dA2 * (1.0 - A2**2)
        gW2 = A1.T @ dZ2
        gb2 = dZ2.sum(0)
        dA1 = dZ2 @ W2.T

        dZ1 = dA1 * (1.0 - A1**2)
        gW1 = cc.T @ dZ1
        gb1 = dZ1.sum(0)

        return loss, [gW1, gb1, gW2, gb2, gW3, gb3]

    # ── Initialise parameters ─────────────────────────────────────────────────
    W1 = _rng.normal(0, np.sqrt(2.0 / 2),   (2,   _H))
    b1 = np.zeros(_H)
    W2 = _rng.normal(0, np.sqrt(2.0 / _H),  (_H, _H))
    b2 = np.zeros(_H)
    W3 = _rng.normal(0, np.sqrt(2.0 / _H),  (_H,  2))
    b3 = np.full(2, 0.5)   # positive bias → softplus output starts positive

    _params = [W1, b1, W2, b2, W3, b3]

    # ── Adam state ────────────────────────────────────────────────────────────
    _ms = [np.zeros_like(p) for p in _params]
    _vs = [np.zeros_like(p) for p in _params]
    _lr, _beta1, _beta2, _eps_adam = 1e-3, 0.9, 0.999, 1e-8
    _batch = 256

    loss_history = []
    for _t in range(1, epochs_slider.value + 1):
        _cc = _rng.uniform(0.1, 3.0, (_batch, 2))
        _loss, _grads = _step(_cc, *_params)
        loss_history.append(_loss)
        for _i, _g in enumerate(_grads):
            _ms[_i] = _beta1 * _ms[_i] + (1 - _beta1) * _g
            _vs[_i] = _beta2 * _vs[_i] + (1 - _beta2) * _g**2
            _mh = _ms[_i] / (1 - _beta1**_t)
            _vh = _vs[_i] / (1 - _beta2**_t)
            _params[_i] -= _lr * _mh / (np.sqrt(_vh) + _eps_adam)

    W1, b1, W2, b2, W3, b3 = _params
    final_loss = loss_history[-1]
    return W1, W2, W3, b1, b2, b3, final_loss, loss_history


@app.cell
def _(W1, W2, W3, b1, b2, b3, c2_slider, final_loss, loss_history, np, plt):
    def _softplus(z):
        return np.log1p(np.exp(np.clip(z, -30, 30)))

    def _predict(cc):
        A1 = np.tanh(cc @ W1 + b1)
        A2 = np.tanh(A1 @ W2 + b2)
        return _softplus(A2 @ W3 + b3)

    def _closed_form(c1_vals, c2_val):
        denom = (2.0 + c1_vals) * (2.0 + c2_val) - 1.0
        return (1.0 + c2_val) / denom, (1.0 + c1_vals) / denom

    c2_val = c2_slider.value
    c1_range = np.linspace(0.1, 3.0, 120)
    cc_plot  = np.column_stack([c1_range, np.full(120, c2_val)])

    out_nn = _predict(cc_plot)
    x1_nn, x2_nn = out_nn[:, 0], out_nn[:, 1]
    x1_cf, x2_cf = _closed_form(c1_range, c2_val)
    p_nn = 1.0 - x1_nn - x2_nn
    p_cf = 1.0 - x1_cf - x2_cf

    _fig, (_ax1, _ax2, _ax3) = plt.subplots(1, 3, figsize=(13, 4))

    # Loss curve
    _ax1.plot(loss_history, color="steelblue", lw=1.6)
    _ax1.set_xlabel("Iteration")
    _ax1.set_ylabel("FOC loss")
    _ax1.set_title("Training loss")

    # x1(c1) comparison
    _ax2.plot(c1_range, x1_cf, color="black",    lw=2.2, label="Closed form $x_1^*$")
    _ax2.plot(c1_range, x1_nn, color="steelblue", lw=2.0, ls="--", label="Neural network $\\hat x_1$")
    _ax2.set_xlabel("$c_1$")
    _ax2.set_ylabel("Output $x_1$")
    _ax2.set_title(f"Firm 1 equilibrium output  ($c_2 = {c2_val:.1f}$)")
    _ax2.legend(fontsize=9)

    # p(c1) comparison
    _ax3.plot(c1_range, p_cf, color="black",    lw=2.2, label="Closed form $p^*$")
    _ax3.plot(c1_range, p_nn, color="firebrick", lw=2.0, ls="--", label="Neural network $\\hat p$")
    _ax3.set_xlabel("$c_1$")
    _ax3.set_ylabel("Price $p$")
    _ax3.set_title("Equilibrium price")
    _ax3.legend(fontsize=9)

    plt.suptitle(f"Final FOC loss: {final_loss:.5f}", fontsize=10)
    plt.tight_layout()
    _fig
    return


@app.cell
def _(final_loss, mo):
    if final_loss < 0.002:
        _kind = "success"
        _msg = f"""
            **Excellent fit** — final FOC loss = {final_loss:.5f}

            The network has learned the Cournot equilibrium mapping well.
            Its predicted quantities closely match the analytical solution across
            the full range of cost parameters.
            """
    elif final_loss < 0.01:
        _kind = "neutral"
        _msg = f"""
            **Good fit** — final FOC loss = {final_loss:.5f}

            The network approximates the equilibrium reasonably well.
            Try increasing the number of training iterations to reduce the loss further.
            """
    else:
        _kind = "warn"
        _msg = f"""
            **High loss** — final FOC loss = {final_loss:.5f}

            The network has not converged well. Try more training iterations.
            """
    mo.callout(mo.md(_msg), kind=_kind)
    return


if __name__ == "__main__":
    app.run()
