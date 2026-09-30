# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy",
# ]
# ///

import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np

    return mo, np


@app.cell
def _(mo):
    mo.md(r"""
    # Interactive Matrix Operations

    Enter values in the two $3 \times 3$ matrices $A$ and $B$ below via:

    - (implicit) sliders to change values or
    - double click the number and type the value you want.

    The app computes several matrix operations in real time. Addition and subtraction are obvious, matrix multiplication you have (hopefully) seen before (if not, check out the [wikipedia page](https://en.wikipedia.org/wiki/Matrix_multiplication)), $A^T$ "reverses" rows and columns and the inverse $A^{-1}$ is defined as $A^{-1} A = I$, where $I$ is the identity matrix. The inverse only exists if the matrix has full rank, that is all rows are independent. A quick way to check whether a matrix is invertible is to compute its determinant: if the determinant is nonzero, the inverse exists.

    **Try this:** make one row of matrix $A$ equal to a multiple of another row; the determinant will drop to zero and the inverse becomes undefined.
    """)
    return


@app.cell
def _(mo):
    A_ui = mo.ui.matrix(
        [[2, 1, 0], [1, 3, 1], [0, 1, 2]],
        label="Matrix A",
    )
    B_ui = mo.ui.matrix(
        [[1, 0, 0], [0, 2, 0], [0, 0, 3]],
        label="Matrix B",
    )
    mo.hstack([A_ui, B_ui], gap=1)
    return A_ui, B_ui


@app.cell
def _(A_ui, B_ui, np):
    A = np.array(A_ui.value, dtype=float)
    B = np.array(B_ui.value, dtype=float)

    rank_A = int(np.linalg.matrix_rank(A))
    det_A  = float(np.linalg.det(A))
    full_rank_A = (rank_A == 3)
    A_inv  = np.linalg.inv(A) if full_rank_A else None
    return A, A_inv, B, det_A, full_rank_A, rank_A


@app.cell
def _(A, A_inv, B, det_A, full_rank_A, mo, np, rank_A):
    def fmt(arr, label):
        s = np.array2string(arr, precision=3, suppress_small=True,
                            floatmode="fixed")
        return f"**{label}**\n```\n{s}\n```"

    if full_rank_A:
        inv_block = fmt(A_inv, "A⁻¹  (inverse of A)")
    else:
        inv_block = (
            f"**A⁻¹ — not defined**\n\n"
            f"Rank of A = {rank_A} < 3.  "
            f"At least one row is a linear combination of the others, "
            f"so det(A) ≈ 0 and A cannot be inverted."
        )

    mo.vstack([
        mo.hstack([
            mo.md(fmt(A + B, "A + B")),
            mo.md(fmt(A - B, "A − B")),
        ], gap=3),
        mo.hstack([
            mo.md(fmt(A @ B, "A @ B  (matrix product)")),
            mo.md(fmt(B @ A, "B @ A  (note: ≠ A @ B in general)")),
        ], gap=3),
        mo.hstack([
            mo.md(fmt(A.T, "Aᵀ  (transpose)")),
            mo.md(
                f"**Rank:** {rank_A} / 3   |   "
                f"**det(A):** {det_A:.4f}\n\n{inv_block}"
            ),
        ], gap=3),
    ])
    return


@app.cell
def _(A_inv, det_A, full_rank_A, mo, np, rank_A):
    if full_rank_A:
        # Verify A @ A_inv ≈ I
        I_approx = np.linalg.norm(np.eye(3) - np.array(A_inv) @ np.eye(3))
        kind = "success"
        body = f"""
            **Matrix A has full rank ({rank_A}/3)** — the inverse exists.

            - det(A) = **{det_A:.4f}** (non-zero confirms invertibility)
            - A square matrix is invertible if and only if its rows (and columns)
              are linearly independent — equivalently, det(A) ≠ 0 and rank = n.

            Change A so that one row is a multiple of another to see the inverse
            become undefined.
            """
    else:
        kind = "warn"
        body = f"""
            **Matrix A is rank-deficient (rank = {rank_A}/3)** — the inverse does not exist.

            - det(A) = **{det_A:.4f}** (zero or near-zero)
            - The system $A x = b$ may have no solution or infinitely many.
            - In statistics this corresponds to **perfect multicollinearity**:
              one predictor is a linear combination of the others, making OLS
              ill-defined.
            """

    mo.callout(mo.md(body), kind=kind)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If the inverse of matrix $A$ exists, we can solve a system of equations $A x = b$ as $x = A^{-1}b$.

    In previous lectures we have seen this in the context of OLS, where the first order condition (of minimizing the sum of squared residuals) can be written as $X^\top X \hat\beta = X^\top y$, giving
    \[
    \hat\beta = (X^\top X)^{-1}X^\top y
    \]
    when $X^\top X$ is invertible. In case of multicollinearity the matrix $X^\top X$ is not invertible, so the inverse does not exist. In this case your statistical software will give a warning or even an error.
    """)
    return


if __name__ == "__main__":
    app.run()
