"""Random matrix generators backing the rotation, Hadamard, and JL-sketch preconditioners.

Produces the numpy-side matrices that
:mod:`veloxquant_mlx.preconditioners.rotation` and
:mod:`veloxquant_mlx.preconditioners.jl_sketch` wrap as MLX arrays:
Haar-distributed orthogonal rotations via QR decomposition
(``make_rotation_matrix``), i.i.d. Gaussian JL projection matrices
(``make_jl_matrix``), and the ±1 diagonal for the randomized Hadamard
transform (``make_hadamard_diagonal``), plus a compatibility gate
(``is_hadamard_compatible``) for ``mx.hadamard_transform``'s dimension
constraint.
"""

from __future__ import annotations

import numpy as np

# mx.hadamard_transform requires d = m * 2^k where m is in this set.
# m=20 and m=28 use a Paley construction that is orthogonal but NOT symmetric,
# so H(H(x)) != x; their multiples (40, 56, 80, 112, ...) have the same property
# and silently corrupt the round-trip in apply_inverse. Drop them so those sizes
# fall back to the QR rotation. See issue #608.
_HADAMARD_VALID_M = {1, 12}


def is_hadamard_compatible(d: int) -> bool:
    """Return True if d is supported by mx.hadamard_transform AND self-inverse.

    MLX supports d = m * 2^k where m in {1, 12, 20, 28} and k >= 0.  This
    gate only admits m in {1, 12}: the m=20 and m=28 Paley-construction
    Hadamards are orthogonal but NOT symmetric, so H^T != H and H(H(x)) != x.
    Passing those sizes to HadamardPreconditioner.apply_inverse applies H a
    second time instead of H^T, corrupting the round-trip with relative error
    ~1.7 (see issue #608).  The affected sizes (40, 56, 80, 112, 160, 224, …)
    fall back to the QR rotation.

    Additionally, bare non-trivial m values (d == 12 exactly, i.e. k=0) are
    rejected because MLX <= 0.32.0 crashes on them during Metal shader
    compilation (#67).

    k=0 with m=1 (d=1) is unaffected (identity transform).  All powers of 2
    work (m=1, k>=0).  Examples that do NOT work: 576=9*64, 192=3*64.
    """
    if d < 1:
        return False
    for m in _HADAMARD_VALID_M:
        if d % m == 0:
            remainder = d // m
            if remainder & (remainder - 1) == 0:  # power of 2
                if m != 1 and remainder == 1:
                    continue
                return True
    return False


def make_hadamard_diagonal(d: int, seed: int = 42) -> np.ndarray:
    """Generate a random ±1 diagonal vector for randomized Hadamard transform.

    The randomized Hadamard preconditioner applies H @ diag(D) where H is the
    Walsh-Hadamard matrix and D is this ±1 diagonal. Only D needs to be stored
    (d scalars vs d² for QR rotation).

    d must satisfy mx.hadamard_transform's constraint: d = m * 2^k where
    m in {1, 12, 20, 28}. All powers of 2 (64, 128, 256, ...) satisfy this.

    Args:
        d: Vector dimension.
        seed: NumPy random seed for reproducibility.

    Returns:
        Float32 array of shape (d,) with entries in {-1, +1}.
    """
    if d < 1:
        raise ValueError(f"make_hadamard_diagonal: d must be >= 1, got {d}")
    rng = np.random.default_rng(seed)
    return rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=d)


def make_rotation_matrix(d: int, seed: int = 42) -> np.ndarray:
    """Generate a d×d random orthogonal rotation matrix via QR decomposition.

    Draws G ~ N(0, 1)^{d×d} and returns Q from G = QR (economy QR).
    The resulting Q is a Haar-distributed orthogonal matrix.

    Args:
        d: Matrix dimension (must be >= 1).
        seed: NumPy random seed for reproducibility.

    Returns:
        Float64 array of shape (d, d) with orthonormal rows (Q @ Q.T ≈ I).

    Raises:
        ValueError: If d < 1.
    """
    if d < 1:
        raise ValueError(f"make_rotation_matrix: d must be >= 1, got {d}")
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((d, d)).astype(np.float64)
    Q, _ = np.linalg.qr(G)
    return Q.astype(np.float64)


def make_jl_matrix(d: int, m: int, seed: int = 42) -> np.ndarray:
    """Generate an m×d Gaussian JL projection matrix.

    Each row is drawn i.i.d. from N(0, I_d). This is the correct
    construction for the QJL sign-based inner product estimator to be
    unbiased:

        E[sqrt(pi/2)/m * ||k|| * sum_i sign(s_i·k)(s_i·q)] = <q, k>

    Unlike orthogonal JL, Gaussian JL allows m > d.

    Args:
        d: Input dimension (must be >= 1).
        m: Sketch dimension (must be >= 1).
        seed: NumPy random seed for reproducibility.

    Returns:
        Float64 array of shape (m, d) with i.i.d. N(0,1) entries.

    Raises:
        ValueError: If d < 1 or m < 1.
    """
    if d < 1:
        raise ValueError(f"make_jl_matrix: d must be >= 1, got {d}")
    if m < 1:
        raise ValueError(f"make_jl_matrix: m must be >= 1, got {m}")

    rng = np.random.default_rng(seed + 1)
    return rng.standard_normal((m, d)).astype(np.float64)
