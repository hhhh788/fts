"""Utility to fit exchange parameters from CSV dataset."""
import argparse
import csv
import math
import os
from typing import Dict, Iterable, List, Sequence, Tuple

Vector = Tuple[float, float, float]
Matrix = List[List[float]]

# Lattice matrices from the Mathematica script
_A1: Matrix = [
    [1.0, 0.0, 0.0],
    [-0.5, math.sqrt(3.0) / 2.0, 0.0],
    [0.0, 0.0, 0.0],
]

_B1: Matrix = [
    [1.0, 1.0 / math.sqrt(3.0), 0.0],
    [0.0, 2.0 / math.sqrt(3.0), 0.0],
    [0.0, 0.0, 0.0],
]

_MIRROR: Matrix = [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, -1.0],
]

_N1 = 6
_N2 = 6
_N3 = 6

# Base nearest-neighbour vectors
_R1NN_DIFF = (-0.0000000000014, -0.000000000002, 0.4999999999998)
_R1NN_BASE = (-0.0000000000014, 0.99999999999998, 0.49999999999998)

_R2NN_DIFF = (0.9999999999986, -0.000000000002, 0.4999999999998)
_R2NN_BASE = (-0.000000000014, 0.9999999999998, 0.4999999999998)

_R3NN_DIFF = (0.000000000014, -1.000000000002, 0.4999999999998)
_R3NN_BASE = (-0.000000000014, 0.9999999999998, 0.4999999999998)


# --- Basic linear algebra helpers -----------------------------------------------------

def dot(u: Sequence[float], v: Sequence[float]) -> float:
    return sum(ui * vi for ui, vi in zip(u, v))


def cross(u: Sequence[float], v: Sequence[float]) -> Vector:
    return (
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    )


def add_vectors(u: Sequence[float], v: Sequence[float]) -> Vector:
    return (u[0] + v[0], u[1] + v[1], u[2] + v[2])


def scale_vector(s: float, v: Sequence[float]) -> Vector:
    return (s * v[0], s * v[1], s * v[2])


def row_vec_mat_mul(v: Sequence[float], m: Matrix) -> Vector:
    cols = len(m[0])
    return tuple(sum(v[k] * m[k][j] for k in range(len(v))) for j in range(cols))  # type: ignore[return-value]


def mat_vec_mul(m: Matrix, v: Sequence[float]) -> Vector:
    return tuple(sum(m[i][k] * v[k] for k in range(len(v))) for i in range(len(m)))  # type: ignore[return-value]


def mat_mul(a: Matrix, b: Matrix) -> Matrix:
    rows, shared, cols = len(a), len(a[0]), len(b[0])
    return [
        [sum(a[i][k] * b[k][j] for k in range(shared)) for j in range(cols)]
        for i in range(rows)
    ]


def transpose(m: Matrix) -> Matrix:
    return [[m[i][j] for i in range(len(m))] for j in range(len(m[0]))]


def matrix_add(a: Matrix, b: Matrix) -> Matrix:
    return [[a[i][j] + b[i][j] for j in range(len(a[0]))] for i in range(len(a))]


def matrix_scale(s: float, m: Matrix) -> Matrix:
    return [[s * m[i][j] for j in range(len(m[0]))] for i in range(len(m))]


def outer(u: Sequence[float], v: Sequence[float]) -> Matrix:
    return [[ui * vj for vj in v] for ui in u]


def rotation_matrix_z(theta: float) -> Matrix:
    c, s = math.cos(theta), math.sin(theta)
    return [
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0],
    ]


def transform_matrix(m: Matrix, transform: Matrix) -> Matrix:
    return mat_mul(mat_mul(transform, m), transpose(transform))


# --- Model specific helper routines ---------------------------------------------------

def initial_nn_vectors() -> Tuple[List[Vector], List[Vector], List[Vector]]:
    def subtract(u: Sequence[float], v: Sequence[float]) -> Vector:
        return (u[0] - v[0], u[1] - v[1], u[2] - v[2])

    r1nn = row_vec_mat_mul(subtract(_R1NN_DIFF, _R1NN_BASE), _A1)
    r2nn = row_vec_mat_mul(subtract(_R2NN_DIFF, _R2NN_BASE), _A1)
    r3nn = row_vec_mat_mul(subtract(_R3NN_DIFF, _R3NN_BASE), _A1)

    def rotated_list(base: Vector, count: int) -> List[Vector]:
        return [mat_vec_mul(rotation_matrix_z(i * 2.0 * math.pi / count), base) for i in range(count)]

    return rotated_list(r1nn, _N1), rotated_list(r2nn, _N2), rotated_list(r3nn, _N3)


_R1NNS, _R2NNS, _R3NNS = initial_nn_vectors()


# s-tensors that define bilinear couplings
_S1S1 = [
    [[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0]],
]

_S1S21 = [
    [
        [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.346410161522],
        [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.346410161522],
        [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.599999999998],
    ],
    [
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, -1.15470053843],
    ],
    [
        [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.577350269186],
        [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.577350269186],
    ],
    [[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0]],
]

_S1S22 = [
    [
        [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0],
        [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
    ]
]

_S1S23 = [
    [
        [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0],
        [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
    ]
]


def build_base_matrices(spec: Iterable[Iterable[Sequence[float]]]) -> List[Matrix]:
    matrices: List[Matrix] = []
    for group in spec:
        accum = [[0.0, 0.0, 0.0] for _ in range(3)]
        for entry in group:
            v1 = entry[0:3]
            v2 = entry[3:6]
            coeff = entry[6]
            term = outer(v1, v2)
            for i in range(3):
                for j in range(3):
                    accum[i][j] += coeff * term[i][j]
        matrices.append(accum)
    return matrices


_BASE_J0 = build_base_matrices(_S1S1)
_BASE_J1 = build_base_matrices(_S1S21)
_BASE_J2 = build_base_matrices(_S1S22)
_BASE_J3 = build_base_matrices(_S1S23)


# Precompute symmetry-equivalent matrices for J1, J2, J3

def build_transformed_matrices(base_matrices: List[Matrix]) -> List[List[Matrix]]:
    transformed: List[List[Matrix]] = []
    for base in base_matrices:
        matrices_for_param: List[Matrix] = []
        for idx in range(6):
            theta = idx * math.pi / 3.0
            rotation = rotation_matrix_z(theta)
            if idx % 2 == 1:
                transform = mat_mul(rotation, _MIRROR)
            else:
                transform = rotation
            matrices_for_param.append(transform_matrix(base, transform))
        transformed.append(matrices_for_param)
    return transformed


_TRANSFORMED_J1 = build_transformed_matrices(_BASE_J1)
_TRANSFORMED_J2 = build_transformed_matrices(_BASE_J2)
_TRANSFORMED_J3 = build_transformed_matrices(_BASE_J3)


# --- Energy contributions -------------------------------------------------------------

def sq_vector(sp: Vector, rn: Vector, qvec: Vector, nvec: Vector) -> Vector:
    phase = dot(rn, qvec)
    return add_vectors(scale_vector(math.cos(phase), sp), scale_vector(math.sin(phase), cross(nvec, sp)))


def j0_contribution(sp: Vector) -> float:
    matrix = _BASE_J0[0]
    return dot(sp, mat_vec_mul(matrix, sp))


def jn_contribution(sp: Vector, qvec: Vector, nvec: Vector, r_list: List[Vector], transformed_mats: List[List[Matrix]]) -> List[float]:
    # The order of transformed matrices matches the order of parameters.
    sq_vectors = [sq_vector(sp, rn, qvec, nvec) for rn in r_list]
    contributions: List[float] = []
    for param_idx, matrices_for_param in enumerate(transformed_mats):
        total = 0.0
        for mat, sq_vec in zip(matrices_for_param, sq_vectors):
            total += dot(sp, mat_vec_mul(mat, sq_vec))
        contributions.append(total)
    return contributions


def sq1nn_vectors(sp: Vector, qvec: Vector, nvec: Vector) -> List[Vector]:
    return [sq_vector(sp, rn, qvec, nvec) for rn in _R1NNS]


def j14_single(sp: Vector, sq_vec: Vector) -> float:
    spx, spy, spz = sp
    sqx, sqy, sqz = sq_vec
    return (
        spx * spx * sqx * sqx
        + 0.5 * spx * spx * sqy * sqy
        - 0.5 * spx * spx
        + spx * spy * sqx * sqy
        + spx * spz * sqx * sqz
        + 0.5 * spy * spy * sqx * sqx
        + spy * spy * sqy * sqy
        - 0.5 * spy * spy
        + spy * spz * sqy * sqz
        - 0.5 * sqx * sqx
        - 0.5 * sqy * sqy
    )


def j14_contributions(sp: Vector, qvec: Vector, nvec: Vector) -> List[float]:
    return [j14_single(sp, sq_vec) for sq_vec in sq1nn_vectors(sp, qvec, nvec)]


# --- Least squares solver ------------------------------------------------------------

def normal_equations(features: List[List[float]], targets: List[float]) -> Tuple[List[List[float]], List[float]]:
    param_count = len(features[0])
    xtx = [[0.0 for _ in range(param_count)] for _ in range(param_count)]
    xty = [0.0 for _ in range(param_count)]
    for row, target in zip(features, targets):
        for i in range(param_count):
            xty[i] += row[i] * target
            for j in range(param_count):
                xtx[i][j] += row[i] * row[j]
    return xtx, xty


def solve_linear_system(a: List[List[float]], b: List[float]) -> List[float]:
    # Gaussian elimination with partial pivoting
    n = len(a)
    # Augment matrix
    for i in range(n):
        a[i] = a[i][:] + [b[i]]

    for col in range(n):
        # pivot selection
        pivot_row = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[pivot_row][col]) < 1e-12:
            raise ValueError("Matrix is singular or ill-conditioned.")
        if pivot_row != col:
            a[col], a[pivot_row] = a[pivot_row], a[col]
        pivot = a[col][col]
        # normalize row
        for j in range(col, n + 1):
            a[col][j] /= pivot
        # eliminate
        for row in range(n):
            if row == col:
                continue
            factor = a[row][col]
            if abs(factor) < 1e-12:
                continue
            for j in range(col, n + 1):
                a[row][j] -= factor * a[col][j]
    return [a[i][n] for i in range(n)]


# --- Feature construction -------------------------------------------------------------

def compute_features(sp: Vector, qvec: Vector, nvec: Vector) -> List[float]:
    q_cart = row_vec_mat_mul(qvec, _B1)
    q_cart = (2.0 * math.pi * q_cart[0], 2.0 * math.pi * q_cart[1], 2.0 * math.pi * q_cart[2])

    features: List[float] = [1.0]  # E0 contribution

    # J0 parameters
    features.append(j0_contribution(sp))

    # J1 parameters (four of them)
    features.extend(jn_contribution(sp, q_cart, nvec, _R1NNS, _TRANSFORMED_J1))

    # J2 parameters (single entry in spec)
    features.extend(jn_contribution(sp, q_cart, nvec, _R2NNS, _TRANSFORMED_J2))

    # J3 parameters (single entry in spec)
    features.extend(jn_contribution(sp, q_cart, nvec, _R3NNS, _TRANSFORMED_J3))

    # B1 parameters (one per nearest neighbour)
    features.extend(j14_contributions(sp, q_cart, nvec))

    return features


# --- Dataset handling -----------------------------------------------------------------

def parse_float(value: str) -> float:
    value = value.strip()
    if value.endswith(";"):
        value = value[:-1]
    if not value:
        raise ValueError("Encountered empty numeric field")
    return float(value)


def load_dataset(path: str) -> Tuple[List[Dict[str, float]], List[str]]:
    with open(path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: List[Dict[str, float]] = []
        for raw_row in reader:
            row: Dict[str, float] = {}
            for key, value in raw_row.items():
                if key is None:
                    continue
                key = key.strip()
                if not key:
                    continue
                row[key] = parse_float(value)
            rows.append(row)
        return rows, reader.fieldnames if reader.fieldnames is not None else []


# --- Main CLI -------------------------------------------------------------------------

def determine_sp_vector(row: Dict[str, float], args: argparse.Namespace) -> Vector:
    if {"Spx", "Spy", "Spz"}.issubset(row):
        return (row["Spx"], row["Spy"], row["Spz"])
    if {"Sqx", "Sqy", "Sqz"}.issubset(row):
        return (row["Sqx"], row["Sqy"], row["Sqz"])
    if args.sp is not None:
        return args.sp
    raise ValueError("Unable to determine (Spx, Spy, Spz); provide --spx/--spy/--spz.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit exchange parameters to energy data")
    parser.add_argument("data", help="Path to the CSV dataset (e.g. ../data_sets/fit_data.csv)")
    parser.add_argument("--output", help="Optional path to save fitted parameters as CSV")
    parser.add_argument("--spx", type=float, help="Override Spx component if not present in data")
    parser.add_argument("--spy", type=float, help="Override Spy component if not present in data")
    parser.add_argument("--spz", type=float, help="Override Spz component if not present in data")
    parser.add_argument("--no-header", action="store_true", help="Treat the CSV file as header-less")
    parser.add_argument("--limit", type=int, help="Use only the first N rows")
    args = parser.parse_args()
    if args.no_header:
        raise NotImplementedError("Header-less CSV files are not supported in this implementation.")
    if args.spx is not None or args.spy is not None or args.spz is not None:
        if None in (args.spx, args.spy, args.spz):
            parser.error("All of --spx, --spy, and --spz must be provided together.")
        args.sp = (args.spx, args.spy, args.spz)
    else:
        args.sp = None
    return args


def main() -> None:
    args = parse_args()
    dataset, header = load_dataset(args.data)
    if not dataset:
        raise ValueError("Dataset is empty.")

    required_fields = {"qx", "qy", "qz", "nx", "ny", "nz", "E"}
    missing = required_fields - set(dataset[0].keys())
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    features: List[List[float]] = []
    targets: List[float] = []

    for idx, row in enumerate(dataset):
        if args.limit is not None and idx >= args.limit:
            break
        sp = determine_sp_vector(row, args)
        qvec = (row["qx"], row["qy"], row["qz"])
        nvec = (row["nx"], row["ny"], row["nz"])
        features.append(compute_features(sp, qvec, nvec))
        targets.append(row["E"])

    xtx, xty = normal_equations(features, targets)
    params = solve_linear_system(xtx, xty)

    names = [
        "E0",
        "J0_1",
        "J1_1",
        "J1_2",
        "J1_3",
        "J1_4",
        "J2_1",
        "J3_1",
        "B1_1",
        "B1_2",
        "B1_3",
        "B1_4",
        "B1_5",
        "B1_6",
    ]

    if len(params) != len(names):
        raise RuntimeError("Parameter vector length mismatch.")

    print("Fitted parameters:\n")
    width = max(len(name) for name in names)
    for name, value in zip(names, params):
        print(f"{name:<{width}} = {value: .12e}")

    # Evaluate residual statistics
    residuals = []
    for feats, actual in zip(features, targets):
        predicted = sum(p * f for p, f in zip(params, feats))
        residuals.append(actual - predicted)
    mse = sum(r * r for r in residuals) / len(residuals)
    rmse = math.sqrt(mse)
    print(f"\nRMSE: {rmse:.6e} ({len(residuals)} samples)")

    if args.output:
        output_path = args.output
        with open(output_path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["parameter", "value"])
            writer.writerows(zip(names, params))
        print(f"\nSaved parameters to {os.path.abspath(output_path)}")


if __name__ == "__main__":
    main()
