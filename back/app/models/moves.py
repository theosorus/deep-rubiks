from __future__ import annotations
from abc import ABC, abstractmethod
import random
import numpy as np

from models.faces import Face              


# ─────────────────────────── utilitaire interne ─────────────────────────
def _rot(face: np.ndarray, k: int = 1) -> np.ndarray:
    """Tourne une face d’un quart horaire (`k` > 0) ou antihoraire (`k` < 0)."""
    return np.rot90(face, -k)              # np.rot90 = sens antihoraire ⇒ on inverse le signe


# ─────────────────────────── classe de base ────────────────────────────
class Move(ABC):
    @abstractmethod
    def make_move_on_cube(self, cube: "Cube") -> None: ...


# ────────────────────────── rotations de faces ──────────────────────────
class U(Move):
    def make_move_on_cube(self, cube):
        s = cube.state
        s[Face.U] = _rot(s[Face.U])
        f, r, b, l = s[Face.F][0].copy(), s[Face.R][0].copy(), s[Face.B][0].copy(), s[Face.L][0].copy()
        s[Face.R][0], s[Face.B][0], s[Face.L][0], s[Face.F][0] = f, r, b, l


class U_PRIME(Move):
    def make_move_on_cube(self, cube): [U().make_move_on_cube(cube) for _ in range(3)]


class U2(Move):
    def make_move_on_cube(self, cube): [U().make_move_on_cube(cube) for _ in range(2)]


class D(Move):
    def make_move_on_cube(self, cube):
        s = cube.state
        s[Face.D] = _rot(s[Face.D])
        f, l, b, r = s[Face.F][2].copy(), s[Face.L][2].copy(), s[Face.B][2].copy(), s[Face.R][2].copy()
        s[Face.L][2], s[Face.B][2], s[Face.R][2], s[Face.F][2] = f, l, b, r


class D_PRIME(Move):
    def make_move_on_cube(self, cube): [D().make_move_on_cube(cube) for _ in range(3)]


class D2(Move):
    def make_move_on_cube(self, cube): [D().make_move_on_cube(cube) for _ in range(2)]


class L(Move):
    def make_move_on_cube(self, cube):
        s = cube.state
        s[Face.L] = _rot(s[Face.L])
        u, f, d, b = s[Face.U][:, 0].copy(), s[Face.F][:, 0].copy(), s[Face.D][:, 0].copy(), s[Face.B][:, 2][::-1].copy()
        s[Face.F][:, 0], s[Face.D][:, 0], s[Face.B][:, 2], s[Face.U][:, 0] = u, f, d[::-1], b


class L_PRIME(Move):
    def make_move_on_cube(self, cube): [L().make_move_on_cube(cube) for _ in range(3)]


class L2(Move):
    def make_move_on_cube(self, cube): [L().make_move_on_cube(cube) for _ in range(2)]


class R(Move):
    def make_move_on_cube(self, cube):
        s = cube.state
        s[Face.R] = _rot(s[Face.R])
        u, f, d, b = s[Face.U][:, 2].copy(), s[Face.F][:, 2].copy(), s[Face.D][:, 2].copy(), s[Face.B][:, 0][::-1].copy()
        s[Face.F][:, 2], s[Face.D][:, 2], s[Face.B][:, 0], s[Face.U][:, 2] = u, f, d[::-1], b


class R_PRIME(Move):
    def make_move_on_cube(self, cube): [R().make_move_on_cube(cube) for _ in range(3)]


class R2(Move):
    def make_move_on_cube(self, cube): [R().make_move_on_cube(cube) for _ in range(2)]


class F(Move):
    def make_move_on_cube(self, cube):
        s = cube.state
        s[Face.F] = _rot(s[Face.F])
        u, r, d, l = s[Face.U][2].copy(), s[Face.R][:, 0].copy(), s[Face.D][0].copy(), s[Face.L][:, 2].copy()
        s[Face.R][:, 0], s[Face.D][0], s[Face.L][:, 2], s[Face.U][2] = u[::-1], r, d[::-1], l


class F_PRIME(Move):
    def make_move_on_cube(self, cube): [F().make_move_on_cube(cube) for _ in range(3)]


class F2(Move):
    def make_move_on_cube(self, cube): [F().make_move_on_cube(cube) for _ in range(2)]


class B(Move):
    def make_move_on_cube(self, cube):
        s = cube.state
        s[Face.B] = _rot(s[Face.B])
        u, l, d, r = s[Face.U][0].copy(), s[Face.L][:, 0].copy(), s[Face.D][2].copy(), s[Face.R][:, 2].copy()
        s[Face.L][:, 0], s[Face.D][2], s[Face.R][:, 2], s[Face.U][0] = u, l[::-1], d, r[::-1]


class B_PRIME(Move):
    def make_move_on_cube(self, cube): [B().make_move_on_cube(cube) for _ in range(3)]


class B2(Move):
    def make_move_on_cube(self, cube): [B().make_move_on_cube(cube) for _ in range(2)]


# ──────────────────────── rotations de tranches ─────────────────────────
class M(Move):                                   # équivalent L′
    def make_move_on_cube(self, cube):
        s = cube.state
        u, f, d, b = s[Face.U][:, 1].copy(), s[Face.F][:, 1].copy(), s[Face.D][:, 1].copy(), s[Face.B][:, 1][::-1].copy()
        s[Face.F][:, 1], s[Face.D][:, 1], s[Face.B][:, 1], s[Face.U][:, 1] = u, f, d[::-1], b


class M_PRIME(Move):
    def make_move_on_cube(self, cube): [M().make_move_on_cube(cube) for _ in range(3)]


class M2(Move):
    def make_move_on_cube(self, cube): [M().make_move_on_cube(cube) for _ in range(2)]


class E(Move):                                   # équivalent D′
    def make_move_on_cube(self, cube):
        s = cube.state
        f, l, b, r = s[Face.F][1].copy(), s[Face.L][1].copy(), s[Face.B][1].copy(), s[Face.R][1].copy()
        s[Face.L][1], s[Face.B][1], s[Face.R][1], s[Face.F][1] = f, l, b, r


class E_PRIME(Move):
    def make_move_on_cube(self, cube): [E().make_move_on_cube(cube) for _ in range(3)]


class E2(Move):
    def make_move_on_cube(self, cube): [E().make_move_on_cube(cube) for _ in range(2)]


class S(Move):                                   # équivalent F
    def make_move_on_cube(self, cube):
        s = cube.state
        u, r, d, l = s[Face.U][1].copy(), s[Face.R][:, 1].copy(), s[Face.D][1].copy(), s[Face.L][:, 1].copy()
        s[Face.R][:, 1], s[Face.D][1], s[Face.L][:, 1], s[Face.U][1] = u[::-1], r, d[::-1], l


class S_PRIME(Move):
    def make_move_on_cube(self, cube): [S().make_move_on_cube(cube) for _ in range(3)]


class S2(Move):
    def make_move_on_cube(self, cube): [S().make_move_on_cube(cube) for _ in range(2)]


# ──────────────────────── outils de tirage ──────────────────────────────
BASE_MOVES = [
    U, U_PRIME, D, D_PRIME, L, L_PRIME, R, R_PRIME,
    F, F_PRIME, B, B_PRIME, M, M_PRIME, E, E_PRIME, S, S_PRIME
]
MOVES = BASE_MOVES + [U2, D2, L2, R2, F2, B2, M2, E2, S2]

def get_random_move() -> Move:
    return random.choice(MOVES)()

def get_random_moves(n: int) -> list[Move]:
    return [random.choice(MOVES)() for _ in range(n)]


# ──────────────────── table “notation → classe” ─────────────────────────
NOTATION_TO_MOVE = {
    "U": U,  "U'": U_PRIME,  "U2": U2,
    "D": D,  "D'": D_PRIME,  "D2": D2,
    "L": L,  "L'": L_PRIME,  "L2": L2,
    "R": R,  "R'": R_PRIME,  "R2": R2,
    "F": F,  "F'": F_PRIME,  "F2": F2,
    "B": B,  "B'": B_PRIME,  "B2": B2,
    "M": M,  "M'": M_PRIME,  "M2": M2,
    "E": E,  "E'": E_PRIME,  "E2": E2,
    "S": S,  "S'": S_PRIME,  "S2": S2,
}