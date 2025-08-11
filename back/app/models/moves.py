# models/moves.py
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
import random

from models.colors import FACE_INDEX

# Petite aide: rotation d'une face (clockwise = k=1)
def _rot(face: np.ndarray, k: int = 1) -> np.ndarray:
    # np.rot90 tourne CCW pour k>0, donc on inverse le signe pour avoir CW par défaut
    return np.rot90(face, -k)


class Move(ABC):
    @abstractmethod
    def make_move_on_cube(self, cube) -> None:
        ...


# ───────────────────────────
# U / U' / U2
# ───────────────────────────
class U(Move):
    def make_move_on_cube(self, cube):
        s = cube.state
        s[FACE_INDEX["U"]] = _rot(s[FACE_INDEX["U"]])
        f, r, b, l = s[FACE_INDEX["F"]][0].copy(), s[FACE_INDEX["R"]][0].copy(), s[FACE_INDEX["B"]][0].copy(), s[FACE_INDEX["L"]][0].copy()
        s[FACE_INDEX["R"]][0], s[FACE_INDEX["B"]][0], s[FACE_INDEX["L"]][0], s[FACE_INDEX["F"]][0] = f, r, b, l

class U_PRIME(Move):
    def make_move_on_cube(self, cube): [U().make_move_on_cube(cube) for _ in range(3)]

class U2(Move):
    def make_move_on_cube(self, cube): [U().make_move_on_cube(cube) for _ in range(2)]


# ───────────────────────────
# D / D' / D2
# ───────────────────────────
class D(Move):
    def make_move_on_cube(self, cube):
        s = cube.state
        s[F["D"]] = _rot(s[F["D"]])
        f, l, b, r = s[F["F"]][2].copy(), s[F["L"]][2].copy(), s[F["B"]][2].copy(), s[F["R"]][2].copy()
        # D (clockwise vu depuis D) => F -> L -> B -> R
        s[F["L"]][2], s[F["B"]][2], s[F["R"]][2], s[F["F"]][2] = f, l, b, r

class D_PRIME(Move):
    def make_move_on_cube(self, cube): [D().make_move_on_cube(cube) for _ in range(3)]

class D2(Move):
    def make_move_on_cube(self, cube): [D().make_move_on_cube(cube) for _ in range(2)]


# ───────────────────────────
# R / R' / R2
# ───────────────────────────
class R(Move):
    def make_move_on_cube(self, cube):
        s = cube.state
        s[F["R"]] = _rot(s[F["R"]])
        u, f, d, b = s[F["U"]][:, 2].copy(), s[F["F"]][:, 2].copy(), s[F["D"]][:, 2].copy(), s[F["B"]][:, 0].copy()
        s[F["F"]][:, 2] = u
        s[F["D"]][:, 2] = f
        s[F["B"]][:, 0] = d[::-1]
        s[F["U"]][:, 2] = b[::-1]

class R_PRIME(Move):
    def make_move_on_cube(self, cube): [R().make_move_on_cube(cube) for _ in range(3)]

class R2(Move):
    def make_move_on_cube(self, cube): [R().make_move_on_cube(cube) for _ in range(2)]


# ───────────────────────────
# L / L' / L2
# ───────────────────────────
class L(Move):
    def make_move_on_cube(self, cube):
        s = cube.state
        s[F["L"]] = _rot(s[F["L"]])
        u, b, d, f = s[F["U"]][:, 0].copy(), s[F["B"]][:, 2].copy(), s[F["D"]][:, 0].copy(), s[F["F"]][:, 0].copy()
        s[F["B"]][:, 2] = d[::-1]
        s[F["D"]][:, 0] = f
        s[F["F"]][:, 0] = u
        s[F["U"]][:, 0] = b[::-1]

class L_PRIME(Move):
    def make_move_on_cube(self, cube): [L().make_move_on_cube(cube) for _ in range(3)]

class L2(Move):
    def make_move_on_cube(self, cube): [L().make_move_on_cube(cube) for _ in range(2)]


# ───────────────────────────
# F / F' / F2
# ───────────────────────────
class F(Move):
    def make_move_on_cube(self, cube):
        s = cube.state
        s[F["F"]] = _rot(s[F["F"]])
        u, r, d, l = s[F["U"]][2].copy(), s[F["R"]][:, 0].copy(), s[F["D"]][0].copy(), s[F["L"]][:, 2].copy()
        s[F["R"]][:, 0] = u
        s[F["D"]][0]   = r[::-1]
        s[F["L"]][:, 2] = d
        s[F["U"]][2]   = l[::-1]

class F_PRIME(Move):
    def make_move_on_cube(self, cube): [F().make_move_on_cube(cube) for _ in range(3)]

class F2(Move):
    def make_move_on_cube(self, cube): [F().make_move_on_cube(cube) for _ in range(2)]


# ───────────────────────────
# B / B' / B2
# ───────────────────────────
class B(Move):
    def make_move_on_cube(self, cube):
        s = cube.state
        s[F["B"]] = _rot(s[F["B"]])
        u, l, d, r = s[F["U"]][0].copy(), s[F["L"]][:, 0].copy(), s[F["D"]][2].copy(), s[F["R"]][:, 2].copy()
        s[F["L"]][:, 0] = d[::-1]
        s[F["D"]][2]   = r
        s[F["R"]][:, 2] = u[::-1]
        s[F["U"]][0]   = l

class B_PRIME(Move):
    def make_move_on_cube(self, cube): [B().make_move_on_cube(cube) for _ in range(3)]

class B2(Move):
    def make_move_on_cube(self, cube): [B().make_move_on_cube(cube) for _ in range(2)]


# ───────────────────────────
# M / M' / M2  (milieu vertical; comme L)
# ───────────────────────────
class M(Move):
    def make_move_on_cube(self, cube):
        s = cube.state
        u, f, d, b = s[F["U"]][:, 1].copy(), s[F["F"]][:, 1].copy(), s[F["D"]][:, 1].copy(), s[F["B"]][:, 1].copy()
        s[F["F"]][:, 1] = u
        s[F["D"]][:, 1] = f
        s[F["B"]][:, 1] = d[::-1]
        s[F["U"]][:, 1] = b[::-1]

class M_PRIME(Move):
    def make_move_on_cube(self, cube): [M().make_move_on_cube(cube) for _ in range(3)]

class M2(Move):
    def make_move_on_cube(self, cube): [M().make_move_on_cube(cube) for _ in range(2)]


# ───────────────────────────
# E / E' / E2  (équateur; comme D)
# ───────────────────────────
class E(Move):
    def make_move_on_cube(self, cube):
        s = cube.state
        f, r, b, l = s[F["F"]][1].copy(), s[F["R"]][1].copy(), s[F["B"]][1].copy(), s[F["L"]][1].copy()
        s[F["R"]][1], s[F["B"]][1], s[F["L"]][1], s[F["F"]][1] = f, r, b, l

class E_PRIME(Move):
    def make_move_on_cube(self, cube): [E().make_move_on_cube(cube) for _ in range(3)]

class E2(Move):
    def make_move_on_cube(self, cube): [E().make_move_on_cube(cube) for _ in range(2)]


# ───────────────────────────
# S / S' / S2  (couche debout; comme F)
# ───────────────────────────
class S(Move):
    def make_move_on_cube(self, cube):
        s = cube.state
        u, r, d, l = s[F["U"]][1].copy(), s[F["R"]][:, 1].copy(), s[F["D"]][1].copy(), s[F["L"]][:, 1].copy()
        s[F["R"]][:, 1] = u
        s[F["D"]][1]   = r[::-1]
        s[F["L"]][:, 1] = d
        s[F["U"]][1]   = l[::-1]

class S_PRIME(Move):
    def make_move_on_cube(self, cube): [S().make_move_on_cube(cube) for _ in range(3)]

class S2(Move):
    def make_move_on_cube(self, cube): [S().make_move_on_cube(cube) for _ in range(2)]


# ───────────────────────────
# Registre des mouvements & utilitaires
# ───────────────────────────
MOVE_CLASSES = {
    "U": U, "U'": U_PRIME, "U2": U2,
    "D": D, "D'": D_PRIME, "D2": D2,
    "L": L, "L'": L_PRIME, "L2": L2,
    "R": R, "R'": R_PRIME, "R2": R2,
    "F": F, "F'": F_PRIME, "F2": F2,
    "B": B, "B'": B_PRIME, "B2": B2,
    "M": M, "M'": M_PRIME, "M2": M2,
    "E": E, "E'": E_PRIME, "E2": E2,
    "S": S, "S'": S_PRIME, "S2": S2,
}

AVAILABLE_MOVES = list(MOVE_CLASSES.keys())

def get_move_by_name(name: str) -> Move:
    cls = MOVE_CLASSES.get(name)
    if not cls:
        raise ValueError(f"Unknown move: {name}")
    return cls()

def get_random_moves(n: int) -> list[Move]:
    return [get_move_by_name(random.choice(AVAILABLE_MOVES)) for _ in range(n)]