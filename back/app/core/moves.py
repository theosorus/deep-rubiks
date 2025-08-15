# models/moves.py
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
import random

from core.face import FACE_INDEX


def _rot(face: np.ndarray, k: int = 1) -> np.ndarray:
    return np.rot90(face, -k)


# ──────────────────────────────────────────────────────────────────────────────
# Base
# ──────────────────────────────────────────────────────────────────────────────

class Move(ABC):
    @staticmethod
    @abstractmethod
    def make_move_on_cube(cube) -> None: ...

# ──────────────────────────────────────────────────────────────────────────────
# U / U' / U2 
# ──────────────────────────────────────────────────────────────────────────────
class U(Move):
    @staticmethod
    def make_move_on_cube(cube): 
        state = cube.state
        temp = state[FACE_INDEX["F"]][0].copy()
        state[FACE_INDEX["F"]][0] = state[FACE_INDEX["R"]][0].copy()
        state[FACE_INDEX["R"]][0] = state[FACE_INDEX["B"]][0].copy()
        state[FACE_INDEX["B"]][0] = state[FACE_INDEX["L"]][0].copy()
        state[FACE_INDEX["L"]][0] = temp
        
        state[FACE_INDEX["U"]] = _rot(state[FACE_INDEX["U"]], 1)
        
        
           
class U_PRIME(Move):
    @staticmethod
    def make_move_on_cube(cube): 
        state = cube.state
        temp = state[FACE_INDEX["F"]][0].copy()
        state[FACE_INDEX["F"]][0] = state[FACE_INDEX["L"]][0].copy()
        state[FACE_INDEX["L"]][0] = state[FACE_INDEX["B"]][0].copy()
        state[FACE_INDEX["B"]][0] = state[FACE_INDEX["R"]][0].copy()
        state[FACE_INDEX["R"]][0] = temp
    
        state[FACE_INDEX["U"]] = _rot(state[FACE_INDEX["U"]], -1)

class U2(Move):
    @staticmethod
    def make_move_on_cube( cube): 
        U.make_move_on_cube(cube)
        U.make_move_on_cube(cube)

# ──────────────────────────────────────────────────────────────────────────────
# D / D' / D2 
# ──────────────────────────────────────────────────────────────────────────────
class D(Move):
    @staticmethod
    def make_move_on_cube( cube): 
        state = cube.state
        temp = state[FACE_INDEX["F"]][2].copy()
        state[FACE_INDEX["F"]][2] = state[FACE_INDEX["L"]][2].copy()
        state[FACE_INDEX["L"]][2] = state[FACE_INDEX["B"]][2].copy()
        state[FACE_INDEX["B"]][2] = state[FACE_INDEX["R"]][2].copy()
        state[FACE_INDEX["R"]][2] = temp
        
        state[FACE_INDEX["D"]] = _rot(state[FACE_INDEX["D"]], 1)
        

class D_PRIME(Move):
    @staticmethod
    def make_move_on_cube( cube):
        state = cube.state
        temp = state[FACE_INDEX["F"]][2].copy()
        state[FACE_INDEX["F"]][2] = state[FACE_INDEX["R"]][2].copy()
        state[FACE_INDEX["R"]][2] = state[FACE_INDEX["B"]][2].copy()
        state[FACE_INDEX["B"]][2] = state[FACE_INDEX["L"]][2].copy()
        state[FACE_INDEX["L"]][2] = temp 
        
        state[FACE_INDEX["D"]] = _rot(state[FACE_INDEX["D"]], -1)
        
        
class D2(Move):
    @staticmethod
    def make_move_on_cube( cube): 
        D.make_move_on_cube(cube)
        D.make_move_on_cube(cube)

# ──────────────────────────────────────────────────────────────────────────────
# R / R' / R2 
# ──────────────────────────────────────────────────────────────────────────────
class R(Move):
    @staticmethod
    def make_move_on_cube( cube): 
        state = cube.state
        temp = state[FACE_INDEX["F"]][:, 2].copy()
        state[FACE_INDEX["F"]][:, 2] = state[FACE_INDEX["D"]][:, 2].copy()
        state[FACE_INDEX["D"]][:, 2] = state[FACE_INDEX["B"]][:, 0][::-1].copy()
        state[FACE_INDEX["B"]][:, 0] = state[FACE_INDEX["U"]][:, 2][::-1].copy()
        state[FACE_INDEX["U"]][:, 2] = temp
        
        state[FACE_INDEX["R"]] = _rot(state[FACE_INDEX["R"]], 1)
        

class R_PRIME(Move):
    @staticmethod
    def make_move_on_cube( cube): 
        state = cube.state
        temp = state[FACE_INDEX["F"]][:, 2].copy()
        state[FACE_INDEX["F"]][:, 2] = state[FACE_INDEX["U"]][:, 2].copy()
        state[FACE_INDEX["U"]][:, 2] = state[FACE_INDEX["B"]][:, 0][::-1].copy()
        state[FACE_INDEX["B"]][:, 0] = state[FACE_INDEX["D"]][:, 2][::-1].copy()
        state[FACE_INDEX["D"]][:, 2] = temp
        
        state[FACE_INDEX["R"]] = _rot(state[FACE_INDEX["R"]], -1)
        

class R2(Move):
    @staticmethod
    def make_move_on_cube( cube): 
        R.make_move_on_cube(cube)
        R.make_move_on_cube(cube)

# ──────────────────────────────────────────────────────────────────────────────
# L / L' / L2 
# ──────────────────────────────────────────────────────────────────────────────
class L(Move):
    @staticmethod
    def make_move_on_cube( cube): 
        state = cube.state
        temp = state[FACE_INDEX["F"]][:, 0].copy()
        state[FACE_INDEX["F"]][:, 0] = state[FACE_INDEX["U"]][:, 0].copy()
        state[FACE_INDEX["U"]][:, 0] = state[FACE_INDEX["B"]][:, 2][::-1].copy()
        state[FACE_INDEX["B"]][:, 2] = state[FACE_INDEX["D"]][:, 0][::-1].copy()
        state[FACE_INDEX["D"]][:, 0] = temp
        
        state[FACE_INDEX["L"]] = _rot(state[FACE_INDEX["L"]], 1)

class L_PRIME(Move):
    @staticmethod
    def make_move_on_cube( cube): 
        state = cube.state
        temp = state[FACE_INDEX["F"]][:, 0].copy()
        state[FACE_INDEX["F"]][:, 0] = state[FACE_INDEX["D"]][:, 0].copy()
        state[FACE_INDEX["D"]][:, 0] = state[FACE_INDEX["B"]][:, 2][::-1].copy()
        state[FACE_INDEX["B"]][:, 2] = state[FACE_INDEX["U"]][:, 0][::-1].copy()
        state[FACE_INDEX["U"]][:, 0] = temp
        
        state[FACE_INDEX["L"]] = _rot(state[FACE_INDEX["L"]], -1)

class L2(Move):
    @staticmethod
    def make_move_on_cube( cube): 
        L.make_move_on_cube(cube)
        L.make_move_on_cube(cube)

# ──────────────────────────────────────────────────────────────────────────────
# F / F' / F2 
# ──────────────────────────────────────────────────────────────────────────────
class F(Move):
    @staticmethod
    def make_move_on_cube( cube): 
        state = cube.state
        temp = state[FACE_INDEX["R"]][:, 0].copy()
        state[FACE_INDEX["R"]][:, 0] = state[FACE_INDEX["U"]][2].copy()
        state[FACE_INDEX["U"]][2] = state[FACE_INDEX["L"]][:, 2][::-1].copy()
        state[FACE_INDEX["L"]][:, 2] = state[FACE_INDEX["D"]][0].copy()
        state[FACE_INDEX["D"]][0] = temp[::-1].copy()
        
        state[FACE_INDEX["F"]] = _rot(state[FACE_INDEX["F"]], 1)

class F_PRIME(Move):
    @staticmethod
    def make_move_on_cube( cube): 
        state = cube.state
        temp = state[FACE_INDEX["R"]][:, 0].copy()
        state[FACE_INDEX["R"]][:, 0] = state[FACE_INDEX["D"]][0][::-1].copy()
        state[FACE_INDEX["D"]][0] = state[FACE_INDEX["L"]][:, 2].copy()
        state[FACE_INDEX["L"]][:, 2] = state[FACE_INDEX["U"]][2][::-1].copy()
        state[FACE_INDEX["U"]][2] = temp
        
        state[FACE_INDEX["F"]] = _rot(state[FACE_INDEX["F"]], -1)
       

class F2(Move):
    @staticmethod
    def make_move_on_cube( cube): 
        F.make_move_on_cube(cube)
        F.make_move_on_cube(cube)

# ──────────────────────────────────────────────────────────────────────────────
# B / B' / B2 
# ──────────────────────────────────────────────────────────────────────────────
class B(Move):
    @staticmethod
    def make_move_on_cube( cube): 
        state = cube.state
        temp = state[FACE_INDEX["R"]][:, 2].copy()
        state[FACE_INDEX["R"]][:, 2] = state[FACE_INDEX["D"]][2][::-1].copy()
        state[FACE_INDEX["D"]][2] = state[FACE_INDEX["L"]][:, 0].copy()
        state[FACE_INDEX["L"]][:, 0] = state[FACE_INDEX["U"]][0][::-1].copy()
        state[FACE_INDEX["U"]][0] = temp
        
        state[FACE_INDEX["B"]] = _rot(state[FACE_INDEX["B"]], 1)
        
        
class B_PRIME(Move):
    @staticmethod
    def make_move_on_cube( cube): 
        state = cube.state
        temp = state[FACE_INDEX["R"]][:, 2].copy()
        state[FACE_INDEX["R"]][:, 2] = state[FACE_INDEX["U"]][0].copy()
        state[FACE_INDEX["U"]][0] = state[FACE_INDEX["L"]][:, 0][::-1].copy()
        state[FACE_INDEX["L"]][:, 0] = state[FACE_INDEX["D"]][2].copy()
        state[FACE_INDEX["D"]][2] = temp[::-1].copy()
        
        state[FACE_INDEX["B"]] = _rot(state[FACE_INDEX["B"]], -1)

class B2(Move):
    @staticmethod
    def make_move_on_cube( cube): 
        B.make_move_on_cube(cube)
        B.make_move_on_cube(cube)

# ──────────────────────────────────────────────────────────────────────────────
# M / M' / M2  
# ──────────────────────────────────────────────────────────────────────────────
class M(Move):
    @staticmethod
    def make_move_on_cube( cube): 
        state = cube.state
        temp = state[FACE_INDEX["F"]][:, 1].copy()
        state[FACE_INDEX["F"]][:, 1] = state[FACE_INDEX["U"]][:, 1].copy()
        state[FACE_INDEX["U"]][:, 1] = state[FACE_INDEX["B"]][:, 1][::-1].copy()
        state[FACE_INDEX["B"]][:, 1] = state[FACE_INDEX["D"]][:, 1][::-1].copy()
        state[FACE_INDEX["D"]][:, 1] = temp

class M_PRIME(Move):
    @staticmethod
    def make_move_on_cube( cube): 
        state = cube.state
        temp = state[FACE_INDEX["F"]][:, 1].copy()
        state[FACE_INDEX["F"]][:, 1] = state[FACE_INDEX["D"]][:, 1].copy()
        state[FACE_INDEX["D"]][:, 1] = state[FACE_INDEX["B"]][:, 1][::-1].copy()
        state[FACE_INDEX["B"]][:, 1] = state[FACE_INDEX["U"]][:, 1][::-1].copy()
        state[FACE_INDEX["U"]][:, 1] = temp

class M2(Move):
    @staticmethod
    def make_move_on_cube( cube): 
        M.make_move_on_cube(cube)
        M.make_move_on_cube(cube)

# ──────────────────────────────────────────────────────────────────────────────
# E / E' / E2  
# ──────────────────────────────────────────────────────────────────────────────
class E(Move):
    @staticmethod
    def make_move_on_cube( cube): 
        state = cube.state
        temp = state[FACE_INDEX["F"]][1].copy()
        state[FACE_INDEX["F"]][1] = state[FACE_INDEX["L"]][1].copy()
        state[FACE_INDEX["L"]][1] = state[FACE_INDEX["B"]][1].copy()
        state[FACE_INDEX["B"]][1] = state[FACE_INDEX["R"]][1].copy()
        state[FACE_INDEX["R"]][1] = temp

class E_PRIME(Move):
    @staticmethod
    def make_move_on_cube( cube): 
        state = cube.state
        temp = state[FACE_INDEX["F"]][1].copy()
        state[FACE_INDEX["F"]][1] = state[FACE_INDEX["R"]][1].copy()
        state[FACE_INDEX["R"]][1] = state[FACE_INDEX["B"]][1].copy()
        state[FACE_INDEX["B"]][1] = state[FACE_INDEX["L"]][1].copy()
        state[FACE_INDEX["L"]][1] = temp 

class E2(Move):
    @staticmethod
    def make_move_on_cube( cube):
        E.make_move_on_cube(cube)
        E.make_move_on_cube(cube)

# ──────────────────────────────────────────────────────────────────────────────
# S / S' / S2  
# ──────────────────────────────────────────────────────────────────────────────
class S(Move):
    @staticmethod
    def make_move_on_cube( cube): 
        state = cube.state
        temp = state[FACE_INDEX["R"]][:, 1].copy()
        state[FACE_INDEX["R"]][:, 1] = state[FACE_INDEX["U"]][1].copy()
        state[FACE_INDEX["U"]][1] = state[FACE_INDEX["L"]][:, 1][::-1].copy()
        state[FACE_INDEX["L"]][:, 1] = state[FACE_INDEX["D"]][1].copy()
        state[FACE_INDEX["D"]][1] = temp[::-1].copy()

class S_PRIME(Move):
    @staticmethod
    def make_move_on_cube( cube): 
        state = cube.state
        temp = state[FACE_INDEX["R"]][:, 1].copy()
        state[FACE_INDEX["R"]][:, 1] = state[FACE_INDEX["D"]][1][::-1].copy()
        state[FACE_INDEX["D"]][1] = state[FACE_INDEX["L"]][:, 1].copy()
        state[FACE_INDEX["L"]][:, 1] = state[FACE_INDEX["U"]][1][::-1].copy()
        state[FACE_INDEX["U"]][1] = temp

class S2(Move):
    @staticmethod
    def make_move_on_cube( cube): 
        S.make_move_on_cube(cube)
        S.make_move_on_cube(cube)

# ──────────────────────────────────────────────────────────────────────────────
# Registre & utilitaires
# ──────────────────────────────────────────────────────────────────────────────
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
    return cls

def get_random_moves(n: int) -> list[Move]:
    return [get_move_by_name(random.choice(AVAILABLE_MOVES)) for _ in range(n)]


def _apply_moves_on_cube(cube, moves: list[Move]) -> None:
    """Apply a list of moves on the given cube."""
    for move in moves:
        move.make_move_on_cube(cube)
        
def apply_moves(cube, moves: list[str]) -> None:
    """Apply a list of move names on the given cube."""
    move_objects = [get_move_by_name(move) for move in moves]
    _apply_moves_on_cube(cube, move_objects)