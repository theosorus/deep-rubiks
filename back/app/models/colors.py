

COLORS = {
    "white": 0,
    "blue": 1,
    "red": 2,
    "yellow": 3,
    "green": 4,
    "orange": 5
}

INT_TO_COLORS = {v: k for k, v in COLORS.items()}
    
DEFAULT_FACE_ORDER = ["U", "F", "B", "R", "L", "D"]
FACE_INDEX = {face: idx for idx, face in enumerate(DEFAULT_FACE_ORDER)}

DEFAULT_FACE_COLORS = {
    "U": "white",
    "F": "blue",
    "B": "red",
    "R": "yellow",
    "L": "green",
    "D": "orange"
}

COLORS_TO_INT = {color: idx for idx, color in enumerate(DEFAULT_FACE_COLORS.values())}
COLORS_TO_NAME = {idx: color for color, idx in COLORS_TO_INT.items()}



