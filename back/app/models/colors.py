




DEFAULT_FACE_ORDER = ["U", "F", "B", "R", "L", "D"]
FACE_INDEX = {face: idx for idx, face in enumerate(DEFAULT_FACE_ORDER)}

DEFAULT_FACE_COLORS = {
    "U": "white",
    "F": "green",
    "B": "blue",
    "R": "red",
    "L": "orange",
    "D": "yellow"
}

COLORS = {name: idx for idx, name in enumerate(DEFAULT_FACE_COLORS.values())}
INT_TO_COLORS = {v: k for k, v in COLORS.items()}
COLORS_TO_INT = COLORS
COLORS_TO_NAME = {idx: color for color, idx in COLORS_TO_INT.items()}



