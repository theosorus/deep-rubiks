from core.face import  DEFAULT_FACE_COLORS

COLORS_TO_INT = {name: idx for idx, name in enumerate(DEFAULT_FACE_COLORS.values())}
INT_TO_COLORS = {v: k for k, v in COLORS_TO_INT.items()}
COLORS_TO_NAME = {idx: color for color, idx in COLORS_TO_INT.items()}



