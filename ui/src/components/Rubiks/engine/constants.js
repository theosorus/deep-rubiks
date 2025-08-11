// constants.js
export const moveMapping = {
  U: { face: "top", dir: -1 }, "U'": { face: "top", dir: 1 }, U2: { face: "top", dir: -1, double: true },
  R: { face: "right", dir: -1 }, "R'": { face: "right", dir: 1 }, R2: { face: "right", dir: -1, double: true },
  F: { face: "front", dir: -1 }, "F'": { face: "front", dir: 1 }, F2: { face: "front", dir: -1, double: true },
  L: { face: "left", dir: -1 }, "L'": { face: "left", dir: 1 }, L2: { face: "left", dir: -1, double: true },
  B: { face: "back", dir: -1 }, "B'": { face: "back", dir: 1 }, B2: { face: "back", dir: -1, double: true },
  D: { face: "bottom", dir: -1 }, "D'": { face: "bottom", dir: 1 }, D2: { face: "bottom", dir: -1, double: true },
  M: { face: "m", dir: -1 }, "M'": { face: "m", dir: 1 }, M2: { face: "m", dir: -1, double: true },
  E: { face: "e", dir: -1 }, "E'": { face: "e", dir: 1 }, E2: { face: "e", dir: -1, double: true },
  S: { face: "s", dir: -1 }, "S'": { face: "s", dir: 1 }, S2: { face: "s", dir: -1, double: true },
};

export const rotateConditions = {
  right: { axis: "x", value: 1 }, left: { axis: "x", value: -1 },
  top: { axis: "y", value: 1 }, bottom: { axis: "y", value: -1 },
  front: { axis: "z", value: 1 }, back: { axis: "z", value: -1 },
  m: { axis: "x", value: 0 }, e: { axis: "y", value: 0 }, s: { axis: "z", value: 0 },
};