// colors.js
export const getFaceColorFactory = (data) => {
  const DEFAULT_FACE_ORDER = ["U","R","F","D","L","B"];
  const FACE_ORDER = Array.isArray(data?.face_order) && data.face_order.length === 6
    ? data.face_order : DEFAULT_FACE_ORDER;

  const idx = (name) => {
    const i = FACE_ORDER.indexOf(name);
    return i === -1 ? 0 : i;
  };

  return (x, y, z, faceIndex) => {
    const conv = (n) => n + 1; // [-1..1] -> [0..2]
    const faceMap = [
      { face: idx("R"), row: 2 - conv(y), col: 2 - conv(z) }, // right (flip rows)
      { face: idx("L"), row: 2 - conv(y), col: conv(z) },     // left  (flip rows)
 { face: idx("U"), row:      conv(z), col: conv(x) }, // top
 { face: idx("D"), row: 2 - conv(z), col: conv(x) }, // bottom
      { face: idx("F"), row: 2 - conv(y), col: conv(x) },     // front (flip rows)
      { face: idx("B"), row: 2 - conv(y), col: 2 - conv(x) }, // back  (flip rows)
    ];
    const visible = (fi, xx, yy, zz) =>
      (fi===0 && xx===1) || (fi===1 && xx===-1) ||
      (fi===2 && yy===1) || (fi===3 && yy===-1) ||
      (fi===4 && zz===1) || (fi===5 && zz===-1);

    if (!visible(faceIndex, x, y, z)) return "gray";
    const m = faceMap[faceIndex];
    const colorNumber = data.state?.[m.face]?.[m.row]?.[m.col];
    return data.colors?.[String(colorNumber)] || "gray";
  };
};