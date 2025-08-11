// RubiksCube.jsx
import React, { forwardRef, useImperativeHandle, useRef } from "react";
import "./RubiksCube.css";
import { useRubiksEngine } from "./useRubiksEngine";

const RubiksCube = forwardRef(({ cubeData }, ref) => {
  const mountRef = useRef(null);
  const api = useRubiksEngine({ mountRef, cubeData });

  useImperativeHandle(ref, () => ({
    addCubeRotation: api.addCubeRotation,
    pause: api.pause,
    resume: api.resume,
    clearQueue: api.clearQueue,
    setFaceLabelsVisible: api.setFaceLabelsVisible,

  }));

  if (!cubeData) return <div id="loading-message">Chargement du cube…</div>;

  return (
    <div
      ref={mountRef}
      style={{ width: "100vw", height: "100vh", overflow: "hidden", background: "#000" }}
    />
  );
});

export default RubiksCube;