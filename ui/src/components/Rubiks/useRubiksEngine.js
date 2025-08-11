// useRubiksEngine.js
import { useEffect, useRef } from "react";
import { createRubiksEngine } from "./engine/engine";

export function useRubiksEngine({ mountRef, cubeData }){
  const apiRef = useRef(null);

  useEffect(()=>{
    if (!mountRef.current || !cubeData) return;
    apiRef.current = createRubiksEngine({ mountEl: mountRef.current, cubeData });
    return ()=>{ apiRef.current?.dispose(); apiRef.current = null; };
  }, [cubeData, mountRef]);

  return {
    addCubeRotation: (m)=> apiRef.current?.addCubeRotation(m),
    pause: ()=> apiRef.current?.pause(),
    resume: ()=> apiRef.current?.resume(),
    clearQueue: ()=> apiRef.current?.clearQueue(),
    setFaceLabelsVisible: (v)=> apiRef.current?.setFaceLabelsVisible?.(v),
  };
}