// App.js
import React, { useEffect, useRef, useState } from 'react';
import { cubeApi } from './api/api_method';
import RubiksCube from './components/Rubiks/RubiksCube';
import MoveButtons from './components/MoveButtons/MoveButtons';
import FunctionnalsButtons from './components/FunctionnalsButtons/FunctionnalesButtons';

function App() {
  const [cubeData, setCubeData] = useState(null);
  const [moves, setMoves] = useState([]);
  const [isResetting, setIsResetting] = useState(false);
  const cubeRef = useRef(null);

  useEffect(() => {
    (async () => {
      try { setCubeData(await cubeApi.getCube()); } catch(e){ console.error(e); }
      try { setMoves(await cubeApi.getMoves());   } catch(e){ console.error(e); }
    })();
  }, []);

  const handleMove = async (move) => {
    cubeRef.current?.addCubeRotation(move);
    try { await cubeApi.rotate(move); } catch (err){ console.error("Erreur API :", err); }
  };


  const handleReset = async () => {
    setIsResetting(true);
    try {
      await cubeApi.reset();  
      const freshCube = await cubeApi.getCube();
      console.log("Cube réinitialisé :", freshCube);
      setCubeData(freshCube);                   
    } catch (e) {
      console.error("Reset échoué :", e);
    } finally {
      setIsResetting(false);
    }
  };

  return (
    <div className="App">
      <MoveButtons moves={moves} onClick={handleMove} />
      <FunctionnalsButtons onReset={handleReset} isLoading={isResetting} />
      <RubiksCube ref={cubeRef} cubeData={cubeData} />
    </div>
  );
}

export default App;