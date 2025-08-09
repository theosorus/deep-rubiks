import React, { useEffect, useRef, useState } from 'react';
import { cubeApi } from './api/api_method';
import RubiksCube from './components/Rubiks/RubiksCube';
import MoveButtons from './components/MoveButtons/MoveButtons';

function App() {
  const [cubeData, setCubeData] = useState(null);
  const [moves, setMoves]       = useState([]);
  const cubeRef = useRef(null);          // <-- réf vers RubiksCube

  /* -------- RÉCUPÉRATION DES DONNÉES AU CHARGEMENT -------- */
  useEffect(() => {
    (async () => {
      try { setCubeData(await cubeApi.getCube()); } catch(e){ console.error(e); }
      try { setMoves(await cubeApi.getMoves());   } catch(e){ console.error(e); }
    })();
  }, []);

  /* -------- APPEL AU BACK + ROTATION GRAPHIQUE -------- */
  const handleMove = async (move) => {
    /* 1) On demande tout de suite la rotation visuelle  */
    cubeRef.current?.addCubeRotation(move);

    /* 2) On notifie l’API (async/await pour gérer l’erreur proprement) */
    try {
      await cubeApi.rotate(move);
    } catch (err){
      console.error("Erreur API :", err);
      /* Optionnel : rollback visuel si l’appel échoue */
    }
  };

  /* -------- RENDER -------- */
  return (
    <div className="App">
      <MoveButtons moves={moves} onClick={handleMove} />
      <RubiksCube ref={cubeRef} cubeData={cubeData} />
    </div>
  );
}

export default App;