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
  const [showFaceNames, setShowFaceNames] = useState(false);
  const cubeRef = useRef(null);

  // Chargement initial
  useEffect(() => {
    (async () => {
      try { setCubeData(await cubeApi.getCube()); } catch (e) { console.error(e); }
      try { setMoves(await cubeApi.getMoves()); } catch (e) { console.error(e); }
    })();
  }, []);

  useEffect(() => {
  cubeRef.current?.setFaceLabelsVisible?.(showFaceNames);
}, [cubeData, showFaceNames]);

  // Un seul coup
  const handleMove = async (move) => {
    // Joue l’anim immédiatement côté client
    cubeRef.current?.addCubeRotation(move);
    // Puis notifie le backend
    try {
      await cubeApi.rotate(move);
    } catch (err) {
      console.error('Erreur API :', err);
    }
  };

  // Enchaîner une séquence de coups (optionnel)
  const handleSequence = async (sequence = []) => {
    for (const mv of sequence) {
      cubeRef.current?.addCubeRotation(mv);
      try { await cubeApi.rotate(mv); } catch (e) { console.error(e); }
    }
  };

  // Reset complet
  const handleReset = async () => {
    setIsResetting(true);
    // stoppe/efface toute anim locale encore en attente
    cubeRef.current?.clearQueue();
    try {
      await cubeApi.reset();
      const freshCube = await cubeApi.getCube();
      setCubeData(freshCube);  // le moteur se remontera proprement
    } catch (e) {
      console.error('Reset échoué :', e);
    } finally {
      setIsResetting(false);
    }
  };

  const handleShuffle = async () => {
    try {
      await cubeApi.shuffle(100); 
      const freshCube = await cubeApi.getCube();
      setCubeData(freshCube);
    }
    catch (e) {
      console.error('Mélange échoué :', e);
    }
  }


  return (
    <div className="App">
      <MoveButtons moves={moves} onClick={handleMove} />
      <FunctionnalsButtons
        onReset={handleReset}
        isLoading={isResetting}
        // exemples d’API moteur (si tu veux des boutons dédiés) :
        onPause={() => cubeRef.current?.pause?.()}
        onResume={() => cubeRef.current?.resume?.()}
        onClear={() => cubeRef.current?.clearQueue?.()}
        shuffle={handleShuffle}
        showFaceNames={showFaceNames}
        onToggleFaceNames={() => {
          const next = !showFaceNames;
          setShowFaceNames(next);
          cubeRef.current?.setFaceLabelsVisible?.(next);
        }}
      />
      <RubiksCube ref={cubeRef} cubeData={cubeData} />
    </div>
  );
}

export default App;