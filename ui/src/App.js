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

  // initial data fetch
  useEffect(() => {
    (async () => {
      try { setCubeData(await cubeApi.getCube()); } catch (e) { console.error(e); }
      try { setMoves(await cubeApi.getMoves()); } catch (e) { console.error(e); }
    })();
  }, []);

  useEffect(() => {
  cubeRef.current?.setFaceLabelsVisible?.(showFaceNames);
}, [cubeData, showFaceNames]);

  const handleMove = async (move) => {
    cubeRef.current?.addCubeRotation(move);
    try {
      await cubeApi.rotate(move);
    } catch (err) {
      console.error(' API error :', err);
    }
  };

  const handleSequence = async (sequence = []) => {
    for (const mv of sequence) {
      cubeRef.current?.addCubeRotation(mv);
      try { await cubeApi.rotate(mv); } catch (e) { console.error(e); }
    }
  };

const handleSolve = async () => {
  try {
    const result = await cubeApi.solve();
    console.log('Solve success:', result);
    
    if (result.solved && result.solution_moves && result.solution_moves.length > 0) {
      console.log(`Executing solution: ${result.solution_moves.join(' ')}`);
      console.log(`Solution found in ${result.search_time.toFixed(3)}s`);
      console.log(`Nodes expanded: ${result.nodes_expanded}`);
      
      for (const move of result.solution_moves) {
        cubeRef.current?.addCubeRotation(move);
        
      }
    } else if (!result.solved) {
      console.log('No solution found');
      alert('No solution found');
    }
    
  } catch (e) {
    console.error('Solve failed:', e);
    alert('Error');
  }
};


  const handleReset = async () => {
    setIsResetting(true);
    cubeRef.current?.clearQueue();
    try {
      await cubeApi.reset();
      const freshCube = await cubeApi.getCube();
      setCubeData(freshCube);  
    } catch (e) {
      console.error('reset fail :', e);
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
      console.error('shuflle success :', e);
    }
  }


  return (
    <div className="App">
      <MoveButtons moves={moves} onClick={handleMove} />
      <FunctionnalsButtons
        onReset={handleReset}
        isLoading={isResetting}
        shuffle={handleShuffle}
        showFaceNames={showFaceNames}
        onSolve={handleSolve}
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