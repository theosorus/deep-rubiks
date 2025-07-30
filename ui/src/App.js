import React, { useEffect, useState } from 'react';
import { cubeApi } from './api/api_method';
import RubiksCube from './components/RubiksCube';

function App() {
  const [cubeData, setCubeData] = useState(null);

  useEffect(() => {
    const fetchCube = async () => {
      try {
        const data = await cubeApi.getCube();
        setCubeData(data);
      } catch (error) {
        console.error('Erreur:', error.message);
      }
    };
    fetchCube();
  }, []);

  useEffect(() => {
    console.log(cubeData);
  }, [cubeData]);

  return (
    <div className="App">
      <RubiksCube cubeData={cubeData} />
    </div>
  );
}

export default App;