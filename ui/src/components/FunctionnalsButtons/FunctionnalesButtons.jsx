import React from 'react';
import { cubeApi } from '../../api/api_method';
import './FunctionnalsButtons.css';


export default function FunctionnalsButtons({
  onReset, isLoading, shuffle ,onPause, onResume, onClear,
  showFaceNames, onToggleFaceNames
}) {
  return (
    <div className="functional-buttons">
      <button onClick={onReset} disabled={isLoading}>
        {isLoading ? 'Reset…' : 'Reset'}
      </button>

      <button onClick={onToggleFaceNames}>
       {showFaceNames ? "Masquer noms faces" : "Afficher noms faces"}
     </button>

     <button onClick={shuffle}>
       {isLoading ? 'Mélanger…' : 'Mélanger'}
     </button>
    </div>
  );
}