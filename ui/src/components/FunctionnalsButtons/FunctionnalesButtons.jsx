import './FunctionnalsButtons.css';


export default function FunctionnalsButtons({
  onReset, isLoading, shuffle,
  showFaceNames, onToggleFaceNames,onSolve
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

     <button onClick={onSolve}>
       {isLoading ? 'Solve…' : 'Solve'}
     </button>
    </div>
  );
}