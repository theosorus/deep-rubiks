import './FunctionnalsButtons.css';


export default function FunctionnalsButtons({
  onReset, isLoading, shuffle,
  showFaceNames, onToggleFaceNames,onSolve
}) {
  return (
    <div className="functional-buttons">
      <button onClick={onReset} disabled={isLoading}>
        {isLoading ? 'Resetting…' : 'Reset'}
      </button>

      <button onClick={onToggleFaceNames}>
        {showFaceNames ? "Hide face names" : "Show face names"}
      </button>

      <button onClick={shuffle}>
        {isLoading ? 'Shuffling…' : 'Shuffle'}
      </button>

      <button onClick={onSolve}>
        {isLoading ? 'Solving…' : 'Solve'}
      </button>
    </div>
  );
}