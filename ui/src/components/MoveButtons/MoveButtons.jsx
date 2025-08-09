import React from 'react';
import './MoveButtons.css';

const MoveButtons = ({ moves, onClick }) => (
  <div className="move-panel">
    {moves.map(m => (
      <button key={m} onClick={() => onClick(m)}>
        {m}
      </button>
    ))}
  </div>
);

export default MoveButtons;