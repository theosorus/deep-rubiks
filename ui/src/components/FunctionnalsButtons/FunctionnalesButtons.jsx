import React from 'react';
import { cubeApi } from '../../api/api_method';


export default function FunctionnalsButtons({ onReset, isLoading }) {
  return (
    <div className="functional-buttons">
      <button onClick={onReset} disabled={isLoading}>
        {isLoading ? 'Reset…' : 'Reset'}
      </button>
    </div>
  );
}