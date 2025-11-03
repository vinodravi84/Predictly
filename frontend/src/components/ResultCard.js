import React from 'react';
import '../styles/ResultCard.css';
export default function ResultCard({ result }) {
  if (!result) return null;
  return (
    <div className="result-card">
      <div className="price-value">{result.combined_pred}</div>
    </div>
  );
}
