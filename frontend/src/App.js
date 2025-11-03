import React from 'react';
import PredictForm from './components/PredictForm';
import './styles/App.css';

export default function App() {
  return (
    <div className="app-root">
      <header className="topbar">
        <h1>Smart Price Prediction</h1>
        <p className="tagline">Upload item image or paste the description — get an instant estimated price.</p>
      </header>

      <main>
        <PredictForm />
      </main>

      <footer className="footer">Built for Smart Price Prediction • Backend at <code>{process.env.REACT_APP_API_BASE}</code> — prices shown in USD</footer>
    </div>
  );
}
