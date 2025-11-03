import React, { useState } from 'react';
import { predictBatch } from '../api/api';
import ResultCard from './ResultCard';

export default function BatchForm() {
  const [jsonText, setJsonText] = useState('[{"catalog_content":"sample 1"},{"catalog_content":"sample 2"}]');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);

  async function onSubmit(e) {
    e.preventDefault();
    setLoading(true);
    try {
      const items = JSON.parse(jsonText);
      const res = await predictBatch(items);
      setResults(res.results || res);
    } catch (err) {
      alert('Error: ' + (err.message || err));
    } finally { setLoading(false); }
  }

  return (
    <section className="panel">
      <form onSubmit={onSubmit}>
        <label>Batch JSON
          <textarea value={jsonText} onChange={e => setJsonText(e.target.value)} rows={8} />
        </label>
        <div className="actions">
          <button type="submit" disabled={loading}>{loading ? 'Running…' : 'Run Batch'}</button>
        </div>
      </form>

      {results && (
        <div className="batch-results">
          {results.map((r, idx) => <ResultCard key={idx} result={r} />)}
        </div>
      )}
    </section>
  );
}
