import React, { useState, useRef } from 'react';
import { predictSingle } from '../api/api';
import Loader from './Loader';
import { previewFileAsDataURL } from '../utils/filePreview';
import '../styles/PredictForm.css';

// fixed image weight (user can't change)
// you can tweak this default if needed (0.0 = ignore image, 1.0 = image-only)
const FIXED_IMAGE_WEIGHT = 0.2;

export default function PredictForm() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [catalog, setCatalog] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [showDebug, setShowDebug] = useState(false);
  const fileRef = useRef();

  async function onFileChange(e) {
    const f = e.target.files[0];
    setFile(f || null);
    if (f) setPreview(await previewFileAsDataURL(f));
    else setPreview(null);
  }

  function clear() {
    setFile(null); setPreview(null); setCatalog(''); if (fileRef.current) fileRef.current.value = null;
    setResult(null);
  }

  async function onSubmit(e) {
    e.preventDefault();
    if ((!file) && (!catalog || !catalog.trim())) {
      alert('Please upload an image or paste a product description.');
      return;
    }
    setLoading(true); setResult(null);
    try {
      const res = await predictSingle({ file, catalog_content: catalog, image_weight: FIXED_IMAGE_WEIGHT });
      // we only show the final combined_pred (friendly UX)
      setResult(res);
    } catch (err) {
      console.error(err);
      alert('Prediction failed: ' + (err.message || err));
    } finally {
      setLoading(false);
    }
  }

  // format price in USD (two decimals)
  function fmtPriceUSD(p) {
    try {
      const n = Number(p);
      if (!isFinite(n)) return String(p);
      return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 2 }).format(n);
    } catch {
      return String(p);
    }
  }

  return (
    <section className="panel center">
      <form onSubmit={onSubmit} className="form-grid compact">
        <label className="full">Product description
          <textarea value={catalog} onChange={e => setCatalog(e.target.value)} placeholder="Paste title, bullet points, or product description (optional)"/>
        </label>

        <label>Upload image (optional)
          <input ref={fileRef} type="file" accept="image/*" onChange={onFileChange} />
        </label>
        {preview && <img src={preview} alt="preview" className="img-preview" />}

       

        <div className="actions">
          <button type="submit" disabled={loading}>{loading ? 'Predicting…' : 'Get Estimate'}</button>
          <button type="button" onClick={clear} className="muted">Reset</button>
        </div>
      </form>

      {loading && <Loader />}

      {result && (
        <div className="result-big">
          <div className="price-label">Estimated price</div>
          <div className="price-value">{fmtPriceUSD(result.combined_pred)}</div>
          <div className="meta">Response time: {Number(result.time_s).toFixed(2)}s</div>

          <div className="debug-toggle">
            <label><input type="checkbox" checked={showDebug} onChange={e => setShowDebug(e.target.checked)} /> Show debug</label>
          </div>

          {showDebug && (
            <details>
              <summary>Raw response</summary>
              <pre>{JSON.stringify(result, null, 2)}</pre>
            </details>
          )}
        </div>
      )}
    </section>
  );
}
