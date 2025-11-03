const API_BASE = process.env.REACT_APP_API_BASE || 'http://localhost:8000';

export async function predictSingle({ file, catalog_content, image_weight = 0.5 }) {
  // Always send FormData for consistent server handling
  const fd = new FormData();
  if (file) fd.append('image', file);
  fd.append('catalog_content', catalog_content || '');
  // fixed weight sent from frontend (user cannot change)
  fd.append('image_weight', String(image_weight));

  const res = await fetch(`${API_BASE}/predict`, { method: 'POST', body: fd });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}
