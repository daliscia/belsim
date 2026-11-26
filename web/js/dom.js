// Tiny DOM helpers. Text always goes in as text nodes (labels come from the API: untrusted).
const SVG_NS = 'http://www.w3.org/2000/svg';

function fill(el, attrs, children) {
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === 'class') el.setAttribute('class', v);
    else if (k === 'style' && typeof v === 'object') Object.assign(el.style, v);
    else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2), v);
    else if (k === 'value' && 'value' in el) el.value = v;
    else if (k === 'checked' || k === 'disabled' || k === 'selected' || k === 'open' || k === 'hidden') el[k] = !!v;
    else el.setAttribute(k, v === true ? '' : v);
  }
  // a table row that reacts to a click is also reachable and usable from the keyboard
  if (el.tagName === 'TR' && attrs && typeof attrs.onclick === 'function') {
    el.tabIndex = 0;
    el.addEventListener('keydown', e => {
      if ((e.key === 'Enter' || e.key === ' ') && e.target === el) { e.preventDefault(); el.click(); }
    });
  }
  for (const c of children.flat(Infinity)) {
    if (c == null || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

export const h = (tag, attrs, ...children) => fill(document.createElement(tag), attrs, children);
export const s = (tag, attrs, ...children) => fill(document.createElementNS(SVG_NS, tag), attrs, children);
export const $ = (sel, root = document) => root.querySelector(sel);

export function toast(msg, isError = false) {
  const t = h('div', { class: 'toast' + (isError ? ' err' : '') }, msg);
  $('#toasts').append(t);
  setTimeout(() => t.remove(), isError ? 7000 : 2600);
}

export function download(filename, text, type = 'text/plain') {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = h('a', { href: url, download: filename });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function debounce(fn, ms = 150) {
  let t;
  return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
}
