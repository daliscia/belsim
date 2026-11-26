// The "How it works" page. The prose lives in web/wp/<lang>.html (one fragment per language, same
// structure); every figure that comes from the model is a placeholder filled here from the API:
//
//   <span class="live" data-v="p.count"></span>     a named value (see `value`)
//   <div data-block="val-table"></div>              a table built here
import { h, $ } from './dom.js';
import { initI18n, setLang, lang, locale, t, LANGS } from './i18n.js';

// A few comparisons with official figures, good and bad, for the short table. `when` picks a
// baseline row; `n` the n-th target of a costed measure (0-based, in the order of the reference
// file), 'last' the last one. The verdict shown is computed from the live gap, never typed.
const SELECTION = [
  { id: 'fpb2026_debt', when: '2031' },
  { id: 'sca2026_pensions', when: '2070' },
  { id: 'arizona_allowance_15600', n: 'last' },
  { id: 'dc2024_vat_food_zero', n: 0 },
  { id: 'hermes2018_employer_ssc_cut_general', n: 4 },
  { id: 'dc2024_pit_45_to_40', n: 0 },
  { id: 'cit_reform_2018_rate_cut', n: 0 },
  { id: 'hermes_vat_increase', n: 'last' },
];
const BUCKETS = [5, 10, 25, 50, Infinity];
const BUCKET_KEYS = ['validation.bucket.b0', 'validation.bucket.b1', 'validation.bucket.b2', 'validation.bucket.b3', 'validation.bucket.b4'];
const data = { version: null, meta: null, parties: null, validation: {}, failed: new Set() };
let fragmentFor = null;

// ------------------------------------------------------------------ formats
const nf = (dec = 0) => new Intl.NumberFormat(locale(), { minimumFractionDigits: dec, maximumFractionDigits: dec });
const int = v => nf(0).format(v);
const fixed = (v, dec) => nf(dec).format(v).replace('-', '−');
const pctText = (v, dec = 1) => fixed(v, dec) + (lang() === 'fr' ? ' %' : '%');
function dateText(iso) {
  const d = iso ? new Date(iso) : null;
  return d && !isNaN(d) ? new Intl.DateTimeFormat(locale(), { day: 'numeric', month: 'long', year: 'numeric' }).format(d) : null;
}
// ------------------------------------------------------------------ data
async function getJson(path, opts) {
  const resp = await fetch(path + (path.includes('?') ? '&' : '?') + 'lang=' + lang(), opts);
  const body = await resp.json().catch(() => null);
  if (!resp.ok) throw new Error((body && body.error) || `${resp.status}`);
  return body;
}
const optional = (name, promise) => promise.catch(() => { data.failed.add(name); return null; });

async function loadCore() {
  [data.version, data.meta, data.parties] = await Promise.all([
    optional('version', getJson('/api/version')),
    optional('meta', getJson('/api/meta')),
    optional('parties', getJson('/api/scenarios?kind=party')),
  ]);
}

/** The validation report in the page language; null while the server computes it. */
async function loadValidation() {
  const id = lang();
  if (data.validation[id]) return data.validation[id];
  const body = await getJson('/api/validation?wait=0');
  if (body && body.status === 'computing') return null;
  data.validation[id] = body;
  return body;
}

// ------------------------------------------------------------------ named values
/** The party scenarios, and the reference rows whose parameter was sized to the official figure. */
function partyCounts() {
  const p = data.parties;
  if (!p || !Array.isArray(p.scenarios)) return null;
  const sized = new Set();
  for (const sc of p.scenarios) {
    for (const m of sc.measures || []) if (m.validation_id && m.method === 'sized') sized.add(m.validation_id);
  }
  return { n: p.scenarios.length, sized };
}
/** Official costings of measures the model was not fitted to: how many are compared, and how close. */
function independent(v) {
  const pc = partyCounts();
  const sized = pc ? pc.sized : new Set();
  const cmp = v.measures.filter(r => !sized.has(r.id) && r.gap_pct != null);
  const within = m => (cmp.length ? 100 * cmp.filter(r => Math.abs(r.gap_pct) <= m).length / cmp.length : null);
  return { n: cmp.length, w10: within(10), w25: within(25), w50: within(50) };
}
const sharePct = x => (x == null ? null : pctText(x, 0));

function value(key) {
  const v = data.version, m = data.meta, pc = partyCounts();
  const val = data.validation[lang()];
  const ind = val ? independent(val) : null;
  const table = {
    'v.commit': () => v && v.commit_short,
    'v.date': () => v && dateText(v.commit_date),
    'p.count': () => v && v.params != null && int(v.params),
    'p.assumptions': () => v && v.assumptions != null && int(v.assumptions),
    'm.end': () => (v && v.end_year ? String(v.end_year) : m && String(m.end_year)),
    'party.n': () => pc && int(pc.n),
    'val.n': () => ind && int(ind.n),
    'val.w10': () => ind && sharePct(ind.w10),
    'val.w25': () => ind && sharePct(ind.w25),
    'val.w50': () => ind && sharePct(ind.w50),
  };
  const fn = table[key];
  const out = fn ? fn() : null;
  return out === false || out === undefined || out === '' ? null : out;
}
/** Values that arrive later than the rest: shown as "computing" rather than "not available". */
const isPending = key => !data.loaded || (/^val\./.test(key) && !data.validation[lang()] && !data.failed.has('validation'));

function fillValues(root) {
  for (const el of root.querySelectorAll('[data-v]')) {
    const key = el.dataset.v, v = value(key);
    el.classList.remove('pending', 'missing');
    if (v != null) { el.textContent = v; el.removeAttribute('title'); continue; }
    if (isPending(key)) { el.textContent = '…'; el.classList.add('pending'); el.title = t('wp.pending'); continue; }
    el.textContent = t('wp.na'); el.classList.add('missing');
  }
}

// ------------------------------------------------------------------ blocks
const table = (cls, caption, head, rows) => h('div', { class: 'tbl-wrap' }, h('table', { class: cls },
  caption ? h('caption', null, caption) : null, h('thead', null, h('tr', null, head)), h('tbody', null, rows)));

const BLOCKS = {
  'val-table'() {
    const v = data.validation[lang()];
    if (!v) return validationPlaceholder();
    const pick = sel => {
      let rows = v.baseline.filter(r => r.id === sel.id);
      if (!rows.length) rows = v.measures.filter(r => r.id === sel.id);
      if (!rows.length) return null;
      if (sel.when != null) return rows.find(r => String(r.when) === sel.when) || null;
      return sel.n === 'last' ? rows[rows.length - 1] : rows[sel.n] || null;
    };
    const num = (x, ref) => (x == null ? '–' : fixed(x, Math.abs(ref ?? x) >= 1000 ? 0 : Math.abs(ref ?? x) >= 100 ? 1 : 2));
    const body = SELECTION.map(pick).filter(r => r && r.gap_pct != null).map(r => {
      const g = r.gap_pct, i = BUCKETS.findIndex(b => Math.abs(g) <= b);
      return h('tr', null,
        h('td', { title: r.description || null }, brief(r.description || r.id), h('span', { class: 'sub' }, [r.label || r.when, r.unit].filter(Boolean).join(', '))),
        h('td', { class: 'n' }, num(r.ref, r.ref)), h('td', { class: 'n' }, num(r.model, r.ref)),
        h('td', { class: 'n' }, (g > 0 ? '+' : '') + pctText(g, Math.abs(g) >= 10 ? 0 : 1)),
        h('td', { class: 'v' }, h('span', { class: 'verdict q' + i }, t(BUCKET_KEYS[i]))));
    });
    if (!body.length) return null;
    return table('', t('wp.val.caption'),
      [h('th', null, t('wp.val.col.what')), h('th', { class: 'n' }, t('wp.val.col.official')), h('th', { class: 'n' }, t('wp.val.col.model')), h('th', { class: 'n' }, t('wp.val.col.gap')), h('th', null, '')],
      body);
  },
};
/** The first clause of a reference's description ("Institution - what"); the whole text is the cell's tooltip. */
function brief(text) {
  const i = text.indexOf(' - ');
  if (i < 0) return text;
  return text.slice(0, i) + ' - ' + text.slice(i + 3).split(/, | \(|; |\. /)[0];
}
function validationPlaceholder() {
  if (data.failed.has('validation')) return h('p', { class: 'computing' }, t('wp.val.failed'));
  return h('p', { class: 'computing', role: 'status' }, t('wp.val.computing'));
}
function fillBlocks(root) {
  for (const el of root.querySelectorAll('[data-block]')) {
    const fn = BLOCKS[el.dataset.block];
    let node = null;
    try { node = fn ? fn() : null; } catch (e) { console.error(e); }
    el.replaceChildren(...(node ? [node] : []));
  }
}

// ------------------------------------------------------------------ page
function renderChrome() {
  document.title = t('wp.doc_title');
  const desc = document.querySelector('meta[name="description"]');
  if (desc) desc.content = t('wp.meta_description');
  $('#skip').textContent = t('common.skip');
  $('#top-sub').textContent = t('nav.methodology');
  $('#app-link').textContent = t('wp.open_app');
  $('#print-btn').textContent = t('wp.print');
  const dark = document.documentElement.dataset.theme === 'dark';
  $('#theme-btn').textContent = dark ? t('bar.light_theme') : t('bar.dark_theme');
  const seg = $('#lang-seg');
  seg.setAttribute('aria-label', t('bar.lang_aria'));
  seg.replaceChildren(...LANGS.map(l => h('button', { type: 'button', 'aria-pressed': String(lang() === l.id), title: l.name, lang: l.id, onclick: () => changeLang(l.id) }, l.label)));
  $('#paper-foot').replaceChildren(h('span', null, t('foot.disclaimer')), h('a', { href: './' }, t('wp.open_app')));
  const failed = [...data.failed].filter(x => x !== 'validation');
  const notice = $('#notice');
  notice.hidden = !failed.length;
  notice.className = failed.length ? 'err' : '';
  notice.textContent = failed.length ? t('wp.live_failed') : '';
}

function fill() {
  const article = $('#article');
  fillBlocks(article);
  fillValues(article);
}

async function loadFragment() {
  const id = lang();
  const resp = await fetch(`wp/${id}.html`, { cache: 'no-cache' });
  if (!resp.ok) throw new Error(`wp/${id}.html: ${resp.status}`);
  // our own static file, parsed inert: no script in it runs, and none is expected
  const doc = new DOMParser().parseFromString(await resp.text(), 'text/html');
  doc.querySelectorAll('script, style, link, iframe, object').forEach(el => el.remove());
  const article = $('#article');
  article.replaceChildren(...[...doc.body.childNodes].map(n => document.importNode(n, true)));
  fragmentFor = id;
  fill();
  if (location.hash.length > 1) { const el = document.getElementById(location.hash.slice(1)); if (el) el.scrollIntoView(); }
}

let polling = null;
async function pollValidation() {
  clearTimeout(polling);
  try {
    const v = await loadValidation();
    if (!v) { polling = setTimeout(pollValidation, 5000); return; }
    data.failed.delete('validation');
  } catch (e) {
    data.failed.add('validation');
  }
  if (fragmentFor === lang()) fill();
}
async function changeLang(id) {
  if (id === lang()) return;
  await setLang(id);
  const url = new URL(location.href);
  url.searchParams.set('lang', id);
  history.replaceState(null, '', url);
  renderChrome();
  await loadFragment().catch(e => console.error(e));
  // texts the server translates (validation rows) are fetched again in the new language
  pollValidation();
}

async function start() {
  await initI18n();
  const q = new URLSearchParams(location.search).get('lang');
  if (q && q !== lang() && LANGS.some(l => l.id === q)) await setLang(q);
  renderChrome();
  $('#theme-btn').addEventListener('click', () => {
    const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem('belsim.theme', next); } catch (e) { /* ignore */ }
    renderChrome();
  });
  $('#print-btn').addEventListener('click', () => window.print());
  $('#skip').addEventListener('click', e => { e.preventDefault(); $('#paper').focus(); });

  await loadFragment().catch(e => {
    $('#article').replaceChildren(h('p', null, String(e.message)));
  });
  await loadCore();
  data.loaded = true;
  renderChrome();
  fill();
  pollValidation();
}
start();
