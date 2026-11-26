// Application state, API access, unit conversion and formatting.
// Everything about *which* series, lines or parameters exist comes from the API.
import { t, lang, locale } from './i18n.js';

export const state = {
  meta: null,            // /api/meta
  params: null,          // /api/params
  library: { measures: [], scenarios: [], errors: [] },
  saved: [],             // /api/scenarios
  parties: null,         // /api/scenarios?kind=party (party programmes of the 2024 costing)
  years: [],
  base: {},              // baseline series (full horizon)
  run: null,             // last /api/run response, or null
  runFor: null,          // JSON of the scenario that produced `run`
  running: false,
  runError: null,
  band: null,            // last /api/run_band response (low / central / high / static runs)
  bandFor: null,         // scenario signature that produced `band`
  bandLoading: false,
  bandError: null,
  compare: [],           // named scenarios kept for the Compare view: {name, description, measures, slot}
  scenario: { name: 'My scenario', description: '', measures: [] },
  year: 2030,
  unit: 'bn',            // bn | real | pct
  view: 'overview',
  panelOpen: true,
  ui: {},                // per-view UI state (selected line, tabs, collapsed groups, ...)
};

const listeners = new Set();
export const subscribe = fn => listeners.add(fn);
let queued = false;
export function emit() {
  if (queued) return;
  queued = true;
  queueMicrotask(() => { queued = false; persist(); listeners.forEach(fn => fn()); });
}

// ------------------------------------------------------------------ persistence
const LS = 'belsim.state.v1';
export const MAX_COMPARE = 7;
export function persist() {
  try {
    localStorage.setItem(LS, JSON.stringify({
      scenario: state.scenario, year: state.year, unit: state.unit, panelOpen: state.panelOpen, ui: state.ui,
      compare: state.compare,
    }));
  } catch (e) { /* storage unavailable: the session still works */ }
}
export function restore() {
  try {
    const d = JSON.parse(localStorage.getItem(LS) || 'null');
    if (!d) return;
    if (d.scenario && Array.isArray(d.scenario.measures)) state.scenario = d.scenario;
    if (Number.isInteger(d.year)) state.year = d.year;
    if (['bn', 'real', 'pct'].includes(d.unit)) state.unit = d.unit;
    if (typeof d.panelOpen === 'boolean') state.panelOpen = d.panelOpen;
    if (d.ui && typeof d.ui === 'object') state.ui = d.ui;
    if (Array.isArray(d.compare)) state.compare = d.compare.filter(c => c && c.name && Array.isArray(c.measures)).slice(0, MAX_COMPARE);
  } catch (e) { /* ignore corrupt storage */ }
}

// ------------------------------------------------------------------ API
/** Every request carries the interface language: the server translates the text it provides. */
export async function api(path, opts) {
  let resp;
  try {
    resp = await fetch(path + (path.includes('?') ? '&' : '?') + 'lang=' + lang(), opts);
  } catch (e) {
    throw new Error(t('common.server_unreachable', { cmd: 'uv run python -m belsim.api' }));
  }
  let body = null;
  try { body = await resp.json(); } catch (e) { /* non-JSON error page */ }
  if (!resp.ok) {
    const detail = body && (body.error || body.detail);
    const err = new Error(typeof detail === 'string' ? detail : detail ? JSON.stringify(detail) : `${resp.status} ${resp.statusText}`);
    err.status = resp.status;
    throw err;
  }
  return body;
}
/**
 * A slow report the server computes in the background: asks without waiting (`wait=0`) and asks
 * again every few seconds while the answer is 202 {"status": "computing"}.
 */
export async function apiWhenReady(path, { every = 4000, tries = 90 } = {}) {
  const url = path + (path.includes('?') ? '&' : '?') + 'wait=0';
  for (let i = 0; i < tries; i++) {
    const body = await api(url);
    if (!body || body.status !== 'computing') return body;
    await new Promise(r => setTimeout(r, every));
  }
  throw new Error(t('common.still_computing'));
}
export const post = (path, data) =>
  api(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) });

/**
 * Data a view needs but the boot does not load (validation, rankings, distribution ...).
 * Returns {status: 'loading' | 'ok' | 'error', data, error}; the view re-renders when it settles.
 */
const lazyStore = new Map();
export function lazy(key, loader) {
  let e = lazyStore.get(key);
  if (!e) {
    e = { status: 'loading', data: null, error: null };
    lazyStore.set(key, e);
    if (lazyStore.size > 80) lazyStore.delete(lazyStore.keys().next().value);
    Promise.resolve().then(loader).then(d => { e.status = 'ok'; e.data = d; }, err => { e.status = 'error'; e.error = err; }).finally(emit);
  }
  return e;
}
export function lazyReset(prefix = '') {
  for (const k of [...lazyStore.keys()]) if (k.startsWith(prefix)) lazyStore.delete(k);
}
/** True when an endpoint does not exist (yet): another module may still be under construction. */
export const isMissing = err => !!err && (err.status === 404 || err.status === 405 || err.status === 501);

const arrayCache = new Map();
export function runId(which = 'scenario') { return which === 'scenario' && state.run ? state.run.id : 'baseline'; }
export function fetchArrays(which, year) {
  const key = `${state.meta.params_signature}:${runId(which)}:${year}`;
  if (!arrayCache.has(key)) {
    if (arrayCache.size > 400) arrayCache.clear();
    const p = api(`/api/arrays?run=${encodeURIComponent(runId(which))}&year=${year}`);
    p.catch(() => arrayCache.delete(key));
    arrayCache.set(key, p);
  }
  return arrayCache.get(key);
}
export function fetchPyramid(which) {
  const key = `${state.meta.params_signature}:pyr:${runId(which)}`;
  if (!arrayCache.has(key)) {
    const p = api(`/api/pyramid?run=${encodeURIComponent(runId(which))}`);
    p.catch(() => arrayCache.delete(key));
    arrayCache.set(key, p);
  }
  return arrayCache.get(key);
}

// ------------------------------------------------------------------ scenario
export function activeScenario() {
  return { name: state.scenario.name, description: state.scenario.description || '', measures: state.scenario.measures };
}
export const enabledMeasures = () => state.scenario.measures.filter(m => m.enabled !== false && (m.changes || []).length);
export const sigOf = measures => JSON.stringify((measures || []).filter(m => m.enabled !== false && (m.changes || []).length).map(m => m.changes));
const sig = () => sigOf(state.scenario.measures);
export const scenarioSig = sig;
export const hasRun = () => !!state.run;
export const isStale = () => (state.run ? state.runFor !== sig() : enabledMeasures().length > 0);

export async function runScenario() {
  if (state.running) return;
  if (!enabledMeasures().length) {
    state.run = null; state.runFor = null; state.runError = null; state.band = null; state.bandFor = null;
    emit();
    return;
  }
  const mine = sig();
  state.running = true; state.runError = null;
  emit();
  try {
    state.run = await post('/api/run', { ...activeScenario(), options: {} });
    state.runFor = mine;
    if (state.bandFor !== mine) { state.band = null; state.bandFor = null; }
  } catch (e) {
    state.runError = e.message;
  } finally {
    state.running = false;
    emit();
  }
  if (state.ui.band && state.run && state.runFor === mine) loadBand();
}

// Sensitivity band: the same scenario with behavioural responses scaled 0.5 / 1 / 1.5 and switched off.
export const BAND_RUNS = ['low', 'central', 'high', 'static'];
export const bandReady = () => !!(state.ui.band && state.band && state.run && state.bandFor === state.runFor);
export async function loadBand() {
  if (state.bandLoading || !state.run) return;
  const mine = state.runFor;
  if (state.band && state.bandFor === mine) return;
  state.bandLoading = true; state.bandError = null;
  emit();
  try {
    const r = await post('/api/run_band', { ...activeScenario(), options: {} });
    if (state.runFor === mine) { state.band = r; state.bandFor = mine; }
  } catch (e) {
    state.bandError = e.message;
  } finally {
    state.bandLoading = false;
    emit();
  }
}
export function setBand(on) {
  state.ui.band = !!on;
  emit();
  if (on && state.run && !isStale()) loadBand();
}

// ------------------------------------------------------------------ series access
export const yearIndex = (y = state.year) => state.years.indexOf(y);
export function raw(key, which = 'scenario') {
  if (which === 'scenario' && state.run) return state.run.scenario.series[key] || null;
  if (which !== 'scenario' && which !== 'baseline') {
    // one of the sensitivity runs (low / central / high / static)
    const r = state.band && state.band.runs && state.band.runs[which];
    return (r && r.series[key]) || null;
  }
  return state.base[key] || null;
}
export const has = key => !!state.base[key] || !!(state.run && state.run.scenario.series[key]);
export const kindOf = key => (state.meta.units[key] || guessKind(key));
function guessKind(key) {
  if (/_pct_gdp/.test(key)) return 'pct';
  if (/^(revenue|expenditure|fiscal|va)\./.test(key)) return 'eur_mn';
  return 'number';
}

/** How a key is displayed under the current unit setting: label, decimals, and converter. */
export function unitFor(key, unit = state.unit, kind = kindOf(key)) {
  const base = state.meta ? state.meta.base_year : 2024;
  switch (kind) {
    case 'eur_mn':
      if (unit === 'pct') return { label: t('unit.pct_gdp'), short: t('unit.pct_gdp_short'), dec: 2, pct: true, conv: (v, i, w) => div(v * 100, at('macro.gdp_nominal', w, i)) };
      if (unit === 'real') return { label: t('unit.bn_real', { year: base }), short: t('unit.bn'), dec: 1, conv: (v, i, w) => div(v / 1000, at('macro.cpi', w, i)) };
      return { label: t('unit.bn'), short: t('unit.bn'), dec: 1, conv: v => v / 1000 };
    case 'eur_mn_real': return { label: t('unit.bn_real', { year: base }), short: t('unit.bn'), dec: 1, conv: v => v / 1000 };
    case 'eur':
      if (unit === 'real') return { label: t('unit.eur_real', { year: base }), short: '€', dec: 0, conv: (v, i, w) => div(v, at('macro.cpi', w, i)) };
      return { label: '€', short: '€', dec: 0, conv: v => v };
    case 'persons': return { label: t('unit.thousand_persons'), short: t('unit.k'), dec: 1, conv: v => v / 1000 };
    case 'rate': return { label: '%', short: '%', dec: 2, pct: true, conv: v => v * 100 };
    case 'pct': return { label: /_pct_gdp/.test(key) ? t('unit.pct_gdp') : '%', short: '%', dec: 2, pct: true, conv: v => v };
    case 'index': return { label: t('unit.index', { year: base }), short: '', dec: 3, conv: v => v };
    case 'years': return { label: t('unit.years'), short: t('unit.y'), dec: 1, conv: v => v };
    default: return { label: '', short: '', dec: 3, conv: v => v };
  }
}
const div = (a, b) => (b ? a / b : null);
function at(key, which, i) { const s = raw(key, which); return s ? s[i] : null; }

/** Converted series for display; null entries where the model gives none. */
export function conv(key, which = 'scenario', unit = state.unit) {
  const s = raw(key, which);
  if (!s) return null;
  const u = unitFor(key, unit);
  return s.map((v, i) => (v == null ? null : u.conv(v, i, which)));
}
export function valueAt(key, which = 'scenario', year = state.year, unit = state.unit) {
  const s = raw(key, which);
  const i = yearIndex(year);
  if (!s || i < 0 || s[i] == null) return null;
  return unitFor(key, unit).conv(s[i], i, which);
}
/** Scenario minus baseline in display units (so in % of GDP mode it is a difference of ratios). */
export function deltaAt(key, year = state.year, unit = state.unit) {
  if (!state.run) return null;
  const a = valueAt(key, 'baseline', year, unit), b = valueAt(key, 'scenario', year, unit);
  return a == null || b == null ? null : b - a;
}
export function deltaSeries(key, unit = state.unit) {
  const a = conv(key, 'baseline', unit), b = conv(key, 'scenario', unit);
  if (!a || !b) return null;
  return b.map((v, i) => (v == null || a[i] == null ? null : v - a[i]));
}
/** Sum of several keys, converted (used for groups of budget lines). */
export function sumAt(keys, which, year = state.year, unit = state.unit) {
  let t = 0, any = false;
  for (const k of keys) { const v = valueAt(k, which, year, unit); if (v != null) { t += v; any = true; } }
  return any ? t : null;
}
export function sumSeries(keys, which, unit = state.unit) {
  const out = state.years.map(() => null);
  for (const k of keys) {
    const s = conv(k, which, unit);
    if (!s) continue;
    s.forEach((v, i) => { if (v != null) out[i] = (out[i] || 0) + v; });
  }
  return out;
}

// ------------------------------------------------------------------ formatting
// Numbers follow the interface language: 1,234.5 (en), 1 234,5 (fr), 1.234,5 (nl).
const nf = new Map();
function nfOf(dec) {
  const id = locale() + ':' + dec;
  if (!nf.has(id)) nf.set(id, new Intl.NumberFormat(locale(), { minimumFractionDigits: dec, maximumFractionDigits: dec }));
  return nf.get(id);
}
const isZero = (v, dec) => Math.abs(v) < 0.5 * 10 ** -dec;
export function num(v, dec = 1) {
  if (v == null || !isFinite(v)) return '–';
  return nfOf(dec).format(isZero(v, dec) ? 0 : v);
}
export function signed(v, dec = 1) {
  if (v == null || !isFinite(v)) return '–';
  if (isZero(v, dec)) return nfOf(dec).format(0);
  return (v > 0 ? '+' : '−') + nfOf(dec).format(Math.abs(v));
}
/** A number in the shortest exact form (up to `max` decimals), for parameter values. */
export function plain(v, max = 6) {
  if (v == null || !isFinite(v)) return '–';
  return new Intl.NumberFormat(locale(), { maximumFractionDigits: max, useGrouping: Math.abs(v) >= 10000 }).format(v);
}
/** "3.2%" / "3,2 %": a formatted number with the percent sign of the language. */
export const pct = (v, dec = 1) => (v == null || !isFinite(v) ? '–' : t('fmt.pct', { v: num(v, dec) }));
export const signedPct = (v, dec = 1) => (v == null || !isFinite(v) ? '–' : t('fmt.pct', { v: signed(v, dec) }));
/** "€ 1,234" / "1 234 €": an amount in euros. `text` may be an already formatted (signed) number. */
export const money = (v, dec = 0) => (v == null || !isFinite(v) ? '–' : t('fmt.eur', { v: num(v, dec) }));
export const signedMoney = (v, dec = 0) => (v == null || !isFinite(v) ? '–' : t('fmt.eur', { v: signed(v, dec) }));
/** "€ 3.2 bn" / "3,2 Mrd €" (the value is already in billions). */
export const moneyBn = (v, dec = 1) => (v == null || !isFinite(v) ? '–' : t('fmt.eur_bn', { v: num(v, dec) }));
/** "12.3k": thousands, for chart axes and tooltips. */
export const kilo = (v, dec = 1) => (v == null || !isFinite(v) ? '–' : num(v / 1000, dec) + t('unit.k'));
/** An ISO date (or Date) in the language's long form: 25 November 2026 / 25 novembre 2026. */
export function fmtDate(d) {
  const date = d instanceof Date ? d : new Date(String(d).length <= 10 ? d + 'T00:00:00' : d);
  if (isNaN(date)) return String(d);
  return new Intl.DateTimeFormat(locale(), { day: 'numeric', month: 'long', year: 'numeric' }).format(date);
}
/** Compares strings in the language's alphabetical order. */
export const compareText = (a, b) => String(a).localeCompare(String(b), locale());
export const fmtKey = (key, v) => num(v, unitFor(key).dec);
/** Formatter for chart axes/tooltips of a key under the current unit. */
export function formatter(key) {
  const u = unitFor(key);
  return v => (u.pct ? pct(v, u.dec) : num(v, u.dec));
}

/** +1 if an increase is good news, -1 if bad, 0 if neutral. Only used to colour deltas. */
export function polarity(key) {
  if (/unemploy|debt|interest$|\.interest\b/.test(key)) return -1;
  if (/unemp/.test(key)) return 0;
  if (/balance|gdp_nominal|gdp_real|\.gdp(\.|$)|gdp_per_capita|household_income|employment|activity_rate|productivity/.test(key)) return 1;
  return 0;
}

// ------------------------------------------------------------------ labels
// Labels of series, lines and dimension members come from the server in the interface language
// (`meta.series_labels`, `meta.labels`); `humanise` is the last resort for a key it does not know.
const ACR = { gdp: 'GDP', cpi: 'CPI', tfr: 'Total fertility rate', pct: '%', vat: 'VAT', pit: 'PIT', cit: 'CIT', ssc: 'SSC', eu: 'EU', '10y': '10-year' };
export function humanise(code) {
  const words = String(code).split(/[_\s]+/).filter(Boolean).map(w => ACR[w.toLowerCase()] || w);
  const text = words.join(' ').replace('% GDP', '(% of GDP)');
  return text.charAt(0).toUpperCase() + text.slice(1);
}
export function dimLabel(dim, code) {
  const table = state.meta.labels[dim];
  return (table && table[code]) || humanise(code);
}
export function label(key) {
  const m = state.meta;
  if (m.series_labels && m.series_labels[key]) return m.series_labels[key];
  const [sec, ...rest] = key.split('.');
  const tail = rest.join('.');
  const last = rest[rest.length - 1];
  if (sec === 'revenue' && m.revenue_lines[tail]) return m.revenue_lines[tail].label;
  if (sec === 'expenditure' && m.expenditure_lines[tail]) return m.expenditure_lines[tail].label;
  if (sec === 'status') return dimLabel('statuses', tail);
  if (sec === 'emp' && rest[0] === 'sector') return dimLabel('sectors', rest[1]);
  if (sec === 'emp' && rest[0] === 'jobtype') return dimLabel('jobtypes', rest[1]);
  if (sec === 'va' && rest[0] === 'sector') return dimLabel('sectors', rest[1]);
  if (sec === 'pop' && rest[0] === 'region') return dimLabel('regions', rest[1]);
  if (sec === 'pop' && rest[0] === 'sex') return dimLabel('sexes', rest[1]);
  if (sec === 'macro' && rest[0] === 'cons') return dimLabel('cons_cats', rest.slice(1).join('.'));
  if (rest.length > 1 && m.dims.regions.includes(last)) return `${label([sec, ...rest.slice(0, -1)].join('.'))}, ${dimLabel('regions', last)}`;
  if (rest.length > 1 && (m.labels.entities || {})[last]) return `${label([sec, ...rest.slice(0, -1)].join('.'))}, ${dimLabel('entities', last)}`;
  return humanise(rest.length ? rest.join(' ') : sec);
}

// ------------------------------------------------------------------ regions
export const regionCodes = () => (state.meta ? state.meta.dims.regions : []);
/** Fixed colour per region (by its position in the model's list, never by rank). */
export const regionColor = code => `var(--c${(Math.max(0, regionCodes().indexOf(code)) % 8) + 1})`;
/** Territories a parameter can be keyed by: the regions, and the federated entities (communities). */
export const territoryCodes = () => [...regionCodes(), ...Object.entries((state.meta && state.meta.entity_detail) || {}).filter(([, d]) => !d || d.kind !== 'consolidation').map(([e]) => e)];
export const territoryLabel = code => (regionCodes().includes(code) ? dimLabel('regions', code) : dimLabel('entities', code));
/** A parameter that holds one value per region (or per community): a regional competence. */
export function isRegional(p) {
  if (!p) return false;
  const v = p.current, codes = territoryCodes();
  return !!v && typeof v === 'object' && !Array.isArray(v) && Object.keys(v).length > 0 && Object.keys(v).every(k => codes.includes(k));
}
/** Territory code a change addresses ('FLA' from key 'FLA' or 'FLA.2'), or null. */
export function regionOfKey(key) {
  if (key == null) return null;
  const head = String(key).split('.')[0];
  return territoryCodes().includes(head) ? head : null;
}
/** Regions a territory code covers: itself for a region, the regions served for a community. */
export function regionsOf(code) {
  if (regionCodes().includes(code)) return [code];
  const info = ((state.meta && state.meta.entity_detail) || {})[code];
  return (info && Array.isArray(info.regions) ? info.regions : []).filter(r => regionCodes().includes(r));
}
/** Metrics published per region: [{id, label, regions:[codes present]}], from every `regional.<metric>.<REG>` key. */
export function regionalMetrics() {
  const regs = regionCodes(), out = new Map();
  const keys = new Set(Object.keys(state.base));
  if (state.run) Object.keys(state.run.scenario.series).forEach(k => keys.add(k));
  for (const k of keys) {
    if (!k.startsWith('regional.')) continue;
    const parts = k.split('.'), reg = parts[parts.length - 1];
    if (parts.length < 3 || !regs.includes(reg)) continue;
    const id = parts.slice(1, -1).join('.');
    if (!out.has(id)) out.set(id, { id, label: (state.meta.regional_metrics || {})[id] || humanise(id.replace(/\./g, ' ')), regions: [] });
    out.get(id).regions.push(reg);
  }
  return [...out.values()];
}
export const regionalKey = (metric, reg) => `regional.${metric}.${reg}`;

// ------------------------------------------------------------------ colours
export const color = i => `var(--c${(i % 8) + 1})`;
export const OTHER = 'var(--other)';

/** Keep at most `n` items by weight (colour by their original order), fold the rest into "Other". */
export function fold(items, weight, n = 7) {
  if (items.length <= n + 1) return { keep: items, rest: [] };
  const ranked = [...items].sort((a, b) => weight(b) - weight(a));
  const top = new Set(ranked.slice(0, n));
  return { keep: items.filter(x => top.has(x)), rest: items.filter(x => !top.has(x)) };
}
