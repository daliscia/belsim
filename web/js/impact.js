// Measure impact table: scenario minus baseline for chosen years, plus optional breakdowns
// by region and by measure (each measure simulated on its own).
import { h, toast } from './dom.js';
import {
  state, emit, post, raw, has, yearIndex, num, signed, activeScenario, enabledMeasures, isStale, bandReady, setBand,
} from './store.js';
import { t } from './i18n.js';
import { card, deltaSpan, notice } from './components.js';
import { regionalBreakdown, runGetter, runCpi } from './regional.js';

const DEFAULT_YEARS = [2025, 2026, 2027, 2028, 2029, 2030, 2035, 2040, 2050, 2060, 2070];
export { yearsEditor };

export function impactYears() {
  const want = Array.isArray(state.ui.impactYears) && state.ui.impactYears.length ? state.ui.impactYears : DEFAULT_YEARS;
  return want.filter(y => state.years.includes(y));
}

const v = (s, i) => (s && s[i] != null ? s[i] : null);
const realFactor = (which, i) => (state.unit === 'real' ? v(raw('macro.cpi', which), i) || 1 : 1);
export const bnUnit = () => (state.unit === 'real' ? t('unit.bn_real', { year: state.meta.base_year }) : t('unit.bn'));

// Each row computes its own delta from raw series so that it does not depend on the unit switch
// (except € bn rows, which follow current vs 2024 prices).
function rowsSpec() {
  const eur = key => i => {
    const b = v(raw(key, 'baseline'), i), sc = v(raw(key, 'scenario'), i);
    return b == null || sc == null ? null : (sc / realFactor('scenario', i) - b / realFactor('baseline', i)) / 1000;
  };
  const ratio = key => i => {
    const b = v(raw(key, 'baseline'), i), sc = v(raw(key, 'scenario'), i);
    const gb = v(raw('macro.gdp_nominal', 'baseline'), i), gs = v(raw('macro.gdp_nominal', 'scenario'), i);
    return b == null || sc == null || !gb || !gs ? null : (sc / gs - b / gb) * 100;
  };
  const rel = key => i => {
    const b = v(raw(key, 'baseline'), i), sc = v(raw(key, 'scenario'), i);
    return !b || sc == null ? null : (sc / b - 1) * 100;
  };
  const diff = (key, scale) => i => {
    const b = v(raw(key, 'baseline'), i), sc = v(raw(key, 'scenario'), i);
    return b == null || sc == null ? null : (sc - b) * scale;
  };
  const bn = bnUnit(), pts = t('common.pts_gdp');
  return [
    { head: t('impact.group.budget') },
    { name: t('impact.row.revenue'), unit: bn, key: 'fiscal.revenue_total', f: eur('fiscal.revenue_total'), dec: 2, compact: true },
    { name: t('impact.row.expenditure'), unit: bn, key: 'fiscal.expenditure_total', f: eur('fiscal.expenditure_total'), dec: 2, pol: 0, compact: true },
    { name: t('impact.row.balance'), unit: bn, key: 'fiscal.balance', f: eur('fiscal.balance'), dec: 2, strong: true },
    { name: t('impact.row.revenue'), unit: pts, key: 'fiscal.revenue_total', f: ratio('fiscal.revenue_total'), dec: 2, compact: true },
    { name: t('impact.row.expenditure'), unit: pts, key: 'fiscal.expenditure_total', f: ratio('fiscal.expenditure_total'), dec: 2, compact: true },
    { name: t('impact.row.balance'), unit: pts, key: 'fiscal.balance', f: ratio('fiscal.balance'), dec: 2, strong: true },
    { name: t('impact.row.debt'), unit: bn, key: 'fiscal.debt', f: eur('fiscal.debt'), dec: 1, compact: true },
    { name: t('impact.row.debt'), unit: pts, key: 'fiscal.debt', f: ratio('fiscal.debt'), dec: 2, compact: true },
    { head: t('impact.group.economy') },
    { name: t('impact.row.gdp_nominal'), unit: '%', key: 'macro.gdp_nominal', f: rel('macro.gdp_nominal'), dec: 2 },
    { name: t('impact.row.gdp_real'), unit: '%', key: 'macro.gdp_real', f: rel('macro.gdp_real'), dec: 2, compact: true },
    { name: t('impact.row.employment'), unit: t('common.thousand'), key: 'labour.employment', f: diff('labour.employment', 1 / 1000), dec: 1, compact: true },
    { name: t('impact.row.employment_rate'), unit: t('common.pts'), key: 'labour.employment_rate_20_64', f: diff('labour.employment_rate_20_64', 100), dec: 2 },
    { name: t('impact.row.unemployment_rate'), unit: t('common.pts'), key: 'labour.unemployment_rate', f: diff('labour.unemployment_rate', 100), dec: 2 },
  ];
}

function yearsEditor() {
  const input = h('input', { type: 'text', value: impactYears().join(', '), style: { width: '230px' }, 'aria-label': t('impact.years_aria') });
  const apply = () => {
    const ys = [...new Set((input.value.match(/\d{4}/g) || []).map(Number))].filter(y => state.years.includes(y)).sort();
    if (!ys.length) { toast(t('impact.years_error', { first: state.years[0], last: state.years[state.years.length - 1] }), true); return; }
    state.ui.impactYears = ys; emit();
  };
  input.addEventListener('change', apply);
  return h('label', { class: 'row small muted' }, t('impact.years'), input);
}

async function loadBreakdown() {
  state.ui.breakdownLoading = true; emit();
  try {
    const r = await post('/api/run', { ...activeScenario(), options: { per_measure: true } });
    breakdown = { id: r.id, per: r.per_measure || [] };
  } catch (e) {
    toast(e.message, true);
  } finally {
    state.ui.breakdownLoading = false; emit();
  }
}
let breakdown = null;

function breakdownTable(ys) {
  if (!breakdown || !state.run || breakdown.id !== state.run.id) return null;
  const real = state.unit === 'real';
  const cpiB = raw('macro.cpi', 'baseline');
  const val = (m, i) => {
    const d = v(m.delta['fiscal.balance'], i);
    if (d == null) return null;
    if (!real) return d / 1000;
    // deflate each run with its own CPI: (scenario balance / scenario cpi) - (baseline / baseline cpi)
    const sc = v(m.scenario['fiscal.balance'], i), cs = v(m.scenario['macro.cpi'], i) || 1, cb = v(cpiB, i) || 1;
    return (sc / cs - (sc - d) / cb) / 1000;
  };
  const total = i => {
    const b = v(raw('fiscal.balance', 'baseline'), i), sc = v(raw('fiscal.balance', 'scenario'), i);
    return b == null || sc == null ? null : (sc / realFactor('scenario', i) - b / realFactor('baseline', i)) / 1000;
  };
  const idx = ys.map(y => yearIndex(y));
  const body = breakdown.per.map(m => h('tr', null, h('td', null, m.name),
    idx.map(i => h('td', { class: 'n' }, deltaSpan('fiscal.balance', val(m, i), 2)))));
  const sums = idx.map(i => breakdown.per.reduce((tot, m) => tot + (val(m, i) || 0), 0));
  body.push(h('tr', null, h('td', { class: 'sub' }, t('impact.breakdown.interaction')),
    idx.map((i, j) => h('td', { class: 'n sub' }, total(i) == null ? '–' : signed(total(i) - sums[j], 2)))));
  body.push(h('tr', { class: 'total' }, h('td', null, t('impact.breakdown.all')),
    idx.map(i => h('td', { class: 'n' }, deltaSpan('fiscal.balance', total(i), 2)))));
  return h('div', null,
    h('div', { class: 'card-sub', style: { padding: '10px 14px 4px' } }, t('impact.breakdown.sub', { unit: bnUnit() })),
    h('div', { class: 'tbl-wrap tall' }, h('table', { class: 'tbl dense' },
      h('thead', null, h('tr', null, h('th', null, t('common.measure')), ys.map(y => h('th', { class: 'n' }, y)))),
      h('tbody', null, body))));
}

/** Change against the baseline in each region: employment, GDP, regional government balance. */
export function regionRows(groups, idx, cols) {
  const rows = [];
  for (const g of groups) {
    rows.push(h('tr', { class: 'group', style: { cursor: 'default' } }, h('td', { colspan: cols }, `${g.label}, ${g.unit}`)));
    for (const r of g.rows) {
      rows.push(h('tr', null,
        h('td', null, h('span', { class: 'key-box', style: { background: r.color, marginRight: '7px' } }), r.label),
        idx.map(i => { const d = r.f(i); return h('td', { class: 'n' }, d == null ? '–' : deltaSpan(g.polKey, d, g.dec)); })));
    }
  }
  return rows;
}

function regionTable(ys) {
  if (state.ui.impactRegions === false) return null;
  const groups = regionalBreakdown(runGetter, { real: state.unit === 'real', cpi: runCpi() });
  if (!groups.length) return null;
  const idx = ys.map(y => yearIndex(y));
  return h('div', null,
    h('div', { class: 'card-sub', style: { padding: '12px 14px 4px' } }, h('b', { style: { color: 'var(--ink)' } }, t('impact.regions.title'))),
    h('div', { class: 'tbl-wrap tall' }, h('table', { class: 'tbl dense' },
      h('thead', null, h('tr', null, h('th', null, t('impact.change_vs_baseline')), ys.map(y => h('th', { class: 'n' }, y)))),
      h('tbody', null, regionRows(groups, idx, ys.length + 1)))),
    groups.some(g => g.entities) ? h('p', { class: 'hint', style: { padding: '6px 14px 0' } }, t('impact.regions.entities_note')) : null);
}
const hasRegionBreakdown = () => regionalBreakdown(runGetter, {}).length > 0;
function regionToggle() {
  if (!hasRegionBreakdown()) return null;
  return h('label', { class: 'row small check' },
    h('input', { type: 'checkbox', checked: state.ui.impactRegions !== false, onchange: e => { state.ui.impactRegions = e.target.checked; emit(); } }),
    t('impact.regions.toggle'));
}

/**
 * The same scenario under four assumptions about behaviour: none (static), low, central, high.
 * Shows how much of the result rests on the behavioural parameters, which are assumptions.
 */
function sensitivityTable(ys) {
  if (!state.ui.band) return null;
  const head = h('div', { class: 'card-sub', style: { padding: '12px 14px 4px' } },
    h('b', { style: { color: 'var(--ink)' } }, t('impact.band.title')));
  if (state.bandError) return h('div', null, head, h('p', { class: 'hint', style: { padding: '0 14px 8px', color: 'var(--bad)' } }, t('impact.band.failed', { error: state.bandError })));
  if (!bandReady()) return h('div', null, head, h('p', { class: 'hint', style: { padding: '0 14px 8px' } }, state.bandLoading ? t('impact.band.loading') : t('impact.band.run_first')));
  const runs = ['static', 'low', 'central', 'high'].filter(r => state.band.runs[r]);
  const cpi = (which, i) => (state.unit === 'real' ? v(raw('macro.cpi', which), i) || 1 : 1);
  const metrics = [
    { name: t('impact.row.balance'), unit: bnUnit(), key: 'fiscal.balance', dec: 2, f: (w, i) => {
      const b = v(raw('fiscal.balance', 'baseline'), i), sc = v(raw('fiscal.balance', w), i);
      return b == null || sc == null ? null : (sc / cpi(w, i) - b / cpi('baseline', i)) / 1000;
    } },
    { name: t('impact.row.balance'), unit: t('common.pts_gdp'), key: 'fiscal.balance', dec: 2, f: (w, i) => {
      const b = v(raw('fiscal.balance_pct_gdp', 'baseline'), i), sc = v(raw('fiscal.balance_pct_gdp', w), i);
      return b == null || sc == null ? null : sc - b;
    } },
    { name: t('impact.row.gdp_real'), unit: '%', key: 'macro.gdp_real', dec: 2, f: (w, i) => {
      const b = v(raw('macro.gdp_real', 'baseline'), i), sc = v(raw('macro.gdp_real', w), i);
      return !b || sc == null ? null : (sc / b - 1) * 100;
    } },
    { name: t('impact.row.employment'), unit: t('common.thousand'), key: 'labour.employment', dec: 1, f: (w, i) => {
      const b = v(raw('labour.employment', 'baseline'), i), sc = v(raw('labour.employment', w), i);
      return b == null || sc == null ? null : (sc - b) / 1000;
    } },
  ].filter(m => has(m.key));
  const idx = ys.map(y => yearIndex(y));
  const rows = [];
  for (const m of metrics) {
    rows.push(h('tr', { class: 'group', style: { cursor: 'default' } }, h('td', { colspan: ys.length + 1 }, `${m.name}, ${m.unit}`)));
    for (const r of runs) {
      rows.push(h('tr', { class: r === 'central' ? 'total' : '' },
        h('td', null, t(`impact.band.run.${r}`)),
        idx.map(i => { const d = m.f(r, i); return h('td', { class: 'n' }, d == null ? '–' : deltaSpan(m.key, d, m.dec)); })));
    }
  }
  return h('div', null, head,
    h('div', { style: { padding: '0 14px 8px' } }, notice('warn', t('impact.band.notice'))),
    h('div', { class: 'tbl-wrap tall' }, h('table', { class: 'tbl dense' },
      h('thead', null, h('tr', null, h('th', null, t('impact.change_vs_baseline')), ys.map(y => h('th', { class: 'n' }, y)))),
      h('tbody', null, rows))));
}

function bandToggle() {
  return h('label', { class: 'row small check', title: t('impact.band.toggle_title') },
    h('input', { type: 'checkbox', checked: !!state.ui.band, onchange: e => setBand(e.target.checked) }),
    t('impact.band.toggle'));
}

export function impactCard({ compact = false } = {}) {
  if (!state.run) {
    return card({
      title: t('impact.title'),
      body: h('div', { class: 'empty' }, enabledMeasures().length
        ? h('span', null, h('b', null, t('impact.empty.run_lead')), ' ' + t('impact.empty.run_rest'))
        : h('span', null, h('b', null, t('impact.empty.add_lead')), ' ' + t('impact.empty.add_rest'))),
    });
  }
  const ys = impactYears();
  const idx = ys.map(y => yearIndex(y));
  const rows = [];
  for (const r of rowsSpec()) {
    if (r.head) { rows.push(h('tr', { class: 'group', style: { cursor: 'default' } }, h('td', { colspan: ys.length + 2 }, r.head))); continue; }
    if (!has(r.key) || (compact && !r.strong && !r.compact)) continue;
    rows.push(h('tr', { class: r.strong ? 'total' : '' },
      h('td', null, r.name), h('td', { class: 'sub' }, r.unit),
      idx.map(i => {
        const d = r.f(i);
        return h('td', { class: 'n' }, d == null ? '–' : deltaSpan(r.pol === 0 ? '' : r.key, d, r.dec) || signed(d, r.dec));
      })));
  }
  const table = h('div', { class: 'tbl-wrap tall' }, h('table', { class: 'tbl dense' },
    h('thead', null, h('tr', null, h('th', null, t('impact.change_vs_baseline')), h('th', null, t('common.unit')), ys.map(y => h('th', { class: 'n' }, y)))),
    h('tbody', null, rows)));
  const many = enabledMeasures().length > 1;
  const tools = compact ? h('div', { class: 'row' }, regionToggle(), bandToggle()) : h('div', { class: 'row' }, regionToggle(), bandToggle(), yearsEditor(),
    many ? h('button', { class: 'btn sm', type: 'button', disabled: !!state.ui.breakdownLoading || isStale(), onclick: loadBreakdown },
      state.ui.breakdownLoading ? t('impact.breakdown.loading') : t('impact.breakdown.button')) : null);
  return card({
    title: t('impact.title'), id: 'impact' + (compact ? '-c' : ''),
    sub: t('impact.sub', { name: state.scenario.name || t('common.scenario') }) + (isStale() ? ' ' + t('impact.sub_stale') : ''),
    tools, flush: true,
    body: h('div', null, table, regionTable(ys), sensitivityTable(ys), compact ? null : breakdownTable(ys)),
  });
}

export { num };
