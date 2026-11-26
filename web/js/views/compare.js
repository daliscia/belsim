// Several named scenarios side by side: headline table, overlay charts, per-region breakdown.
import { h, toast } from '../dom.js';
import {
  state, emit, post, lazy, lazyReset, enabledMeasures, yearIndex, num, pct, color, MAX_COMPARE, sigOf,
} from '../store.js';
import { t, tn } from '../i18n.js';
import { card, deltaSpan, seg, sortTable, lazyBody, pickYear, notice } from '../components.js';
import { lineChart } from '../charts.js';
import { exportable, keepForComparison, inComparison, loadScenario } from '../scenario.js';
import { impactYears, yearsEditor, bnUnit } from '../impact.js';
import { regionalBreakdown } from '../regional.js';
import { compareSection } from '../parties.js';

/** Scenarios shown: the kept set, plus the active scenario when it is not one of them. */
function entries() {
  const out = state.compare.map(c => ({ ...c, color: color(c.slot + 1), kept: true }));
  const twin = inComparison();
  if (twin) out.find(e => e.name === twin.name).active = true;
  else if (enabledMeasures().length) {
    out.unshift({ name: t('compare.being_edited', { name: state.scenario.name || t('common.scenario') }), measures: exportable().measures, color: 'var(--scenario)', active: true, kept: false });
  }
  return out;
}

const at = (s, i) => (s && s[i] != null ? s[i] : null);

/** Metrics as functions of (scenario series, baseline series, year index). */
function metrics() {
  const real = state.unit === 'real';
  const bn = bnUnit(), pts = t('common.pts_gdp');
  const eur = key => (s, b, i) => {
    const x = at(s[key], i), y = at(b[key], i);
    if (x == null || y == null) return null;
    const cs = real ? at(s['macro.cpi'], i) || 1 : 1, cb = real ? at(b['macro.cpi'], i) || 1 : 1;
    return (x / cs - y / cb) / 1000;
  };
  const diff = (key, scale = 1) => (s, b, i) => {
    const x = at(s[key], i), y = at(b[key], i);
    return x == null || y == null ? null : (x - y) * scale;
  };
  const rel = key => (s, b, i) => {
    const x = at(s[key], i), y = at(b[key], i);
    return x == null || !y ? null : (x / y - 1) * 100;
  };
  return [
    { id: 'bal', label: t('compare.metric.balance'), unit: bn, key: 'fiscal.balance', f: eur('fiscal.balance'), dec: 2 },
    { id: 'balpct', label: t('compare.metric.balance'), unit: pts, key: 'fiscal.balance_pct_gdp', f: diff('fiscal.balance_pct_gdp'), dec: 2 },
    { id: 'rev', label: t('compare.metric.revenue'), unit: bn, key: 'fiscal.revenue_total', f: eur('fiscal.revenue_total'), dec: 2, pol: 0 },
    { id: 'exp', label: t('compare.metric.expenditure'), unit: bn, key: 'fiscal.expenditure_total', f: eur('fiscal.expenditure_total'), dec: 2, pol: 0 },
    { id: 'gdp', label: t('compare.metric.gdp_real'), unit: '%', key: 'macro.gdp_real', f: rel('macro.gdp_real'), dec: 2 },
    { id: 'emp', label: t('compare.metric.employment'), unit: t('common.thousand'), key: 'labour.employment', f: diff('labour.employment', 1 / 1000), dec: 1 },
    { id: 'debt', label: t('compare.metric.debt'), unit: pts, key: 'fiscal.debt_pct_gdp', f: diff('fiscal.debt_pct_gdp'), dec: 2 },
  ];
}

function swatch(e) {
  return h('span', { class: 'key-line', style: { color: e.color, marginRight: '7px', verticalAlign: 'middle' } });
}

function setCard(list) {
  const lib = state.library;
  const add = h('select', { 'aria-label': t('compare.set.add_aria') },
    h('option', { value: '' }, t('compare.set.add')),
    state.saved.length ? h('optgroup', { label: t('compare.set.saved') }, state.saved.map((s, i) => h('option', { value: `saved:${i}` }, s.scenario.name))) : null,
    lib.scenarios.length ? h('optgroup', { label: t('compare.set.prebuilt') }, lib.scenarios.map((s, i) => h('option', { value: `lib:${i}` }, s.name))) : null,
    lib.measures.length ? h('optgroup', { label: t('compare.set.one_measure') }, lib.measures.map((m, i) => h('option', { value: `m:${i}` }, m.name))) : null);
  add.addEventListener('change', () => {
    const [kind, i] = add.value.split(':');
    const scen = kind === 'saved' ? state.saved[i].scenario : kind === 'lib' ? lib.scenarios[i]
      : kind === 'm' ? { name: lib.measures[i].name, description: lib.measures[i].description, measures: [lib.measures[i]] } : null;
    add.value = '';
    if (scen) keepForComparison(scen);
  });
  const edit = e => {
    if (enabledMeasures().length && !inComparison() && !confirm(t('compare.set.replace_confirm', { current: state.scenario.name, name: e.name }))) return;
    loadScenario({ name: e.name, description: e.description, measures: e.measures });
    state.panelOpen = true;
    toast(t('compare.set.now_in_panel', { name: e.name }));
  };
  const rows = list.map(e => h('tr', null,
    h('td', null, swatch(e), h('b', null, e.name), e.active && e.kept ? h('span', { class: 'tag', style: { marginLeft: '6px' } }, t('compare.set.in_panel')) : null),
    h('td', { class: 'sub', title: e.measures.map(m => m.name).join('\n') },
      `${tn('common.n_measures', e.measures.length)}${t('common.colon')}${e.measures.map(m => m.name).join(', ')}`),
    h('td', { class: 'n' }, e.kept
      ? [h('button', { class: 'linkbtn', type: 'button', onclick: () => edit(e) }, t('compare.set.edit')), ' ',
        h('button', { class: 'x', type: 'button', title: t('compare.set.remove_title'), 'aria-label': t('compare.set.remove', { name: e.name }),
          onclick: () => { state.compare = state.compare.filter(c => c.name !== e.name); emit(); } }, '×')]
      : h('button', { class: 'btn sm', type: 'button', onclick: () => keepForComparison() }, t('panel.keep')))));
  return card({
    title: t('compare.set.title'), flush: true, id: 'cmp-set',
    sub: t('compare.set.sub', { max: MAX_COMPARE }),
    tools: h('div', { class: 'row' }, add,
      state.compare.length ? h('button', { class: 'linkbtn', type: 'button', onclick: () => { if (confirm(t('compare.set.remove_all_confirm'))) { state.compare = []; emit(); } } }, t('compare.set.remove_all')) : null),
    body: list.length ? h('div', { class: 'tbl-wrap tall' }, h('table', { class: 'tbl cmp-set' }, h('tbody', null, rows)))
      : h('div', { class: 'empty', style: { padding: '14px' } }, h('b', null, t('compare.set.empty_lead')), ' ' + t('compare.set.empty_rest')),
  });
}

function headlineTable(list, data) {
  const i = yearIndex(state.year), last = data.years.length - 1, endYear = data.years[last];
  const ms = metrics().filter(m => data.baseline[m.key]);
  const cols = [
    { id: 'name', label: t('common.scenario'), value: r => r.e.name, cell: r => [swatch(r.e), r.e.name, r.error ? h('div', { class: 'sub', style: { color: 'var(--bad)', whiteSpace: 'normal' } }, t('compare.refused', { error: r.error })) : null] },
    ...ms.filter(m => m.id !== 'debt').map(m => ({
      id: m.id, label: `Δ ${m.label}, ${m.unit}`, num: true,
      value: r => (r.series ? m.f(r.series, data.baseline, i) : null),
      cell: r => { const d = r.series ? m.f(r.series, data.baseline, i) : null; return d == null ? null : deltaSpan(m.pol === 0 ? '' : m.key, d, m.dec); },
    })),
  ];
  const debt = ms.find(m => m.id === 'debt');
  if (debt) {
    cols.push({
      id: 'debt', label: t('compare.head.debt_end', { year: endYear, unit: debt.unit }), num: true, title: t('compare.head.debt_end_title'),
      value: r => (r.series ? debt.f(r.series, data.baseline, last) : null),
      cell: r => { const d = r.series ? debt.f(r.series, data.baseline, last) : null; return d == null ? null : deltaSpan(debt.key, d, debt.dec); },
    });
  }
  const rows = list.map((e, j) => ({ e, ...(data.scenarios[j] || {}) }));
  return card({
    title: t('compare.head.title', { year: state.year }), flush: true, id: 'cmp-head',
    sub: t('compare.head.sub'),
    body: sortTable(cols, rows, { id: 'cmp-head', sort: null }),
  });
}

/** Change against the baseline by region, in the selected year: one row per scenario. */
function regionTable(list, data) {
  const i = yearIndex(state.year);
  const real = state.unit === 'real';
  const getter = series => key => ({ s: series ? series[key] : null, b: data.baseline[key] });
  const shape = regionalBreakdown(getter(null), { real });
  if (!shape.length || i < 0) return null;
  const rows = list.map((e, j) => {
    const s = (data.scenarios[j] || {}).series;
    const groups = s ? regionalBreakdown(getter(s), { real, cpi: { s: s['macro.cpi'], b: data.baseline['macro.cpi'] } }) : [];
    return h('tr', null, h('td', null, swatch(e), e.name),
      shape.flatMap(g => g.rows.map(r => {
        const gg = groups.find(x => x.id === g.id), rr = gg && gg.rows.find(x => x.code === r.code);
        const d = rr ? rr.f(i) : null;
        return h('td', { class: 'n' }, d == null ? '–' : deltaSpan(g.polKey, d, g.dec));
      })));
  });
  return card({
    title: t('compare.regions.title', { year: state.year }), flush: true, id: 'cmp-regions',
    sub: t('compare.regions.sub'),
    body: h('div', null, h('div', { class: 'tbl-wrap tall' }, h('table', { class: 'tbl dense' },
      h('thead', null,
        h('tr', null, h('th', null, ''), shape.map(g => h('th', { class: 'n span', colspan: g.rows.length }, `${g.label}, ${g.unit}`))),
        h('tr', null, h('th', null, t('common.scenario')), shape.flatMap(g => g.rows.map(r => h('th', { class: 'n' }, r.label))))),
      h('tbody', null, rows))),
      shape.some(g => g.entities) ? h('p', { class: 'hint', style: { padding: '6px 14px 0' } }, t('impact.regions.entities_note')) : null),
  });
}

function byYearTable(list, data) {
  const ms = metrics().filter(m => data.baseline[m.key]);
  const pick = seg('cmpMetric', ms.map(m => ({ id: m.id, label: `${m.label}, ${m.unit}` })), 'bal');
  const m = ms.find(x => x.id === pick.cur) || ms[0];
  const ys = impactYears().filter(y => data.years.includes(y));
  const rows = list.map((e, j) => {
    const r = data.scenarios[j] || {};
    return h('tr', null, h('td', null, swatch(e), e.name),
      ys.map(y => {
        const d = r.series ? m.f(r.series, data.baseline, data.years.indexOf(y)) : null;
        return h('td', { class: 'n' }, d == null ? '–' : deltaSpan(m.pol === 0 ? '' : m.key, d, m.dec));
      }));
  });
  return card({
    title: t('compare.years.title'), flush: true, id: 'cmp-years',
    sub: `${m.label}, ${m.unit}`,
    tools: yearsEditor(),
    body: h('div', null, h('div', { style: { padding: '0 14px 8px', overflowX: 'auto' } }, pick.el),
      h('div', { class: 'tbl-wrap tall' }, h('table', { class: 'tbl dense' },
        h('thead', null, h('tr', null, h('th', null, t('common.scenario')), ys.map(y => h('th', { class: 'n' }, y)))),
        h('tbody', null, rows)))),
  });
}

const CHARTS = [
  { id: 'balance', key: 'fiscal.balance_pct_gdp', unit: 'unit.pct_gdp', dunit: 'common.points_gdp', scale: 1, dec: 2, pct: true, zero: true },
  { id: 'debt', key: 'fiscal.debt_pct_gdp', unit: 'unit.pct_gdp', dunit: 'common.points_gdp', scale: 1, dec: 2, pct: true },
  { id: 'employment', key: 'labour.employment', unit: 'unit.thousand_persons', dunit: 'unit.thousand_persons', scale: 1 / 1000, dec: 1 },
];

function overlay(list, data, c, mode) {
  const b = data.baseline[c.key];
  if (!b) return null;
  const diff = mode === 'delta';
  const series = [];
  if (!diff) series.push({ name: t('common.baseline'), values: b.map(v => (v == null ? null : v * c.scale)), color: 'var(--baseline)', width: 1.5 });
  list.forEach((e, j) => {
    const s = (data.scenarios[j] || {}).series;
    if (!s || !s[c.key]) return;
    series.push({ name: e.name, color: e.color, values: s[c.key].map((v, i) => (v == null || b[i] == null ? null : (diff ? v - b[i] : v) * c.scale)) });
  });
  const fmt = v => (c.pct && !diff ? pct(v, c.dec) : num(v, c.dec));
  const title = t(`compare.chart.${c.id}`);
  return card({
    title, id: 'cmp-' + c.key,
    sub: diff ? t('compare.chart.diff_sub', { unit: t(c.dunit) }) : t(c.unit),
    body: lineChart({ x: data.years, series, fmt, zero: diff || c.zero, height: 240, mark: state.year, onPick: pickYear, forceLegend: true, title }),
    table: () => ({
      head: [t('common.year'), ...series.map(s => s.name)],
      rows: data.years.map((y, i) => [String(y), ...series.map(s => num(s.values[i], c.dec))]),
    }),
  });
}

export function render(root) {
  const list = entries();
  const top = setCard(list);
  if (!list.length) { root.append(h('div', { class: 'stack' }, top, compareSection())); return; }
  const key = 'cmp:' + state.meta.params_signature + ':' + list.map(e => sigOf(e.measures)).join('|');
  const entry = lazy(key, () => post('/api/run_batch', { scenarios: list.map(e => ({ name: e.name, measures: e.measures })), options: {} }));
  const mode = seg('cmpMode', [{ id: 'level', label: t('compare.mode.levels') }, { id: 'delta', label: t('compare.mode.delta') }], 'level');
  root.append(h('div', { class: 'stack' }, top, lazyBody(entry, {
    loading: tn('compare.simulating', list.length),
    retry: () => { lazyReset('cmp:'); emit(); },
    ok: data => h('div', { class: 'stack' },
      data.scenarios.some(s => s.error) ? notice('warn', t('compare.some_refused') + ' ',
        data.scenarios.filter(s => s.error).map(s => `${s.name} (${s.error})`).join('; ')) : null,
      headlineTable(list, data),
      regionTable(list, data),
      h('div', { class: 'row' }, h('span', { class: 'hint' }, t('compare.charts_show')), mode.el),
      h('div', { class: 'grid g3' }, CHARTS.map(c => overlay(list, data, c, mode.cur))),
      byYearTable(list, data)),
  }), compareSection()));
}
