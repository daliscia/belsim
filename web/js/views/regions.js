// Flanders, Wallonia and Brussels side by side. Built from whatever the model publishes per
// region: every `regional.<metric>.<REG>` series, plus other series that end in a region code.
import { h } from '../dom.js';
import {
  state, has, conv, valueAt, deltaAt, unitFor, formatter, num, label, dimLabel, regionCodes, regionColor,
  regionalMetrics, regionalKey, polarity, humanise,
} from '../store.js';
import { t, tn } from '../i18n.js';
import { card, deltaSpan, seg, pickYear, scenarioName } from '../components.js';
import { lineChart } from '../charts.js';
import { federatedEntities } from '../regional.js';

// Reading order of the metrics the regional model is expected to publish; anything else follows
// in the model's own order. Matching is by pattern, so a renamed or new metric still shows.
const ORDER = [/^population/, /^employment(?!_rate)/, /^employment_rate/, /^unemployment_rate/, /^unemploy/, /^activity_rate/,
  /^gdp(?!_per)/, /^gdp_per/, /^household_income/, /^(gov|government)_revenue/, /^(gov|government)_expenditure/,
  /^(gov|government)_balance/, /^(gov|government)_debt/, /transfer/];
const HEADLINE = [/^population$/, /^employment_rate/, /^unemployment_rate/, /^gdp_per_capita/, /^household_income_per/, /^(gov|government)_balance(_pct_gdp)?$/];
const rank = id => { const i = ORDER.findIndex(rx => rx.test(id)); return i < 0 ? ORDER.length : i; };

/** Metrics by region: {id, label, key: region => series key, regions, section: 'main' | 'revenue' | 'expenditure' | 'other'}. */
function metrics() {
  const regs = regionCodes();
  const lineLabel = id => {
    const [side, ...rest] = id.split('.');
    const meta = (side === 'revenue' ? state.meta.revenue_lines : state.meta.expenditure_lines)[rest.join('.')];
    return meta ? meta.label : null;
  };
  const out = regionalMetrics().map((m, i) => {
    const section = /^revenue\./.test(m.id) ? 'revenue' : /^expenditure\./.test(m.id) ? 'expenditure' : 'main';
    return { ...m, i, key: r => regionalKey(m.id, r), section, label: (section !== 'main' && lineLabel(m.id)) || m.label };
  });
  const known = new Set(out.map(m => m.id.split('.').pop()));
  // national modules also publish a few series per region (demo.tfr.FLA, labour.unemployment_rate.FLA ...)
  const extra = new Map();
  for (const k of Object.keys(state.base)) {
    const parts = k.split('.'), reg = parts[parts.length - 1];
    if (parts[0] === 'regional' || parts.length < 3 || !regs.includes(reg) || (parts[0] === 'pop' && parts[1] === 'region')) continue;
    const prefix = parts.slice(0, -1).join('.');
    if (known.has(parts[parts.length - 2])) continue;
    if (!extra.has(prefix)) {
      const full = label(k), suffix = ', ' + dimLabel('regions', reg);
      extra.set(prefix, { id: prefix, label: full.endsWith(suffix) ? full.slice(0, -suffix.length) : humanise(parts.slice(1, -1).join(' ')), regions: [], key: r => `${prefix}.${r}`, section: 'other', i: 1000 + extra.size });
    }
    extra.get(prefix).regions.push(reg);
  }
  return [...out, ...extra.values()].sort((a, b) => (a.section === 'main' && b.section === 'main' ? rank(a.id) - rank(b.id) || a.i - b.i : a.i - b.i));
}

const firstKey = m => m.key(m.regions[0]);
const unitText = m => unitFor(firstKey(m)).label;

function valueCell(m, r, mode) {
  const key = m.key(r);
  if (!has(key)) return h('td', { class: 'n' }, '–');
  const u = unitFor(key);
  const d = state.run ? deltaAt(key) : null;
  if (mode === 'delta') return h('td', { class: 'n' }, d == null ? '–' : deltaSpan(key, d, u.dec));
  return h('td', { class: 'n' }, num(valueAt(key), u.dec),
    d != null && Math.abs(d) >= 0.5 * 10 ** -u.dec ? h('div', { class: 'sub' }, deltaSpan(key, d, u.dec)) : null);
}

/** One column per region: its headline figures for the selected year. */
function columns(all) {
  const regs = regionCodes();
  const main = all.filter(m => m.section !== 'revenue' && m.section !== 'expenditure');
  let picks = HEADLINE.map(rx => main.find(m => rx.test(m.id.split('.').pop()))).filter(Boolean);
  picks = [...new Set(picks)];
  for (const m of main) { if (picks.length >= 6) break; if (!picks.includes(m)) picks.push(m); }
  return h('div', { class: 'region-cols' }, regs.map(r => h('section', { class: 'region-col', style: { '--region': regionColor(r) } },
    h('h3', null, dimLabel('regions', r)),
    h('dl', null, picks.filter(m => has(m.key(r))).map(m => {
      const key = m.key(r), u = unitFor(key), d = state.run ? deltaAt(key) : null;
      return h('div', { class: 'stat' },
        h('dt', null, m.label),
        h('dd', null, h('span', { class: 'v' }, num(valueAt(key), u.dec)), u.label ? h('span', { class: 'u' }, u.label) : null,
          d != null && Math.abs(d) >= 0.5 * 10 ** -u.dec ? h('div', { class: 'd' }, deltaSpan(key, d, u.dec), ' ' + t('common.vs_baseline')) : null));
    })))));
}

function table(list, id, title, sub, mode) {
  const regs = regionCodes();
  const rows = list.map(m => h('tr', null, h('td', null, m.label), h('td', { class: 'sub' }, unitText(m)), regs.map(r => valueCell(m, r, mode))));
  return card({
    title, sub, flush: true, id,
    body: h('div', { class: 'tbl-wrap tall' }, h('table', { class: 'tbl' },
      h('thead', null, h('tr', null, h('th', null, t('regions.table.metric')), h('th', null, t('common.unit')),
        regs.map(r => h('th', { class: 'n' }, h('span', { class: 'key-box', style: { background: regionColor(r), marginRight: '6px' } }), dimLabel('regions', r))))),
      h('tbody', null, rows))),
    table: () => ({
      head: [t('regions.table.metric'), t('common.unit'), ...regs.flatMap(r => (state.run ? [`${dimLabel('regions', r)} (${t('common.baseline')})`, `${dimLabel('regions', r)} (${scenarioName()})`] : [dimLabel('regions', r)]))],
      rows: list.map(m => [m.label, unitText(m), ...regs.flatMap(r => {
        const key = m.key(r), dec = has(key) ? unitFor(key).dec : 1;
        return state.run ? [num(valueAt(key, 'baseline'), dec), num(valueAt(key, 'scenario'), dec)] : [num(valueAt(key, 'baseline'), dec)];
      })]),
    }),
  });
}

/** One line per region; with a run, the baseline of each region is its dashed twin. */
function chart(m, mode) {
  const regs = regionCodes().filter(r => has(m.key(r)));
  const key0 = m.key(regs[0]), u = unitFor(key0);
  const series = [];
  for (const r of regs) {
    const key = m.key(r), name = dimLabel('regions', r), sc = conv(key, 'scenario'), b = conv(key, 'baseline');
    if (mode === 'delta') {
      series.push({ name, color: regionColor(r), values: sc && b ? sc.map((v, i) => (v == null || b[i] == null ? null : v - b[i])) : null });
    } else {
      series.push({ name, color: regionColor(r), values: sc });
      if (state.run && sc && b && sc.some((v, i) => v != null && b[i] != null && Math.abs(v - b[i]) > 1e-9)) {
        series.push({ name: t('regions.chart.baseline_of', { region: name }), color: regionColor(r), values: b, dash: true, width: 1.5, twin: true });
      }
    }
  }
  const head = [t('common.year'), ...series.map(s => s.name)];
  return card({
    title: m.label, id: 'reg-' + m.id,
    sub: mode === 'delta' ? t('regions.chart.delta_sub', { unit: u.label || t('common.units') }) : u.label || null,
    body: lineChart({ x: state.years, series, fmt: formatter(key0), height: 180, mark: state.year, onPick: pickYear, zero: mode === 'delta', forceLegend: true, title: m.label }),
    table: () => ({ head, rows: state.years.map((y, i) => [String(y), ...series.map(s => num(s.values && s.values[i], u.dec))]) }),
  });
}

export function render(root) {
  const all = metrics();
  if (!all.length) {
    root.append(h('div', { class: 'empty' }, h('b', null, t('regions.empty.lead')), ' ' + t('regions.empty.rest')));
    return;
  }
  const mode = state.run ? seg('regMode', [{ id: 'level', label: t('regions.mode.levels') }, { id: 'delta', label: t('regions.mode.delta') }], 'level') : { cur: 'level', el: null };
  const main = all.filter(m => m.section === 'main'), other = all.filter(m => m.section === 'other');
  const rev = all.filter(m => m.section === 'revenue'), exp = all.filter(m => m.section === 'expenditure');
  const fed = federatedEntities();
  const charted = [...main, ...other];
  root.append(h('div', { class: 'stack' },
    h('div', { class: 'row' },
      h('span', { class: 'hint' }, state.run ? t('regions.intro_run', { name: scenarioName(), year: state.year }) : t('regions.intro', { year: state.year })),
      h('span', { class: 'spacer' }), mode.el),
    columns(all),
    main.length ? table(main, 'reg-main', t('regions.table.title', { year: state.year }), tn('regions.table.sub', main.length), mode.cur) : null,
    other.length ? table(other, 'reg-other', t('regions.table.other_title', { year: state.year }), t('regions.table.other_sub'), mode.cur) : null,
    h('div', { class: 'grid g3' }, charted.map(m => chart(m, mode.cur))),
    rev.length ? table(rev, 'reg-rev', t('regions.table.revenue_title', { year: state.year }), t('regions.table.lines_sub'), mode.cur) : null,
    exp.length ? table(exp, 'reg-exp', t('regions.table.expenditure_title', { year: state.year }), t('regions.table.lines_sub'), mode.cur) : null,
    h('p', { class: 'hint' }, fed.length ? t('regions.entities_hint') : t('regions.entities_pending'),
      ' ', h('a', { href: '#/budget', onclick: () => { state.ui.budgetTab = 'entities'; } }, t('regions.entities_link')))));
}
