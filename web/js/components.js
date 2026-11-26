// Shared view building blocks: cards with a chart/table switch, deltas, baseline-vs-scenario series.
import { h, download } from './dom.js';
import { t } from './i18n.js';
import {
  state, emit, conv, has, label, unitFor, formatter, num, signed, polarity, color, deltaSeries, sumSeries, bandReady, isMissing,
  compareText,
} from './store.js';
import { lineChart, stackChart } from './charts.js';

export const scenarioName = () => state.scenario.name || t('common.scenario');
export const pickYear = y => { state.year = y; emit(); };

export function deltaSpan(key, d, dec, suffix = '') {
  if (d == null) return null;
  const pol = polarity(key), zero = Math.abs(d) < 0.5 * 10 ** -dec;
  const cls = zero || !pol ? '' : (d > 0) === (pol > 0) ? ' good' : ' bad';
  return h('span', { class: 'delta' + cls }, (zero ? '' : d > 0 ? '▲ ' : '▼ ') + signed(d, dec) + suffix);
}

function toCsv(head, rows) {
  const q = v => (v == null ? '' : /[",\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : String(v));
  return [head, ...rows].map(r => r.map(q).join(',')).join('\n');
}

export function dataTable(head, rows, cls = 'tbl dense') {
  return h('div', { class: 'tbl-wrap' }, h('table', { class: cls },
    h('thead', null, h('tr', null, head.map((c, i) => h('th', { class: i ? 'n' : '' }, c)))),
    h('tbody', null, rows.map(r => h('tr', null, r.map((c, i) =>
      h('td', { class: i ? 'n' : '' }, typeof c === 'number' ? num(c, Math.abs(c) >= 1000 ? 0 : 2) : (c == null ? '–' : c))))))));
}

/** File name stem from a (possibly accented) title: "Dépenses 2030" -> "depenses-2030". */
const fileSlug = title => String(title).normalize('NFD').replace(/[\u0300-\u036f]/g, '').toLowerCase().replace(/[^a-z0-9]+/g, '-');

/**
 * A titled card. `table` (optional) is () => ({head, rows}); it adds a Chart/Table switch and a
 * CSV download so every chart's numbers are reachable without hovering.
 */
export function card({ title, sub, body, table, tools, flush, id }) {
  const tid = 'tbl:' + (id || title);
  const showTable = !!(table && state.ui[tid]);
  const head = h('div', { class: 'card-head' },
    h('div', null, h('h3', { class: 'card-title' }, title), sub ? h('div', { class: 'card-sub' }, sub) : null),
    h('div', { class: 'card-tools' }, tools || null,
      table ? h('button', {
        class: 'linkbtn', type: 'button',
        onclick: () => { state.ui[tid] = !showTable; emit(); },
      }, showTable ? t('cmp.show_chart') : t('cmp.show_table')) : null));
  let content = body;
  if (showTable) {
    const tb = table();
    content = h('div', null, dataTable(tb.head, tb.rows),
      h('div', { style: { padding: '6px 14px 0' } }, h('button', {
        class: 'linkbtn', type: 'button',
        onclick: () => download(`${fileSlug(title)}.csv`, toCsv(tb.head, tb.rows), 'text/csv'),
      }, t('cmp.download_csv'))));
  }
  return h('section', { class: 'card' }, head, h('div', { class: 'card-body' + (flush || showTable ? ' flush' : '') }, content));
}

/** Baseline vs scenario for one key: baseline in ink, scenario in the scenario colour. */
export function cmpSeries(key) {
  const out = [{ name: t('common.baseline'), values: conv(key, 'baseline'), color: 'var(--baseline)', width: state.run ? 1.5 : 2 }];
  if (state.run) out.push({ name: scenarioName(), values: conv(key, 'scenario'), color: 'var(--scenario)' });
  if (bandReady()) {
    const st = conv(key, 'static');
    if (st) out.push({ name: t('cmp.static'), values: st, color: 'var(--scenario)', dash: true, width: 1.5 });
  }
  return out;
}

export const bandName = () => t('cmp.band_name');
/** Shaded range between the low- and high-response runs of the active scenario, for one key. */
export function bandFor(key) {
  if (!bandReady()) return null;
  const a = conv(key, 'low'), b = conv(key, 'high');
  if (!a || !b) return null;
  return {
    lo: a.map((v, i) => (v == null || b[i] == null ? null : Math.min(v, b[i]))),
    hi: a.map((v, i) => (v == null || b[i] == null ? null : Math.max(v, b[i]))),
    color: 'var(--scenario)', name: bandName(),
  };
}

/** Several keys on one axis: one colour per key; with a run, the baseline is the dashed twin. */
export function multiSeries(keys, names) {
  const out = [];
  keys.forEach((k, i) => {
    if (!has(k)) return;
    const nm = (names && names[i]) || label(k);
    out.push({ name: nm, values: conv(k, 'scenario'), color: color(i) });
    if (state.run) {
      const d = deltaSeries(k);
      if (d && d.some(v => v != null && Math.abs(v) > 1e-9)) {
        out.push({ name: t('cmp.name_baseline', { name: nm }), values: conv(k, 'baseline'), color: color(i), dash: true, width: 1.5, twin: true });
      }
    }
  });
  return out;
}

export function seriesTable(keys, names) {
  const present = keys.filter(has);
  const head = [t('common.year')];
  present.forEach(k => {
    const nm = (names && names[keys.indexOf(k)]) || label(k);
    if (state.run) head.push(t('cmp.col_baseline', { name: nm }), t('cmp.col_scenario', { name: nm }));
    else head.push(nm);
  });
  const cols = present.map(k => [conv(k, 'baseline'), conv(k, 'scenario')]);
  const rows = state.years.map((y, i) => {
    const r = [String(y)];
    cols.forEach(([b, sc], j) => {
      const dec = unitFor(present[j]).dec;
      r.push(num(b && b[i], dec));
      if (state.run) r.push(num(sc && sc[i], dec));
    });
    return r;
  });
  return { head, rows };
}

/** Card with a line chart of one key (baseline vs scenario) or several keys on a shared unit. */
export function lineCard(title, keys, opts = {}) {
  const list = (Array.isArray(keys) ? keys : [keys]).filter(has);
  const single = !Array.isArray(keys);
  const first = list[0];
  const unitLabel = first ? unitFor(first).label : '';
  return card({
    title, id: opts.id,
    sub: [opts.sub, unitLabel].filter(Boolean).join(t('common.list_sep')) || null,
    body: lineChart({
      x: state.years,
      series: first ? (single ? cmpSeries(first) : multiSeries(list, opts.names && list.map(k => opts.names[keys.indexOf(k)]))) : [],
      refs: opts.refs, fmt: first ? formatter(first) : undefined, height: opts.height,
      mark: state.year, onPick: pickYear, zero: opts.zero, title,
      emptyText: opts.emptyText, band: single && first ? bandFor(first) : null,
    }),
    table: first ? () => seriesTable(list, opts.names && list.map(k => opts.names[keys.indexOf(k)])) : null,
  });
}

/** Card with a stacked area of several keys (scenario values), baseline total as dashed outline. */
export function stackCard(title, items, opts = {}) {
  // items: [{name, keys:[...]}] — each layer is the sum of its keys
  const layers = items.map((it, i) => ({ name: it.name, color: it.color || color(i), values: sumSeries(it.keys, 'scenario') }))
    .filter(l => l.values.some(v => v != null));
  const allKeys = items.flatMap(it => it.keys);
  const first = allKeys.find(has);
  let outline = null;
  if (state.run && first) {
    const b = sumSeries(allKeys, 'baseline'), sc = sumSeries(allKeys, 'scenario');
    if (b.some((v, i) => v != null && Math.abs(v - sc[i]) > 1e-9)) outline = { name: t('common.baseline_total'), values: b };
  }
  const fmt = first ? formatter(first) : undefined;
  return card({
    title, id: opts.id,
    sub: [opts.sub, first ? unitFor(first).label : ''].filter(Boolean).join(t('common.list_sep')) || null,
    body: stackChart({ x: state.years, layers, fmt, mark: state.year, onPick: pickYear, outline, height: opts.height, title }),
    table: first ? () => ({
      head: [t('common.year'), ...layers.map(l => l.name), t('common.total')],
      rows: state.years.map((y, i) => {
        const dec = unitFor(first).dec;
        return [String(y), ...layers.map(l => num(l.values[i], dec)), num(layers.reduce((acc, l) => acc + (l.values[i] || 0), 0), dec)];
      }),
    }) : null,
  });
}

export function tabs(id, options, fallback) {
  const cur = options.some(o => o.id === state.ui[id]) ? state.ui[id] : (fallback || options[0].id);
  const el = h('div', { class: 'tabs', role: 'tablist' }, options.map(o => h('button', {
    type: 'button', role: 'tab', 'aria-selected': String(o.id === cur),
    onclick: () => { state.ui[id] = o.id; emit(); },
  }, o.label)));
  return { el, cur };
}

export function seg(id, options, fallback, cls = 'seg sm') {
  const cur = options.some(o => o.id === state.ui[id]) ? state.ui[id] : (fallback || options[0].id);
  const el = h('div', { class: cls, role: 'group' }, options.map(o => h('button', {
    type: 'button', 'aria-pressed': String(o.id === cur),
    onclick: () => { state.ui[id] = o.id; emit(); },
  }, o.label)));
  return { el, cur };
}

export function missingNote(keys) {
  const miss = keys.filter(k => !has(k));
  if (!miss.length) return null;
  return h('p', { class: 'hint' }, t('cmp.missing', { keys: miss.join(', ') }));
}

// ------------------------------------------------------------------ shared pieces of the newer views

/** A boxed note. kind: 'warn' (method caveats, assumptions) | 'info'. */
export function notice(kind, ...children) {
  return h('div', { class: `notice ${kind || 'info'}`, role: kind === 'warn' ? 'note' : null }, children);
}

/** Placeholder while a slow request runs. */
export function progress(text, detail) {
  return h('div', { class: 'progress', role: 'status' },
    h('div', { class: 'progress-bar' }, h('span')),
    h('div', null, h('b', null, text), detail ? h('div', { class: 'hint' }, detail) : null));
}

/**
 * Renders the state of a `lazy()` entry: progress, "not available yet" for a missing endpoint,
 * an error box, or `ok(data)`.
 */
export function lazyBody(entry, { loading, detail, missing, retry, ok }) {
  if (entry.status === 'loading') return progress(loading || t('common.loading'), detail);
  if (entry.status === 'error') {
    if (isMissing(entry.error)) {
      return h('div', { class: 'empty' }, h('b', null, t('cmp.not_available'), ' '), missing || t('cmp.server_missing'),
        retry ? h('div', { style: { marginTop: '8px' } }, h('button', { class: 'btn sm', type: 'button', onclick: retry }, t('cmp.check_again'))) : null);
    }
    return h('div', { class: 'errbox' }, h('h2', null, t('cmp.compute_failed')), h('pre', null, entry.error.message),
      retry ? h('button', { class: 'btn', type: 'button', onclick: retry }, t('cmp.try_again')) : null);
  }
  try {
    return ok(entry.data);
  } catch (e) {
    console.error(e);
    return h('div', { class: 'errbox' }, h('h2', null, t('cmp.display_failed')), h('pre', null, String(e && e.stack || e)));
  }
}

/**
 * Sortable table. cols: [{id, label, num, value: row => sortable value, cell: row => node|string, title}]
 * opts: { id, sort: default column id, desc, rowClass, foot }
 */
export function sortTable(cols, rows, opts = {}) {
  const sid = 'sort:' + opts.id;
  const cur = state.ui[sid] && cols.some(c => c.id === state.ui[sid].col) ? state.ui[sid] : { col: opts.sort || null, desc: opts.desc !== false };
  const col = cols.find(c => c.id === cur.col);
  let sorted = rows;
  if (col && col.value) {
    const dir = cur.desc ? -1 : 1;
    sorted = [...rows].sort((a, b) => {
      const x = col.value(a), y = col.value(b);
      if (x == null && y == null) return 0;
      if (x == null) return 1;
      if (y == null) return -1;
      return (typeof x === 'string' ? compareText(x, y) : x - y) * dir;
    });
  }
  const head = cols.map(c => {
    const active = col && c.id === col.id;
    const btn = c.value ? h('button', {
      class: 'th-sort', type: 'button', title: c.title || t('cmp.sort_by', { label: c.label }),
      onclick: () => { state.ui[sid] = { col: c.id, desc: active ? !cur.desc : !!c.num }; emit(); },
    }, c.label, h('span', { class: 'arrow', 'aria-hidden': 'true' }, active ? (cur.desc ? ' ▼' : ' ▲') : '')) : c.label;
    return h('th', { class: c.num ? 'n' : '', 'aria-sort': active ? (cur.desc ? 'descending' : 'ascending') : null, title: c.title || null }, btn);
  });
  return h('div', { class: 'tbl-wrap' + (opts.tall === false ? '' : ' tall') }, h('table', { class: opts.cls || 'tbl' },
    h('thead', null, h('tr', null, head)),
    h('tbody', null, sorted.map(r => h('tr', { class: opts.rowClass ? opts.rowClass(r) : null }, cols.map(c => {
      const v = c.cell ? c.cell(r) : c.value ? c.value(r) : null;
      return h('td', { class: c.num ? 'n' : (c.cls || '') }, v == null ? '–' : v);
    }))), opts.foot || null)));
}

/** Text input bound to state.ui[id]; keeps focus across re-renders through its DOM id. */
export function searchBox(id, placeholder, width = '260px') {
  const input = h('input', { type: 'search', id: 'q-' + id, placeholder, value: state.ui[id] || '', style: { width } });
  let timer;
  input.addEventListener('input', () => { clearTimeout(timer); timer = setTimeout(() => { state.ui[id] = input.value; emit(); }, 140); });
  return input;
}
