// Payslip explorer: net income and marginal rate against gross income, baseline vs scenario,
// and the full breakdown at one gross amount. The endpoints belong to another module.
import { h } from '../dom.js';
import { state, emit, api, post, lazy, lazyReset, num, pct, money, kilo, humanise, enabledMeasures, scenarioSig } from '../store.js';
import { card, deltaSpan, lazyBody, notice, scenarioName } from '../components.js';
import { lineChart } from '../charts.js';
import { exportable } from '../scenario.js';
import { t, tOr } from '../i18n.js';

const isNum = v => typeof v === 'number' && isFinite(v);
const eur = v => money(v, 0);
const keur = v => (Math.abs(v) >= 1000 ? kilo(v, v % 1000 ? 1 : 0) : num(v, 0));

/** Options as [{id, label}] from a list of strings, a list of objects or an {id: label} map. */
function choices(v) {
  if (Array.isArray(v)) {
    return v.map(x => (x && typeof x === 'object'
      ? { id: String(x.id ?? x.key ?? x.value ?? x.name), label: String(x.label ?? x.title ?? x.name ?? x.id), raw: x }
      : { id: String(x), label: humanise(String(x)) }));
  }
  if (v && typeof v === 'object') return Object.entries(v).map(([k, x]) => ({ id: k, label: typeof x === 'string' ? x : (x && x.label) || humanise(k), raw: x }));
  return [];
}

function settings(opts) {
  const u = state.ui.pay || {};
  const d = (opts && opts.defaults) || {};
  const kinds = choices(opts && (opts.kinds || opts.kind)), hhs = choices(opts && (opts.households || opts.household));
  const pick = (cur, list, def) => (list.some(o => o.id === cur) ? cur : list.some(o => o.id === String(def)) ? String(def) : list.length ? list[0].id : null);
  const n = (v, def) => (isNum(v) ? v : def);
  const gmin = n(u.gross_min, n(d.gross_min, 0)), gmax = Math.max(gmin + 1000, n(u.gross_max, n(d.gross_max, 120000)));
  const regions = choices(opts && opts.household_fields && opts.household_fields.region);
  // one point per € 1,000 where the server allows it, so the breakdown lands on round amounts
  const maxPoints = n(opts && opts.max_points, 400);
  return {
    kinds, hhs, regions,
    kind: pick(u.kind, kinds, d.kind), household: pick(u.household, hhs, d.household),
    region: regions.length ? pick(u.region, regions, regions[0].id) : null,
    gross_min: gmin, gross_max: gmax, points: Math.round(Math.min(maxPoints, Math.max(11, (gmax - gmin) / 1000 + 1))),
    at: Math.min(gmax, Math.max(gmin, n(u.at, n(d.breakdown_gross ?? d.at, Math.round((gmin + gmax) / 2 / 1000) * 1000)))),
  };
}
const setUi = patch => { state.ui.pay = { ...(state.ui.pay || {}), ...patch }; emit(); };

function controls(set) {
  const sel = (id, label, list, cur) => list.length ? h('label', null, label,
    h('select', { id: 'pay-' + id, onchange: e => setUi({ [id]: e.target.value }) }, list.map(o => h('option', { value: o.id, selected: o.id === cur }, o.label)))) : null;
  const numIn = (id, label, val, step) => h('label', null, label, h('input', {
    type: 'number', id: 'pay-' + id, value: val, step, min: 0,
    onchange: e => { const v = parseFloat(e.target.value); if (Number.isFinite(v) && v >= 0) setUi({ [id]: v }); else e.target.value = val; },
  }));
  return h('div', { class: 'form pay-form' },
    sel('kind', t('payslip.form.kind'), set.kinds, set.kind),
    sel('household', t('payslip.form.household'), set.hhs, set.household),
    sel('region', t('payslip.form.region'), set.regions, set.region),
    numIn('gross_min', t('payslip.form.gross_min'), set.gross_min, 1000),
    numIn('gross_max', t('payslip.form.gross_max'), set.gross_max, 1000),
    numIn('at', t('payslip.form.at'), set.at, 500));
}

/** Reference wage (a model code: min_wage, p25, median ...) in the interface language. */
const refName = k => tOr('payslip.ref.' + k, humanise(k));
/** Breakdown line: the dotted key ("tax.federal" -> payslip.item.tax_federal), else its last segment, else the code in words. */
const itemName = k => tOr('payslip.item.' + k.replace(/\./g, '_'), tOr('payslip.item.' + k.split('.').pop(), humanise(k.replace(/\./g, ' '))));
const nearest = (grid, x) => grid.reduce((best, g, i) => (Math.abs(g - x) < Math.abs(grid[best] - x) ? i : best), 0);

/** Numeric leaves of a breakdown as {dotted key: number}. Lists aligned with the grid are read at index `i`. */
function flatten(obj, i, n, out = {}, prefix = '') {
  if (Array.isArray(obj)) {
    if (obj.length === n && obj.every(x => x == null || isNum(x))) { if (isNum(obj[i])) out[prefix] = obj[i]; }
    else if (obj.length === n && obj[i] && typeof obj[i] === 'object') flatten(obj[i], i, n, out, prefix);
    else obj.forEach((x, j) => { if (x && typeof x === 'object' && (x.label || x.name || x.key) && isNum(x.value ?? x.amount)) out[(prefix ? prefix + '.' : '') + (x.key || x.label || x.name)] = x.value ?? x.amount; });
  } else if (obj && typeof obj === 'object') {
    for (const [k, v] of Object.entries(obj)) {
      const p = prefix ? `${prefix}.${k}` : k;
      if (isNum(v)) out[p] = v; else if (v && typeof v === 'object') flatten(v, i, n, out, p);
    }
  }
  return out;
}

function breakdownCard(d, set, i) {
  const n = d.grid.length;
  const b = flatten((d.baseline || {}).breakdown_at ?? (d.baseline || {}).breakdown, i, n);
  const s = flatten((d.scenario || {}).breakdown_at ?? (d.scenario || {}).breakdown, i, n);
  const keys = [...new Set([...Object.keys(b), ...Object.keys(s)])].filter(k => !/^reference_wages/.test(k));
  const at = isNum(b.gross) ? b.gross : d.grid[i];
  if (!keys.length) return card({ title: t('payslip.breakdown.title'), body: h('div', { class: 'empty' }, t('payslip.breakdown.none')) });
  const isRate = k => /rate|share|marginal/.test(k) && Math.max(Math.abs(b[k] || 0), Math.abs(s[k] || 0)) <= 1.5;
  const fmt = (k, v) => (!isNum(v) ? '–' : isRate(k) ? pct(v * 100, 1) : num(v, 0));
  const strong = k => /^(gross|net|employer_cost)$/.test(k.split('.').pop());
  const changed = keys.some(k => isNum(b[k]) && isNum(s[k]) && Math.abs(s[k] - b[k]) > 1e-6);
  return card({
    title: t('payslip.breakdown.title_at', { amount: eur(at) }), flush: true, id: 'pay-break',
    sub: Math.abs(at - set.at) > 1 ? t('payslip.breakdown.sub_nearest', { amount: eur(set.at) }) : t('payslip.breakdown.sub'),
    body: h('div', { class: 'tbl-wrap tall' }, h('table', { class: 'tbl' },
      h('thead', null, h('tr', null, h('th', null, t('payslip.breakdown.item')), h('th', { class: 'n' }, t('common.baseline')), changed ? [h('th', { class: 'n' }, scenarioName()), h('th', { class: 'n' }, t('common.change'))] : null)),
      h('tbody', null, keys.map(k => h('tr', { class: strong(k) ? 'total' : '' },
        h('td', { title: k }, itemName(k)),
        h('td', { class: 'n' }, fmt(k, b[k])),
        changed ? [h('td', { class: 'n' }, fmt(k, s[k])),
          h('td', { class: 'n' }, isNum(b[k]) && isNum(s[k]) ? (isRate(k) ? deltaSpan('', (s[k] - b[k]) * 100, 1, ' ' + t('common.pts')) : deltaSpan('', s[k] - b[k], 0)) : '–')] : null))))),
  });
}

function charts(d, set) {
  if (!d || !Array.isArray(d.grid) || !d.grid.length) throw new Error(t('payslip.no_grid'));
  const grid = d.grid, b = d.baseline || {}, s = d.scenario || {};
  const arr = v => (Array.isArray(v) && v.length === grid.length ? v : null);
  const differs = (x, y) => !!(x && y && x.some((v, i) => isNum(v) && isNum(y[i]) && Math.abs(v - y[i]) > 1e-6));
  const pair = (key, mul = 1) => {
    const x = arr(b[key]), y = arr(s[key]);
    const out = [];
    if (x) out.push({ name: t('common.baseline'), values: x.map(v => (isNum(v) ? v * mul : null)), color: 'var(--baseline)', width: differs(x, y) ? 1.5 : 2 });
    if (y && differs(x, y)) out.push({ name: scenarioName(), values: y.map(v => (isNum(v) ? v * mul : null)), color: 'var(--scenario)' });
    return out;
  };
  const i = nearest(grid, set.at);
  const pickAt = x => setUi({ at: x });
  const refs = b.reference_wages && typeof b.reference_wages === 'object' ? Object.entries(b.reference_wages).filter(([, v]) => isNum(v) && v >= grid[0] && v <= grid[grid.length - 1]) : [];
  const refBar = refs.length ? h('div', { class: 'row small' }, h('span', { class: 'muted' }, t('payslip.ref_bar', { year: d.year ?? state.year })),
    refs.map(([k, v]) => h('button', { class: 'btn sm', type: 'button', title: eur(v), onclick: () => pickAt(Math.round(v)) }, t('payslip.ref_button', { name: refName(k), amount: eur(v) })))) : null;
  const mmax = Math.max(...[...(arr(b.marginal) || []), ...(arr(s.marginal) || [])].filter(isNum).map(Math.abs), 0);
  const mmul = mmax <= 1.5 ? 100 : 1;
  const same = !differs(arr(b.net), arr(s.net)) && !differs(arr(b.marginal), arr(s.marginal));
  return h('div', { class: 'stack' },
    enabledMeasures().length && same ? notice('info', t('payslip.same')) : null,
    h('div', { class: 'grid g2' },
      card({
        title: t('payslip.net.title'), sub: t('payslip.net.sub', { year: d.year ?? state.year }), id: 'pay-net',
        body: lineChart({ x: grid, series: pair('net'), fmt: eur, xFmt: keur, xPrefix: t('payslip.gross_prefix'), height: 260, mark: grid[i], onPick: pickAt, zero: true, forceLegend: true, title: t('payslip.net.title') }),
        table: () => ({ head: [t('payslip.gross'), t('payslip.net.head_baseline'), t('payslip.net.head_scenario')], rows: grid.map((g, j) => [num(g, 0), num((arr(b.net) || [])[j], 0), num((arr(s.net) || [])[j], 0)]) }),
      }),
      card({
        title: t('payslip.marg.title'), sub: t('payslip.marg.sub'), id: 'pay-marg',
        body: lineChart({ x: grid, series: pair('marginal', mmul), fmt: v => pct(v, 1), xFmt: keur, xPrefix: t('payslip.gross_prefix'), height: 260, mark: grid[i], onPick: pickAt, zero: true, forceLegend: true, title: t('payslip.marg.title') }),
        table: () => ({ head: [t('payslip.gross'), t('payslip.marg.head_baseline'), t('payslip.marg.head_scenario')], rows: grid.map((g, j) => [num(g, 0), num(((arr(b.marginal) || [])[j] ?? NaN) * mmul, 1), num(((arr(s.marginal) || [])[j] ?? NaN) * mmul, 1)]) }),
      })),
    refBar,
    breakdownCard(d, set, i));
}

export function render(root) {
  const optEntry = lazy('payopt:' + state.meta.params_signature, () => api('/api/payslip/options'));
  const retry = () => { lazyReset('pay'); emit(); };
  root.append(lazyBody(optEntry, {
    loading: t('payslip.options_loading'),
    missing: t('payslip.options_missing'),
    retry,
    ok: opts => {
      const set = settings(opts);
      const scen = exportable();
      // a household is a preset id, or the preset with fields overridden (here: the region)
      const household = set.region && set.regions.length && set.region !== set.regions[0].id ? { preset: set.household, region: set.region } : set.household;
      const body = { scenario: scen, year: state.year, kind: set.kind, household, gross_min: set.gross_min, gross_max: set.gross_max, points: set.points };
      const key = `pay:${state.meta.params_signature}:${scenarioSig()}:${JSON.stringify({ ...body, scenario: null })}`;
      const entry = lazy(key, () => post('/api/payslip', body));
      const cav = opts && (opts.caveats || (opts.meta && opts.meta.caveats));
      return h('div', { class: 'stack' },
        card({ title: t('payslip.whose'), sub: scen.measures.length ? t('payslip.whose_sub_scenario', { year: state.year, name: scenarioName() }) : t('payslip.whose_sub_baseline', { year: state.year }), body: controls(set) }),
        cav ? notice('warn', (Array.isArray(cav) ? cav : [cav]).map(c => h('div', null, String(c)))) : null,
        lazyBody(entry, {
          loading: t('payslip.loading'), retry,
          missing: t('payslip.missing'),
          ok: d => h('div', { class: 'stack' },
            d.meta && d.meta.caveats ? notice('warn', (Array.isArray(d.meta.caveats) ? d.meta.caveats : [d.meta.caveats]).map(c => h('div', null, String(c)))) : null,
            charts(d, set)),
        }));
    },
  }));
}
