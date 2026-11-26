// "Balance the budget": gap between the active scenario and a target, and what each prebuilt
// measure contributes on its own. No optimiser: a transparent ranking.
import { h, toast } from '../dom.js';
import {
  state, emit, api, lazy, lazyReset, raw, has, yearIndex, num, signed, pct, moneyBn, hasRun, dimLabel, regionCodes,
} from '../store.js';
import { t, tn } from '../i18n.js';
import { card, deltaSpan, seg, sortTable, lazyBody, pickYear, scenarioName, bandFor, searchBox } from '../components.js';
import { lineChart } from '../charts.js';
import { addMeasure, inScenario, measureTags, regionFilter } from '../scenario.js';
import { impactYears, yearsEditor, bnUnit } from '../impact.js';
import { regionalBreakdown } from '../regional.js';

const at = (s, i) => (s && s[i] != null ? s[i] : null);

function settings() {
  const u = state.ui;
  const deficit = Number.isFinite(u.balDeficit) ? u.balDeficit : 3;
  return { kind: u.balKind === 'debt' ? 'debt' : 'deficit', deficit };
}

/**
 * Target balance in % of GDP for every year, for one run.
 * - deficit: a constant (−x % of GDP)
 * - debt: the balance that keeps debt / GDP at its level of the year before:
 *         b*_t = −debt_{t−1} × g_t / GDP_t, with g_t nominal GDP growth (stock-flow adjustments ignored)
 */
function targetSeries(which, set) {
  const n = state.years.length;
  if (set.kind === 'deficit') return state.years.map(() => -set.deficit);
  const debt = raw('fiscal.debt', which), gdp = raw('macro.gdp_nominal', which);
  if (!debt || !gdp) return new Array(n).fill(null);
  return state.years.map((y, i) => {
    if (!i || debt[i - 1] == null || !gdp[i - 1] || !gdp[i]) return null;
    const g = gdp[i] / gdp[i - 1] - 1;
    return (-debt[i - 1] * g / gdp[i]) * 100;
  });
}

function controls(set, headline) {
  const kind = seg('balKind', [{ id: 'deficit', label: t('balance.kind.deficit') }, { id: 'debt', label: t('balance.kind.debt') }], 'deficit', 'seg');
  const deficit = h('input', { type: 'number', id: 'bal-deficit', step: 0.1, min: -5, max: 20, value: set.deficit, style: { width: '70px' },
    onchange: e => { const v = parseFloat(e.target.value); state.ui.balDeficit = Number.isFinite(v) ? v : 3; emit(); } });
  const year = h('input', { type: 'number', id: 'bal-year', min: state.years[0], max: state.years[state.years.length - 1], value: state.year, style: { width: '76px' },
    onchange: e => { const v = parseInt(e.target.value, 10); if (state.years.includes(v)) { state.year = v; emit(); } else e.target.value = state.year; } });
  return h('div', { class: 'targetbar' },
    h('span', { class: 'lead' }, t('balance.target')),
    kind.el,
    set.kind === 'deficit'
      ? h('label', { class: 'row small' }, t('balance.deficit_at_most'), deficit, t('unit.pct_gdp'))
      : h('span', { class: 'small muted' }, t('balance.debt_rule')),
    h('label', { class: 'row small' }, t('balance.in_year'), year),
    h('span', { class: 'spacer' }),
    h('span', { class: 'hint' }, hasRun() ? t('balance.measured_on', { name: scenarioName() }) : t('balance.measured_on_baseline')),
    h('div', { class: 'verdict' + (headline.miss ? ' miss' : '') }, headline.text));
}

function gapCard(set) {
  const which = hasRun() ? 'scenario' : 'baseline';
  const bal = raw('fiscal.balance_pct_gdp', which), gdp = raw('macro.gdp_nominal', which), cpi = raw('macro.cpi', which);
  const target = targetSeries(which, set);
  const real = state.unit === 'real';
  const gapPct = i => (at(bal, i) == null || target[i] == null ? null : target[i] - bal[i]);     // > 0: effort still needed
  const gapBn = i => { const g = gapPct(i); return g == null || at(gdp, i) == null ? null : (g / 100) * gdp[i] / 1000 / (real ? at(cpi, i) || 1 : 1); };
  const ys = [...new Set([...impactYears(), state.year])].sort();
  const idx = ys.map(y => yearIndex(y));
  const cls = y => 'n' + (y === state.year ? ' hot' : '');
  const effort = (i, f, dec) => {
    const g = f(i);
    if (g == null) return '–';
    return h('span', { class: 'delta ' + (g > 0.005 ? 'bad' : 'good') }, g > 0.005 ? signed(g, dec) : g < -0.005 ? t('balance.spare', { v: num(-g, dec) }) : num(0, dec));
  };
  const rows = [
    h('tr', null, h('td', null, hasRun() ? t('balance.gap.balance_scenario') : t('balance.gap.balance_baseline')), h('td', { class: 'sub' }, t('unit.pct_gdp')), ys.map((y, j) => h('td', { class: cls(y) }, num(at(bal, idx[j]), 2)))),
    h('tr', null, h('td', null, t('balance.gap.target')), h('td', { class: 'sub' }, t('unit.pct_gdp')), ys.map((y, j) => h('td', { class: cls(y) }, num(target[idx[j]], 2)))),
    h('tr', { class: 'total' }, h('td', null, t('balance.gap.to_close')), h('td', { class: 'sub' }, t('common.pts_gdp')), ys.map((y, j) => h('td', { class: cls(y) }, effort(idx[j], gapPct, 2)))),
    h('tr', { class: 'total' }, h('td', null, t('balance.gap.to_close')), h('td', { class: 'sub' }, bnUnit()), ys.map((y, j) => h('td', { class: cls(y) }, effort(idx[j], gapBn, 1)))),
  ];
  const series = [{ name: t('common.baseline'), values: raw('fiscal.balance_pct_gdp', 'baseline'), color: 'var(--baseline)', width: hasRun() ? 1.5 : 2 }];
  if (hasRun()) series.push({ name: scenarioName(), values: bal, color: 'var(--scenario)' });
  series.push({ name: set.kind === 'deficit' ? t('balance.series.target_deficit', { v: pct(set.deficit, 1) }) : t('balance.series.target_debt'), values: target, color: 'var(--bad)', dash: true, width: 1.5 });
  const i = yearIndex(), g = gapPct(i), gb = gapBn(i);
  const who = hasRun() ? 'scenario' : 'baseline';
  const prices = real ? ' ' + t('common.in_prices_of', { year: state.meta.base_year }) : '';
  const headline = g == null ? { text: t('balance.verdict.none', { year: state.year }) }
    : g > 0.005 ? { miss: true, text: t(`balance.verdict.miss_${who}`, { year: state.year, pts: num(g, 2), amount: moneyBn(gb, 1), prices }) }
    : { text: t(`balance.verdict.meet_${who}`, { year: state.year, pts: num(-g, 2), amount: moneyBn(-gb, 1) }) };
  return {
    gapBn: gb, gapPct: g, headline,
    el: h('div', { class: 'grid g-tree' },
      card({
        title: t('balance.gap.title'), flush: true, id: 'bal-gap',
        sub: t('balance.gap.sub'), tools: yearsEditor(),
        body: h('div', null,
          h('div', { class: 'tbl-wrap tall' }, h('table', { class: 'tbl dense' },
            h('thead', null, h('tr', null, h('th', null, ''), h('th', null, t('common.unit')), ys.map(y => h('th', { class: cls(y) }, y)))),
            h('tbody', null, rows))),
          set.kind === 'debt' ? h('p', { class: 'hint', style: { padding: '8px 14px 0' } },
            t('balance.gap.debt_note')) : null),
      }),
      card({
        title: t('balance.chart.title'), sub: t('unit.pct_gdp'), id: 'bal-chart',
        body: lineChart({
          x: state.years, series, fmt: v => pct(v, 2), zero: true, height: 215, mark: state.year, onPick: pickYear,
          band: hasRun() ? bandFor('fiscal.balance_pct_gdp') : null, title: t('balance.chart.title'),
        }),
      })),
  };
}

function rankCard(gap) {
  const entry = lazy('effects:' + state.meta.params_signature, () => api('/api/measures/effects'));
  const real = state.unit === 'real';
  // optional: the effect of each measure in one region (employment, GDP, regional government balance)
  let regionSeg = null, regionFilt = null;
  const body = lazyBody(entry, {
    loading: tn('balance.rank.loading', state.library.measures.length),
    detail: t('balance.rank.loading_detail'),
    retry: () => { lazyReset('effects:'); emit(); },
    ok: data => {
      const i = data.years.indexOf(state.year);
      const cpiB = raw('macro.cpi', 'baseline');
      const q = (state.ui.balQ || '').trim().toLowerCase();
      regionFilt = regionFilter('balLibRegion', data.measures);
      // series of one measure rebuilt from its change: scenario = baseline + delta
      const getter = d => key => {
        const base = raw(key, 'baseline');
        return { b: base, s: base && d[key] ? base.map((v, j) => (v == null || d[key][j] == null ? null : v + d[key][j])) : null };
      };
      const shape = regionalBreakdown(getter({}), { real });
      const codes = [...new Set(shape.flatMap(g => g.rows.map(r => r.code)))];
      if (codes.length) {
        regionSeg = seg('balRegion', [{ id: 'none', label: t('balance.rank.no_region') },
          ...codes.map(c => ({ id: c, label: regionCodes().includes(c) ? dimLabel('regions', c) : dimLabel('entities', c) }))], 'none');
      }
      const reg = regionSeg && regionSeg.cur !== 'none' ? regionSeg.cur : null;
      const rows = data.measures
        .filter(m => (!regionFilt || regionFilt.keep(m)) && (!q || `${m.name} ${m.description} ${(m.tags || []).join(' ')} ${Object.values(m.tag_labels || {}).join(' ')}`.toLowerCase().includes(q)))
        .map(m => {
          const d = m.delta || {};
          // 2024 prices: the measure's own price effect is second-order here, so deflate with the baseline CPI
          const bn = at(d['fiscal.balance'], i) == null ? null : d['fiscal.balance'][i] / 1000 / (real ? at(cpiB, i) || 1 : 1);
          const regional = {};
          if (reg) {
            for (const g of regionalBreakdown(getter(d), { real, cpi: { s: cpiB, b: cpiB } })) {
              const r = g.rows.find(x => x.code === reg);
              if (r) regional[g.id] = r.f(i);
            }
          }
          return { m, bn, regional, pct: at(d['fiscal.balance_pct_gdp'], i), emp: at(d['labour.employment'], i) == null ? null : d['labour.employment'][i] / 1000,
            gdp: at(d['macro.gdp_real'], i) == null || !at(raw('macro.gdp_real', 'baseline'), i) ? null : d['macro.gdp_real'][i] / raw('macro.gdp_real', 'baseline')[i] * 100 };
        });
      const maxAbs = Math.max(1e-9, ...rows.map(r => Math.abs(r.bn || 0)));
      const bar = r => {
        if (r.bn == null) return null;
        const w = Math.abs(r.bn) / maxAbs * 50;
        return h('span', { class: 'divbar', 'aria-hidden': 'true' }, h('span', { class: r.bn >= 0 ? 'pos' : 'neg', style: { width: w + '%', [r.bn >= 0 ? 'left' : 'right']: '50%' } }));
      };
      const regName = reg ? (regionCodes().includes(reg) ? dimLabel('regions', reg) : dimLabel('entities', reg)) : '';
      const regionCols = reg ? shape.filter(g => g.rows.some(r => r.code === reg)).map(g => ({
        id: 'reg:' + g.id, label: `Δ ${g.label}, ${regName}, ${g.unit}`, num: true, cls: 'hot',
        value: r => (r.regional[g.id] == null ? null : r.regional[g.id]), cell: r => (r.regional[g.id] == null ? null : deltaSpan(g.polKey, r.regional[g.id], g.dec)),
      })) : [];
      const cols = [
        { id: 'name', label: t('balance.rank.col_measure'), value: r => r.m.name, cls: 'wrap',
          cell: r => [h('b', null, r.m.name), h('div', { class: 'sub' }, r.m.error ? h('span', { style: { color: 'var(--bad)' } }, t('balance.rank.failed', { error: r.m.error })) : r.m.description,
            measureTags(r.m))] },
        { id: 'bn', label: t('balance.rank.col_balance', { unit: bnUnit() }), num: true, value: r => r.bn, cell: r => (r.bn == null ? null : [bar(r), deltaSpan('fiscal.balance', r.bn, 2)]) },
        { id: 'pct', label: t('balance.rank.col_balance', { unit: t('common.pts_gdp') }), num: true, value: r => r.pct, cell: r => (r.pct == null ? null : deltaSpan('fiscal.balance', r.pct, 2)) },
        { id: 'share', label: t('balance.rank.col_share'), num: true, title: t('balance.rank.col_share_title'),
          value: r => (r.pct == null || !(gap.gapPct > 0.005) ? null : r.pct / gap.gapPct * 100), cell: r => (r.pct == null || !(gap.gapPct > 0.005) ? null : pct(r.pct / gap.gapPct * 100, 0)) },
        { id: 'emp', label: t('balance.rank.col_employment'), num: true, value: r => r.emp, cell: r => (r.emp == null ? null : deltaSpan('labour.employment', r.emp, 1)) },
        { id: 'gdp', label: t('balance.rank.col_gdp'), num: true, value: r => r.gdp, cell: r => (r.gdp == null ? null : deltaSpan('macro.gdp_real', r.gdp, 2)) },
        ...regionCols,
        { id: 'add', label: '', cell: r => { const used = inScenario(r.m); return h('button', {
          class: 'btn sm', type: 'button', disabled: used || !!r.m.error,
          onclick: () => { addMeasure(r.m); state.panelOpen = true; toast(t('balance.rank.added', { name: r.m.name })); },
        }, used ? t('common.in_scenario') : t('common.add_to_scenario')); } },
      ];
      if (i < 0) return h('div', { class: 'empty', style: { padding: '14px' } }, t('balance.rank.no_year', { year: state.year }));
      return h('div', null,
        regionSeg || regionFilt ? h('div', { class: 'row', style: { padding: '0 14px 8px' } },
          regionFilt ? [h('span', { class: 'hint' }, t('balance.rank.measures_for')), regionFilt.el] : null,
          regionSeg ? [h('span', { class: 'hint' }, t('balance.rank.effect_in')), regionSeg.el] : null) : null,
        sortTable(cols, rows, { id: 'bal-rank', sort: 'bn', desc: true, rowClass: r => (inScenario(r.m) ? 'dim' : null) }),
        h('p', { class: 'hint', style: { padding: '8px 14px 0', maxWidth: '920px' } }, t('balance.rank.note')));
    },
  });
  return card({
    title: t('balance.rank.title', { year: state.year }), flush: true, id: 'bal-rank',
    sub: t('balance.rank.sub'),
    tools: searchBox('balQ', t('balance.rank.filter'), '180px'),
    body,
  });
}

export function render(root) {
  if (!has('fiscal.balance_pct_gdp') || !has('macro.gdp_nominal')) {
    root.append(h('div', { class: 'empty' }, t('balance.unavailable')));
    return;
  }
  const set = settings();
  const gap = gapCard(set);
  root.append(h('div', { class: 'stack' },
    controls(set, gap.headline), gap.el,
    rankCard(gap)));
}
