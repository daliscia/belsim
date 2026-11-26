import { h } from '../dom.js';
import {
  state, emit, has, raw, valueAt, sumAt, unitFor, num, moneyBn, label, dimLabel, color, OTHER, fold,
} from '../store.js';
import { t, tn } from '../i18n.js';
import { card, deltaSpan, lineCard, stackCard, tabs } from '../components.js';

const SIDES = {
  revenue: { meta: m => m.revenue_lines, total: 'fiscal.revenue_total' },
  expenditure: { meta: m => m.expenditure_lines, total: 'fiscal.expenditure_total' },
};
const sideTitle = side => t(`budget.side.${side}`);

/** Lines of one side grouped as the model's LINES metadata says; lines without metadata go to "Unclassified".
 *  `id` is the stable (English) group name, `name` the label in the interface language. */
function groupsOf(side) {
  const meta = SIDES[side].meta(state.meta);
  const keys = new Set(Object.keys(meta));
  const all = new Set([...Object.keys(state.base), ...(state.run ? Object.keys(state.run.scenario.series) : [])]);
  for (const k of all) if (k.startsWith(side + '.')) keys.add(k.slice(side.length + 1));
  const groups = new Map();
  for (const id of keys) {
    const key = `${side}.${id}`;
    if (!has(key)) continue;
    const m = meta[id];
    const gid = (m && (m.group_id || m.group)) || 'Unclassified';
    if (!groups.has(gid)) groups.set(gid, { id: gid, name: m && m.group ? m.group : t('budget.unclassified'), lines: [] });
    groups.get(gid).lines.push({ id, key, meta: m || { label: label(key), esa: '', entity_shares: {} } });
  }
  return [...groups.values()];
}

function tree(side) {
  const groups = groupsOf(side);
  const u = unitFor(SIDES[side].total);
  const allKeys = groups.flatMap(g => g.lines.map(l => l.key));
  // shares and % of GDP are computed from current-price amounts whatever the unit switch says
  const total = sumAt(allKeys, 'scenario', state.year, 'bn');
  const closed = state.ui['closed:' + side] || {};
  const sel = state.ui.budgetLine;
  const pick = key => { state.ui.budgetLine = key; emit(); };
  const cells = (keys, key) => {
    const v = sumAt(keys, 'scenario'), bn = sumAt(keys, 'scenario', state.year, 'bn'), pct = sumAt(keys, 'scenario', state.year, 'pct');
    const b = sumAt(keys, 'baseline');
    const d = state.run && v != null && b != null ? v - b : null;
    const share = total ? (bn / total) * 100 : null;
    return [
      h('td', { class: 'n' }, num(v, u.dec)),
      h('td', { class: 'n' }, num(pct, 2)),
      h('td', { class: 'n' }, h('span', { class: 'sharebar', style: { width: Math.max(0, Math.min(100, share || 0)) * 0.4 + 'px' } }), num(share, 1)),
      state.run ? h('td', { class: 'n' }, d == null ? '–' : deltaSpan(key, d, 2)) : null,
    ];
  };
  const rows = [];
  for (const g of groups) {
    const gkeys = g.lines.map(l => l.key);
    const gid = `group:${side}:${g.id}`;
    const toggle = evt => {
      evt.stopPropagation();
      state.ui['closed:' + side] = { ...closed, [g.id]: !closed[g.id] }; emit();
    };
    rows.push(h('tr', { class: 'group' + (sel === gid ? ' sel' : ''), onclick: () => pick(gid) },
      h('td', null,
        h('button', { class: 'caret', type: 'button', onclick: toggle, 'aria-expanded': String(!closed[g.id]), 'aria-label': t('budget.tree.fold_group', { group: g.name }) }, closed[g.id] ? '▸' : '▾'),
        g.name, h('span', { class: 'sub' }, `  ${g.lines.length}`)),
      cells(gkeys, '')));
    if (closed[g.id]) continue;
    g.lines.sort((a, b) => (valueAt(b.key, 'scenario', state.year, 'bn') || 0) - (valueAt(a.key, 'scenario', state.year, 'bn') || 0));
    for (const l of g.lines) {
      rows.push(h('tr', { class: 'line' + (sel === l.key ? ' sel' : ''), onclick: () => pick(l.key), title: l.meta.description || null },
        h('td', null, l.meta.label, l.meta.esa ? h('span', { class: 'esa' }, l.meta.esa) : null), cells([l.key], '')));
    }
  }
  rows.push(h('tr', { class: 'total' }, h('td', null, t(`budget.tree.total_${side}`)), cells(allKeys, '')));
  const reported = valueAt(SIDES[side].total, 'scenario', state.year, 'bn');
  const gap = reported != null && total != null ? reported - total : null;
  const allClosed = groups.every(g => closed[g.id]);
  return card({
    title: t('budget.tree.title', { side: sideTitle(side), year: state.year }), flush: true, id: 'budget-tree-' + side,
    sub: t('budget.tree.sub', { groups: tn('budget.n_groups', groups.length), lines: tn('budget.n_lines', allKeys.length) }),
    tools: h('div', { class: 'row' },
      h('button', { class: 'linkbtn', type: 'button', onclick: () => {
        state.ui['closed:' + side] = Object.fromEntries(groups.map(g => [g.id, !allClosed])); emit();
      } }, allClosed ? t('budget.tree.unfold_all') : t('budget.tree.fold_all'))),
    body: h('div', null,
      h('div', { class: 'tbl-wrap tall' }, h('table', { class: 'tbl' },
        h('thead', null, h('tr', null, h('th', null, t('budget.tree.col_line')), h('th', { class: 'n' }, u.label),
          h('th', { class: 'n' }, t('unit.pct_gdp')), h('th', { class: 'n' }, t('budget.tree.col_share')), state.run ? h('th', { class: 'n' }, t('common.delta_vs_baseline')) : null)),
        h('tbody', null, rows))),
      gap != null && Math.abs(gap) > 0.05 ? h('p', { class: 'hint', style: { padding: '8px 14px 0' } },
        t('budget.tree.gap', { sum: moneyBn(total, 1), key: SIDES[side].total, reported: moneyBn(reported, 1), year: state.year })) : null),
  });
}

function detail(side) {
  const groups = groupsOf(side);
  let sel = state.ui.budgetLine;
  const g = sel && sel.startsWith(`group:${side}:`) ? groups.find(x => `group:${side}:${x.id}` === sel) : null;
  if (!g && !(sel && sel.startsWith(side + '.') && has(sel))) sel = SIDES[side].total;
  if (g) {
    return stackCard(t('budget.detail.by_line', { group: g.name }), layerItems(g.lines.map(l => ({ name: l.meta.label, keys: [l.key] }))), { id: 'budget-detail' });
  }
  return lineCard(sel === SIDES[side].total ? sideTitle(side) : label(sel), sel, {
    id: 'budget-detail', sub: sel === SIDES[side].total ? t('common.total') : sel, zero: false, height: 220,
  });
}

/** At most 7 layers + "Other", ranked by size in the base year so colours stay put. */
function layerItems(items) {
  const w = it => it.keys.reduce((tot, k) => tot + Math.abs((raw(k, 'baseline') || [0])[0] || 0), 0);
  const { keep, rest } = fold(items, w, 7);
  const out = keep.map((it, i) => ({ ...it, color: color(i) }));
  if (rest.length) out.push({ name: t('common.other_n', { n: rest.length }), keys: rest.flatMap(r => r.keys), color: OTHER });
  return out;
}

function sideView(side) {
  const groups = groupsOf(side);
  if (!groups.length) {
    return h('div', { class: 'empty' }, t(`budget.empty_${side}`));
  }
  return h('div', { class: 'grid g-tree' },
    tree(side),
    h('div', { class: 'stack' },
      detail(side),
      stackCard(t('budget.by_group', { side: sideTitle(side) }), layerItems(groups.map(g => ({ name: g.name, keys: g.lines.map(l => l.key) }))),
        { id: 'budget-stack-' + side })));
}

// ------------------------------------------------------------------ entities
const MEASURES = ['revenue', 'expenditure', 'balance', 'debt'];
const TOTAL_KEYS = { revenue: 'fiscal.revenue_total', expenditure: 'fiscal.expenditure_total', balance: 'fiscal.balance', debt: 'fiscal.debt' };

function entityValue(measure, entity, which, year, unit) {
  const key = `fiscal.${measure}.${entity}`;
  if (has(key)) return { v: valueAt(key, which, year, unit), derived: false };
  if ((measure !== 'revenue' && measure !== 'expenditure') || !state.meta.dims.entities.includes(entity)) return { v: null, derived: false };
  // fall back on the lines' entity shares (before inter-governmental transfers)
  const meta = SIDES[measure].meta(state.meta);
  let tot = 0, any = false;
  for (const [id, m] of Object.entries(meta)) {
    const share = (m.entity_shares || {})[entity];
    const v = valueAt(`${measure}.${id}`, which, year, unit);
    if (share && v != null) { tot += share * v; any = true; }
  }
  return { v: any ? tot : null, derived: any };
}

/** Federated entities the model details under a subsector: {parent: [entity ids that have figures]}. */
function childrenOf() {
  const detail = state.meta.entity_detail || {};
  const out = {};
  for (const [id, info] of Object.entries(detail)) {
    if (!MEASURES.some(m => has(`fiscal.${m}.${id}`))) continue;
    const parent = (info && info.parent) || 'regions_communities';
    (out[parent] = out[parent] || []).push(id);
  }
  return out;
}

function entityView() {
  const ents = state.meta.dims.entities;
  const kids = childrenOf();
  const u = unitFor('fiscal.balance');
  const closed = state.ui.entClosed || {};
  let derived = false;
  const cells = e => MEASURES.flatMap(m => {
    const sc = entityValue(m, e, 'scenario', state.year), b = entityValue(m, e, 'baseline', state.year);
    derived = derived || sc.derived;
    const d = state.run && sc.v != null && b.v != null ? sc.v - b.v : null;
    return [h('td', { class: 'n' }, num(sc.v, u.dec), sc.derived ? '*' : ''),
      h('td', { class: 'n' }, d == null ? '' : deltaSpan(m === 'balance' ? 'balance' : m === 'debt' ? 'debt' : '', d, 2))];
  });
  const rows = [];
  for (const e of ents) {
    const sub = kids[e] || [];
    const name = dimLabel('entities', e);
    rows.push(h('tr', { class: sub.length ? 'parent' : null },
      h('td', null,
        sub.length ? h('button', { class: 'caret', type: 'button', 'aria-expanded': String(!closed[e]), 'aria-label': t('budget.tree.fold_group', { group: name }),
          onclick: () => { state.ui.entClosed = { ...closed, [e]: !closed[e] }; emit(); } }, closed[e] ? '▸' : '▾') : null,
        name, h('span', { class: 'esa' }, '  ' + (state.meta.dims.entity_esa[e] || ''))),
      cells(e)));
    if (closed[e]) continue;
    for (const c of sub) {
      const info = state.meta.entity_detail[c] || {};
      // the consolidation item is the difference with the consolidated subsector, not a government
      rows.push(h('tr', { class: 'line child' + (info.kind === 'consolidation' ? ' dim' : '') },
        h('td', null, dimLabel('entities', c), info.esa ? h('span', { class: 'esa' }, info.esa) : null), cells(c)));
    }
  }
  rows.push(h('tr', { class: 'total' }, h('td', null, t('budget.entities.general_government')), MEASURES.flatMap(m => {
    const sc = valueAt(TOTAL_KEYS[m], 'scenario'), b = valueAt(TOTAL_KEYS[m], 'baseline');
    return [h('td', { class: 'n' }, num(sc, u.dec)), h('td', { class: 'n' }, state.run && sc != null && b != null ? deltaSpan(TOTAL_KEYS[m], sc - b, 2) : '')];
  })));
  const missing = MEASURES.filter(m => !ents.some(e => has(`fiscal.${m}.${e}`)));
  const order = ['balance', 'debt', 'revenue', 'expenditure'];
  const charts = order.filter(m => ents.some(e => has(`fiscal.${m}.${e}`)))
    .map(m => lineCard(t(`budget.entities.chart_${m}`), ents.map(e => `fiscal.${m}.${e}`),
      { names: ents.map(e => dimLabel('entities', e)), zero: m === 'balance', id: 'ent-' + m }));
  // the communities and regions one by one, when the regional model details them
  const fed = Object.values(kids).flat().filter(e => (state.meta.entity_detail[e] || {}).kind !== 'consolidation');
  const fedCharts = fed.length ? order.filter(m => fed.some(e => has(`fiscal.${m}.${e}`)))
    .map(m => lineCard(t(`budget.entities.fed_chart_${m}`), fed.map(e => `fiscal.${m}.${e}`),
      { names: fed.map(e => dimLabel('entities', e)), zero: m === 'balance', id: 'fed-' + m })) : [];
  return h('div', { class: 'stack' },
    card({
      title: t('budget.entities.title', { year: state.year }), sub: t('budget.entities.sub', { unit: u.label }), flush: true, id: 'budget-entities',
      body: h('div', null,
        h('div', { class: 'tbl-wrap tall' }, h('table', { class: 'tbl' },
          h('thead', null, h('tr', null, h('th', null, t('budget.entities.col_entity')),
            MEASURES.flatMap(m => [h('th', { class: 'n' }, t(`budget.measure.${m}`)), h('th', { class: 'n' }, 'Δ')]))),
          h('tbody', null, rows))),
        derived ? h('p', { class: 'hint', style: { padding: '8px 14px 0' } }, t('budget.entities.derived')) : null,
        Object.keys(kids).length ? h('p', { class: 'hint', style: { padding: '4px 14px 0' } }, t('budget.entities.fed_note')) : null,
        missing.length ? h('p', { class: 'hint', style: { padding: '4px 14px 0' } },
          t('budget.entities.missing', { list: missing.map(m => t(`budget.measure.${m}`)).join(', ') })) : null),
    }),
    h('div', { class: 'grid g2' }, charts),
    fedCharts.length ? h('div', { class: 'grid g2' }, fedCharts) : null);
}

export function render(root) {
  const tab = tabs('budgetTab', [
    { id: 'revenue', label: t('budget.side.revenue') }, { id: 'expenditure', label: t('budget.side.expenditure') }, { id: 'entities', label: t('budget.tab_entities') },
  ]);
  root.append(tab.el, tab.cur === 'entities' ? entityView() : sideView(tab.cur));
}
