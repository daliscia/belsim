import { h, download, debounce } from '../dom.js';
import { t, tn, tOr } from '../i18n.js';
import { state, emit, has, raw, label, kindOf, unitFor, deltaSeries, formatter, humanise } from '../store.js';
import { card, lineCard, seg, pickYear } from '../components.js';
import { lineChart } from '../charts.js';

const MAX = 12;
// Section headings: catalog entries `explorer.section.<first part of the key>`, else the humanised code.
const sectionName = sec => tOr('explorer.section.' + sec, humanise(sec));

function allKeys() {
  const keys = new Set(Object.keys(state.base));
  if (state.run) Object.keys(state.run.scenario.series).forEach(k => keys.add(k));
  return [...keys].sort();
}

function keyList(selected, toggle) {
  const list = h('div', { class: 'keylist' });
  const draw = q => {
    q = q.trim().toLowerCase();
    const groups = new Map();
    for (const k of allKeys()) {
      const lab = label(k);
      if (q && !k.toLowerCase().includes(q) && !lab.toLowerCase().includes(q)) continue;
      const sec = k.split('.')[0];
      if (!groups.has(sec)) groups.set(sec, []);
      groups.get(sec).push([k, lab]);
    }
    const nodes = [];
    for (const [sec, items] of groups) {
      nodes.push(h('h4', null, `${sectionName(sec)} (${items.length})`));
      for (const [k, lab] of items) {
        nodes.push(h('label', null,
          h('input', { type: 'checkbox', checked: selected.includes(k), onchange: () => toggle(k) }),
          h('span', null, lab), h('span', { class: 'k' }, k)));
      }
    }
    list.replaceChildren(...(nodes.length ? nodes : [h('div', { class: 'empty', style: { padding: '14px' } }, t('explorer.no_match', { q }))]));
  };
  const input = h('input', {
    type: 'search', id: 'explorer-search', placeholder: t('explorer.search_placeholder'), value: state.ui.explorerQ || '',
    style: { width: '100%' }, oninput: debounce(() => { state.ui.explorerQ = input.value; draw(input.value); }, 80),
  });
  draw(state.ui.explorerQ || '');
  return h('div', null, h('div', { style: { padding: '0 14px 8px' } }, input), list);
}

function exportCsv(keys) {
  // machine-readable export: column names and run ids stay in English whatever the interface language
  const head = ['key', 'label', 'unit', 'run', ...state.years];
  const rows = [];
  for (const k of keys) {
    const b = raw(k, 'baseline') || [], sc = state.run ? raw(k, 'scenario') || [] : null;
    rows.push([k, label(k), kindOf(k), 'baseline', ...b.map(v => v ?? '')]);
    if (sc) {
      rows.push([k, label(k), kindOf(k), 'scenario', ...sc.map(v => v ?? '')]);
      rows.push([k, label(k), kindOf(k), 'delta', ...sc.map((v, i) => (v == null || b[i] == null ? '' : v - b[i]))]);
    }
  }
  const q = v => (/[",\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : String(v));
  download('belsim-series.csv', [head, ...rows].map(r => r.map(q).join(',')).join('\n'), 'text/csv');
}

export function render(root) {
  const selected = (state.ui.explorerKeys || []).filter(has);
  const toggle = k => {
    const cur = (state.ui.explorerKeys || []).filter(has);
    state.ui.explorerKeys = cur.includes(k) ? cur.filter(x => x !== k) : [...cur, k];
    emit();
  };
  const mode = seg('explorerMode', [{ id: 'level', label: t('explorer.mode.level') }, { id: 'delta', label: t('explorer.mode.delta') }], 'level');
  const shown = selected.slice(0, MAX);
  const charts = shown.map(k => {
    const x = h('button', { class: 'x', type: 'button', title: t('common.remove'), 'aria-label': t('explorer.remove_aria', { label: label(k) }), onclick: () => toggle(k) }, '×');
    if (mode.cur === 'delta') {
      const d = state.run ? deltaSeries(k) : null;
      return card({
        title: label(k), sub: t('explorer.delta_sub', { key: k, unit: unitFor(k).label || t('common.units') }), tools: x, id: 'exp-d-' + k,
        body: d ? lineChart({
          x: state.years, series: [{ name: t('explorer.delta_series'), values: d, color: 'var(--scenario)' }],
          fmt: formatter(k), zero: true, height: 170, mark: state.year, onPick: pickYear,
        }) : h('div', { class: 'empty' }, t('explorer.run_first')),
      });
    }
    const c = lineCard(label(k), k, { sub: k, height: 170, id: 'exp-' + k });
    c.querySelector('.card-tools').prepend(x);
    return c;
  });
  root.append(h('div', { class: 'browser' },
    card({ title: t('explorer.all.title'), sub: tn('explorer.all.sub', allKeys().length), flush: true, body: keyList(selected, toggle) }),
    h('div', { class: 'stack' },
      h('div', { class: 'row' }, mode.el, h('span', { class: 'spacer' }),
        h('span', { class: 'hint' }, selected.length > MAX ? tn('explorer.selected_capped', selected.length, { max: MAX }) : tn('explorer.selected', selected.length)),
        h('button', { class: 'btn sm', type: 'button', disabled: !selected.length, onclick: () => exportCsv(selected) }, t('explorer.download')),
        h('button', { class: 'btn sm', type: 'button', disabled: !selected.length, onclick: () => { state.ui.explorerKeys = []; emit(); } }, t('explorer.clear'))),
      charts.length ? h('div', { class: 'grid g2' }, charts)
        : card({ title: t('explorer.none.title'), body: h('div', { class: 'empty' }, h('b', null, t('explorer.none.lead')), ' ', t('explorer.none.rest')) }))));
}
