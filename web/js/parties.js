// Party programmes of the 2024 costing (DC2024): list, coverage, detail of one party, all parties.
// Everything about the parties comes from /api/scenarios?kind=party (state.parties).
import { h, toast } from './dom.js';
import { state, emit, api, lazy, lazyReset, num, signed, enabledMeasures, compareText } from './store.js';
import { t, tn } from './i18n.js';
import { card, notice, lazyBody, sortTable, deltaSpan, seg } from './components.js';
import { hbars } from './charts.js';
import { loadScenario, keepForComparison, inComparison, registerPartiesSection } from './scenario.js';

const STATUS_COLOR = { modelled: 'var(--c3)', partly_modelled: 'var(--c4)', not_modelled: 'var(--ink-3)', already_in_baseline: 'var(--c1)' };
const STATUSES = ['modelled', 'partly_modelled', 'already_in_baseline', 'not_modelled'];

export const partyList = () => (state.parties && state.parties.scenarios) || [];
export const partyById = id => partyList().find(p => p.id === id) || null;

/** Parties by language community (order given by the server), alphabetical inside a group. */
export function partyGroups() {
  const order = (state.parties && state.parties.communities) || [];
  return order.map(c => ({
    id: c, label: t(`parties.group.${c}`),
    parties: partyList().filter(p => p.community === c).sort((a, b) => compareText(a.name, b.name)),
  })).filter(g => g.parties.length);
}

export function caveatNotice() {
  if (!state.parties) return null;
  const src = state.parties.source || {};
  return notice('warn', h('b', null, t('parties.caveat_lead') + ' '), state.parties.caveat, ' ',
    src.url ? h('a', { href: src.url, target: '_blank', rel: 'noopener' }, t('parties.source_link')) : null);
}

const pctText = v => t('fmt.pct', { v: num(v * 100, 0) });

function coverageText(cov) {
  return STATUSES.map(s => `${t(`parties.status.${s}`)}${t('common.colon')}${cov.count[s]} (${pctText(cov.impulse_share[s])})`).join(' · ');
}

/** Share of the ex-ante impulse by status, as one stacked bar. */
export function coverageBar(p, width = '120px') {
  const cov = p.coverage;
  return h('div', { class: 'covbar', style: { width }, role: 'img', title: coverageText(cov),
    'aria-label': t('parties.coverage_aria', { name: p.name, text: coverageText(cov) }) },
  STATUSES.map(s => (cov.impulse_share[s] > 0 ? h('span', { style: { width: cov.impulse_share[s] * 100 + '%', background: STATUS_COLOR[s] } }) : null)));
}

export function coverageLegend() {
  return h('div', { class: 'legend', style: { margin: '0' } }, STATUSES.map(s =>
    h('span', { class: 'item' }, h('span', { class: 'key-box', style: { background: STATUS_COLOR[s] } }), t(`parties.status.${s}`))));
}

function statusTag(status) {
  return h('span', { class: 'tag status', style: { borderColor: STATUS_COLOR[status] } },
    h('span', { class: 'dot', style: { background: STATUS_COLOR[status] } }), t(`parties.status.${status}`));
}

// ------------------------------------------------------------------ actions
export function loadParty(p) {
  if (enabledMeasures().length && !inComparison()
    && !confirm(t('compare.set.replace_confirm', { current: state.scenario.name, name: p.scenario.name }))) return;
  loadScenario(p.scenario);
  state.panelOpen = true;
  toast(t('compare.set.now_in_panel', { name: p.scenario.name }));
}
export const keepParty = p => keepForComparison(p.scenario);
export function openParty(p) {
  state.ui.party = p.id;
  if (state.view !== 'parties') location.hash = '#/parties';
  emit();
}

function actions(p, compact) {
  const none = !p.scenario.measures.length;
  return h('span', { class: 'row', style: { gap: '6px', flexWrap: 'nowrap', justifyContent: 'flex-end' } },
    h('button', { class: 'btn sm', type: 'button', disabled: none, title: t('parties.load_title', { name: p.name }), onclick: () => loadParty(p) }, t('panel.load')),
    h('button', { class: 'btn sm', type: 'button', disabled: none, title: t('parties.keep_title', { name: p.name }), onclick: () => keepParty(p) }, t('parties.keep')),
    h('button', { class: compact ? 'linkbtn' : 'btn sm', type: 'button', onclick: () => openParty(p) }, t('parties.details')));
}

// ------------------------------------------------------------------ scenario panel section
export function panelSection() {
  if (!partyList().length) return null;
  const groups = partyGroups().map(g => [
    h('div', { class: 'hint', style: { marginTop: '8px', fontWeight: 600 } }, g.label),
    ...g.parties.map(p => h('div', { class: 'lib-item' },
      h('b', null, p.name), actions(p, true),
      h('div', { class: 'd' }, h('div', { class: 'row', style: { gap: '8px', flexWrap: 'nowrap' } }, coverageBar(p, '90px'),
        h('span', null, t('parties.panel_line', { n: p.coverage.measures, with: p.scenario.measures.length })))))),
  ]);
  return h('details', { class: 'fold', id: 'panel-parties', open: !!state.ui.partiesOpen, ontoggle: e => { state.ui.partiesOpen = e.target.open; } },
    h('summary', null, t('parties.panel_title', { n: partyList().length })),
    h('div', null,
      h('div', { class: 'notice warn', style: { fontSize: '12px', margin: '4px 0' } }, state.parties.caveat),
      groups,
      h('div', { style: { marginTop: '8px' } },
        h('button', { class: 'linkbtn', type: 'button', onclick: () => { state.ui.partiesAll = true; location.hash = '#/parties'; emit(); } }, t('parties.all.button')))));
}

// ------------------------------------------------------------------ list card (Compare view, Parties view)
export function listCard() {
  if (!state.parties) return null;
  if (!partyList().length) return card({ title: t('parties.title'), body: h('div', { class: 'empty' }, t('parties.none')) });
  const refYear = 2029;
  const rows = partyGroups().flatMap(g => [
    h('tr', { class: 'group' }, h('th', { colspan: 5, scope: 'colgroup' }, g.label)),
    ...g.parties.map(p => h('tr', { class: state.ui.party === p.id && state.view === 'parties' ? 'sel' : null },
      h('td', null, h('b', null, p.name), p.full_name !== p.name ? h('div', { class: 'sub' }, p.full_name) : null),
      h('td', { class: 'n', title: coverageText(p.coverage) }, `${p.coverage.costed} / ${p.coverage.measures}`),
      h('td', null, h('div', { class: 'row', style: { gap: '8px', flexWrap: 'nowrap' } }, coverageBar(p),
        h('span', { class: 'sub' }, t('parties.cov_short', { m: pctText(p.coverage.impulse_share.modelled), p: pctText(p.coverage.impulse_share.partly_modelled) })))),
      h('td', { class: 'n' }, p.official.impulse ? signed(p.official.impulse[refYear], 0) : '–'),
      h('td', { class: 'n' }, actions(p)))),
  ]);
  const all = !!state.ui.partiesAll;
  return card({
    title: t('parties.title'), flush: true, id: 'parties-list',
    sub: t('parties.list_sub'),
    tools: h('button', { class: 'btn sm' + (all ? '' : ' primary'), type: 'button', id: 'parties-all-btn',
      onclick: () => { state.ui.partiesAll = !all; emit(); } }, all ? t('parties.all.hide') : t('parties.all.button')),
    body: h('div', null,
      h('div', { style: { padding: '0 14px 10px' } }, caveatNotice()),
      h('div', { class: 'tbl-wrap full' }, h('table', { class: 'tbl parties' },
        h('thead', null, h('tr', null,
          h('th', null, t('parties.col.party')),
          h('th', { class: 'n', title: t('parties.col.measures_title') }, t('parties.col.measures')),
          h('th', null, t('parties.col.coverage')),
          h('th', { class: 'n', title: t('parties.col.impulse_title') }, t('parties.col.impulse', { year: refYear })),
          h('th', null, ''))),
        h('tbody', null, rows))),
      h('div', { class: 'row', style: { padding: '8px 14px 0', gap: '14px' } }, coverageLegend()),
      h('details', { class: 'fold', style: { margin: '6px 14px 0' } }, h('summary', null, t('parties.method_title')),
        h('p', { class: 'hint', style: { maxWidth: '980px', lineHeight: 1.5 } }, state.parties.method))),
  });
}

// ------------------------------------------------------------------ detail of one party
/** How a measure is modelled, in words (built from the codes of the party file). */
function methodNotes(m) {
  const out = [];
  if (m.method === 'baseline') out.push(t('parties.method.baseline', { what: t(`parties.baseline.${m.baseline}`) }));
  else if (m.method === 'sized') out.push(m.status === 'partly_modelled' ? t('parties.method.proxy') : t('parties.method.sized'));
  else out.push(t(`parties.method.${m.method}`));
  if (m.increment) out.push(t('parties.method.increment'));
  if (m.reversal) out.push(t('parties.method.reversal'));
  if (m.sizing_residual != null) out.push(t('parties.method.residual', { gap: num(Math.abs(m.sizing_residual), 0) }));
  if (m.lever) out.push(t('parties.method.missing', { lever: t(`parties.lever.${m.lever}`) }));
  return out.join(' ');
}

const at = (s, i) => (s && s[i] != null ? s[i] : null);

function detailTable(p, fx) {
  const maxYear = fx.years[fx.years.length - 1];
  const year = Math.min(Math.max(state.year, 2027), maxYear), yi = fx.years.indexOf(year);
  const T = Math.min(Math.max(year - 2, 2025), 2029);
  const gdp0 = at(fx.gdp, fx.years.indexOf(2024)), gdpY = at(fx.gdp, yi);
  const byFiche = Object.fromEntries(fx.measures.map(r => [r.fiche, r]));
  const filter = seg('partyFilter', ['all', 'changes', 'none'].map(id => ({ id, label: t(`parties.detail.filter.${id}`) })), 'all');
  const rowsAll = p.measures.map(m => {
    const r = byFiche[m.fiche] || {};
    const off = m.official_impulse ? m.official_impulse[T] : null;
    return {
      m, off, scaled: off != null && gdp0 ? off * gdpY / gdp0 : null,
      stat: r.static ? at(r.static['fiscal.primary_balance'], yi) : null,
      beh: r.behavioural ? at(r.behavioural['fiscal.primary_balance'], yi) : null,
      emp: r.behavioural ? at(r.behavioural['labour.employment'], yi) : null,
      error: r.error,
    };
  });
  const rows = rowsAll.filter(r => filter.cur === 'all' || (filter.cur === 'changes') === (r.m.changes.length > 0));
  const cols = [
    { id: 'fiche', label: t('parties.detail.col.fiche'), value: r => r.m.fiche,
      cell: r => h('a', { href: r.m.source.url, target: '_blank', rel: 'noopener', title: t('parties.detail.source') }, r.m.source.fiche) },
    { id: 'name', label: t('parties.detail.col.measure'), value: r => r.m.name, cls: 'wrap',
      cell: r => [h('div', null, r.m.name), h('div', { class: 'sub', lang: p.community === 'fr' ? 'fr' : 'nl', title: t('parties.detail.title_in_fiche') }, `« ${r.m.title} »`)] },
    { id: 'status', label: t('parties.detail.col.status'), value: r => STATUSES.indexOf(r.m.status), cls: 'wrap',
      cell: r => [statusTag(r.m.status), h('div', { class: 'sub' }, methodNotes(r.m)),
        r.error ? h('div', { class: 'sub', style: { color: 'var(--bad)' } }, r.error) : null] },
    { id: 'off', label: t('parties.detail.col.official', { year: T }), num: true, title: t('parties.detail.col.official_title', { T, year }), value: r => r.off,
      cell: r => (r.off == null ? null : [signed(r.off, 0), h('div', { class: 'sub' }, t('parties.detail.scaled', { v: signed(r.scaled, 0), year }))]) },
    { id: 'stat', label: t('parties.detail.col.static', { year }), num: true, title: t('parties.detail.col.model_title', { year }), value: r => r.stat,
      cell: r => (r.stat == null ? null : deltaSpan('fiscal.balance', r.stat, 0)) },
    { id: 'beh', label: t('parties.detail.col.behav', { year }), num: true, title: t('parties.detail.col.model_title', { year }), value: r => r.beh,
      cell: r => (r.beh == null ? null : deltaSpan('fiscal.balance', r.beh, 0)) },
    { id: 'emp', label: t('parties.detail.col.employment', { year }), num: true, value: r => r.emp,
      cell: r => (r.emp == null ? null : deltaSpan('labour.employment', r.emp / 1000, 1)) },
  ];
  const pk = fx.package || {};
  const pv = (variant, key, scale = 1) => { const v = pk[variant] ? at(pk[variant][key], yi) : null; return v == null ? null : v * scale; };
  const offSum = p.official.impulse ? p.official.impulse[T] : null;
  const foot = h('tr', { class: 'foot' },
    h('td', null, ''), h('td', null, h('b', null, t('parties.detail.package')), h('div', { class: 'sub' }, t('parties.detail.package_sub'))), h('td', null, ''),
    h('td', { class: 'n' }, offSum == null ? '–' : [signed(offSum, 0), h('div', { class: 'sub' }, t('parties.detail.scaled', { v: signed(offSum * gdpY / gdp0, 0), year }))]),
    h('td', { class: 'n' }, deltaSpan('fiscal.balance', pv('static', 'fiscal.primary_balance'), 0)),
    h('td', { class: 'n' }, deltaSpan('fiscal.balance', pv('behavioural', 'fiscal.primary_balance'), 0)),
    h('td', { class: 'n' }, deltaSpan('labour.employment', pv('behavioural', 'labour.employment', 1 / 1000), 1)));
  const o = p.official, oy = T;
  const offLine = [
    o.balance_pct_gdp ? `${t('compare.metric.balance')} ${signed(o.balance_pct_gdp[oy], 2)} ${t('common.pts_gdp')}` : null,
    o.gdp_pct ? `${t('compare.metric.gdp_real')} ${t('fmt.pct', { v: signed(o.gdp_pct[oy], 2) })}` : null,
    o.employment ? `${t('compare.metric.employment')} ${signed(o.employment[oy] / 1000, 1)} ${t('common.thousand')}` : null,
  ].filter(Boolean).join(' · ');
  const modLine = [
    pv('behavioural', 'fiscal.balance_pct_gdp') == null ? null : `${t('compare.metric.balance')} ${signed(pv('behavioural', 'fiscal.balance_pct_gdp'), 2)} ${t('common.pts_gdp')}`,
    pv('behavioural', 'labour.employment') == null ? null : `${t('compare.metric.employment')} ${signed(pv('behavioural', 'labour.employment', 1 / 1000), 1)} ${t('common.thousand')}`,
  ].filter(Boolean).join(' · ');
  return h('div', null,
    h('div', { class: 'row', style: { padding: '0 14px 8px', gap: '14px' } }, filter.el,
      h('span', { class: 'hint' }, t('parties.detail.year_note', { T, y: year, max: maxYear }))),
    sortTable(cols, rows, { id: 'party-detail', sort: null, foot, cls: 'tbl party-detail' }),
    h('div', { class: 'hint', style: { padding: '8px 14px 0', lineHeight: 1.6 } },
      h('div', null, h('b', null, t('parties.detail.official_results', { year: oy }) + t('common.colon')), offLine || '–'),
      h('div', null, h('b', null, t('parties.detail.model_results', { year }) + t('common.colon')), modLine || '–'),
      h('div', null, t('parties.detail.units'))));
}

export function detailCard() {
  const list = partyList();
  if (!list.length) return null;
  const p = partyById(state.ui.party) || null;
  const pick = h('select', { id: 'party-pick', 'aria-label': t('parties.detail.pick') },
    h('option', { value: '' }, t('parties.detail.pick_one')),
    partyGroups().map(g => h('optgroup', { label: g.label }, g.parties.map(x => h('option', { value: x.id, selected: p && x.id === p.id }, x.name)))));
  pick.addEventListener('change', () => { state.ui.party = pick.value || null; emit(); });
  if (!p) {
    return card({ title: t('parties.detail.none_title'), id: 'party-detail', tools: pick,
      body: h('div', { class: 'empty' }, t('parties.detail.none')) });
  }
  const key = `party:${state.meta.params_signature}:${p.id}`;
  const entry = lazy(key, () => api(`/api/parties/${encodeURIComponent(p.id)}/effects`));
  const cov = p.coverage;
  return card({
    title: t('parties.detail.title', { name: p.name }), flush: true, id: 'party-detail',
    sub: t('parties.detail.sub', { full: p.full_name, n: cov.measures, costed: cov.costed }),
    tools: h('div', { class: 'row' }, pick, actions(p).childNodes.length ? [
      h('button', { class: 'btn sm', type: 'button', disabled: !p.scenario.measures.length, onclick: () => loadParty(p) }, t('parties.load_long')),
      h('button', { class: 'btn sm', type: 'button', disabled: !p.scenario.measures.length, onclick: () => keepParty(p) }, t('parties.keep_long'))] : null),
    body: h('div', null,
      h('div', { style: { padding: '0 14px 10px' } }, caveatNotice()),
      h('div', { class: 'row', style: { padding: '0 14px 10px', gap: '12px' } },
        coverageBar(p, '220px'), h('span', { class: 'hint' }, coverageText(cov)),
        h('a', { class: 'hint', href: p.source.fiches_url, target: '_blank', rel: 'noopener' }, t('parties.detail.source')),
        h('a', { class: 'hint', href: p.source.workbook_url, target: '_blank', rel: 'noopener' }, t('parties.detail.workbook'))),
      p.unknown_params && p.unknown_params.length ? h('div', { style: { padding: '0 14px 10px' } }, notice('warn', t('panel.lib.unknown_params', { list: p.unknown_params.join(', ') }))) : null,
      lazyBody(entry, {
        loading: t('parties.detail.loading', { name: p.name }),
        retry: () => { lazyReset('party:'); emit(); },
        ok: fx => detailTable(p, fx),
      })),
  });
}

// ------------------------------------------------------------------ all parties side by side
function orderOf(data) {
  const ids = partyGroups().flatMap(g => g.parties.map(p => p.id));
  return [...data.scenarios].sort((a, b) => ids.indexOf(a.party) - ids.indexOf(b.party));
}

function allTable(data, runs, yi, year, oy) {
  const last = data.years.length - 1, endYear = data.years[last];
  const b = data.baseline;
  const d = (r, key, i, scale = 1) => { const x = at(r.series && r.series[key], i), y = at(b[key], i); return x == null || y == null ? null : (x - y) * scale; };
  const rel = (r, key, i) => { const x = at(r.series && r.series[key], i), y = at(b[key], i); return x == null || !y ? null : (x / y - 1) * 100; };
  const off = (r, f, scale = 1) => (r.official && r.official[f] && r.official[f][oy] != null ? r.official[f][oy] * scale : null);
  const pair = (model, bureau, key, dec) => [deltaSpan(key, model, dec), h('div', { class: 'sub' }, bureau == null ? '–' : `${t('parties.all.bureau')} ${signed(bureau, dec)}`)];
  const rows = runs.map(r => h('tr', null,
    h('td', null, h('button', { class: 'linkbtn', type: 'button', onclick: () => openParty(partyById(r.party) || { id: r.party }) }, h('b', null, r.name)),
      r.error ? h('div', { class: 'sub', style: { color: 'var(--bad)', whiteSpace: 'normal' } }, t('compare.refused', { error: r.error })) : null),
    h('td', null, partyById(r.party) ? coverageBar(partyById(r.party), '90px') : null),
    h('td', { class: 'n' }, pair(d(r, 'fiscal.balance_pct_gdp', yi), off(r, 'balance_pct_gdp'), 'fiscal.balance', 2)),
    h('td', { class: 'n' }, pair(d(r, 'labour.employment', yi, 1 / 1000), off(r, 'employment', 1 / 1000), 'labour.employment', 1)),
    h('td', { class: 'n' }, pair(rel(r, 'macro.gdp_real', yi), off(r, 'gdp_pct'), 'macro.gdp_real', 2)),
    h('td', { class: 'n' }, pair(d(r, 'fiscal.debt_pct_gdp', yi), off(r, 'debt_pct_gdp'), 'fiscal.debt', 2)),
    h('td', { class: 'n' }, deltaSpan('fiscal.debt', d(r, 'fiscal.debt_pct_gdp', last), 2))));
  let group = null;
  const body = [];
  runs.forEach((r, i) => {
    if (r.community !== group) { group = r.community; body.push(h('tr', { class: 'group' }, h('th', { colspan: 7, scope: 'colgroup' }, t(`parties.group.${group}`)))); }
    body.push(rows[i]);
  });
  return h('div', { class: 'tbl-wrap full' }, h('table', { class: 'tbl parties', id: 'parties-all-table' },
    h('thead', null, h('tr', null,
      h('th', null, t('parties.col.party')), h('th', null, t('parties.col.coverage')),
      h('th', { class: 'n' }, t('parties.all.col.balance', { year })), h('th', { class: 'n' }, t('parties.all.col.employment', { year })),
      h('th', { class: 'n' }, t('parties.all.col.gdp', { year })), h('th', { class: 'n' }, t('parties.all.col.debt', { year })),
      h('th', { class: 'n', title: t('compare.head.debt_end_title') }, t('parties.all.col.debt', { year: endYear })))),
    h('tbody', null, body)));
}

function distBlock(year) {
  if (!state.ui.partiesDist) {
    return h('div', null, h('button', { class: 'btn', type: 'button', id: 'parties-dist-btn', onclick: () => { state.ui.partiesDist = true; emit(); } }, t('parties.all.dist_button')));
  }
  const y = Math.min(Math.max(year, 2027), 2040);
  const entry = lazy(`partydist:${state.meta.params_signature}:${y}`, () => api(`/api/parties/distribution?year=${y}`));
  return card({
    title: t('parties.all.dist_title', { year: y }), flush: true, id: 'parties-dist', sub: t('parties.all.dist_sub'),
    tools: h('button', { class: 'linkbtn', type: 'button', onclick: () => { state.ui.partiesDist = false; emit(); } }, t('parties.all.dist_hide')),
    body: lazyBody(entry, {
      loading: t('parties.all.dist_loading', { n: partyList().length }),
      retry: () => { lazyReset('partydist:'); emit(); },
      ok: data => {
        const rows = orderOf({ scenarios: data.parties });
        const cell = v => h('td', { class: 'n' }, v == null ? '–' : deltaSpan('', v, 1));
        return h('div', { class: 'tbl-wrap full' }, h('table', { class: 'tbl dense parties', id: 'parties-dist-table' },
          h('thead', null, h('tr', null, h('th', null, t('parties.col.party')), h('th', null, ''),
            Array.from({ length: 10 }, (_, i) => h('th', { class: 'n' }, t('parties.all.decile', { n: i + 1 }))))),
          h('tbody', null, rows.flatMap(r => [
            h('tr', null, h('td', { rowspan: r.official_deciles_pct ? 2 : 1 }, h('b', null, r.name),
              r.error ? h('div', { class: 'sub', style: { color: 'var(--bad)', whiteSpace: 'normal' } }, r.error) : null),
            h('td', { class: 'sub' }, t('parties.all.model')), Array.from({ length: 10 }, (_, i) => cell(r.deciles_pct ? r.deciles_pct[i] : null))),
            r.official_deciles_pct ? h('tr', { class: 'quiet' }, h('td', { class: 'sub' }, t('parties.all.bureau')),
              r.official_deciles_pct.map(v => h('td', { class: 'n sub' }, signed(v, 1)))) : null,
          ]))));
      },
    }),
  });
}

export function allBlock() {
  if (!partyList().length) return null;
  const entry = lazy(`parties:${state.meta.params_signature}`, () => api('/api/parties/compare'));
  return lazyBody(entry, {
    loading: t('parties.all.loading', { n: partyList().length }),
    retry: () => { lazyReset('parties:'); emit(); },
    ok: data => {
      const runs = orderOf(data);
      const year = Math.max(state.year, 2027), yi = data.years.indexOf(year);
      const oy = Math.min(Math.max(year - 2, 2025), 2029);
      const b = data.baseline;
      const val = (r, key, scale, relative) => {
        const x = at(r.series && r.series[key], yi), y = at(b[key], yi);
        if (x == null || y == null) return null;
        return relative ? (y ? (x / y - 1) * 100 : null) : (x - y) * scale;
      };
      const chart = (id, key, scale, dec, relative) => card({
        title: t(`parties.all.chart.${id}`, { year }), id: 'parties-chart-' + id, sub: t(`parties.all.chart_unit.${id}`),
        body: hbars(runs.map(r => ({ label: r.name, value: val(r, key, scale, relative) })),
          { fmt: v => signed(v, dec), plain: true, valueName: t('parties.all.model') }),
        table: () => ({ head: [t('parties.col.party'), t(`parties.all.chart_unit.${id}`)], rows: runs.map(r => [r.name, num(val(r, key, scale, relative), dec)]) }),
      });
      return h('div', { class: 'stack', id: 'parties-all' },
        card({ title: t('parties.all.title', { year }), flush: true, id: 'parties-all-card', sub: t('parties.all.sub', { oyear: oy, year }),
          body: h('div', null, allTable(data, runs, yi, year, oy),
            h('p', { class: 'hint', style: { padding: '8px 14px 0', maxWidth: '980px' } }, t('parties.all.note'))) }),
        h('div', { class: 'grid g2' },
          chart('balance', 'fiscal.balance_pct_gdp', 1, 2), chart('debt', 'fiscal.debt_pct_gdp', 1, 2),
          chart('employment', 'labour.employment', 1 / 1000, 1), chart('gdp', 'macro.gdp_real', 1, 2, true)),
        distBlock(year));
    },
  });
}

/** The party section of the Compare view: list, and the all-party comparison on request. */
export function compareSection() {
  if (!partyList().length) return null;
  return h('div', { class: 'stack', id: 'cmp-parties' }, listCard(), state.ui.partiesAll ? allBlock() : null);
}

registerPartiesSection(panelSection);
