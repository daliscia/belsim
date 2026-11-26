// Distribution of a scenario over households: deciles, typical households, inequality indicators.
// The endpoint belongs to another module: every field is read defensively.
import { h } from '../dom.js';
import { state, emit, post, lazy, lazyReset, num, signed, pct, signedPct, money, signedMoney, humanise, enabledMeasures, isStale, hasRun, scenarioSig } from '../store.js';
import { card, deltaSpan, sortTable, lazyBody, notice, scenarioName, seg, searchBox } from '../components.js';
import { columnChart } from '../charts.js';
import { exportable } from '../scenario.js';
import { t, tn, tOr, locale } from '../i18n.js';

const isNum = v => typeof v === 'number' && isFinite(v);
const list = v => (Array.isArray(v) ? v : v == null ? [] : typeof v === 'object' ? Object.entries(v).map(([k, x]) => humanise(k) + t('common.colon') + x) : [String(v)]);
const eur = v => (isNum(v) ? money(v, 0) : '–');
/** Cause of the income change (a model code) in the interface language; `lower` for use inside a phrase. */
const causeName = c => tOr('distribution.cause.' + c, humanise(c));
const causeLower = c => causeName(c).toLocaleLowerCase(locale());
const pctOf = v => (isNum(v) ? v * (Math.abs(v) <= 1.5 ? 100 : 1) : null);   // rates may come as fractions or percent

function caveats(meta) {
  meta = meta && typeof meta === 'object' ? meta : {};
  const cav = list(meta.caveats);
  const method = typeof meta.method === 'string' ? meta.method : list(meta.method).join(' ');
  if (!method && !cav.length) return null;
  return notice('warn',
    h('b', null, t('distribution.read_first'), ' '), method,
    cav.length ? h('ul', { class: 'plain', style: { marginTop: '6px' } }, cav.map(c => h('li', null, String(c)))) : null,
    typeof meta.income_concept === 'string' ? h('div', { style: { marginTop: '6px' } }, h('b', null, t('distribution.income_measured'), ' '), meta.income_concept) : null);
}

/**
 * Inequality indicators. The stylised household set compresses inequality, so the headline is the
 * change between baseline and scenario; model levels and the observed statistic sit underneath.
 */
function indicatorTiles(ind) {
  if (!ind || typeof ind !== 'object') return null;
  const tiles = Object.entries(ind).map(([k, v]) => {
    if (!v || typeof v !== 'object') return null;
    const b = v.baseline, sc = v.scenario;
    if (!isNum(b) && !isNum(sc)) return null;
    const rate = /rate|share/.test(k) && Math.max(Math.abs(b || 0), Math.abs(sc || 0)) <= 1.5;
    const mul = rate ? 100 : 1, dec = rate ? 2 : 3;
    const d = isNum(b) && isNum(sc) ? (sc - b) * mul : null;
    const obs = v.observed && isNum(v.observed.value) ? v.observed : null;
    const lvl = x => (rate ? pct(x, dec) : num(x, dec));
    const seen = obs ? lvl(obs.value * mul) : '';
    return h('div', { class: 'tile' },
      h('div', { class: 'tile-label' }, t('distribution.indicator_change', { name: tOr('distribution.indicator.' + k, humanise(k)) })),
      h('div', { class: 'tile-value' }, d == null ? '–' : signed(d, dec), h('span', { class: 'u' }, rate ? t('common.points') : '')),
      h('div', { class: 'tile-delta' }, t('distribution.model_levels', { baseline: lvl(isNum(b) ? b * mul : null), scenario: lvl(isNum(sc) ? sc * mul : null) })),
      obs ? h('div', { class: 'tile-delta' }, (obs.year ? t('distribution.observed_year', { value: seen, year: obs.year }) : t('distribution.observed', { value: seen })) + (obs.source ? ` (${obs.source})` : '')) : null);
  }).filter(Boolean);
  if (!tiles.length) return null;
  return h('div', { class: 'stack', style: { gap: '6px' } },
    h('div', { class: 'tiles' }, tiles),
    h('p', { class: 'hint' }, t('distribution.read_change_hint')));
}

/** At-risk-of-poverty rate by household type and by activity status: observed, baseline, scenario. */
function povertyCard(pov, hasScenario) {
  if (!pov || typeof pov !== 'object') return null;
  const groups = [['by_household_type', 'distribution.poverty.by_household'], ['by_activity_status', 'distribution.poverty.by_activity']]
    .filter(([k]) => Array.isArray(pov[k]) && pov[k].length);
  if (!groups.length) return null;
  const rate = v => (isNum(v) ? pct(v * 100, 1) : '–');
  const body = groups.flatMap(([k, title]) => [
    h('tr', { class: 'group' }, h('th', { colspan: hasScenario ? 6 : 4, scope: 'colgroup', style: { textAlign: 'left' } }, t(title))),
    ...pov[k].map(r => h('tr', null,
      h('td', null, r.label || r.id),
      h('td', { class: 'n' }, isNum(r.population_share) ? pct(r.population_share * 100, 1) : '–'),
      h('td', { class: 'n', title: r.observed && r.observed.source ? `${r.observed.source}${r.observed.year ? ', ' + r.observed.year : ''}` : null }, r.observed ? rate(r.observed.value) : '–'),
      h('td', { class: 'n' }, rate(r.baseline)),
      hasScenario ? h('td', { class: 'n' }, rate(r.scenario)) : null,
      hasScenario ? h('td', { class: 'n' }, isNum(r.delta) ? signed(r.delta * 100, 2) : '–') : null)),
  ]);
  return card({
    title: t('distribution.poverty.title'), flush: true, id: 'dist-poverty', sub: t('distribution.poverty.sub'),
    body: h('div', { class: 'tbl-wrap tall' }, h('table', { class: 'tbl dense' },
      h('thead', null, h('tr', null, h('th', null, ''), h('th', { class: 'n' }, t('distribution.poverty.share')),
        h('th', { class: 'n' }, t('distribution.poverty.observed')), h('th', { class: 'n' }, t('common.baseline')),
        hasScenario ? h('th', { class: 'n' }, t('common.scenario')) : null,
        hasScenario ? h('th', { class: 'n' }, t('distribution.poverty.change')) : null)),
      h('tbody', null, body))),
  });
}

function summaryTiles(sum) {
  if (!sum || typeof sum !== 'object') return null;
  const tile = (label, value, unit, sub) => h('div', { class: 'tile' }, h('div', { class: 'tile-label' }, label),
    h('div', { class: 'tile-value' }, value, unit ? h('span', { class: 'u' }, unit) : null), h('div', { class: 'tile-delta' }, sub || ''));
  const tiles = [
    isNum(sum.gainers_share) ? tile(t('distribution.tile.gaining'), num(pctOf(sum.gainers_share), 0), '%', t('distribution.tile.gaining_sub')) : null,
    isNum(sum.losers_share) ? tile(t('distribution.tile.losing'), num(pctOf(sum.losers_share), 0), '%', t('distribution.tile.losing_sub')) : null,
    isNum(sum.delta_eur_mean) ? tile(t('distribution.tile.avg'), signedMoney(sum.delta_eur_mean, 0), t('distribution.a_year'), isNum(sum.delta_pct) ? t('distribution.tile.avg_sub', { pct: signedPct(sum.delta_pct, 2) }) : '') : null,
    isNum(sum.delta_total_meur) ? tile(t('distribution.tile.all'), t('fmt.eur_bn', { v: signed(sum.delta_total_meur / 1000, 2) }), t('distribution.a_year'), t('distribution.tile.all_sub')) : null,
  ].filter(Boolean);
  return tiles.length ? h('div', { class: 'tiles' }, tiles) : null;
}

function decileCard(deciles, year, causeNotes) {
  const rows = (Array.isArray(deciles) ? deciles : []).filter(d => d && typeof d === 'object');
  if (!rows.length) return card({ title: t('distribution.decile.title'), body: h('div', { class: 'empty' }, t('distribution.decile.none')) });
  const vals = rows.map(d => (isNum(d.delta_pct) ? d.delta_pct : null));
  const causes = [...new Set(rows.flatMap(d => Object.keys(d.by_cause || {})))];
  const share = v => (isNum(v) ? pct(pctOf(v), 0) : '–');
  const flat = vals.every(v => v == null || Math.abs(v) < 1e-9);
  return card({
    title: t('distribution.decile.title_year', { year }), id: 'dist-dec',
    sub: t('distribution.decile.sub'),
    body: flat ? h('div', { class: 'empty' }, hasRun() ? t('distribution.decile.flat_run') : t('distribution.decile.flat_norun'))
      : columnChart({
        cats: rows.map((d, i) => d.decile ?? i + 1), catPrefix: t('distribution.decile.name') + ' ', barWidth: 36, height: 250,
        series: [{ name: t('distribution.decile.series'), values: vals, color: 'var(--scenario)' }],
        fmt: v => signedPct(v, 2), title: t('distribution.decile.title'),
        tip: ci => {
          const d = rows[ci];
          return [
            { color: 'var(--scenario)', value: isNum(d.delta_pct) ? signedPct(d.delta_pct, 2) : '–', name: t('distribution.decile.tip_change') },
            { value: isNum(d.delta_eur) ? signedMoney(d.delta_eur, 0) : '–', name: t('common.per_year') },
            { value: eur(d.baseline_income), name: t('distribution.decile.baseline_income') },
            { value: share(d.gainers_share), name: t('distribution.tile.gaining') },
            { value: share(d.losers_share), name: t('distribution.tile.losing') },
            ...causes.filter(c => isNum((d.by_cause || {})[c])).map(c => ({ value: signedMoney(d.by_cause[c], 0), name: t('distribution.of_which', { cause: causeLower(c) }) })),
          ];
        },
      }),
    table: () => ({
      head: [t('distribution.decile.name'), t('distribution.decile.head.baseline_income_eur'), t('distribution.decile.head.change_eur'), t('distribution.decile.head.change_pct'),
        t('distribution.decile.head.gaining'), t('distribution.decile.head.losing'), ...causes.map(c => t('distribution.of_which_eur', { cause: causeLower(c) }))],
      rows: rows.map((d, i) => [String(d.decile ?? i + 1), num(d.baseline_income, 0), signed(d.delta_eur, 0), signed(d.delta_pct, 2), share(d.gainers_share), share(d.losers_share),
        ...causes.map(c => signed((d.by_cause || {})[c], 0))]),
    }),
  });
}

function householdCard(households, year, causeNotes) {
  const hh = (Array.isArray(households) ? households : []).filter(x => x && typeof x === 'object');
  if (!hh.length) return card({ title: t('distribution.hh.title'), body: h('div', { class: 'empty' }, t('distribution.hh.none')) });
  const causes = [...new Set(hh.flatMap(x => Object.keys(((x.delta || {}).by_cause) || {})))];
  // the filter id is the group in English (`group_id`, added by the server when it translates `group`)
  const gid = x => x.group_id || x.group;
  const groups = [...new Map(hh.filter(x => x.group).map(x => [gid(x), x.group])).entries()];
  const gsel = groups.length > 1 ? seg('distGroup', [{ id: 'all', label: t('common.all') }, ...groups.map(([id, name]) => ({ id, label: humanise(name) }))], 'all') : null;
  const q = (state.ui.distQ || '').trim().toLowerCase();
  const rows = hh.filter(x => (!gsel || gsel.cur === 'all' || gid(x) === gsel.cur) && (!q || `${x.label} ${x.group} ${x.id}`.toLowerCase().includes(q)));
  const g = (x, ...path) => path.reduce((o, k) => (o && typeof o === 'object' ? o[k] : undefined), x);
  const amount = (id, label, f, title) => ({ id, label, num: true, title, value: x => (isNum(f(x)) ? f(x) : null), cell: x => (isNum(f(x)) ? num(f(x), 0) : null) });
  const rv = (x, k) => (isNum(g(x, 'rates', k)) ? g(x, 'rates', k) * 100 : null);     // fractions; marginal and participation may be null
  const rate = (id, label, k, title) => ({ id, label, num: true, title: t('distribution.col.in_scenario_note', { title }), value: x => rv(x, k), cell: x => {
    const v = rv(x, k), b0 = isNum(g(x, 'baseline', 'rates', k)) ? g(x, 'baseline', 'rates', k) * 100 : null;
    return v == null ? null : [pct(v, 1), b0 != null && Math.abs(v - b0) >= 0.05 ? h('span', { class: 'sub' }, ` (${signed(v - b0, 1)})`) : null];
  } });
  const cols = [
    { id: 'label', label: t('distribution.col.household'), cls: 'wrap', value: x => String(x.label || x.id || ''), cell: x => [h('b', null, x.label || x.id), x.group ? h('span', { class: 'tag', style: { marginLeft: '6px' } }, humanise(x.group)) : null] },
    ...(hh.some(x => isNum(x.weight)) ? [{ id: 'weight', label: t('distribution.col.households'), num: true, title: t('distribution.col.households_title'), value: x => (isNum(x.weight) ? x.weight : null), cell: x => (isNum(x.weight) ? num(x.weight, 0) : null) }] : []),
    amount('gross', t('distribution.col.gross'), x => g(x, 'baseline', 'gross')),
    amount('net0', t('distribution.col.net_baseline'), x => g(x, 'baseline', 'net')),
    amount('net1', t('distribution.col.net_scenario'), x => g(x, 'scenario', 'net')),
    { id: 'dnom', label: t('distribution.col.change_before_prices'), num: true, title: t('distribution.col.change_before_prices_title'),
      value: x => (isNum(g(x, 'delta', 'nominal_eur')) ? g(x, 'delta', 'nominal_eur') : null), cell: x => (isNum(g(x, 'delta', 'nominal_eur')) ? signed(g(x, 'delta', 'nominal_eur'), 0) : null) },
    { id: 'deur', label: t('distribution.col.change_eur'), num: true, title: t('distribution.col.change_eur_title'), value: x => (isNum(g(x, 'delta', 'eur')) ? g(x, 'delta', 'eur') : null), cell: x => (isNum(g(x, 'delta', 'eur')) ? deltaSpan('', g(x, 'delta', 'eur'), 0) : null) },
    { id: 'dpct', label: t('distribution.col.change_pct'), num: true, value: x => (isNum(g(x, 'delta', 'pct')) ? g(x, 'delta', 'pct') : null), cell: x => (isNum(g(x, 'delta', 'pct')) ? signedPct(g(x, 'delta', 'pct'), 2) : null) },
    ...causes.map(c => ({ id: 'c:' + c, label: t('distribution.of_which_eur', { cause: causeLower(c) }), num: true, title: (causeNotes && typeof causeNotes[c] === 'string' && causeNotes[c]) || t('distribution.due_to', { cause: causeLower(c) }),
      value: x => (isNum(g(x, 'delta', 'by_cause', c)) ? g(x, 'delta', 'by_cause', c) : null), cell: x => (isNum(g(x, 'delta', 'by_cause', c)) ? signed(g(x, 'delta', 'by_cause', c), 0) : null) })),
    rate('avg', t('distribution.col.avg'), 'average', t('distribution.col.avg_title')),
    rate('marg', t('distribution.col.marg'), 'marginal', t('distribution.col.marg_title')),
    rate('part', t('distribution.col.part'), 'participation', t('distribution.col.part_title')),
  ].filter(c => c.id === 'label' || rows.some(x => c.value(x) != null) || hh.some(x => c.value(x) != null));
  return card({
    title: t('distribution.hh.title_year', { year }), flush: true, id: 'dist-hh',
    sub: tn('distribution.hh.sub', hh.length, { shown: num(rows.length, 0) }),
    tools: h('div', { class: 'row' }, gsel ? gsel.el : null, searchBox('distQ', t('distribution.hh.filter'), '170px')),
    body: sortTable(cols, rows, { id: 'dist-hh', sort: null, cls: 'tbl dense' }),
  });
}

export function render(root) {
  const scen = exportable();
  const stale = isStale() && enabledMeasures().length;
  const key = `dist:${state.meta.params_signature}:${scenarioSig()}:${state.year}`;
  const entry = lazy(key, () => post('/api/distribution', { scenario: scen, year: state.year }));
  root.append(h('div', { class: 'stack' },
    h('p', { class: 'hint' }, scen.measures.length ? t(stale ? 'distribution.hint_scenario_stale' : 'distribution.hint_scenario', { name: scenarioName(), year: state.year })
      : t('distribution.hint_baseline', { year: state.year })),
    lazyBody(entry, {
      loading: t('distribution.loading'),
      missing: t('distribution.missing'),
      retry: () => { lazyReset('dist:'); emit(); },
      ok: d => {
        if (!d || typeof d !== 'object') throw new Error(t('distribution.unexpected', { answer: JSON.stringify(d).slice(0, 200) }));
        return h('div', { class: 'stack' },
          caveats(d.meta),
          scen.measures.length ? summaryTiles(d.summary) : null,
          decileCard(d.deciles, d.year ?? state.year, (d.meta || {}).causes),
          indicatorTiles(d.indicators),
          povertyCard(d.poverty, scen.measures.length > 0),
          householdCard(d.households, d.year ?? state.year, (d.meta || {}).causes));
      },
    })));
}
