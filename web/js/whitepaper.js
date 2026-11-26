// The methodology page. The prose lives in web/wp/<lang>.html (one fragment per language, same
// structure); every figure that comes from the model is a placeholder filled here from the API,
// so the page follows the parameters of the running build:
//
//   <span class="live" data-v="p.count"></span>                     a named value (see VALUES)
//   <span class="live" data-v="b:fiscal.debt_pct_gdp@2031:pct1">    baseline series at a year
//   <span class="live" data-v="d:align_off:KEY@2031:abs1">          scenario minus baseline
//   <span class="live" data-v="s:align_off:KEY@2031:abs1">          scenario value
//   <div data-block="val-table"></div>                              a table or figure built here
//   <span data-optional> … </span>                                  dropped when a figure inside is missing
import { h, $ } from './dom.js';
import { initI18n, setLang, lang, locale, t, LANGS } from './i18n.js';

// Scenarios that put a number on two baseline conventions (one batch request).
const SCENARIOS = {
  align_off: [{ param: 'rev.fpb_alignment_enabled', value: 0, start: 2025 }],
  price_link: [{ param: 'core.tax_threshold_wage_link', value: 0, start: 2025 }],
};

// A selection of references for the table of the Validation section, by theme. `when` picks a
// baseline row; `n` the n-th target of a costed measure (0-based, in the order of the reference
// file), 'last' the last one. The verdict shown is computed from the live gap, never typed.
const SELECTION = [
  { sec: 'near', rows: [
    { id: 'fpb2026_gdp_nominal_index', when: '2031' }, { id: 'fpb2026_employment', when: '2031' },
    { id: 'fpb2026_revenue', when: '2031' }, { id: 'fpb2026_expenditure', when: '2031' },
    { id: 'fpb2026_balance', when: '2031' }, { id: 'fpb2026_debt', when: '2031' },
    { id: 'fpb2026_balance_regions', when: '2031' }, { id: 'nbb2026_debt', when: '2028' },
  ] },
  { sec: 'long', rows: [
    { id: 'sca2026_pensions', when: '2070' }, { id: 'sca2026_pensions_change', when: '2050-2070' },
    { id: 'sca2026_health_total', when: '2070' }, { id: 'sca2026_employment', when: '2070' },
    { id: 'pop2026_67_plus', n: 'last' },
  ] },
  { sec: 'static', rows: [
    { id: 'arizona_allowance_15600', n: 'last' }, { id: 'dc2024_vat_food_zero', n: 0 },
    { id: 'vat_standard_22_press_2026', n: 0 }, { id: 'dc2024_pit_45_to_40', n: 0 },
    { id: 'cit_reform_2018_rate_cut', n: 0 }, { id: 'dc2024_abolish_special_ssc', n: 0 },
    { id: 'arizona_vat_hotels_12', n: 0 }, { id: 'dc2024_mr_802', n: 0 },
  ] },
  { sec: 'dynamic', rows: [
    { id: 'hermes2018_employer_ssc_cut_general', n: 4 }, { id: 'hermes2018_employer_ssc_cut_general', n: 5 },
    { id: 'hermes2018_employer_ssc_cut_low_wage', n: 5 }, { id: 'hermes_vat_increase', n: 3 }, { id: 'hermes_vat_increase', n: 'last' },
    { id: 'hermes_pit_increase', n: 2 }, { id: 'hermes_employee_ssc_increase', n: 2 }, { id: 'hermes_employee_ssc_increase', n: 'last' },
    { id: 'index_jump_2015_fpb', n: 1 }, { id: 'index_jump_2015_nbb', n: 0 },
  ] },
  { sec: 'measures', rows: [
    { id: 'unemployment_time_limit', n: 1 }, { id: 'dc2024_health_norm_1_5', n: 'last' },
    { id: 'pension_bonus_malus_2026', n: 1 }, { id: 'dc2024_pension_age_65', n: 'last' }, { id: 'dc2024_vb_301', n: 0 },
  ] },
];
const BUCKETS = [5, 10, 25, 50, Infinity];
const BUCKET_KEYS = ['validation.bucket.b0', 'validation.bucket.b1', 'validation.bucket.b2', 'validation.bucket.b3', 'validation.bucket.b4'];
const BASE_ROWS = [
  ['macro.gdp_real_growth', 'overview.growth', 'rate'], ['labour.employment_rate_20_64', 'overview.employment_rate', 'rate'],
  ['fiscal.revenue_pct_gdp', 'overview.revenue', 'pct'], ['fiscal.expenditure_pct_gdp', 'overview.expenditure', 'pct'],
  ['fiscal.balance_pct_gdp', 'overview.balance', 'pct'], ['fiscal.debt_pct_gdp', 'overview.debt', 'pct'],
];
const BASE_YEARS = [2024, 2025, 2031, 2040, 2050, 2070];
// The income survey of year N records the incomes of N-1: the 2025 survey is set against the model's 2024.
const DIST_YEAR = 2024;

const data = { version: null, meta: null, base: null, batch: null, parties: null, dist: null, validation: {}, failed: new Set() };
let fragmentFor = null;

// ------------------------------------------------------------------ formats
const nf = (dec = 0) => new Intl.NumberFormat(locale(), { minimumFractionDigits: dec, maximumFractionDigits: dec });
const int = v => nf(0).format(v);
const fixed = (v, dec) => nf(dec).format(v).replace('-', '−');
const pctText = (v, dec = 1) => fixed(v, dec) + (lang() === 'fr' ? ' %' : '%');
function dateText(iso) {
  const d = iso ? new Date(iso) : null;
  return d && !isNaN(d) ? new Intl.DateTimeFormat(locale(), { day: 'numeric', month: 'long', year: 'numeric' }).format(d) : null;
}
const FORMATS = {
  int: v => int(v), pct1: v => pctText(v, 1), abs1: v => fixed(Math.abs(v), 1), abs2: v => fixed(Math.abs(v), 2),
  d1: v => fixed(v, 1), d2: v => fixed(v, 2), sd2: v => (v > 0 ? '+' : '') + fixed(v, 2), abspct1: v => pctText(Math.abs(v), 1),
  rate1: v => pctText(v * 100, 1), bn: v => int(v / 1000), k: v => int(v / 1000),
};

// ------------------------------------------------------------------ data
async function getJson(path, opts) {
  const resp = await fetch(path + (path.includes('?') ? '&' : '?') + 'lang=' + lang(), opts);
  const body = await resp.json().catch(() => null);
  if (!resp.ok) throw new Error((body && body.error) || `${resp.status}`);
  return body;
}
const optional = (name, promise) => promise.catch(() => { data.failed.add(name); return null; });

async function loadCore() {
  const batchBody = { scenarios: Object.entries(SCENARIOS).map(([name, changes]) => ({ name, measures: [{ name, changes }] })) };
  const post = body => ({ method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  [data.version, data.meta, data.base, data.batch, data.parties, data.dist] = await Promise.all([
    optional('version', getJson('/api/version')),
    optional('meta', getJson('/api/meta')),
    optional('baseline', getJson('/api/baseline')),
    optional('scenarios', getJson('/api/run_batch', post(batchBody))),
    optional('parties', getJson('/api/scenarios?kind=party')),
    getJson('/api/distribution', post({ scenario: { name: 'baseline', measures: [] }, year: DIST_YEAR })).catch(() => null),
  ]);
}

/** The validation report in the page language; null while the server computes it. */
async function loadValidation() {
  const id = lang();
  if (data.validation[id]) return data.validation[id];
  const body = await getJson('/api/validation?wait=0');
  if (body && body.status === 'computing') return null;
  data.validation[id] = body;
  return body;
}

// ------------------------------------------------------------------ named values
const at = (series, years, key, year) => {
  const i = years ? years.indexOf(year) : -1;
  const v = series && series[key] && i >= 0 ? series[key][i] : null;
  return v == null ? null : v;
};
function partyCounts() {
  const p = data.parties;
  if (!p || !Array.isArray(p.scenarios)) return null;
  const out = { n: p.scenarios.length, measures: 0, modelled: 0, partly_modelled: 0, not_modelled: 0, already_in_baseline: 0, sized_measures: 0, sized: new Set(), all: new Set() };
  for (const sc of p.scenarios) {
    for (const m of sc.measures || []) {
      out.measures += 1;
      if (m.status in out) out[m.status] += 1;
      if (m.method === 'sized') out.sized_measures += 1;
      if (m.validation_id) { out.all.add(m.validation_id); if (m.method === 'sized') out.sized.add(m.validation_id); }
    }
  }
  return out;
}
/** Rows of the report in three groups: baseline, independent costings, fiches sized by construction. */
function validationGroups(v) {
  const pc = partyCounts();
  const sized = pc ? pc.sized : new Set();
  return { base: v.baseline, ind: v.measures.filter(r => !sized.has(r.id)), sized: v.measures.filter(r => sized.has(r.id)) };
}
function share(rows) {
  const cmp = rows.filter(r => r.gap_pct != null);
  const counts = BUCKETS.map((b, i) => cmp.filter(r => Math.abs(r.gap_pct) <= b && (i === 0 || Math.abs(r.gap_pct) > BUCKETS[i - 1])).length);
  const within = m => (cmp.length ? 100 * cmp.filter(r => Math.abs(r.gap_pct) <= m).length / cmp.length : null);
  return { n: rows.length, compared: cmp.length, counts, w10: within(10), w25: within(25), w50: within(50) };
}

function value(key) {
  const v = data.version, m = data.meta, pc = partyCounts();
  const series = (kind, rest) => {
    const [, scen, k, year, fmt] = /^(?:(\w+):)?([\w.]+)@(\d{4})(?::(\w+))?$/.exec(rest) || [];
    if (!k) return null;
    const y = +year, f = FORMATS[fmt] || FORMATS.d1;
    const b = data.base ? at(data.base.series, data.base.years, k, y) : null;
    if (kind === 'b') return b == null ? null : f(b);
    const run = data.batch && (data.batch.scenarios || []).find(s => s.name === scen && s.series);
    const sv = run ? at(run.series, data.batch.years, k, y) : null;
    const bv = run ? at(data.batch.baseline, data.batch.years, k, y) : null;
    if (sv == null || bv == null) return null;
    return f(kind === 'd' ? sv - bv : sv);
  };
  if (/^[bds]:/.test(key)) return series(key[0], key.slice(2));
  const val = data.validation[lang()];
  const groups = val ? validationGroups(val) : null;
  const dist = data.dist && data.dist.meta ? data.dist.meta : null;
  const table = {
    'v.commit': () => v && v.commit_short,
    'v.date': () => v && dateText(v.commit_date),
    'v.commits': () => v && v.commits != null && int(v.commits),
    'v.tests': () => v && (v.tests != null ? int(v.tests) : null),
    'v.modules': () => v && Array.isArray(v.modules) && v.modules.join(', '),
    'p.count': () => v && v.params != null && int(v.params),
    'p.assumptions': () => v && v.assumptions != null && int(v.assumptions),
    'p.share': () => v && v.params ? pctText(100 * v.assumptions / v.params, 0) : null,
    'm.base': () => m && String(m.base_year),
    'm.end': () => (v && v.end_year ? String(v.end_year) : m && String(m.end_year)),
    'm.rev_lines': () => m && int(Object.keys(m.revenue_lines || {}).length),
    'm.exp_lines': () => m && int(Object.keys(m.expenditure_lines || {}).length),
    'm.series': () => data.base && int(Object.keys(data.base.series).length),
    'party.n': () => pc && int(pc.n),
    'party.measures': () => pc && int(pc.measures),
    'party.modelled': () => pc && int(pc.modelled),
    'party.partly': () => pc && int(pc.partly_modelled),
    'party.not': () => pc && int(pc.not_modelled),
    'party.baseline': () => pc && int(pc.already_in_baseline),
    'party.sized': () => pc && int(pc.sized_measures),
    'val.base.n': () => groups && int(groups.base.length),
    'val.ind.n': () => groups && int(groups.ind.length),
    'val.sized.n': () => groups && int(groups.sized.length),
    'dist.households': () => data.dist && Array.isArray(data.dist.households_all) && data.dist.households_all.length ? int(data.dist.households_all.length) : null,
    'dist.shown': () => dist && Number.isFinite(dist.households_shown) && int(dist.households_shown),
    'dist.pit': () => dist && dist.coverage && Number.isFinite(dist.coverage.income_tax) && pctText(100 * dist.coverage.income_tax, 0),
  };
  const fn = table[key];
  const out = fn ? fn() : null;
  return out === false || out === undefined || out === '' ? null : out;
}
/** Values that arrive later than the rest: shown as "computing" rather than "not available". */
const isPending = key => !data.loaded || (/^val\./.test(key) && !data.validation[lang()] && !data.failed.has('validation'))
  || (key === 'v.tests' && data.version && data.version.tests == null && !data.testsGivenUp);

function fillValues(root) {
  for (const el of root.querySelectorAll('[data-v]')) {
    const key = el.dataset.v, v = value(key);
    el.classList.remove('pending', 'missing');
    el.classList.toggle('list', key === 'v.modules');
    if (v != null) { el.textContent = v; el.removeAttribute('title'); continue; }
    if (isPending(key)) { el.textContent = '…'; el.classList.add('pending'); el.title = t('wp.pending'); continue; }
    el.textContent = t('wp.na'); el.classList.add('missing');
  }
  // an optional aside that depends on a figure the server did not give
  for (const el of root.querySelectorAll('[data-optional]')) el.hidden = !!el.querySelector('.live.missing, .live.pending');
}

// ------------------------------------------------------------------ blocks
const table = (cls, caption, head, rows) => h('div', { class: 'tbl-wrap' }, h('table', { class: cls },
  caption ? h('caption', null, caption) : null, h('thead', null, h('tr', null, head)), h('tbody', null, rows)));

const BLOCKS = {
  build() {
    const v = data.version;
    if (!v) return data.loaded ? h('p', { class: 'note' }, t('wp.build.unknown')) : null;
    const cell = (label, val, small) => h('div', null, h('dt', null, label), h('dd', null, val == null ? '–' : val, small ? h('small', null, small) : null));
    return h('dl', { class: 'build' },
      cell(t('wp.build.commit'), v.commit_short, v.dirty ? t('wp.build.dirty') : null),
      cell(t('wp.build.date'), dateText(v.commit_date)),
      cell(t('wp.build.params'), v.params != null ? int(v.params) : null),
      cell(t('wp.build.assumptions'), v.assumptions != null ? int(v.assumptions) : null),
      cell(t('wp.build.tests'), v.tests != null ? int(v.tests) : (data.testsGivenUp ? null : '…')));
  },
  'params-table'() {
    const v = data.version;
    if (!v || !Array.isArray(v.param_files) || !v.param_files.length) return null;
    const row = (name, p, a, cls) => h('tr', { class: cls || null }, h('td', null, name), h('td', { class: 'n' }, int(p)), h('td', { class: 'n' }, int(a)),
      h('td', { class: 'n' }, p ? pctText(100 * a / p, 0) : '–'));
    return table('', t('wp.params.caption'),
      [h('th', null, t('wp.params.file')), h('th', { class: 'n' }, t('wp.params.count')), h('th', { class: 'n' }, t('wp.params.assumptions')), h('th', { class: 'n' }, t('wp.params.share'))],
      [...v.param_files.map(f => row(h('code', null, 'params/' + f.file), f.params, f.assumptions)), row(t('common.total'), v.params, v.assumptions, 'total')]);
  },
  'baseline-table'() {
    const b = data.base;
    if (!b) return null;
    const years = BASE_YEARS.filter(y => b.years.includes(y));
    const cell = (key, kind, y) => { const v = at(b.series, b.years, key, y); return v == null ? '–' : fixed(kind === 'rate' ? v * 100 : v, 1); };
    return table('', t('wp.base.caption'),
      [h('th', null, ''), years.map(y => h('th', { class: 'n' }, y))],
      BASE_ROWS.filter(([key]) => b.series[key]).map(([key, label, kind]) => h('tr', null,
        h('td', null, t(label), h('span', { class: 'sub' }, kind === 'rate' ? '%' : t('unit.pct_gdp'))), years.map(y => h('td', { class: 'n' }, cell(key, kind, y))))));
  },
  'val-summary'() {
    const v = data.validation[lang()];
    if (!v) return validationPlaceholder();
    const g = validationGroups(v);
    const group = (title, rows, note) => {
      const s = share(rows);
      const total = Math.max(1, s.counts.reduce((a, b) => a + b, 0));
      return h('div', null, h('h4', null, title),
        h('p', null, t('wp.val.line', { n: int(s.n), compared: int(s.compared), w10: pctText(s.w10 ?? 0, 0), w25: pctText(s.w25 ?? 0, 0), w50: pctText(s.w50 ?? 0, 0) }), note ? ' ' + note : ''),
        h('div', { class: 'valbar', role: 'img', 'aria-label': s.counts.map((c, i) => `${int(c)} ${t(BUCKET_KEYS[i])}`).join(', ') },
          s.counts.map((c, i) => (c ? h('span', { class: 'q' + i, style: { flexGrow: c / total }, title: `${int(c)} ${t(BUCKET_KEYS[i])}` }, int(c)) : null))));
    };
    return h('div', { class: 'valsum' },
      group(t('wp.val.group.base'), g.base), group(t('wp.val.group.ind'), g.ind),
      h('div', { class: 'legend' }, BUCKETS.map((b, i) => h('span', null, h('i', { class: 'q' + i }), t(BUCKET_KEYS[i])))));
  },
  'val-table'() {
    const v = data.validation[lang()];
    if (!v) return null;
    const pick = sel => {
      let rows = v.baseline.filter(r => r.id === sel.id);
      if (!rows.length) rows = v.measures.filter(r => r.id === sel.id);
      if (!rows.length) return null;
      if (sel.when != null) return rows.find(r => String(r.when) === sel.when) || null;
      return sel.n === 'last' ? rows[rows.length - 1] : rows[sel.n] || null;
    };
    const num = (x, ref) => (x == null ? '–' : fixed(x, Math.abs(ref ?? x) >= 1000 ? 0 : Math.abs(ref ?? x) >= 100 ? 1 : 2));
    const body = [];
    for (const sec of SELECTION) {
      const rows = sec.rows.map(pick).filter(r => r && r.gap != null);
      if (!rows.length) continue;
      body.push(h('tr', { class: 'sec' }, h('th', { colspan: 5, scope: 'colgroup' }, t('wp.val.sec.' + sec.sec))));
      for (const r of rows) {
        const g = r.gap_pct, i = g == null ? -1 : BUCKETS.findIndex(b => Math.abs(g) <= b);
        const model = r.model;
        body.push(h('tr', null,
          h('td', null, r.description || r.id, h('span', { class: 'sub' }, [r.label || r.when, r.unit].filter(Boolean).join(', '))),
          h('td', { class: 'n' }, num(r.ref, r.ref)), h('td', { class: 'n' }, num(model, r.ref)),
          h('td', { class: 'n' }, g == null ? '–' : (g > 0 ? '+' : '') + pctText(g, Math.abs(g) >= 10 ? 0 : 1)),
          h('td', { class: 'v' }, i < 0 ? '' : h('span', { class: 'verdict q' + i }, t(BUCKET_KEYS[i])))));
      }
    }
    if (!body.length) return null;
    return table('', t('wp.val.caption'),
      [h('th', null, t('wp.val.col.what')), h('th', { class: 'n' }, t('wp.val.col.official')), h('th', { class: 'n' }, t('wp.val.col.model')), h('th', { class: 'n' }, t('wp.val.col.gap')), h('th', null, '')],
      body);
  },
  'dist-table'() {
    const d = data.dist;
    const ind = d && d.indicators;
    if (!ind) return null;
    const obs = x => (x && x.observed && Number.isFinite(x.observed.value) ? x.observed : null);
    const years = new Set();
    const row = (label, x, fmt, cls) => {
      const o = obs(x);
      if (!x || !Number.isFinite(x.baseline) || !o) return null;
      years.add(o.year);
      return h('tr', { class: cls || null }, h('td', null, label), h('td', { class: 'n' }, fmt(x.baseline)), h('td', { class: 'n' }, fmt(o.value)));
    };
    const rate = v => pctText(100 * v, 1);
    const rows = [row(t('wp.dist.gini'), ind.gini, v => fixed(v, 3)), row(t('wp.dist.s80'), ind.s80s20, v => fixed(v, 2)), row(t('wp.dist.poverty'), ind.poverty_rate, rate)].filter(Boolean);
    const types = ((d.poverty || {}).by_household_type || []).map(r => row(r.label || r.id, r, rate)).filter(Boolean);
    if (types.length) rows.push(h('tr', { class: 'sec' }, h('th', { colspan: 3, scope: 'colgroup' }, t('wp.dist.by_type'))), ...types);
    if (!rows.length) return null;
    return table('', t('wp.dist.caption', { year: d.year }),
      [h('th', null, ''), h('th', { class: 'n' }, t('wp.val.col.model')), h('th', { class: 'n' }, t('wp.dist.observed', { year: [...years].sort().join(', ') }))], rows);
  },
  sources() {
    const v = data.validation[lang()];
    if (!v) return validationPlaceholder();
    const seen = new Map();
    for (const r of [...v.baseline, ...v.measures]) {
      const s = r.source || {};
      if (!s.title && !s.url) continue;
      // one row per document: the fiches of one party share a file, and so a link
      const key = /^https?:/.test(s.url || '') ? `${s.institution}|${s.url}` : `${s.institution}|${s.title}`;
      if (!seen.has(key)) seen.set(key, { ...s, ids: new Set(), titles: new Set() });
      seen.get(key).ids.add(r.id);
      if (s.title) seen.get(key).titles.add(s.title);
    }
    for (const doc of seen.values()) {
      const titles = [...doc.titles];
      if (titles.length < 2) continue;
      let n = 0;
      while (n < titles[0].length && titles.every(x => x[n] === titles[0][n])) n += 1;
      const common = titles[0].slice(0, n).replace(/[\s\-–,(:]+$/, '');
      doc.title = common.length >= 6 ? common : titles[0];
    }
    const list = [...seen.values()].sort((a, b) => (a.institution || '').localeCompare(b.institution || '', locale()) || (b.date || '').localeCompare(a.date || ''));
    const safe = u => /^https?:\/\//.test(u || '');
    return table('', t('wp.sources.caption', { n: int(list.length) }),
      [h('th', null, t('wp.sources.institution')), h('th', null, t('wp.sources.document')), h('th', null, t('wp.sources.date')), h('th', { class: 'n' }, t('wp.sources.refs'))],
      list.map(s => h('tr', null, h('td', null, s.institution || '–'),
        h('td', null, safe(s.url) ? h('a', { href: s.url, rel: 'noopener noreferrer' }, s.title || s.url) : (s.title || '–'), safe(s.url) ? h('span', { class: 'url' }, s.url) : null),
        h('td', { class: 'date' }, s.date || ''), h('td', { class: 'n' }, int(s.ids.size)))));
  },
};
function validationPlaceholder() {
  if (data.failed.has('validation')) return h('p', { class: 'computing' }, t('wp.val.failed'));
  return h('p', { class: 'computing', role: 'status' }, t('wp.val.computing'));
}
function fillBlocks(root) {
  for (const el of root.querySelectorAll('[data-block]')) {
    const fn = BLOCKS[el.dataset.block];
    let node = null;
    try { node = fn ? fn() : null; } catch (e) { console.error(e); }
    el.replaceChildren(...(node ? [node] : []));
  }
}

// ------------------------------------------------------------------ page
function buildToc(article) {
  const secs = [...article.querySelectorAll('section[id]')];
  const items = () => secs.map(s => h('li', null, h('a', { href: '#' + s.id }, s.querySelector('h2').textContent)));
  $('#toc-list').replaceChildren(...items());
  const head = article.querySelector('.wp-head');
  if (head) head.after(h('nav', { class: 'toc-print', 'aria-hidden': 'true' }, h('h2', null, t('wp.contents')), h('ol', null, items())));
  if ('IntersectionObserver' in window) {
    if (buildToc.io) buildToc.io.disconnect();
    const links = new Map([...$('#toc-list').querySelectorAll('a')].map(a => [a.getAttribute('href').slice(1), a]));
    buildToc.io = new IntersectionObserver(entries => {
      for (const e of entries) {
        if (!e.isIntersecting) continue;
        for (const a of links.values()) a.removeAttribute('aria-current');
        const a = links.get(e.target.id);
        if (a) a.setAttribute('aria-current', 'true');
      }
    }, { rootMargin: '-15% 0px -70% 0px' });
    secs.forEach(s => buildToc.io.observe(s));
  }
}

function renderChrome() {
  document.title = t('wp.doc_title');
  const desc = document.querySelector('meta[name="description"]');
  if (desc) desc.content = t('wp.meta_description');
  $('#skip').textContent = t('common.skip');
  $('#top-sub').textContent = t('nav.methodology');
  $('#toc-title').textContent = t('wp.contents');
  $('#app-link').textContent = t('wp.open_app');
  $('#print-btn').textContent = t('wp.print');
  const dark = document.documentElement.dataset.theme === 'dark';
  $('#theme-btn').textContent = dark ? t('bar.light_theme') : t('bar.dark_theme');
  const seg = $('#lang-seg');
  seg.setAttribute('aria-label', t('bar.lang_aria'));
  seg.replaceChildren(...LANGS.map(l => h('button', { type: 'button', 'aria-pressed': String(lang() === l.id), title: l.name, lang: l.id, onclick: () => changeLang(l.id) }, l.label)));
  $('#paper-foot').replaceChildren(h('span', null, t('foot.disclaimer')), h('a', { href: './' }, t('wp.open_app')));
  const failed = [...data.failed].filter(x => x !== 'validation');
  const notice = $('#notice');
  notice.hidden = !failed.length;
  notice.className = failed.length ? 'err' : '';
  notice.textContent = failed.length ? t('wp.live_failed') : '';
}

function fill() {
  const article = $('#article');
  fillBlocks(article);
  fillValues(article);
}

async function loadFragment() {
  const id = lang();
  const resp = await fetch(`wp/${id}.html`, { cache: 'no-cache' });
  if (!resp.ok) throw new Error(`wp/${id}.html: ${resp.status}`);
  // our own static file, parsed inert: no script in it runs, and none is expected
  const doc = new DOMParser().parseFromString(await resp.text(), 'text/html');
  doc.querySelectorAll('script, style, link, iframe, object').forEach(el => el.remove());
  const article = $('#article');
  article.replaceChildren(...[...doc.body.childNodes].map(n => document.importNode(n, true)));
  fragmentFor = id;
  buildToc(article);
  fill();
  if (location.hash.length > 1) { const el = document.getElementById(location.hash.slice(1)); if (el) el.scrollIntoView(); }
}

let polling = null;
async function pollValidation() {
  clearTimeout(polling);
  try {
    const v = await loadValidation();
    if (!v) { polling = setTimeout(pollValidation, 5000); return; }
    data.failed.delete('validation');
  } catch (e) {
    data.failed.add('validation');
  }
  if (fragmentFor === lang()) fill();
}
async function pollTests(tries = 0) {
  if (!data.version || data.version.tests != null) return;
  if (tries >= 6) { data.testsGivenUp = true; fill(); return; }
  await new Promise(r => setTimeout(r, 2500));
  const v = await getJson('/api/version').catch(() => null);
  if (v) data.version = v;
  if (data.version.tests != null) fill(); else pollTests(tries + 1);
}

async function changeLang(id) {
  if (id === lang()) return;
  await setLang(id);
  const url = new URL(location.href);
  url.searchParams.set('lang', id);
  history.replaceState(null, '', url);
  renderChrome();
  await loadFragment().catch(e => console.error(e));
  // texts the server translates (validation rows) are fetched again in the new language
  await optional('meta', getJson('/api/meta')).then(m => { if (m) data.meta = m; });
  fill();
  pollValidation();
}

async function start() {
  await initI18n();
  const q = new URLSearchParams(location.search).get('lang');
  if (q && q !== lang() && LANGS.some(l => l.id === q)) await setLang(q);
  renderChrome();
  $('#theme-btn').addEventListener('click', () => {
    const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem('belsim.theme', next); } catch (e) { /* ignore */ }
    renderChrome();
  });
  $('#print-btn').addEventListener('click', () => window.print());
  $('#skip').addEventListener('click', e => { e.preventDefault(); $('#paper').focus(); });
  // the whole contents list is open when printing, whatever its state on a narrow screen
  const narrow = matchMedia('(max-width: 900px)');
  const setToc = () => { $('#toc-box').open = !narrow.matches; };
  setToc(); narrow.addEventListener('change', setToc);
  $('#toc-list').addEventListener('click', () => { if (narrow.matches) $('#toc-box').open = false; });

  await loadFragment().catch(e => {
    $('#article').replaceChildren(h('p', null, String(e.message)));
  });
  await loadCore();
  data.loaded = true;
  renderChrome();
  fill();
  pollValidation();
  pollTests();
}
start();
