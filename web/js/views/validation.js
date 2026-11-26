// Model against official reference figures (belsim.validation): how far to trust a result.
import { h, toast } from '../dom.js';
import { state, emit, apiWhenReady, lazy, lazyReset, num, signed, pct, signedPct, fmtDate, compareText } from '../store.js';
import { card, tabs, seg, lazyBody, notice, searchBox } from '../components.js';
import { addMeasure } from '../scenario.js';
import { t, tn, tOr } from '../i18n.js';

// `label` is a catalog key: read with t() when the view renders
const BUCKETS = [
  { max: 5, label: 'validation.bucket.b0' }, { max: 10, label: 'validation.bucket.b1' }, { max: 25, label: 'validation.bucket.b2' },
  { max: 50, label: 'validation.bucket.b3' }, { max: Infinity, label: 'validation.bucket.b4' },
];
const bucketOf = r => (r.gap_pct == null ? -1 : BUCKETS.findIndex(b => Math.abs(r.gap_pct) <= b.max));
const dec = v => (v == null ? 2 : Math.abs(v) >= 1000 ? 0 : Math.abs(v) >= 100 ? 1 : 2);
const fmt = v => num(v, dec(v));

function sourceLink(src) {
  if (!src) return null;
  const text = [src.institution, src.title, src.date].filter(Boolean).join(', ') || src.url || t('validation.source');
  // only http(s) links are followed; anything else is shown as text
  return /^https?:\/\//.test(src.url || '')
    ? h('a', { href: src.url, target: '_blank', rel: 'noopener noreferrer', class: 'src' }, text)
    : h('span', { class: 'src' }, text);
}
const confName = c => tOr('validation.confidence.' + c, c);
const conf = c => (c ? h('span', { class: `conf ${c}`, title: t('validation.confidence_title', { level: confName(c) }) }, confName(c)) : null);

/** Gap in % of the reference as a bar from the centre, clamped at ±100%. */
function gapBar(pct) {
  if (pct == null) return null;
  const w = Math.min(100, Math.abs(pct)) / 2;
  return h('span', { class: 'divbar', 'aria-hidden': 'true' },
    h('span', { class: pct >= 0 ? 'pos' : 'neg', style: { width: w + '%', [pct >= 0 ? 'left' : 'right']: '50%' } }));
}

/** `what`: 'baseline' or 'costing' (the kind of figures counted). */
function summary(rows, what) {
  const compared = rows.filter(r => r.gap != null);
  const counts = BUCKETS.map((b, i) => compared.filter(r => bucketOf(r) === i).length);
  const noPct = compared.filter(r => r.gap_pct == null).length;
  const not = rows.length - compared.length;
  const abs = compared.filter(r => r.gap_pct != null).map(r => Math.abs(r.gap_pct)).sort((a, b) => a - b);
  const median = abs.length ? abs[Math.floor(abs.length / 2)] : null;
  const total = Math.max(1, counts.reduce((a, b) => a + b, 0));
  return h('div', { class: 'valsum' },
    h('div', { class: 'valsum-text' },
      h('b', null, tn(what === 'costing' ? 'validation.sum.costing_compared' : 'validation.sum.baseline_compared', compared.length)),
      median == null ? '' : t('validation.sum.median', { gap: pct(median, 1) }),
      not ? '. ' + tn('validation.sum.no_equivalent', not) : '', noPct ? '. ' + tn('validation.sum.zero_ref', noPct) : '', '.'),
    h('div', { class: 'valsum-bar', role: 'img', 'aria-label': BUCKETS.map((b, i) => t('validation.bucket_count', { count: num(counts[i], 0), label: t(b.label) })).join(', ') },
      counts.map((c, i) => (c ? h('span', { class: `q q${i}`, style: { flexGrow: c / total }, title: t('validation.bucket_count', { count: num(c, 0), label: b_label(i) }) }, c) : null))),
    h('div', { class: 'legend' }, BUCKETS.map((b, i) => h('span', { class: 'item' }, h('span', { class: `key-box q${i}` }), t('validation.bucket_legend', { label: t(b.label), count: num(counts[i], 0) })))));
}
const b_label = i => t(BUCKETS[i].label);

function filters(rows, idPrefix) {
  const insts = [...new Set(rows.map(r => (r.source || {}).institution).filter(Boolean))].sort(compareText);
  const confSeg = seg(idPrefix + 'Conf', [{ id: 'all', label: t('validation.filter.any_confidence') }, { id: 'high', label: t('validation.filter.high') },
    { id: 'medium', label: t('validation.filter.medium') }, { id: 'low', label: t('validation.filter.low') }], 'all');
  const order = seg(idPrefix + 'Order', [{ id: 'ref', label: t('validation.filter.by_reference') }, { id: 'gap', label: t('validation.filter.largest_gaps') }], 'ref');
  const inst = h('select', { 'aria-label': t('validation.filter.institution'), style: { maxWidth: '260px' } }, h('option', { value: '' }, t('validation.filter.any_institution')),
    insts.map(x => h('option', { value: x, selected: state.ui[idPrefix + 'Inst'] === x }, x)));
  inst.addEventListener('change', () => { state.ui[idPrefix + 'Inst'] = inst.value; emit(); });
  const only = h('label', { class: 'row small check' },
    h('input', { type: 'checkbox', checked: !!state.ui[idPrefix + 'Only'], onchange: e => { state.ui[idPrefix + 'Only'] = e.target.checked; emit(); } }), t('validation.filter.only_compared'));
  const q = (state.ui[idPrefix + 'Q'] || '').trim().toLowerCase();
  const keep = r => (confSeg.cur === 'all' || r.confidence === confSeg.cur)
    && (!state.ui[idPrefix + 'Inst'] || (r.source || {}).institution === state.ui[idPrefix + 'Inst'])
    && (!state.ui[idPrefix + 'Only'] || r.gap != null)
    && (!q || `${r.id} ${r.description} ${r.label || ''} ${r.unit} ${(r.source || {}).institution} ${(r.source || {}).title} ${r.note || ''}`.toLowerCase().includes(q));
  return {
    el: h('div', { class: 'row filters' }, searchBox(idPrefix + 'Q', t('validation.filter.search')), confSeg.el, inst, only, h('span', { class: 'spacer' }), order.el),
    rows: rows.filter(keep), byGap: order.cur === 'gap',
  };
}

function groupRows(rows, byGap) {
  if (byGap) return [...rows].sort((a, b) => (Math.abs(b.gap_pct ?? -1)) - (Math.abs(a.gap_pct ?? -1))).map(r => ({ head: r, rows: [r] }));
  const groups = new Map();
  for (const r of rows) { if (!groups.has(r.id)) groups.set(r.id, { head: r, rows: [] }); groups.get(r.id).rows.push(r); }
  return [...groups.values()];
}

/** A status other than ok: a known text ("no model equivalent") in the interface language, else the server's message as it is. */
const statusText = st => tOr('validation.status.' + String(st).toLowerCase().replace(/[^a-z0-9]+/g, '_'), st);
const statusCell = r => (r.status && r.status !== 'ok' ? h('span', { class: 'sub', title: statusText(r.status) }, statusText(r.status)) : null);
const gapCells = r => [
  h('td', { class: 'n' }, r.gap == null ? '–' : signed(r.gap, dec(r.ref))),
  h('td', { class: 'n gapcell' }, r.gap_pct == null ? (r.gap == null ? '' : '–') : [gapBar(r.gap_pct), signedPct(r.gap_pct, Math.abs(r.gap_pct) >= 100 ? 0 : 1)]),
];

function baselineTab(v) {
  const f = filters(v.baseline, 'valB');
  const body = [];
  for (const g of groupRows(f.rows, f.byGap)) {
    const r0 = g.head;
    body.push(h('tr', { class: 'group ref' }, h('td', { colspan: 6 },
      h('div', { class: 'ref-head' }, h('span', null, r0.description || r0.id), conf(r0.confidence)),
      h('div', { class: 'ref-src' }, sourceLink(r0.source), r0.note ? h('span', { class: 'note' }, ` ${r0.note}`) : null))));
    for (const r of g.rows) {
      body.push(h('tr', null,
        h('td', { class: 'when' }, r.when), h('td', { class: 'sub' }, r.unit),
        h('td', { class: 'n' }, fmt(r.ref)), h('td', { class: 'n' }, r.model == null ? statusCell(r) || '–' : num(r.model, dec(r.ref))),
        gapCells(r)));
    }
  }
  return h('div', { class: 'stack' },
    summary(v.baseline, 'baseline'),
    card({
      title: t('validation.base.title'), flush: true, id: 'val-base',
      sub: tn('validation.base.sub', v.baseline.length, { shown: num(f.rows.length, 0) }),
      body: h('div', null, h('div', { style: { padding: '0 14px 10px' } }, f.el),
        f.rows.length ? h('div', { class: 'tbl-wrap tall' }, h('table', { class: 'tbl val' },
          h('thead', null, h('tr', null, h('th', null, t('validation.col.when')), h('th', null, t('common.unit')), h('th', { class: 'n' }, t('validation.col.official')), h('th', { class: 'n' }, t('validation.col.model')),
            h('th', { class: 'n' }, t('validation.col.gap')), h('th', { class: 'n' }, t('validation.col.gap_pct')))),
          h('tbody', null, body))) : h('div', { class: 'empty', style: { padding: '14px' } }, t('validation.base.empty'))),
    }));
}

function measuresTab(v) {
  const f = filters(v.measures, 'valM');
  const used = new Set(state.scenario.measures.map(m => m.name));
  const body = [];
  for (const g of groupRows(f.rows, f.byGap)) {
    const r0 = g.head;
    const m = r0.measure && Array.isArray(r0.measure.changes) ? { name: r0.measure.name || r0.description || r0.id, description: t('validation.meas.official_costing', { institution: (r0.source || {}).institution || '' }).trim(), changes: r0.measure.changes } : null;
    body.push(h('tr', { class: 'group ref' }, h('td', { colspan: 8 },
      h('div', { class: 'ref-head' }, h('span', null, r0.description || r0.id), conf(r0.confidence),
        m ? h('button', { class: 'btn sm', type: 'button', disabled: used.has(m.name), style: { marginLeft: 'auto' },
          onclick: () => { addMeasure(m); state.panelOpen = true; toast(t('validation.meas.added', { name: m.name })); } }, used.has(m.name) ? t('common.in_scenario') : t('validation.meas.add')) : null),
      h('div', { class: 'ref-src' }, sourceLink(r0.source), r0.note ? h('span', { class: 'note' }, ` ${r0.note}`) : null))));
    for (const r of g.rows) {
      const cmp = r.variant === 'static' ? 'static' : 'behavioural';
      const cell = (val, which) => h('td', { class: 'n' + (which === cmp && val != null ? ' used' : '') }, val == null ? '–' : num(val, dec(r.ref)));
      body.push(h('tr', null,
        h('td', { class: 'wrap' }, r.label || '–', r.target_note ? h('div', { class: 'sub' }, r.target_note) : null, statusCell(r) ? h('div', null, statusCell(r)) : null),
        h('td', { class: 'when' }, r.ref_year === r.model_year ? r.ref_year : t('validation.meas.model_year', { ref: r.ref_year, model: r.model_year })),
        h('td', { class: 'sub' }, r.unit),
        h('td', { class: 'n' }, fmt(r.ref), h('div', { class: 'sub' }, r.variant ? tOr('validation.variant.' + r.variant, r.variant) : r.variant)),
        cell(r.static, 'static'), cell(r.behavioural, 'behavioural'), gapCells(r)));
    }
  }
  return h('div', { class: 'stack' },
    summary(v.measures, 'costing'),
    card({
      title: t('validation.meas.title'), flush: true, id: 'val-meas',
      sub: tn('validation.meas.sub', v.measures.length, { shown: num(f.rows.length, 0) }),
      body: h('div', null, h('div', { style: { padding: '0 14px 10px' } }, f.el),
        f.rows.length ? h('div', { class: 'tbl-wrap tall' }, h('table', { class: 'tbl val' },
          h('thead', null, h('tr', null, h('th', null, t('validation.col.figure')), h('th', null, t('common.year')), h('th', null, t('common.unit')), h('th', { class: 'n' }, t('validation.col.official')),
            h('th', { class: 'n' }, t('validation.col.model_static')), h('th', { class: 'n' }, t('validation.col.model_behavioural')), h('th', { class: 'n' }, t('validation.col.gap')), h('th', { class: 'n' }, t('validation.col.gap_pct')))),
          h('tbody', null, body))) : h('div', { class: 'empty', style: { padding: '14px' } }, t('validation.meas.empty'))),
    }));
}

function otherTab(v) {
  return h('div', { class: 'stack' },
    card({
      title: t('validation.rest.not_found_title'), flush: true,
      sub: t('validation.rest.not_found_sub'),
      body: v.not_found.length ? h('div', { class: 'tbl-wrap tall' }, h('table', { class: 'tbl' },
        h('thead', null, h('tr', null, h('th', null, t('validation.col.reference')), h('th', null, t('validation.col.searched')))),
        h('tbody', null, v.not_found.map(n => h('tr', null, h('td', { class: 'wrap' }, n.description || n.id), h('td', { class: 'wrap sub' }, n.searched || '')))))) : h('div', { class: 'empty', style: { padding: '14px' } }, t('validation.rest.none')),
    }),
    card({
      title: t('validation.rest.problems_title'),
      body: v.problems.length ? h('ul', { class: 'plain' }, v.problems.map(p => h('li', null, p))) : h('div', { class: 'empty' }, t('validation.rest.problems_none')),
    }));
}

export function render(root) {
  const entry = lazy('validation:' + state.meta.params_signature, () => apiWhenReady('/api/validation'));
  root.append(lazyBody(entry, {
    loading: t('validation.loading'),
    detail: t('validation.loading_detail'),
    retry: () => { lazyReset('validation:'); emit(); },
    missing: t('validation.missing'),
    ok: v => {
      const tab = tabs('valTab', [
        { id: 'baseline', label: t('validation.tab.baseline', { n: num(v.baseline.length, 0) }) }, { id: 'measures', label: t('validation.tab.measures', { n: num(v.measures.length, 0) }) },
        { id: 'other', label: t('validation.tab.not_found', { n: num(v.not_found.length + v.problems.length, 0) }) },
      ]);
      const meta = v.meta || {};
      return h('div', null,
        notice('info', h('b', null, t('validation.how_title'), ' '), t('validation.how_text'), ' ',
          meta.retrieved ? t('validation.retrieved', { date: fmtDate(meta.retrieved) }) + (v.file ? ` (${v.file})` : '') + '. ' : v.file ? `${v.file}. ` : '',
          v.seconds ? t('validation.computed_in', { seconds: num(v.seconds, 0) }) : '',
          meta.conventions ? h('div', { style: { marginTop: '4px' } }, String(meta.conventions).trim()) : null),
        tab.el, tab.cur === 'measures' ? measuresTab(v) : tab.cur === 'other' ? otherTab(v) : baselineTab(v));
    },
  }));
}
