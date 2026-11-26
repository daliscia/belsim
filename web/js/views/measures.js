import { h, toast, debounce } from '../dom.js';
import { state, emit, num, plain, territoryCodes, territoryLabel, isRegional, regionOfKey } from '../store.js';
import { t, tn, tOr } from '../i18n.js';
import { card, tabs, seg } from '../components.js';
import { lineChart } from '../charts.js';
import { impactCard } from '../impact.js';
import { addChange, paramSpec } from '../scenario.js';

const short = v => {
  if (v == null) return '–';
  if (typeof v === 'number') return Math.abs(v) >= 1000 ? num(v, 0) : plain(+v.toPrecision(6));
  if (typeof v === 'boolean') return String(v);
  if (Array.isArray(v)) return tn('measures.list_of', v.length);
  if (typeof v === 'object') {
    const keys = Object.keys(v);
    if (keys.length && keys.every(k => territoryCodes().includes(k))) return t('measures.by_region');
    return tn('measures.n_keys', keys.length);
  }
  return String(v);
};

/** Addressable numeric leaves of a list/dict parameter: [{path:'rates.2', value}] */
function leaves(v, prefix = '') {
  const out = [];
  const walk = (x, p) => {
    if (Array.isArray(x)) x.forEach((y, i) => walk(y, p ? `${p}.${i}` : String(i)));
    else if (x && typeof x === 'object') Object.entries(x).forEach(([k, y]) => walk(y, p ? `${p}.${k}` : k));
    else if (p) out.push({ path: p, value: x });
  };
  walk(v, prefix);
  return out;
}

function changedParams() {
  const s = new Set();
  for (const m of state.scenario.measures) for (const c of m.changes || []) s.add(c.param);
  return s;
}

function browser() {
  const P = state.params;
  const changed = changedParams();
  const nRegional = P.params.filter(isRegional).length;
  const filter = seg('paramFilter', [
    { id: 'all', label: t('measures.filter.all', { n: P.params.length }) },
    { id: 'assumption', label: t('measures.filter.assumptions', { n: P.params.filter(p => p.assumption).length }) },
    ...(nRegional ? [{ id: 'regional', label: t('measures.filter.regional', { n: nRegional }) }] : []),
    { id: 'changed', label: t('measures.filter.changed', { n: changed.size }) },
  ], 'all');
  const list = h('div', { class: 'plist' });
  const draw = q => {
    q = q.trim().toLowerCase();
    const terms = q.split(/\s+/).filter(Boolean);
    const groups = new Map();
    for (const p of P.params) {
      if (filter.cur === 'assumption' && !p.assumption) continue;
      if (filter.cur === 'changed' && !changed.has(p.name)) continue;
      if (filter.cur === 'regional' && !isRegional(p)) continue;
      const hay = `${p.name} ${p.description} ${p.unit} ${p.source} ${p.file}`.toLowerCase();
      if (!terms.every(term => hay.includes(term))) continue;
      if (!groups.has(p.group)) groups.set(p.group, []);
      groups.get(p.group).push(p);
    }
    const closed = state.ui.pClosed || {};
    const nodes = [...groups].map(([g, ps]) => h('details', {
      class: 'pgroup', open: q || filter.cur === 'regional' ? true : !closed[g],
      ontoggle: e => { if (!q && filter.cur !== 'regional') state.ui.pClosed = { ...(state.ui.pClosed || {}), [g]: !e.target.open }; },
    },
      h('summary', null, h('span', null, g), h('span', { class: 'c' }, `${tn('measures.n_params', ps.length)}, ${[...new Set(ps.map(p => p.file))].join(', ')}`)),
      ps.map(p => h('button', {
        class: 'prow', type: 'button', 'aria-current': String(state.ui.param === p.name),
        onclick: () => { state.ui.param = p.name; emit(); },
      },
        h('span', { class: 'nm' }, p.name, ' ',
          p.assumption ? h('span', { class: 'flag-a', title: p.source }, t('measures.tag.assumption')) : null,
          isRegional(p) ? h('span', { class: 'tag region', style: { marginLeft: '4px' }, title: t('measures.tag.regional_title') }, t('measures.tag.regional')) : null,
          p.wage_link ? h('span', { class: 'tag', style: { marginLeft: '4px' }, title: t('measures.tag.wage_linked_title') }, t('measures.tag.wage_linked')) : null,
          changed.has(p.name) ? h('span', { class: 'tag', style: { marginLeft: '4px' } }, t('measures.tag.in_scenario')) : null),
        h('span', { class: 'cv' }, short(p.current), p.unit && typeof p.current !== 'object' ? ` ${p.unit}` : ''),
        h('span', { class: 'ds' }, p.description || t('measures.no_description'))))));
    list.replaceChildren(...(nodes.length ? nodes : [h('div', { class: 'empty', style: { padding: '14px' } },
      P.params.length ? t('measures.no_match') : t('measures.no_params'))]));
  };
  const input = h('input', {
    type: 'search', id: 'param-search', placeholder: t('measures.search'), value: state.ui.paramQ || '', style: { width: '100%' },
    oninput: debounce(() => { state.ui.paramQ = input.value; draw(input.value); }, 80),
  });
  draw(state.ui.paramQ || '');
  return card({
    title: t('measures.params_title'), flush: true, id: 'param-browser',
    sub: t('measures.params_sub', { params: tn('measures.n_params', P.params.length), files: tn('measures.n_files', P.files.length), year: state.meta.base_year }),
    body: h('div', null, h('div', { style: { padding: '0 14px 8px', display: 'grid', gap: '6px' } }, input, h('div', { style: { overflowX: 'auto' } }, filter.el)), list),
  });
}

function changeForm(p) {
  const years = state.years;
  const isScalar = p.kind === 'scalar' || p.kind === 'bool';
  const regional = isRegional(p);
  const regs = regional ? territoryCodes().filter(r => r in p.current) : [];
  // a regional parameter is addressed by region first; what is left of the path is the element inside one region
  const lv = isScalar ? [] : regional ? leaves(p.current[regs[0]]) : leaves(p.current);
  const ms = state.scenario.measures;
  const f = state.ui.form || {};
  const opSel = h('select', { id: 'cf-op' },
    ['set', 'add', 'mul'].map(v => h('option', { value: v, selected: f.op === v }, t(`measures.form.op.${v}`))));
  const regionSel = regional ? h('select', { id: 'cf-region' },
    regs.map(r => h('option', { value: r, selected: f.region === r }, t('measures.form.region_now', { region: territoryLabel(r), value: short(p.current[r]) }))),
    regs.length > 1 ? h('option', { value: 'all', selected: f.region === 'all' }, regs.length === 3 && regs.every(r => r.length === 3) ? t('common.all_regions') : t('measures.form.all_territories')) : null) : null;
  const keySel = isScalar || (regional && !lv.length) ? null : h('select', { id: 'cf-key' },
    h('option', { value: '' }, regional ? t('measures.form.whole_region') : t('measures.form.whole_value')),
    lv.map(l => h('option', { value: l.path }, t('measures.form.element_now', { path: l.path, value: short(l.value) }))));
  const value = h('input', { type: 'text', id: 'cf-value', placeholder: isScalar ? short(p.current) : regional && !lv.length ? short(p.current[regs[0]]) : t('measures.form.value_placeholder'), autocomplete: 'off' });
  const start = h('input', { type: 'number', id: 'cf-start', min: years[0], max: years[years.length - 1], value: f.start || Math.min(years[years.length - 1], Math.max(years[0] + 2, 2026)) });
  const end = h('input', { type: 'number', id: 'cf-end', min: years[0], max: years[years.length - 1], placeholder: t('measures.form.permanent') });
  const phase = h('input', { type: 'number', id: 'cf-phase', min: 0, max: 40, value: 0 });
  const target = h('select', { id: 'cf-target' },
    ms.map((m, i) => h('option', { value: i, selected: i === ms.length - 1 }, m.name)),
    h('option', { value: 'new', selected: !ms.length }, t('measures.form.new_measure')));
  const newName = h('input', { type: 'text', id: 'cf-newname', placeholder: t('measures.form.new_name_placeholder', { example: p.description ? p.description.slice(0, 32) : p.name }) });
  const newWrap = h('label', { class: 'w2' }, t('measures.form.new_name'), newName);
  const syncNew = () => { newWrap.hidden = target.value !== 'new'; };
  target.addEventListener('change', syncNew);
  syncNew();
  const preview = h('div', { class: 'hint' });
  /** The changes the form describes: one, or one per region for "all three regions". */
  const read = () => {
    const text = value.value.trim();
    if (!text) throw new Error(t('measures.err.no_value'));
    let val;
    if (/^[\[{]/.test(text)) { try { val = JSON.parse(text); } catch (e) { throw new Error(t('measures.err.bad_json')); } }
    else if (text === 'true' || text === 'false') val = text === 'true';
    else { val = Number(text.replace(',', '.')); if (!Number.isFinite(val)) throw new Error(t('measures.err.not_number', { text })); }
    const sub = keySel && keySel.value ? keySel.value : null;
    const st = parseInt(start.value, 10);
    if (!years.includes(st)) throw new Error(t('measures.err.start_range', { first: years[0], last: years[years.length - 1] }));
    const base = { param: p.name, value: val, start: st, op: opSel.value };
    if (end.value !== '') {
      const en = parseInt(end.value, 10);
      if (en < st) throw new Error(t('measures.err.end_before_start'));
      base.end = en;
    }
    const ph = parseInt(phase.value, 10) || 0;
    if (ph > 0) base.phase_in_years = ph;
    if (regional) {
      // a plain number needs an element to land on when each region holds several values
      if (lv.length && !sub && opSel.value === 'set' && typeof val !== 'object') throw new Error(t('measures.err.need_element'));
      const targets = regionSel.value === 'all' ? regs : [regionSel.value];
      return targets.map(r => ({ ...base, key: sub ? `${r}.${sub}` : r }));
    }
    if (!isScalar && !sub && opSel.value === 'set' && typeof val !== 'object') throw new Error(t('measures.err.need_element'));
    return [sub ? { ...base, key: sub } : base];
  };
  const currentOf = c => {
    if (!c.key) return p.current;
    if (regional) {
      const [r, ...rest] = String(c.key).split('.');
      return rest.length ? (leaves(p.current[r]).find(l => l.path === rest.join('.')) || {}).value : p.current[r];
    }
    return (lv.find(l => l.path === c.key) || {}).value;
  };
  const updatePreview = () => {
    try {
      const cs = read();
      const lines = cs.map(c => {
        const cur = currentOf(c);
        if (typeof cur !== 'number' || typeof c.value !== 'number') return null;
        const after = c.op === 'set' ? c.value : c.op === 'add' ? cur + c.value : cur * c.value;
        const reg = regionOfKey(c.key);
        return (reg ? territoryLabel(reg) + t('common.colon') : '') + t('measures.preview.becomes', { from: short(cur), to: short(after), unit: p.unit ? ' ' + p.unit : '' });
      }).filter(Boolean);
      const c = cs[0];
      preview.textContent = !lines.length ? '' : lines.join('; ') + ' '
        + (c.end ? t('measures.preview.from_to', { start: c.start, end: c.end }) : t('measures.preview.from', { start: c.start }))
        + (c.phase_in_years ? ', ' + tn('measures.preview.phased', c.phase_in_years) : '')
        + (p.uprate && p.uprate !== 'none' ? ' ' + t('measures.preview.uprated', { year: state.meta.base_year, index: tOr(`measures.index.${p.uprate}`, p.uprate) }) : '') + '.';
    } catch (e) { preview.textContent = ''; }
  };
  for (const el of [opSel, regionSel, keySel, value, start, end, phase]) if (el) el.addEventListener('input', updatePreview);
  const submit = e => {
    e.preventDefault();
    try {
      const cs = read();
      state.ui.form = { op: cs[0].op, start: cs[0].start, region: regionSel ? regionSel.value : undefined };
      let idx = target.value === 'new' ? -1 : parseInt(target.value, 10);
      for (const c of cs) {
        addChange(idx, newName.value.trim() || p.description || p.name, c);
        if (idx < 0) idx = state.scenario.measures.length - 1;
      }
      state.panelOpen = true;
      toast(cs.length > 1 ? tn('measures.form.added_n', cs.length) : t('measures.form.added'));
    } catch (err) { toast(err.message, true); }
  };
  return h('form', { class: 'form', onsubmit: submit },
    h('label', null, t('measures.form.operation'), opSel),
    regionSel ? h('label', null, t('common.region'), regionSel) : null,
    keySel ? h('label', null, t('measures.form.element'), keySel) : null,
    h('label', { class: keySel || regionSel ? '' : 'w2' }, p.unit ? t('measures.form.value_unit', { unit: p.unit }) : t('common.value'), value),
    h('label', null, t('measures.form.from_year'), start),
    h('label', null, t('measures.form.until_year'), end),
    h('label', null, t('measures.form.phase_in'), phase),
    h('label', null, t('measures.form.add_to'), target),
    newWrap,
    h('div', { class: 'w4 row', style: { gridColumn: '1 / -1' } },
      h('button', { class: 'btn dark', type: 'submit' }, t('measures.form.submit')), preview));
}

function indexationText(p) {
  const up = p.uprate && p.uprate !== 'none' ? t('measures.indexation.uprated', { index: tOr(`measures.index.${p.uprate}`, p.uprate) }) : null;
  const link = paramSpec('core.tax_threshold_wage_link');
  const wl = p.wage_link ? t('measures.indexation.wage_link', { param: link ? 'core.tax_threshold_wage_link' : t('measures.indexation.wage_link_generic') }) : null;
  return [up, wl].filter(Boolean).join('; ') || t('measures.indexation.none');
}

/** Compact, readable rendering of a list- or dict-valued parameter. */
function structured(p) {
  const v = p.current;
  const isNum = x => typeof x === 'number';
  const fmt = x => (isNum(x) ? short(x) : typeof x === 'object' && x !== null ? JSON.stringify(x) : String(x));
  const regional = isRegional(p);
  const keyName = k => (regional ? territoryLabel(k) : k);
  const keyHead = regional ? t('common.region') : t('measures.struct.key');
  let rows = null, head = null;
  if (Array.isArray(v) && v.every(x => !Array.isArray(x) && (typeof x !== 'object' || x === null))) {
    head = [t('measures.struct.element'), t('common.value')]; rows = v.map((x, i) => [String(i), fmt(x)]);
  } else if (Array.isArray(v) && v.every(x => Array.isArray(x) && x.length === 2)) {
    head = [t('measures.struct.element'), t('measures.struct.first'), t('measures.struct.second')]; rows = v.map((x, i) => [String(i), fmt(x[0]), fmt(x[1])]);
  } else if (v && typeof v === 'object' && !Array.isArray(v)) {
    const vals = Object.values(v);
    if (vals.length && vals.every(x => Array.isArray(x) && x.length === vals[0].length && x.every(isNum))) {
      head = [keyHead, ...vals[0].map((_, i) => String(i))]; rows = Object.entries(v).map(([k, x]) => [keyName(k), ...x.map(fmt)]);
    } else { head = [keyHead, t('common.value')]; rows = Object.entries(v).map(([k, x]) => [keyName(k), fmt(x)]); }
  }
  if (!rows) return h('pre', { class: 'val' }, JSON.stringify(v, null, 1));
  if (!rows.length) return h('div', { class: 'hint' }, t('measures.struct.empty'));
  return h('div', { class: 'tbl-wrap', style: { maxHeight: '240px' } }, h('table', { class: 'tbl dense' },
    h('thead', null, h('tr', null, head.map((c, i) => h('th', { class: i ? 'n' : '' }, c)))),
    h('tbody', null, rows.map(r => h('tr', null, r.map((c, i) => h('td', { class: i ? 'n' : '' }, c)))))));
}

function detail() {
  const p = paramSpec(state.ui.param);
  if (!p) {
    return card({ title: t('measures.choose.title'), body: h('div', { class: 'empty' },
      h('b', null, t('measures.choose.lead')), ' ' + t('measures.choose.rest')) });
  }
  const isScalar = p.kind === 'scalar';
  const regional = isRegional(p);
  const dated = Object.entries(p.values);
  const inScenario = state.scenario.measures.flatMap(m => (m.changes || []).filter(c => c.param === p.name).map(c => ({ m, c })));
  const unitSuffix = p.unit ? `, ${p.unit}` : '';
  return h('div', { class: 'stack' },
    card({
      title: p.name, sub: p.description || t('measures.no_description'), id: 'param-detail',
      tools: h('span', null,
        regional ? h('span', { class: 'tag region', title: t('measures.tag.regional_title') }, t('measures.tag.regional')) : null, ' ',
        p.wage_link ? h('span', { class: 'tag', title: t('measures.tag.wage_linked_title') }, t('measures.tag.wage_linked')) : null, ' ',
        p.assumption ? h('span', { class: 'flag-a' }, t('measures.tag.assumption')) : null),
      body: h('div', { class: 'stack' },
        p.assumption ? h('div', { class: 'hint', style: { background: 'var(--warn-bg)', color: 'var(--warn-ink)', padding: '6px 9px', borderRadius: '5px' } },
          t('measures.assumption_note')) : null,
        h('dl', { class: 'kv' },
          h('dt', null, t('common.unit')), h('dd', null, p.unit || t('measures.not_stated')),
          h('dt', null, t('measures.kv.source')), h('dd', null, p.source || t('measures.not_stated')),
          h('dt', null, t('measures.kv.indexation')), h('dd', null, indexationText(p)),
          h('dt', null, t('measures.kv.defined_in')), h('dd', null, `params/${p.file}`),
          h('dt', null, t('measures.kv.legislated')), h('dd', null, dated.some(([, v]) => v && typeof v === 'object')
            ? h('div', { class: 'dated' }, dated.map(([y, v]) => h('div', null, h('b', null, `${y}: `), typeof v === 'object' ? JSON.stringify(v).replace(/([,:])/g, '$1 ') : String(v))))
            : dated.map(([y, v]) => `${y}: ${v}`).join('   '))),
        isScalar && p.baseline ? h('div', null,
          h('div', { class: 'card-sub', style: { marginBottom: '4px' } }, t('measures.baseline_by_year') + unitSuffix),
          lineChart({ x: state.params.years, series: [{ name: t('common.baseline'), values: p.baseline, color: 'var(--baseline)' }],
            fmt: v => short(v), height: 130, mark: state.year }))
          : h('div', null,
            h('div', { class: 'card-sub', style: { marginBottom: '4px' } },
              t('measures.value_in', { year: state.meta.base_year }) + unitSuffix + '. ' + (regional ? t('measures.address_region') : t('measures.address_element'))),
            structured(p)),
        inScenario.length ? h('div', { class: 'hint' }, t('measures.already_changed', { list: [...new Set(inScenario.map(x => x.m.name))].join(', ') })) : null),
    }),
    card({ title: t('measures.form.title'), sub: t('measures.form.sub'), body: changeForm(p), id: 'param-form' }));
}

export function render(root) {
  const tab = tabs('measuresTab', [{ id: 'params', label: t('measures.tab.params') }, { id: 'impact', label: t('measures.tab.impact') }]);
  root.append(tab.el);
  if (tab.cur === 'impact') { root.append(impactCard()); return; }
  if (!state.params) { root.append(h('div', { class: 'empty' }, t('measures.load_failed'))); return; }
  root.append(h('div', { class: 'browser' }, browser(), detail()));
}
