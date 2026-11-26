// Scenario side panel: measures in the current scenario, prebuilt library, run / save / export.
import { h, $, toast, download } from './dom.js';
import yaml from '../vendor/js-yaml.mjs';
import {
  state, emit, api, post, runScenario, enabledMeasures, isStale, raw, label, kindOf, MAX_COMPARE, setBand, sigOf,
  dimLabel, regionCodes, regionOfKey, isRegional, territoryLabel, regionsOf,
} from './store.js';
import { t, tn } from './i18n.js';
import { seg } from './components.js';

/** Party-programme section of the panel; registered by parties.js (which imports this module). */
let partiesSection = null;
export const registerPartiesSection = fn => { partiesSection = fn; };

// ------------------------------------------------------------------ scenario mutations
export const paramSpec = name => (state.params ? state.params.params.find(p => p.name === name) : null);

export function addChange(measureIndex, newName, change) {
  let m = state.scenario.measures[measureIndex];
  if (!m) {
    m = { name: newName || t('panel.measure_n', { n: state.scenario.measures.length + 1 }), description: '', enabled: true, changes: [] };
    state.scenario.measures.push(m);
  }
  m.changes.push(change);
  emit();
}
/** `measure.id` (a library measure) is kept as `lib`, so its name follows the interface language. */
export function addMeasure(measure) {
  const m = {
    name: measure.name, description: measure.description || '', enabled: true,
    changes: (measure.changes || []).map(c => ({ ...c })),
  };
  if (measure.id) m.lib = measure.id;
  state.scenario.measures.push(m);
  emit();
}
/** Is this library measure already in the scenario (whatever the language it was added in)? */
export function inScenario(measure) {
  return state.scenario.measures.some(m => (measure.id && m.lib === measure.id)
    || m.name === measure.name || (measure.name_en && m.name === measure.name_en));
}
/** {library id: {name, description}} in the current language, to rename measures after a language switch. */
export function libraryIndex() {
  return Object.fromEntries((state.library.measures || []).map(m => [m.id, { name: m.name, description: m.description || '' }]));
}
/** After a language switch: library measures whose name was not edited take the new language. */
export function relabelScenario(before) {
  const now = libraryIndex();
  for (const m of state.scenario.measures) {
    const old = m.lib && before[m.lib], cur = m.lib && now[m.lib];
    if (!old || !cur) continue;
    if (m.name === old.name) m.name = cur.name;
    if ((m.description || '') === old.description) m.description = cur.description;
  }
}
/** Regions a measure addresses through its region-keyed changes. */
export function measureRegions(m) {
  if (Array.isArray(m.regions) && m.regions.length) return m.regions;
  const found = new Set((m.changes || []).flatMap(c => { const code = regionOfKey(c.key); return code ? regionsOf(code) : []; }));
  for (const tag of m.tags || []) if (regionCodes().includes(tag)) found.add(tag);
  return regionCodes().filter(r => found.has(r));
}
function cleanChange(c) {
  const out = { param: c.param, value: c.value, start: c.start, op: c.op || 'set' };
  if (c.end != null && c.end !== '') out.end = c.end;
  if (c.phase_in_years) out.phase_in_years = c.phase_in_years;
  if (c.key != null && c.key !== '') out.key = String(c.key);
  return out;
}
/** The scenario as the engine reads it (enabled measures only, core field names only). */
export function exportable() {
  return {
    name: state.scenario.name || 'scenario',
    description: state.scenario.description || '',
    measures: enabledMeasures().map(m => ({ name: m.name, description: m.description || '', changes: m.changes.map(cleanChange) })),
  };
}
function skippedNote() {
  const n = state.scenario.measures.length - enabledMeasures().length;
  return n ? ' ' + tn('panel.skipped', n) : '';
}
export function loadScenario(scen) {
  state.scenario = {
    name: scen.name || t('common.scenario'), description: scen.description || '',
    measures: (scen.measures || []).map(m => ({
      name: m.name || t('common.measure'), description: m.description || '', enabled: m.enabled !== false,
      changes: (m.changes || []).map(cleanChange), ...(m.lib || m.id ? { lib: m.lib || m.id } : {}),
    })),
  };
  emit();
  runScenario();
}
// ------------------------------------------------------------------ scenarios kept for comparison
/** Add a scenario (default: the active one) to the comparison set; same name replaces. */
export function keepForComparison(scen = exportable()) {
  const measures = (scen.measures || []).filter(m => m.enabled !== false && (m.changes || []).length)
    .map(m => ({ name: m.name || t('common.measure'), description: m.description || '', changes: m.changes.map(cleanChange) }));
  if (!measures.length) { toast(t('panel.compare.none'), true); return false; }
  const name = scen.name || t('common.scenario');
  const entry = { name, description: scen.description || '', measures };
  const i = state.compare.findIndex(c => c.name === name);
  if (i >= 0) {
    state.compare[i] = { ...entry, slot: state.compare[i].slot };
    toast(t('panel.compare.updated', { name }));
  } else {
    if (state.compare.length >= MAX_COMPARE) { toast(t('panel.compare.full', { max: MAX_COMPARE }), true); return false; }
    const used = new Set(state.compare.map(c => c.slot));
    let slot = 0;
    while (used.has(slot)) slot += 1;
    state.compare.push({ ...entry, slot });
    toast(t('panel.compare.kept', { name }));
  }
  emit();
  return true;
}
export const inComparison = () => state.compare.find(c => sigOf(c.measures) === sigOf(state.scenario.measures)) || null;

function importDoc(doc) {
  if (Array.isArray(doc)) doc = { measures: doc };
  if (doc && Array.isArray(doc.changes)) { addMeasure(doc); toast(t('panel.import.measure', { name: doc.name || t('common.measure') })); return; }
  if (doc && Array.isArray(doc.measures)) {
    if (doc.name) { loadScenario(doc); toast(t('panel.import.scenario', { name: doc.name })); } else { doc.measures.forEach(addMeasure); toast(tn('panel.import.measures', doc.measures.length)); }
    return;
  }
  throw new Error(t('panel.import.not_scenario'));
}

// ------------------------------------------------------------------ exports
export function resultsCsv() {
  const keys = Object.keys(state.run ? state.run.scenario.series : state.base).sort();
  const head = ['key', 'label', 'unit', 'run', ...state.years];
  const lines = [head];
  for (const k of keys) {
    const b = raw(k, 'baseline') || [];
    lines.push([k, label(k), kindOf(k), 'baseline', ...b.map(v => v ?? '')]);
    if (state.run) {
      const sc = raw(k, 'scenario') || [];
      lines.push([k, label(k), kindOf(k), 'scenario', ...sc.map(v => v ?? '')]);
      lines.push([k, label(k), kindOf(k), 'delta', ...sc.map((v, i) => (v == null || b[i] == null ? '' : v - b[i]))]);
    }
  }
  const q = v => (/[",\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : String(v));
  return lines.map(r => r.map(q).join(',')).join('\n');
}
const slug = s => String(s || 'scenario').toLowerCase().normalize('NFD').replace(/[\u0300-\u036f]/g, '').replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '') || 'scenario';

// ------------------------------------------------------------------ rendering
const OPS = ['set', 'add', 'mul'];
const fmtVal = v => (typeof v === 'object' ? JSON.stringify(v) : String(v));

function parseValue(text) {
  const s = text.trim();
  if (s === '') throw new Error(t('panel.err.no_value'));
  if (/^[\[{]/.test(s)) return JSON.parse(s);
  if (s === 'true' || s === 'false') return s === 'true';
  const n = Number(s.replace(',', '.'));
  if (!Number.isFinite(n)) throw new Error(t('panel.err.not_number', { text: s }));
  return n;
}

/** The region a change applies to, as a chip; for a regional parameter, a selector to move it. */
function regionControl(c, spec, upd) {
  const reg = regionOfKey(c.key);
  if (!isRegional(spec)) return reg ? h('span', { class: 'tag region' }, territoryLabel(reg)) : null;
  if (!reg && c.key == null) return h('span', { class: 'tag region' }, t('common.all_regions'));
  const rest = String(c.key).split('.').slice(1);
  return h('select', { class: 'region', 'aria-label': t('common.region'), onchange: upd(el => { c.key = [el.value, ...rest].join('.'); }) },
    Object.keys(spec.current).map(r => h('option', { value: r, selected: r === reg }, territoryLabel(r))));
}

function changeRow(m, c, ci) {
  const spec = paramSpec(c.param);
  const years = state.years;
  const upd = fn => e => { try { fn(e.target); emit(); } catch (err) { toast(err.message, true); e.target.value = e.target.defaultValue; } };
  const yearInput = (val, on, ph, aria) => h('input', {
    class: 'y', type: 'number', min: years[0], max: years[years.length - 1], value: val ?? '', placeholder: ph, 'aria-label': aria,
    onchange: upd(on),
  });
  const reg = regionOfKey(c.key);
  const keyText = c.key == null ? '' : reg ? (String(c.key).includes('.') ? ` [${String(c.key).split('.').slice(1).join('.')}]` : '') : ` [${c.key}]`;
  return h('div', { class: 'change' },
    h('div', { class: 'row', style: { flexWrap: 'nowrap', alignItems: 'baseline' } },
      h('button', {
        class: 'linkbtn p' + (spec ? '' : ' unknown'), type: 'button', style: { textAlign: 'left', fontSize: '12.5px' },
        title: spec ? spec.description : t('panel.change.unknown_title'),
        onclick: () => { state.ui.param = c.param; state.ui.measuresTab = 'params'; location.hash = '#/measures'; emit(); },
      }, c.param + keyText),
      h('span', { class: 'spacer' }),
      regionControl(c, spec, upd),
      h('button', { class: 'x', type: 'button', title: t('panel.change.remove'), 'aria-label': t('panel.change.remove'), onclick: () => { m.changes.splice(ci, 1); emit(); } }, '×')),
    spec ? null : h('div', { class: 'hint', style: { color: 'var(--bad)' } }, t('panel.change.unknown')),
    h('div', { class: 'ctl' },
      h('select', { 'aria-label': t('panel.change.operation'), onchange: upd(el => { c.op = el.value; }) },
        OPS.map(k => h('option', { value: k, selected: (c.op || 'set') === k }, t(`panel.op.${k}`)))),
      h('input', { class: 'v', type: 'text', value: fmtVal(c.value), 'aria-label': t('common.value'), onchange: upd(el => { c.value = parseValue(el.value); }) }),
      spec && spec.unit ? h('span', null, spec.unit) : null),
    h('div', { class: 'ctl' }, t('panel.change.from'),
      yearInput(c.start, el => { const y = parseInt(el.value, 10); if (!years.includes(y)) throw new Error(t('panel.err.start_outside')); c.start = y; }, null, t('panel.change.from_aria')),
      t('panel.change.to'), yearInput(c.end, el => { if (el.value === '') delete c.end; else c.end = parseInt(el.value, 10); }, t('panel.change.open'), t('panel.change.to_aria')),
      t('panel.change.phase_in'), h('input', { class: 'y', type: 'number', min: 0, max: 40, value: c.phase_in_years || 0, 'aria-label': t('panel.change.phase_aria'),
        onchange: upd(el => { c.phase_in_years = Math.max(0, parseInt(el.value, 10) || 0); }) }), t('unit.y')));
}

function measureCard(m, mi) {
  return h('div', { class: 'measure' + (m.enabled === false ? ' off' : '') },
    h('div', { class: 'measure-head' },
      h('input', { type: 'checkbox', checked: m.enabled !== false, title: t('panel.measure.include_title'), 'aria-label': t('panel.measure.include', { name: m.name }),
        onchange: e => { m.enabled = e.target.checked; emit(); } }),
      h('input', { type: 'text', value: m.name, 'aria-label': t('panel.measure.name'), onchange: e => { m.name = e.target.value || t('common.measure'); emit(); } }),
      h('button', { class: 'x', type: 'button', title: t('panel.measure.remove_title'), 'aria-label': t('panel.measure.remove', { name: m.name }),
        onclick: () => { state.scenario.measures.splice(mi, 1); emit(); } }, '×')),
    m.description ? h('div', { class: 'measure-desc' }, m.description) : null,
    m.changes.length ? m.changes.map((c, ci) => changeRow(m, c, ci))
      : h('div', { class: 'change hint' }, t('panel.measure.empty')));
}

/** Tags of a library measure, with the regions it addresses. */
export function measureTags(m) {
  const regs = measureRegions(m);
  return [
    ...(m.tags || []).filter(tag => !regionCodes().includes(tag)).map(tag => h('span', { class: 'tag', style: { marginLeft: '6px' } }, (m.tag_labels || {})[tag] || tag)),
    ...regs.map(r => h('span', { class: 'tag region', style: { marginLeft: '6px' } }, dimLabel('regions', r))),
  ];
}
export const isRegionalMeasure = m => (m.tags || []).includes('regional') || measureRegions(m).length > 0;
/** Region filter for lists of library measures ('all' or a region code); null when no measure is regional. */
export function regionFilter(id, measures) {
  if (!measures.some(isRegionalMeasure)) return null;
  const pick = seg(id, [{ id: 'all', label: t('common.all') }, { id: 'regional', label: t('panel.lib.regional') },
    ...regionCodes().map(r => ({ id: r, label: dimLabel('regions', r) }))], 'all');
  const keep = m => pick.cur === 'all' || (isRegionalMeasure(m) && (pick.cur === 'regional' || !measureRegions(m).length || measureRegions(m).includes(pick.cur)));
  return { el: pick.el, keep, cur: pick.cur };
}

function librarySection() {
  const lib = state.library;
  const filter = regionFilter('libRegion', lib.measures);
  const shown = filter ? lib.measures.filter(filter.keep) : lib.measures;
  const items = shown.map(m => {
    const used = inScenario(m);
    return h('div', { class: 'lib-item' },
      h('b', null, m.name, m.unknown_params && m.unknown_params.length ? h('span', { class: 'flag-a', style: { marginLeft: '6px' }, title: t('panel.lib.unknown_params', { list: m.unknown_params.join(', ') }) }, t('panel.lib.needs_update')) : null),
      h('button', { class: 'btn sm', type: 'button', disabled: used, onclick: () => { addMeasure(m); toast(t('panel.lib.added_toast', { name: m.name })); } }, used ? t('panel.lib.added') : t('panel.lib.add')),
      h('div', { class: 'd' }, m.description || tn('common.n_changes', m.changes.length), measureTags(m)));
  });
  const scens = lib.scenarios.map(sc => h('div', { class: 'lib-item' },
    h('b', null, sc.name), h('button', { class: 'btn sm', type: 'button', onclick: () => loadScenario(sc) }, t('panel.load')),
    h('div', { class: 'd' }, sc.description || tn('common.n_measures', sc.measures.length))));
  return h('details', { class: 'fold', open: !!state.ui.libOpen, ontoggle: e => { state.ui.libOpen = e.target.open; } },
    h('summary', null, t('panel.lib.title', { n: lib.measures.length })),
    h('div', null,
      filter ? h('div', { style: { overflowX: 'auto' } }, filter.el) : null,
      items.length ? items : h('div', { class: 'hint' }, lib.measures.length ? t('panel.lib.none_match') : t('panel.lib.empty')),
      scens.length ? [h('div', { class: 'hint', style: { marginTop: '6px', fontWeight: 600 } }, t('panel.lib.scenarios')), ...scens] : null,
      (lib.errors || []).map(e => h('div', { class: 'hint', style: { color: 'var(--bad)' } }, `${e.file}: ${e.error}`))));
}

// On a public server nothing is stored there: "Save" keeps the scenario in this browser
// (localStorage), and export / import move it between browsers.
const LS_SAVED = 'belsim.saved.v1';
const MAX_SAVED = 50;
const savesInBrowser = () => !!(state.meta && state.meta.mode && state.meta.mode.save === 'browser');
function browserSaved() {
  try {
    const d = JSON.parse(localStorage.getItem(LS_SAVED) || '[]');
    return Array.isArray(d) ? d.filter(sc => sc && typeof sc.name === 'string' && Array.isArray(sc.measures)).slice(0, MAX_SAVED) : [];
  } catch (e) { return []; }
}
function storeBrowserSaved(list) {
  try { localStorage.setItem(LS_SAVED, JSON.stringify(list.slice(0, MAX_SAVED))); return true; } catch (e) { return false; }
}
async function removeSaved(sv) {
  if (sv.kind === 'browser') { storeBrowserSaved(browserSaved().filter(sc => slug(sc.name) !== sv.slug)); return refreshSaved(); }
  try { await api(`/api/scenarios/${sv.slug}`, { method: 'DELETE' }); await refreshSaved(); } catch (e) { toast(e.message, true); }
}

function savedSection() {
  const items = state.saved.map(sv => h('div', { class: 'lib-item' },
    h('b', null, sv.scenario.name),
    h('span', null,
      h('button', { class: 'btn sm', type: 'button', onclick: () => { loadScenario(sv.scenario); toast(t('panel.saved.loaded', { name: sv.scenario.name })); } }, t('panel.load')),
      h('button', { class: 'x', type: 'button', title: t('panel.saved.delete_title'), 'aria-label': t('panel.saved.delete', { name: sv.scenario.name }), onclick: () => removeSaved(sv) }, '×')),
    h('div', { class: 'd' }, `${tn('common.n_measures', sv.scenario.measures.length)}, ${sv.kind === 'browser' ? t('panel.saved.in_browser') : sv.file}`)));
  return h('details', { class: 'fold', open: !!state.ui.savedOpen, ontoggle: e => { state.ui.savedOpen = e.target.open; } },
    h('summary', null, t('panel.saved.title', { n: state.saved.length })),
    h('div', null, items.length ? items : h('div', { class: 'hint' }, savesInBrowser() ? t('panel.saved.empty_browser') : t('panel.saved.empty'))));
}

export async function refreshSaved() {
  if (savesInBrowser()) {
    state.saved = browserSaved().map(sc => ({ slug: slug(sc.name), kind: 'browser', file: null, scenario: sc }));
    emit();
    return;
  }
  try { state.saved = (await api('/api/scenarios')).scenarios; } catch (e) { state.saved = []; }
  emit();
}

async function save() {
  if (savesInBrowser()) {
    const scen = exportable();
    const rest = browserSaved().filter(sc => slug(sc.name) !== slug(scen.name));
    if (storeBrowserSaved([scen, ...rest])) toast(t('panel.saved.saved_browser') + skippedNote());
    else toast(t('panel.saved.browser_failed'), true);
    state.ui.savedOpen = true;
    return refreshSaved();
  }
  try {
    const r = await post('/api/scenarios', exportable());
    toast(t('panel.saved.saved_to', { file: r.file }) + skippedNote());
    if (r.problems && r.problems.length) toast(t('panel.saved.saved_but', { problem: r.problems[0] }), true);
    refreshSaved();
  } catch (e) { toast(e.message, true); }
}

function importFile() {
  const input = h('input', { type: 'file', accept: '.yaml,.yml,.json' });
  input.addEventListener('change', async () => {
    const f = input.files[0];
    if (!f) return;
    try { importDoc(yaml.load(await f.text())); } catch (e) { toast(t('panel.import.failed', { file: f.name, error: String(e.reason || e.message).split('\n')[0].slice(0, 160) }), true); }
  });
  input.click();
}

export function renderPanel() {
  const panel = $('#panel');
  panel.classList.toggle('closed', !state.panelOpen);
  if (!state.panelOpen || !state.meta) return;
  const ms = state.scenario.measures, n = enabledMeasures().length;
  const nChanges = enabledMeasures().reduce((tot, m) => tot + m.changes.length, 0);
  const status = state.running ? t('panel.status.simulating')
    : state.runError ? t('panel.status.failed')
    : !n ? t('panel.status.none')
    : isStale() ? (state.run ? t('panel.status.edited') : t('panel.status.not_run'))
    : t('panel.status.current');
  const scroll = panel.querySelector('.panel-body')?.scrollTop || 0;
  const active = document.activeElement && panel.contains(document.activeElement) ? document.activeElement.id : null;
  panel.replaceChildren(
    h('div', { class: 'panel-head' },
      h('div', { class: 'row' }, h('span', { class: 'panel-title' }, t('common.scenario')), h('span', { class: 'spacer' }),
        h('button', { class: 'linkbtn', type: 'button', onclick: () => { state.panelOpen = false; emit(); } }, t('panel.hide'))),
      h('input', { class: 'name', type: 'text', id: 'scenario-name', value: state.scenario.name, 'aria-label': t('panel.name_aria'),
        onchange: e => { state.scenario.name = e.target.value || t('common.scenario'); emit(); } }),
      h('div', { class: 'row' },
        h('button', { class: 'btn primary', type: 'button', disabled: state.running || (!n && !state.run), onclick: runScenario },
          state.running ? t('panel.running') : t('bar.run')),
        h('span', { class: 'hint' }, `${tn('common.n_measures', n)}, ${tn('common.n_changes', nChanges)}. ${status}.`)),
      h('label', { class: 'row small check', title: t('panel.band.title') },
        h('input', { type: 'checkbox', checked: !!state.ui.band, onchange: e => setBand(e.target.checked) }),
        h('span', null, t('panel.band.label'),
          state.ui.band && state.bandLoading ? ' ' + t('panel.band.computing') : '',
          state.ui.band && state.bandError ? h('span', { style: { color: 'var(--bad)' } }, ' ' + t('panel.band.failed', { error: state.bandError })) : ''))),
    h('div', { class: 'panel-body' },
      state.runError ? h('div', { class: 'errbox', style: { padding: '10px 12px' } }, h('b', null, t('panel.refused')), h('pre', null, state.runError.split('; ').join('\n'))) : null,
      ms.length ? ms.map(measureCard)
        : h('div', { class: 'empty', style: { padding: '4px 0' } }, h('b', null, t('panel.empty.lead')), ' ' + t('panel.empty.rest')),
      h('div', { class: 'row' },
        h('button', { class: 'btn', type: 'button', onclick: () => { state.ui.measuresTab = 'params'; location.hash = '#/measures'; emit(); } }, t('panel.add_change')),
        h('button', { class: 'btn', type: 'button', onclick: () => { ms.push({ name: t('panel.measure_n', { n: ms.length + 1 }), description: '', enabled: true, changes: [] }); emit(); } }, t('panel.new_measure')),
        ms.length ? h('button', { class: 'linkbtn', type: 'button', onclick: () => {
          if (confirm(t('panel.clear_confirm'))) { state.scenario.measures = []; emit(); runScenario(); }
        } }, t('panel.clear')) : null),
      librarySection(), partiesSection ? partiesSection() : null, savedSection()),
    h('div', { class: 'panel-foot' },
      h('button', { class: 'btn sm', type: 'button', disabled: !n, title: t('panel.keep_title'),
        onclick: () => keepForComparison() }, inComparison() ? t('panel.in_comparison') : t('panel.keep')),
      h('button', { class: 'btn sm', type: 'button', onclick: save, title: savesInBrowser() ? t('panel.saved.in_browser') : null }, t('panel.save')),
      h('button', { class: 'btn sm', type: 'button', onclick: () => { download(`${slug(state.scenario.name)}.yaml`, yaml.dump(exportable(), { lineWidth: 100 }), 'text/yaml'); toast(t('panel.exported', { format: 'YAML' }) + skippedNote()); } }, t('panel.export', { format: 'YAML' })),
      h('button', { class: 'btn sm', type: 'button', onclick: () => { download(`${slug(state.scenario.name)}.json`, JSON.stringify(exportable(), null, 2), 'application/json'); toast(t('panel.exported', { format: 'JSON' }) + skippedNote()); } }, t('panel.export', { format: 'JSON' })),
      h('button', { class: 'btn sm', type: 'button', onclick: importFile }, t('panel.import.button')),
      h('button', { class: 'btn sm', type: 'button', onclick: () => download(`${slug(state.run ? state.scenario.name : 'baseline')}-results.csv`, resultsCsv(), 'text/csv') }, t('panel.results_csv'))));
  const body = panel.querySelector('.panel-body');
  if (body) body.scrollTop = scroll;
  if (active) { const el = document.getElementById(active); if (el) el.focus(); }
}
