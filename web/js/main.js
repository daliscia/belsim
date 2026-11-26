// Boot, routing, top bar (units, year, language, theme) and the render loop.
import { h, $, toast } from './dom.js';
import { state, emit, subscribe, restore, api, runScenario, isStale, enabledMeasures, lazyReset, fmtDate } from './store.js';
import { t, tn, lang, setLang, initI18n, LANGS } from './i18n.js';
import { renderPanel, refreshSaved, relabelScenario, libraryIndex } from './scenario.js';
import { hideTip } from './charts.js';
import * as overview from './views/overview.js';
import * as budget from './views/budget.js';
import * as regions from './views/regions.js';
import * as population from './views/population.js';
import * as labour from './views/labour.js';
import * as economy from './views/economy.js';
import * as measures from './views/measures.js';
import * as explorer from './views/explorer.js';
import * as compare from './views/compare.js';
import * as balance from './views/balance.js';
import * as distribution from './views/distribution.js';
import * as payslip from './views/payslip.js';
import * as validation from './views/validation.js';
import * as partiesView from './views/parties.js';

// `group` only orders the rail: what the model projects, what a policy changes, how far to trust it.
// Titles live in the catalogs: nav.<id>.title (page heading) and nav.<id>.nav (rail entry).
const VIEWS = [
  { id: 'overview', mod: overview, group: 'projection' },
  { id: 'budget', mod: budget, group: 'projection' },
  { id: 'regions', mod: regions, group: 'projection' },
  { id: 'population', mod: population, group: 'projection' },
  { id: 'labour', mod: labour, group: 'projection' },
  { id: 'economy', mod: economy, group: 'projection' },
  { id: 'explorer', mod: explorer, group: 'projection' },
  { id: 'measures', mod: measures, group: 'policy' },
  { id: 'compare', mod: compare, group: 'policy' },
  { id: 'parties', mod: partiesView, group: 'policy', controls: 'year' },
  { id: 'balance', mod: balance, group: 'policy' },
  { id: 'distribution', mod: distribution, group: 'policy', controls: 'year' },
  { id: 'payslip', mod: payslip, group: 'policy', controls: 'year' },
  { id: 'validation', mod: validation, group: 'trust', controls: 'none' },
];
const UNITS = ['bn', 'real', 'pct'];
const unitLabel = u => (u === 'bn' ? t('unit.bn') : u === 'real' ? t('unit.bn_real', { year: state.meta ? state.meta.base_year : 2024 }) : t('unit.pct_gdp'));

let token = { live: false };
let bootError = null;
let ready = false;          // catalogs loaded

function route() {
  const id = (location.hash.match(/^#\/(\w+)/) || [])[1];
  state.view = VIEWS.some(v => v.id === id) ? id : 'overview';
}

const isDark = () => document.documentElement.dataset.theme === 'dark';

function renderChrome() {
  if (!ready) return;
  const view = VIEWS.find(v => v.id === state.view);
  const title = t(`nav.${view.id}.title`);
  $('#view-title').textContent = title;
  document.title = `${title} — belsim`;
  const desc = document.querySelector('meta[name="description"]');
  if (desc) desc.content = t('meta.description');
  $('#skip').textContent = t('common.skip');
  const where = `whitepaper?lang=${lang()}`;
  $('#method-link').textContent = t('nav.methodology'); $('#method-link').href = where;
  $('#foot-disclaimer').textContent = t('foot.disclaimer');
  $('#foot-method').textContent = t('foot.method'); $('#foot-method').href = where;
  $('#foot-source').textContent = t('foot.source');
  const v = state.version;
  $('#foot-build').textContent = v && v.commit_short
    ? t('foot.build', { commit: v.commit_short, date: fmtDate(v.commit_date) }) : '';
  $('#brand-sub').textContent = t('nav.tagline');
  $('#rail').setAttribute('aria-label', t('nav.aria'));
  $('#panel').setAttribute('aria-label', t('panel.aria'));
  const n = enabledMeasures().length;
  const items = [];
  VIEWS.forEach((v, i) => {
    if (!i || VIEWS[i - 1].group !== v.group) items.push(h('li', { class: 'nav-group' }, t(`nav.group.${v.group}`)));
    const badge = v.id === 'measures' && n ? h('span', { class: 'count', title: tn('nav.active_measures', n) }, n)
      : v.id === 'compare' && state.compare.length ? h('span', { class: 'count quiet', title: tn('nav.kept_scenarios', state.compare.length) }, state.compare.length) : null;
    items.push(h('li', null, h('a', { href: `#/${v.id}`, 'aria-current': v.id === state.view ? 'page' : null }, t(`nav.${v.id}.nav`), badge)));
  });
  $('#nav').replaceChildren(...items);
  const unitSeg = $('#unit-seg');
  unitSeg.setAttribute('aria-label', t('bar.units_aria'));
  unitSeg.replaceChildren(...UNITS.map(u => h('button', {
    type: 'button', 'aria-pressed': String(state.unit === u), onclick: () => { state.unit = u; emit(); },
  }, unitLabel(u))));
  const langSeg = $('#lang-seg');
  langSeg.setAttribute('aria-label', t('bar.lang_aria'));
  langSeg.replaceChildren(...LANGS.map(l => h('button', {
    type: 'button', 'aria-pressed': String(lang() === l.id), title: l.name, lang: l.id, onclick: () => changeLang(l.id),
  }, l.label)));
  // the unit switch and the year only appear where the view reads them
  unitSeg.hidden = !!view.controls;
  $('.yearctl').hidden = view.controls === 'none';
  $('.yearctl').setAttribute('aria-label', t('bar.year_aria'));
  const yr = $('#year-range');
  yr.setAttribute('aria-label', t('common.year'));
  if (state.years.length) {
    yr.min = state.years[0]; yr.max = state.years[state.years.length - 1]; yr.step = 1; yr.value = state.year;
  }
  $('#year-out').textContent = state.year;
  const play = $('#year-play');
  play.textContent = playing ? '❚❚' : '▶';
  play.title = t('bar.play'); play.setAttribute('aria-label', t('bar.play'));
  $('#year-prev').setAttribute('aria-label', t('bar.prev_year'));
  $('#year-next').setAttribute('aria-label', t('bar.next_year'));
  $('#panel-btn').hidden = state.panelOpen;
  $('#panel-btn').textContent = n ? t('bar.scenario_n', { n }) : t('common.scenario');
  $('#theme-btn').textContent = isDark() ? t('bar.light_theme') : t('bar.dark_theme');

  const strip = $('#status-strip');
  strip.className = '';
  if (state.runError) {
    strip.hidden = false; strip.className = 'error';
    strip.replaceChildren(h('span', null, h('b', null, t('bar.refused') + ' '), state.runError),
      h('button', { class: 'btn sm', type: 'button', onclick: runScenario }, t('bar.run_again')));
  } else if (state.running) {
    strip.hidden = false; strip.replaceChildren(t('bar.simulating'));
  } else if (isStale()) {
    strip.hidden = false;
    strip.replaceChildren(state.run ? t('bar.stale_run') : t('bar.stale_never'),
      h('button', { class: 'btn sm primary', type: 'button', onclick: runScenario }, t('bar.run')));
  } else strip.hidden = true;

  if (state.meta) {
    const p = state.params ? state.params.params : [];
    const a = p.filter(x => x.assumption).length;
    $('#model-note').replaceChildren(
      h('div', null, t('nav.model_size', { series: Object.keys(state.base).length, params: p.length })),
      a ? h('div', null, tn('nav.assumptions', a)) : null,
      h('button', { class: 'linkbtn', type: 'button', style: { color: 'inherit', fontSize: '11.5px' }, onclick: () => boot(true) }, t('nav.reload')));
  }
}

function renderView() {
  const root = $('#view');
  token.live = false;
  token = { live: true };
  hideTip();
  if (!ready) return;
  if (bootError) {
    root.replaceChildren(h('div', { class: 'errbox' }, h('h2', null, t('bar.boot_failed')), h('pre', null, bootError),
      h('button', { class: 'btn dark', type: 'button', onclick: () => boot(true) }, t('bar.try_again'))));
    return;
  }
  if (!state.meta) return;
  const view = VIEWS.find(v => v.id === state.view);
  const active = document.activeElement && root.contains(document.activeElement) ? document.activeElement : null;
  const focusId = active && active.id, sel = active && 'selectionStart' in active ? [active.selectionStart, active.selectionEnd] : null;
  const scroll = root.dataset.view === view.id ? root.scrollTop : 0;
  const box = h('div');
  try {
    view.mod.render(box, token);
  } catch (e) {
    console.error(e);
    box.replaceChildren(h('div', { class: 'errbox' }, h('h2', null, t('bar.view_failed', { view: t(`nav.${view.id}.title`) })), h('pre', null, String(e && e.stack || e))));
  }
  root.replaceChildren(box);
  root.dataset.view = view.id;
  root.scrollTop = scroll;
  if (focusId) {
    const el = document.getElementById(focusId);
    if (el) { el.focus(); if (sel) { try { el.setSelectionRange(sel[0], sel[1]); } catch (e) { /* not a text input */ } } }
  }
}

function render() { renderChrome(); renderView(); renderPanel(); }

// ------------------------------------------------------------------ year playback
let playing = null;
function stopPlay() { if (playing) { clearInterval(playing); playing = null; } }
function togglePlay() {
  if (playing) { stopPlay(); emit(); return; }
  if (state.year >= state.years[state.years.length - 1]) state.year = state.years[0];
  playing = setInterval(() => {
    if (state.year >= state.years[state.years.length - 1]) { stopPlay(); emit(); return; }
    state.year += 1; emit();
  }, 320);
  emit();
}
function stepYear(d) {
  const i = state.years.indexOf(state.year) + d;
  if (i >= 0 && i < state.years.length) { state.year = state.years[i]; emit(); }
}

// ------------------------------------------------------------------ boot
const bootBox = () => h('div', { class: 'boot' }, h('div', { class: 'progress-bar' }, h('span')), ready ? h('div', null, t('bar.loading_model')) : null);

/** Text the server provides in the interface language: metadata, parameters, measure library. */
async function loadModelText() {
  const [meta, params, library, parties] = await Promise.all([
    api('/api/meta'),
    api('/api/params').catch(e => { toast(t('bar.params_failed', { error: e.message }), true); return null; }),
    api('/api/measures').catch(() => ({ measures: [], scenarios: [], errors: [] })),
    api('/api/scenarios?kind=party').catch(() => null),
  ]);
  state.meta = meta; state.params = params; state.library = library;
  state.parties = parties && Array.isArray(parties.scenarios) ? parties : null;
}

async function boot(reload = false) {
  bootError = null;
  if (reload) $('#view').replaceChildren(bootBox());
  try {
    const [, base] = await Promise.all([loadModelText(), api('/api/baseline')]);
    state.base = base.series; state.years = base.years;
    api('/api/version').then(v => { state.version = v; renderChrome(); }).catch(() => {});
    if (!state.years.includes(state.year)) state.year = Math.min(state.years[state.years.length - 1], Math.max(state.years[0], 2030));
    state.run = null; state.runFor = null; state.band = null; state.bandFor = null;
    lazyReset();
    emit();
    refreshSaved();
    if (enabledMeasures().length) runScenario();
  } catch (e) {
    bootError = e.message;
    render();
  }
}

/**
 * Switch the interface language. The scenario, its results, the year and every view setting are
 * kept: only the catalog and the text the server provides are loaded again.
 */
let switching = false;
async function changeLang(id) {
  if (id === lang() || switching) return;
  switching = true;
  const before = libraryIndex();
  const untitled = state.scenario.name === t('panel.default_name');
  try {
    await setLang(id);
    if (untitled) state.scenario.name = t('panel.default_name');
    renderChrome();
    if (state.meta) {
      await loadModelText();
      relabelScenario(before);
      lazyReset();
    }
  } catch (e) {
    toast(e.message, true);
  } finally {
    switching = false;
    emit();
  }
}

function toggleTheme() {
  const next = isDark() ? 'light' : 'dark';
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem('belsim.theme', next); } catch (e) { /* ignore */ }
  renderChrome();
}

async function start() {
  restore();
  route();
  await initI18n();
  ready = true;
  // a scenario that still carries the built-in name takes the name of the interface language
  // (the built-in name in any of the three languages: the language may have been changed on the methodology page)
  if (['My scenario', 'Mon scénario', 'Mijn scenario'].includes(state.scenario.name) && !state.scenario.measures.length) state.scenario.name = t('panel.default_name');
  $('#view').replaceChildren(bootBox());
  subscribe(render);
  addEventListener('hashchange', () => { route(); $('#view').scrollTop = 0; emit(); });
  $('#year-range').addEventListener('input', e => { stopPlay(); state.year = parseInt(e.target.value, 10); emit(); });
  $('#year-prev').addEventListener('click', () => { stopPlay(); stepYear(-1); });
  $('#year-next').addEventListener('click', () => { stopPlay(); stepYear(1); });
  $('#year-play').addEventListener('click', togglePlay);
  $('#panel-btn').addEventListener('click', () => { state.panelOpen = true; emit(); });
  $('#theme-btn').addEventListener('click', toggleTheme);
  $('#skip').addEventListener('click', e => { e.preventDefault(); $('#view').focus(); });
  addEventListener('keydown', e => {
    if (e.target.closest('input, textarea, select')) return;
    if (e.key === 'ArrowLeft') stepYear(-1);
    if (e.key === 'ArrowRight') stepYear(1);
  });
  renderChrome();
  boot();
}
start();
