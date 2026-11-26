import { h } from '../dom.js';
import { t, lang } from '../i18n.js';
import { state, has, valueAt, deltaAt, unitFor, num } from '../store.js';
import { deltaSpan, lineCard, missingNote } from '../components.js';
import { impactCard } from '../impact.js';

// Headline tiles. `unit` pins a display unit regardless of the global switch where that is
// the convention (ratios stay ratios); otherwise the tile follows the switch. `label` and `u` are catalog keys.
const TILES = [
  { key: 'macro.gdp_nominal', label: 'overview.gdp', unit: 'bn+' },
  { key: 'macro.gdp_real_growth', label: 'overview.growth' },
  { key: 'pop.total', label: 'overview.population', scale: 1 / 1000, u: 'unit.million_persons', dec: 2 },
  { key: 'labour.employment_rate_20_64', label: 'overview.employment_rate' },
  { key: 'labour.unemployment_rate', label: 'overview.unemployment_rate' },
  { key: 'fiscal.revenue_total', label: 'overview.revenue' },
  { key: 'fiscal.expenditure_total', label: 'overview.expenditure' },
  { key: 'fiscal.balance_pct_gdp', label: 'overview.balance' },
  { key: 'fiscal.debt_pct_gdp', label: 'overview.debt' },
];

function tile(tl) {
  if (!has(tl.key)) {
    return h('div', { class: 'tile missing' }, h('div', { class: 'tile-label' }, t(tl.label)),
      h('div', { class: 'tile-value' }, '–'), h('div', { class: 'tile-delta' }, t('common.not_provided')));
  }
  // GDP as % of GDP is meaningless: show it in € bn (current or base-year prices) instead.
  const unit = tl.unit === 'bn+' && state.unit === 'pct' ? 'bn' : state.unit;
  const u = unitFor(tl.key, unit);
  const sc = tl.scale || 1, dec = tl.dec ?? u.dec;
  const val = valueAt(tl.key, 'scenario', state.year, unit);
  const d = deltaAt(tl.key, state.year, unit);
  const isRatio = /%/.test(u.label);
  return h('div', { class: 'tile' },
    h('div', { class: 'tile-label' }, t(tl.label)),
    h('div', { class: 'tile-value' }, num(val == null ? null : val * sc, dec), h('span', { class: 'u' }, tl.u ? t(tl.u) : u.label)),
    h('div', { class: 'tile-delta' }, state.run
      ? [deltaSpan(tl.key, d == null ? null : d * sc, dec, isRatio ? ' ' + t('common.pts') : ''), ' ', t('common.vs_baseline')]
      : t('overview.baseline')));
}

// Card ids keep the table/chart switch (state.ui) independent of the interface language.
/** What the tool is and is not, with the way to the methodology page. */
function aboutCard() {
  return h('section', { class: 'about', 'aria-labelledby': 'about-title' },
    h('div', null,
      h('h2', { id: 'about-title' }, t('overview.about.title')),
      h('p', null, t('overview.about.what')),
      h('p', null, h('b', null, t('overview.about.limits_lead')), ' ', t('overview.about.limits'))),
    h('a', { class: 'btn', href: `whitepaper?lang=${lang()}` }, t('overview.about.link')));
}

export function render(root) {
  root.append(h('div', { class: 'stack' },
    h('div', { class: 'tiles' }, TILES.map(tile)),
    aboutCard(),
    h('div', { class: 'grid g2' },
      lineCard(t('overview.balance_title'), 'fiscal.balance_pct_gdp', {
        id: 'Budget balance', refs: [{ value: -3, label: t('overview.deficit_limit') }], zero: true, height: 230,
      }),
      lineCard(t('overview.debt_title'), 'fiscal.debt_pct_gdp', {
        id: 'Public debt', refs: [{ value: 60, label: t('overview.debt_ref') }], height: 230,
      })),
    impactCard({ compact: true }),
    h('div', { class: 'grid g2' },
      lineCard(t('overview.rev_exp_title'), ['fiscal.revenue_pct_gdp', 'fiscal.expenditure_pct_gdp'],
        { id: 'Revenue and expenditure', names: [t('overview.revenue'), t('overview.expenditure')] }),
      lineCard(t('overview.growth'), 'macro.gdp_real_growth', { id: 'Real GDP growth', zero: true })),
    missingNote(TILES.map(tl => tl.key)),
  ));
}
