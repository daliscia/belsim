import { h } from '../dom.js';
import { t } from '../i18n.js';
import { state, has, valueAt, unitFor, dimLabel, color, OTHER, fold, raw, formatter } from '../store.js';
import { card, lineCard, stackCard, missingNote, deltaSpan } from '../components.js';
import { hbars } from '../charts.js';

function folded(items) {
  const w = it => Math.abs((raw(it.keys[0], 'baseline') || [0])[0] || 0);
  const { keep, rest } = fold(items, w, 7);
  const out = keep.map((it, i) => ({ ...it, color: color(i) }));
  if (rest.length) out.push({ name: t('common.other_n', { n: rest.length }), keys: rest.flatMap(r => r.keys), color: OTHER });
  return out;
}

function vaCard() {
  const d = state.meta.dims;
  const keys = d.sectors.map(s => `va.sector.${s}`).filter(has);
  if (!keys.length) return card({ title: t('economy.va.title'), body: h('div', { class: 'empty' }, t('economy.va.empty')) });
  const u = unitFor(keys[0]), fmt = formatter(keys[0]);
  const rows = d.sectors.filter(s => has(`va.sector.${s}`)).map(s => {
    const key = `va.sector.${s}`, v = valueAt(key, 'scenario'), b = state.run ? valueAt(key, 'baseline') : null;
    return {
      label: dimLabel('sectors', s), title: `${s} — ${dimLabel('sectors', s)}`, value: v, base: b,
      delta: b != null && v != null && Math.abs(v - b) > 5e-3 ? deltaSpan('', v - b, 2) : null,
    };
  }).sort((a, b) => (b.value || 0) - (a.value || 0));
  return card({ title: t('economy.va.title_year', { year: state.year }), sub: u.label, body: hbars(rows, { fmt, plain: !state.run, valueName: t('economy.va.value') }) });
}

// Card ids keep the table/chart switch (state.ui) independent of the interface language.
export function render(root) {
  const d = state.meta.dims;
  const comp = ['macro.consumption_households', 'macro.gov_consumption', 'macro.investment', 'macro.exports', 'macro.imports'];
  const income = ['macro.compensation_employees', 'macro.gross_operating_surplus', 'macro.mixed_income'];
  const bases = ['macro.corporate_profits_taxable', 'macro.household_capital_income', 'macro.property_transactions', 'macro.inheritance_flow'];
  const indices = ['macro.cpi', 'macro.health_index', 'macro.wage_index', 'macro.gdp_deflator'];
  const rates = ['macro.interest_rate_10y', 'macro.interest_rate_short'];
  root.append(h('div', { class: 'stack' },
    h('div', { class: 'grid g3' },
      lineCard(t('economy.gdp_current'), 'macro.gdp_nominal', { id: 'GDP, current prices', sub: state.unit === 'pct' ? t('economy.gdp_share_hint') : null }),
      lineCard(t('economy.gdp_volume'), 'macro.gdp_real', { id: 'GDP, volume' }),
      lineCard(t('economy.growth'), 'macro.gdp_real_growth', { id: 'Real GDP growth', zero: true })),
    h('div', { class: 'grid g2' },
      lineCard(t('economy.components.title'), comp, {
        id: 'Expenditure components of GDP',
        names: [t('economy.components.households'), t('economy.components.government'), t('economy.components.investment'), t('economy.components.exports'), t('economy.components.imports')],
      }),
      stackCard(t('economy.cons_title'), folded(d.cons_cats.map(c => ({ name: dimLabel('cons_cats', c), keys: [`macro.cons.${c}`] }))), { id: 'eco-cons' })),
    h('div', { class: 'grid g3' },
      lineCard(t('economy.prices.title'), indices, {
        id: 'Prices and wage index',
        names: [t('economy.prices.cpi'), t('economy.prices.health_index'), t('economy.prices.wage_index'), t('economy.prices.gdp_deflator')],
      }),
      lineCard(t('economy.inflation'), 'macro.inflation', { id: 'Inflation', zero: true }),
      lineCard(t('economy.rates.title'), rates, { id: 'Interest rates', names: [t('economy.rates.bond_10y'), t('economy.rates.short')] })),
    h('div', { class: 'grid g2' },
      vaCard(),
      h('div', { class: 'stack' },
        lineCard(t('economy.income.title'), income, {
          id: 'Income components', names: [t('economy.income.compensation'), t('economy.income.gos'), t('economy.income.mixed')],
        }),
        lineCard(t('economy.tax_bases'), bases.filter(has), { id: 'Tax bases' }),
        has('macro.productivity_growth') ? lineCard(t('economy.productivity'), 'macro.productivity_growth', { id: 'Productivity growth', zero: true }) : null,
        has('macro.demand_gap') && state.run ? lineCard(t('economy.demand_gap'), 'macro.demand_gap', { id: 'Demand gap from the fiscal impulse', zero: true }) : null)),
    missingNote([...comp, 'macro.investment_housing', ...income, 'macro.wages_salaries', ...bases, 'macro.household_financial_wealth',
      ...indices.slice(0, 3), 'macro.inflation', ...rates, 'macro.productivity_growth'])));
}
