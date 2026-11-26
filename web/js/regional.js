// Per-region reading of a scenario: which `regional.<metric>.<REG>` series stand for employment,
// GDP and the regional government balance, and the change of each against the baseline.
// Nothing here names a series outright: the model's regional module decides what exists.
import { state, regionCodes, regionColor, dimLabel, regionalMetrics, regionalKey, has, raw } from './store.js';
import { t } from './i18n.js';

// For each headline: preferred metric ids first, then a pattern. The first that exists wins.
const HEADLINES = {
  emp: { prefer: ['employment', 'employment_residents', 'employment_residence', 'employment_workplace', 'employed'], rx: /^employ(ment|ed)(?!.*(rate|share|growth))/ },
  gdp: { prefer: ['gdp_real', 'gdp', 'gdp_nominal'], rx: /^gdp(?!.*(per_|share|growth|pct|deflator))/ },
  bal: { prefer: ['gov_balance', 'government_balance', 'entity_balance', 'fiscal_balance', 'balance'], rx: /balance(?!.*(pct|share|primary))/ },
};

/** Metric id used for a headline ('emp' | 'gdp' | 'bal'), or null when the model has none. */
export function headlineMetric(kind, exists = m => regionCodes().some(r => has(regionalKey(m, r)))) {
  const spec = HEADLINES[kind];
  const ids = regionalMetrics().map(m => m.id);
  return spec.prefer.find(m => exists(m)) || ids.find(m => spec.rx.test(m) && exists(m)) || null;
}

/** Federated entities (communities and regions) that the model reports a balance for. */
export function federatedEntities(measure = 'balance') {
  const detail = (state.meta && state.meta.entity_detail) || {};
  return Object.keys(detail).filter(e => detail[e].kind !== 'consolidation' && has(`fiscal.${measure}.${e}`));
}

const at = (s, i) => (s && s[i] != null ? s[i] : null);

/**
 * Rows of a per-region breakdown. `get(key)` returns {s, b}: scenario and baseline series of a key
 * (arrays over years, or null). `cpi` = {s, b} consumer-price series when amounts are in 2024 prices.
 * Returns [{id, label, unit, dec, polKey, rows: [{code, label, color, key, f: yearIndex => change}]}].
 */
export function regionalBreakdown(get, { real = false, cpi = null, exists } = {}) {
  const regs = regionCodes();
  const present = m => regs.some(r => { const x = get(regionalKey(m, r)); return !!(x && x.b); });
  const ok = exists || present;
  const defl = (which, i) => (real && cpi ? at(cpi[which], i) || 1 : 1);
  const diff = (key, scale) => i => { const x = get(key) || {}; const a = at(x.b, i), b = at(x.s, i); return a == null || b == null ? null : (b - a) * scale; };
  const rel = key => i => { const x = get(key) || {}; const a = at(x.b, i), b = at(x.s, i); return !a || b == null ? null : (b / a - 1) * 100; };
  const eur = key => i => { const x = get(key) || {}; const a = at(x.b, i), b = at(x.s, i); return a == null || b == null ? null : (b / defl('s', i) - a / defl('b', i)) / 1000; };
  const bn = real ? t('unit.bn_real', { year: state.meta.base_year }) : t('unit.bn');
  const rows = (metric, f) => regs.filter(r => { const x = get(regionalKey(metric, r)); return !!(x && x.b); })
    .map(r => ({ code: r, label: dimLabel('regions', r), color: regionColor(r), key: regionalKey(metric, r), f: f(regionalKey(metric, r)) }));
  const out = [];
  const emp = headlineMetric('emp', ok), gdp = headlineMetric('gdp', ok), bal = headlineMetric('bal', ok);
  if (emp) out.push({ id: 'emp', label: t('regions.delta.employment'), unit: t('common.thousand'), dec: 1, polKey: 'labour.employment', rows: rows(emp, k => diff(k, 1 / 1000)) });
  if (gdp) out.push({ id: 'gdp', label: t('regions.delta.gdp'), unit: '%', dec: 2, polKey: 'macro.gdp_real', rows: rows(gdp, rel) });
  if (bal) out.push({ id: 'bal', label: t('regions.delta.gov_balance'), unit: bn, dec: 2, polKey: 'fiscal.balance', rows: rows(bal, eur) });
  else {
    // no balance by region: the balance of each community and region (a different cut: entities, not territories)
    const detail = (state.meta && state.meta.entity_detail) || {};
    const ents = Object.keys(detail).filter(e => detail[e].kind !== 'consolidation').filter(e => { const x = get(`fiscal.balance.${e}`); return !!(x && x.b); });
    if (ents.length) {
      out.push({ id: 'bal', label: t('regions.delta.entity_balance'), unit: bn, dec: 2, polKey: 'fiscal.balance', entities: true,
        rows: ents.map((e, i) => ({ code: e, label: dimLabel('entities', e), color: `var(--c${(i % 8) + 1})`, key: `fiscal.balance.${e}`, f: eur(`fiscal.balance.${e}`) })) });
    }
  }
  return out.filter(g => g.rows.length);
}

/** `get` for the active scenario run against the baseline. */
export const runGetter = key => ({ s: raw(key, 'scenario'), b: raw(key, 'baseline') });
export const runCpi = () => ({ s: raw('macro.cpi', 'scenario'), b: raw('macro.cpi', 'baseline') });
