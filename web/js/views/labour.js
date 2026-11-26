import { h } from '../dom.js';
import { t, tOr } from '../i18n.js';
import { state, fetchArrays, has, num, kilo, money, label, dimLabel, color, OTHER, valueAt, signed, raw } from '../store.js';
import { card, lineCard, stackCard, missingNote, dataTable } from '../components.js';
import { stackChart, hbars, columnChart, legend, lineChart } from '../charts.js';

const k = v => kilo(v, 1);
const eur = v => money(v, 0);

/** Fetch scenario (+ baseline when a run exists) arrays for the selected year, then fill `holder`. */
function withArrays(token, holder, fill) {
  const want = [fetchArrays('scenario', state.year)];
  if (state.run) want.push(fetchArrays('baseline', state.year));
  Promise.all(want).then(([sc, base]) => { if (token.live) holder.replaceChildren(...[fill(sc, base || null)].flat()); })
    .catch(e => { if (token.live) holder.replaceChildren(h('div', { class: 'empty' }, t('labour.load_error', { year: state.year, error: e.message }))); });
  return holder;
}
const loading = (minHeight = 200) => h('div', { style: { minHeight: minHeight + 'px' } }, h('div', { class: 'empty' }, t('common.loading')));

// CPI deflator for per-person euro amounts in "base-year prices" mode (wages come from arrays, not series).
function deflator(which) {
  if (state.unit !== 'real') return 1;
  const c = valueAt('macro.cpi', which, state.year, 'bn');
  return c || 1;
}

/** Why employment differs from the baseline: one line per cause the model reports. */
function employmentCauses() {
  if (!state.run) return null;
  const prefix = 'labour.employment_vs_baseline.';
  const keys = Object.keys(state.run.scenario.series).filter(key => key.startsWith(prefix));
  if (!keys.length) return null;
  const series = keys.map((key, i) => ({ name: tOr('labour.cause.' + key.slice(prefix.length), label(key)), color: i < 8 ? color(i) : OTHER, values: raw(key, 'scenario').map(v => (v == null ? null : v / 1000)) }))
    .filter(se => se.values.some(v => v != null && Math.abs(v) > 5e-4));
  const b = raw('labour.employment', 'baseline'), sc = raw('labour.employment', 'scenario');
  if (b && sc) series.unshift({ name: t('labour.causes.total'), color: 'var(--baseline)', width: 2.5, values: sc.map((v, i) => (v == null || b[i] == null ? null : (v - b[i]) / 1000)) });
  return card({
    title: t('labour.causes.title'), sub: t('labour.causes.sub'), id: 'lab-causes',
    body: series.length > 1 ? lineChart({ x: state.years, series, fmt: v => signed(v, 1) + t('unit.k'), zero: true, height: 230, mark: state.year, title: t('labour.causes.aria') })
      : h('div', { class: 'empty' }, t('labour.causes.none')),
    table: () => ({ head: [t('common.year'), ...series.map(se => se.name)], rows: state.years.map((y, i) => [String(y), ...series.map(se => signed(se.values[i], 1))]) }),
  });
}

export function render(root, token) {
  const d = state.meta.dims;
  const statusItems = d.statuses.map((s, i) => ({ name: dimLabel('statuses', s), keys: [`status.${s}`], code: s }));
  // 9 statuses on an 8-colour palette: the last one takes the neutral grey
  const palette = statusItems.map((it, i) => (i < 8 ? color(i) : OTHER));
  statusItems.forEach((it, i) => { it.color = palette[i]; });

  const byAge = withArrays(token, loading(250), sc => stackChart({
    x: d.ages, xTip: a => t('chart.age', { age: a }),
    layers: d.statuses.map((s, i) => ({ name: dimLabel('statuses', s), color: palette[i], values: sc.status_age[i] })),
    fmt: k, height: 250, totalName: t('labour.population'),
  }));

  const jobColors = d.jobtypes.map((j, i) => color(i));
  const sectors = withArrays(token, loading(330), (sc, base) => {
    const rows = d.sectors.map((s, i) => ({
      label: dimLabel('sectors', s), title: `${s} — ${dimLabel('sectors', s)}`,
      value: sc.emp[i].reduce((a, b) => a + b, 0),
      base: base ? base.emp[i].reduce((a, b) => a + b, 0) : null,
      parts: d.jobtypes.map((j, ji) => ({ name: dimLabel('jobtypes', j), value: sc.emp[i][ji], color: jobColors[ji] })),
    })).sort((a, b) => b.value - a.value);
    return [legend(d.jobtypes.map((j, ji) => ({ name: dimLabel('jobtypes', j), color: jobColors[ji], box: true }))
      .concat(base ? [{ name: t('common.baseline_total'), color: 'var(--baseline)' }] : [])), hbars(rows, { fmt: k })];
  });

  const empTable = withArrays(token, loading(330), (sc, base) => {
    const head = [t('labour.sector'), ...d.jobtypes.map(j => dimLabel('jobtypes', j)), t('common.total'), ...(base ? [t('common.delta_vs_baseline')] : [])];
    const rows = d.sectors.map((s, i) => {
      const tot = sc.emp[i].reduce((a, b) => a + b, 0);
      const r = [`${s}  ${dimLabel('sectors', s)}`, ...sc.emp[i].map(v => num(v / 1000, 1)), num(tot / 1000, 1)];
      if (base) r.push(signed((tot - base.emp[i].reduce((a, b) => a + b, 0)) / 1000, 1));
      return r;
    });
    const colTot = d.jobtypes.map((j, ji) => sc.emp.reduce((acc, r) => acc + r[ji], 0));
    const all = colTot.reduce((a, b) => a + b, 0);
    const last = [t('labour.all_sectors'), ...colTot.map(v => num(v / 1000, 1)), num(all / 1000, 1)];
    if (base) last.push(signed((all - base.emp.flat().reduce((a, b) => a + b, 0)) / 1000, 1));
    rows.push(last);
    return dataTable(head, rows);
  });

  const wages = withArrays(token, loading(330), (sc, base) => {
    const fs = deflator('scenario'), fb = deflator('baseline');
    const avg = (a, i) => {
      const n = a.emp[i].reduce((x, y) => x + y, 0);
      return n ? a.emp[i].reduce((acc, e, j) => acc + e * a.wage[i][j], 0) / n : null;
    };
    const rows = d.sectors.map((s, i) => ({
      label: dimLabel('sectors', s), title: `${s} — ${dimLabel('sectors', s)}`,
      value: avg(sc, i) == null ? null : avg(sc, i) / fs, base: base && avg(base, i) != null ? avg(base, i) / fb : null,
    })).sort((a, b) => (b.value || 0) - (a.value || 0));
    return hbars(rows, { fmt: eur, plain: !base, valueName: t('labour.avg_gross_earnings') });
  });

  const deciles = withArrays(token, loading(230), (sc, base) => {
    const fs = deflator('scenario'), fb = deflator('baseline');
    return columnChart({
      cats: d.deciles, catTip: c => t('labour.decile', { n: c }),
      series: d.jobtypes.map((j, ji) => ({ name: dimLabel('jobtypes', j), color: jobColors[ji], values: sc.wage_dist[ji].map(v => v / fs) })),
      base: base ? d.jobtypes.map((j, ji) => ({ values: base.wage_dist[ji].map(v => v / fb) })) : null,
      fmt: eur, height: 230,
    });
  });
  const priceNote = state.unit === 'real' ? t('common.prices_of', { year: state.meta.base_year }) : t('common.current_prices');

  const rates = ['labour.employment_rate_20_64', 'labour.activity_rate_20_64', 'labour.unemployment_rate'];
  const extras = ['labour.flexi_jobs', 'labour.student_jobs', 'labour.service_voucher_workers', 'labour.cross_border_out', 'labour.cross_border_in'];
  const avgW = ['labour.avg_wage_private', 'labour.avg_wage_public', 'labour.avg_income_self_employed'];
  // Card ids keep the table/chart switch (state.ui) independent of the interface language.
  root.append(h('div', { class: 'stack' },
    h('div', { class: 'grid g2' },
      stackCard(t('labour.status.title'), statusItems, { id: 'lab-status', height: 250 }),
      card({ title: t('labour.status_age.title', { year: state.year }), sub: t('labour.per_age'), body: byAge })),
    h('div', { class: 'grid g3' },
      lineCard(t('labour.employment_rate'), 'labour.employment_rate_20_64', { id: 'Employment rate, 20-64' }),
      lineCard(t('labour.unemployment_rate'), 'labour.unemployment_rate', { id: 'Unemployment rate' }),
      lineCard(t('labour.employment'), 'labour.employment', { id: 'Employment' })),
    employmentCauses(),
    h('div', { class: 'grid g2' },
      card({ title: t('labour.by_sector.title', { year: state.year }), sub: t('labour.thousand_persons'), body: sectors }),
      card({ title: t('labour.table.title', { year: state.year }), sub: t('labour.thousand_persons'), body: empTable, flush: true })),
    h('div', { class: 'grid g2' },
      card({ title: t('labour.wages.title', { year: state.year }), sub: t('labour.wages.sub', { prices: priceNote }), body: wages }),
      h('div', { class: 'stack' },
        card({ title: t('labour.deciles.title', { year: state.year }), sub: t('labour.deciles.sub', { prices: priceNote }), body: deciles }),
        avgW.some(has) ? lineCard(t('labour.avg_earnings.title'), avgW, {
          id: 'Average earnings', names: [t('labour.avg_earnings.private'), t('labour.avg_earnings.public'), t('labour.avg_earnings.self_employed')],
        }) : null)),
    h('div', { class: 'grid g3' },
      has('labour.activity_rate_20_64') ? lineCard(t('labour.activity_rate'), 'labour.activity_rate_20_64', { id: 'Activity rate, 20-64' }) : null,
      extras.some(has) ? lineCard(t('labour.special_title'), extras.filter(has), { id: 'Special work arrangements' }) : null,
      ['labour.wage_bill_private', 'labour.wage_bill_public', 'labour.self_employed_income'].some(has)
        ? lineCard(t('labour.wage_bill.title'), ['labour.wage_bill_private', 'labour.wage_bill_public', 'labour.self_employed_income'],
          { id: 'Wage bill and self-employed income', names: [t('labour.wage_bill.private'), t('labour.wage_bill.public'), t('labour.wage_bill.self_employed')] }) : null),
    missingNote([...rates, 'labour.employment', ...avgW, ...extras, 'labour.hours_per_worker', 'labour.blue_collar_share', 'labour.statutory_share'])));
}
