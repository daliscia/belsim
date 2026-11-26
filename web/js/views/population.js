import { h } from '../dom.js';
import { t } from '../i18n.js';
import { state, fetchPyramid, has, num, kilo, dimLabel, raw } from '../store.js';
import { card, lineCard, stackCard, seg, pickYear, missingNote, scenarioName } from '../components.js';
import { pyramid, lineChart } from '../charts.js';

const fmtK = v => kilo(v, 0);

function pyramidCard(token) {
  const regions = state.meta.dims.regions;
  const sel = seg('pyrRegion', [{ id: 'all', label: t('common.belgium') }, ...regions.map(r => ({ id: r, label: dimLabel('regions', r) }))], 'all');
  const holder = h('div', { style: { minHeight: '440px' } }, h('div', { class: 'empty' }, t('population.pyramid.loading')));
  const want = [fetchPyramid('scenario')];
  if (state.run) want.push(fetchPyramid('baseline'));
  Promise.all(want).then(([sc, base]) => {
    if (!token.live) return;
    const pickRun = d => (sel.cur === 'all' ? d.pop : d.pop_region[regions.indexOf(sel.cur)]);
    const data = pickRun(sc), i = sc.years.indexOf(state.year);
    if (i < 0) { holder.replaceChildren(h('div', { class: 'empty' }, t('population.pyramid.no_data', { year: state.year }))); return; }
    // one scale for every year of this selection, so scrubbing shows real growth and ageing
    let max = 0;
    for (const yr of data) for (const row of yr) max = Math.max(max, row[0], row[1]);
    const b = base ? pickRun(base)[base.years.indexOf(state.year)] : null;
    const total = data[i].reduce((acc, r) => acc + r[0] + r[1], 0);
    holder.replaceChildren(
      pyramid({
        ages: state.meta.dims.ages, male: data[i].map(r => r[0]), female: data[i].map(r => r[1]),
        baseMale: b && b.map(r => r[0]), baseFemale: b && b.map(r => r[1]),
        max, fmt: fmtK, labels: { M: dimLabel('sexes', 'M'), F: dimLabel('sexes', 'F') },
      }),
      h('p', { class: 'hint' }, t('population.pyramid.hint', { total: num(total / 1e6, 3), year: state.year })));
  }).catch(e => { if (token.live) holder.replaceChildren(h('div', { class: 'empty' }, t('population.pyramid.load_error', { error: e.message }))); });
  return card({
    title: t('population.pyramid.title', { year: state.year }), sub: t('population.pyramid.sub'),
    tools: sel.el, body: holder,
  });
}

// Card ids keep the table/chart switch (state.ui) independent of the interface language.
function dependencyCard() {
  if (has('demo.old_age_dependency')) {
    return lineCard(t('population.dependency.title'), 'demo.old_age_dependency', { id: 'Old-age dependency ratio', sub: t('population.dependency.sub') });
  }
  // not reported by the model: derive it from the age-group totals
  const ratio = which => {
    const o = raw('pop.age_67_plus', which), w = raw('pop.age_18_66', which);
    return o && w ? o.map((v, i) => (w[i] ? v / w[i] : null)) : null;
  };
  const series = [{ name: t('common.baseline'), values: ratio('baseline'), color: 'var(--baseline)', width: state.run ? 1.5 : 2 }];
  if (state.run) series.push({ name: scenarioName(), values: ratio('scenario'), color: 'var(--scenario)' });
  return card({
    title: t('population.dependency.title'), sub: t('population.dependency.sub_computed'),
    body: lineChart({ x: state.years, series, fmt: v => num(v, 3), mark: state.year, onPick: pickYear }),
  });
}

export function render(root, token) {
  const flows = ['demo.births', 'demo.deaths', 'demo.net_migration'];
  const mig = ['demo.immigration', 'demo.emigration'];
  const regions = state.meta.dims.regions;
  root.append(h('div', { class: 'grid g-main' },
    pyramidCard(token),
    h('div', { class: 'stack' },
      dependencyCard(),
      lineCard(t('population.flows.title'), flows, {
        id: 'Births, deaths and net migration',
        names: [t('population.flows.births'), t('population.flows.deaths'), t('population.flows.net_migration')], zero: true,
      }),
      stackCard(t('population.age_groups.title'), [
        { name: '0-17', keys: ['pop.age_0_17'] }, { name: '18-66', keys: ['pop.age_18_66'] }, { name: t('population.age_groups.over_67'), keys: ['pop.age_67_plus'] },
      ], { id: 'pop-age' }))),
  h('div', { class: 'grid g3', style: { marginTop: '14px' } },
    lineCard(t('population.total_title'), 'pop.total', { id: 'Total population' }),
    stackCard(t('population.by_region_title'), regions.map(r => ({ name: dimLabel('regions', r), keys: [`pop.region.${r}`] })), { id: 'pop-region' }),
    mig.some(has) ? lineCard(t('population.migration.title'), mig, {
      id: 'Immigration and emigration', names: [t('population.migration.immigration'), t('population.migration.emigration')],
    }) : null,
    has('demo.tfr') ? lineCard(t('population.tfr.title'), 'demo.tfr', { id: 'Total fertility rate', sub: t('population.tfr.sub') }) : null,
    ['demo.life_expectancy_M', 'demo.life_expectancy_F'].some(has)
      ? lineCard(t('population.life.title'), ['demo.life_expectancy_M', 'demo.life_expectancy_F'], {
        id: 'Life expectancy at birth', names: [t('population.life.men'), t('population.life.women')],
      }) : null),
  missingNote([...flows, ...mig, 'demo.tfr', 'demo.life_expectancy_M', 'demo.life_expectancy_F', 'demo.old_age_dependency']));
}
