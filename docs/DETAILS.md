# belsim — Belgium economic and public-finance simulator

Annual simulation of Belgium from 2024 to 2070: population, labour market, economy and the
complete general-government budget (every revenue and expenditure line, four levels of
government, debt). Policy measures are dated changes to parameters and are compared against a
no-policy-change baseline.

## Run

```
uv sync
uv run python -m belsim.api        # dashboard on http://127.0.0.1:8000 (restarts on code changes)
uv run python -m belsim.api --public   # hosting mode, see DEPLOY.md
uv run pytest -q                   # tests
```

From Python:

```python
from belsim.core import engine
from belsim.core.params import Scenario, Measure, Change

base = engine.run_baseline()
scen = engine.run(
    Scenario(name="VAT 22%", measures=[Measure(name="VAT", changes=[
        Change(param="vat.rate_standard", value=0.22, start=2027)])]),
    baseline=base,
)
delta = engine.compare(base, scen)       # scenario minus baseline, every series
```

## Layout

| Path | Content |
|---|---|
| `belsim/core/` | state, parameters / measures / scenarios, yearly engine loop |
| `belsim/demographics/` | cohort-component projection by age, sex, region |
| `belsim/labour/`, `belsim/macro/` | status of every person, employment by sector and job type, wages, prices, GDP |
| `belsim/fiscal/revenue/` | 43 revenue lines and a payslip calculator |
| `belsim/fiscal/expenditure/`, `balance.py` | 43 expenditure lines, interest, entities and transfers, debt |
| `belsim/behaviour/` | behavioural responses (elasticities) and short-run demand feedback |
| `belsim/measures/library/` | prebuilt measures |
| `belsim/distribution/` | stylised households: winners and losers by decile, payslips |
| `belsim/regional/` | regional economy, federated entities, transfers between regions |
| `belsim/i18n/`, `web/i18n/` | French and Dutch catalogs (`uv run python -m belsim.i18n.check`) |
| `belsim/validation.py`, `data/reference/` | official reference figures and the comparison report |
| `belsim/api/`, `web/` | HTTP API and dashboard |
| `params/*.yaml` | every legislated value and assumption, with source |
| `belsim/data/fetch_*.py`, `data/base/` | download scripts and the calibration tables they produce |

`ARCHITECTURE.md` is the module contract (units, key names, order of computation).

## Dashboard

Fourteen views in three groups, in English, French and Dutch, light by default. A white paper explaining the method,
the choices made and the limits is at `/whitepaper` ("Methodology").
**Projection:** overview, budget, population, labour, economy, regions, explorer. **Policy:** measures (parameter browser and scenario builder), compare (several
scenarios side by side), balance the budget (gap to a deficit or debt target and what each
library measure contributes), distribution (change in disposable income by decile and household
type), payslip (net pay and marginal rate against gross), party programmes (2024 costing). **Trust:** validation (model against
official figures, with sources). The uncertainty band on impact charts is the range between
half and one-and-a-half times the central behavioural responses, with the static costing as a
dashed line; it is not a statistical confidence interval.

The distribution view uses about 290 synthetic households (26 shown) built on the income
survey's household types and the tax returns' income groups; inequality and poverty are close
to the observed levels (Gini 0.240 against 0.234): read the changes rather than the levels.

## Party programmes

`scenarios/parties/` holds one prebuilt scenario for each of the twelve parties whose 2024
election programme was costed by the Federal Planning Bureau (DC2024, dc2024.be): 342 measures
with their fiche, source link and official ex-ante impulse, translated into changes of this
model's parameters with one method for all parties. The dashboard lists them in the scenario
panel, the Compare view and a "Party programmes" view ("compare all parties").

- A measure whose fiche fixes a value uses that value; otherwise the parameter is sized so the
  model's static budget effect equals the Bureau's impulse (227 measures), often on a broader
  budget line than the proposal ("partly modelled"). 71 measures carry no changes (not costed,
  budget-neutral, or no usable lever such as the minimum wage); 13 were enacted since 2024 and
  are already in the baseline.
- They are the measures each party submitted for costing in 2024 as translated into this
  model, not the parties' current positions, and the results are this model's, not the
  Planning Bureau's: package effects on the balance and on employment differ from the Bureau's
  for several parties (each party's detail panel shows both).
- `scenarios/parties/source/mapping.yaml` holds the translation; rebuild with
  `uv run python -m belsim.api.party_build --validation` after changing parameters.

## Regions

Results are published for Flanders, Wallonia and Brussels (series `regional.<metric>.<REG>`)
and for the federated entities (Flemish Community, Walloon Region, French Community, Brussels
Region). Policy levers that are a regional competence are parameters keyed by region, so a
measure for one region is one change with a key:

```python
Change("labour.nairu", -0.01, 2027, op="add", key="WAL", phase_in_years=5)
```

Education and childcare levers are keyed by Community (`flemish_community`,
`french_community`). The measure library has twelve regional measures.

## How it works

Each year: prices and indices → population → behavioural adjustments → labour market →
output and incomes → expenditure → revenue → balance and debt. Every budget line is
`base × legislated parameters × calibration factor`; factors are fitted so 2024 reproduces the
published accounts (Eurostat / NBB, ESA 2010) and are then held constant. A scenario with no
measures reproduces the baseline exactly.

Add the library measure "Static costing" (or set `beh.enabled` to 0) to see the mechanical
effect of a scenario without behavioural responses; `beh.scale` (0.5 / 1 / 1.5) gives a
sensitivity band.

## What is data and what is assumption

- **Data (downloaded):** 2024 population, fertility, mortality, migration; labour-force status;
  employment, value added and wages by sector; national accounts; all government revenue and
  expenditure by line and subsector; debt.
- **Verified legislation:** tax, contribution and benefit rules checked against official or
  authoritative sources on 25 November 2026, each cited in the parameter's `source`. Some
  rest on secondary sources (noted there). Enacted reforms are dated entries in the baseline:
  income-tax reform (allowance to EUR 15,600 by 2030), pension reform (malus, end of
  perequation, civil-servant reference wage), unemployment time limit, capital gains tax,
  securities-account tax, health growth norm, defence path.
- **Assumptions:** parameters whose `source` starts with `ASSUMPTION` (287 of 475, flagged in
  the dashboard): elasticities, long-run projection paths, structural shares that shape how a
  line reacts, and the legislated values that could not be verified.

## Validation

`uv run python -m belsim.validation` compares the model with official figures collected in
`data/reference/validation.yaml` (Study Committee on Ageing 2026, Federal Planning Bureau
outlook of June 2026 and HERMES variants, National Bank, Court of Audit, Planning Bureau
costings of the 2024 election programmes) and writes `data/reference/validation_report.md`.
`tests/test_validation_tolerances*.py` keep the validated matches from regressing.

Close to the official figures:
- **Baseline to 2031 (Planning Bureau outlook):** nominal and real GDP (within 0.1%),
  inflation, employment, unemployment, interest, debt, revenue (0.07-0.18 point of GDP above),
  total expenditure (within 0.2 point), expenditure and balance of each federated entity, and
  the general-government balance within 0.15 point to 2030 (6.05% of GDP against 6.4% in
  2031). The baseline reproduces the observed 2025 accounts line by line and keeps that level.
- **Long run (Study Committee on Ageing):** pensions by scheme, acute care and long-term
  care to 2070.
- **Taxes (static):** tax-free allowance, special social-security contribution,
  securities-account tax, one point of standard VAT (EUR 1.5 bn, press figure), food at zero
  rate, hotel VAT at 12%, VAT on new housing, the 2018 corporate-tax cut relative to receipts.
- **Dynamic responses (Planning Bureau HERMES variants):** employer contributions, general
  and targeted at low wages (jobs, prices, labour cost, GDP, budget cost after feedback),
  employee contributions and income tax (GDP), public consumption, investment and benefit
  cuts (GDP, prices, balance), VAT (prices, labour cost, GDP, balance), a 2% index jump
  (labour cost, prices, employment in % of employment, balance).
- **Measures:** unemployment time limit (saving, people excluded, social-assistance cost),
  keeping the legal pension age at 65, the pension malus (path 2027-2031), a 1.5% health
  growth norm (like for like), a rise of the minimum pension.

Of the official costings and evaluations compared (excluding the party measures that are
sized on the official figure by construction), about two fifths are within 10%, two thirds
within 25% and seven in eight within 50%; the Validation view and the white paper give the
live figures.

Known gaps (model vs official):
- **VAT rise:** jobs in the first year are about 30% short, and the gain for the balance
  after two to four years is well below HERMES on the plain ratio to GDP (close to it when
  GDP is taken at market prices, including the tax).
- **Employee contributions:** the gain for the balance after four years is 0.21 point of GDP
  against 0.37; a benefit cut lowers GDP somewhat less than in HERMES (about a fifth less
  after four years).
- **National Bank evaluations:** its GDP effects of the index jump (+0.5%) and the 2016
  contribution cut (+0.3%) are reproduced only with the documented alternative value of
  `dyn.wage_cost.net_export_output_elasticity` (0.30); the default follows the Planning Bureau.
- **Upper-bracket income-tax reforms** are 10-18% too expensive against the 2024 election
  costings. The Bureau's implied bases are smaller in the upper brackets (44 against 50 bn at
  45%); the model keeps the 2024 tax returns, on which its schedule reproduces 2024 receipts,
  and offers the Bureau's bases as an alternative (`pit.bracket_base_factor`, off by default).
- **Income-tax offset of employee contributions:** 52 / 34 / 32% at t / t+2 / t+4 against
  35 / 24 / 24.5% in HERMES (same euros of tax; HERMES has more bargained wage growth).
- **Special social-security contribution, 2028 reform:** costs EUR 560 mn against 423 mn; the
  marital quotient is 30% short like for like.
- **Revenue 2026-2031:** the change of each category since 2025 follows the Planning
  Bureau's outlook through a labelled alignment (`rev.fpb_alignment`, switch
  `rev.fpb_alignment_enabled`, recomputed with `python -m belsim.fiscal.revenue.align`); the
  observed 2025 level is kept. With the alignment off, revenue is up to 0.26 point of GDP
  higher by 2031.
- **Long run:** the model has about 50,000 more people aged 67 and over in 2070 than the
  national population projection (cohorts aged 55-69 shrink more slowly), and total population
  about 160,000 higher; employment growth to 2070 is somewhat short of the Study Committee's.
- **Debt** is 0.3-1.1 point of GDP below the National Bank's forecast for 2026-2028.
- **Distribution view:** overall poverty rate 10.1% against 10.9% observed; household types
  within about 2 points, "other inactive" 3.4 points too poor, median income 2.2% low.
- **Party programmes:** package results differ from the Planning Bureau's for several
  parties, in particular on employment (see "Party programmes"); proposals costed in 2024
  against pre-reform law are not like for like where the reform has since been enacted.
- The retirement response to the pension malus is calibrated on the Planning Bureau's report
  and is far above micro studies; the labour-supply response to benefit levels is calibrated
  on duration elasticities from the literature (10% higher unemployment benefits: about
  10,000 fewer jobs after four years).
- Tobacco and fuel excise have no illicit or cross-border response; their volumes decline by
  assumption, which lowers the revenue ratio from 48.2% in 2031 to 46.6% in 2070
  (`excise.volume_growth`).

## Maintenance

- After changing baseline tax bases, recompute the revenue alignment with
  `uv run python -m belsim.fiscal.revenue.align --write` (`tests/test_revenue_align.py` fails
  if it drifts), then rebuild the party scenarios with
  `uv run python -m belsim.api.party_build --validation`.
- Near-term productivity growth is solved so that real GDP follows the Planning Bureau given
  the model's employment path: after changing employment, rerun
  `uv run python -m belsim.macro.outlook` and copy the values into `macro.productivity_growth`
  (`tests/test_macro_outlook.py` fails if it drifts).
- `uv run python -m belsim.i18n.check` lists missing French and Dutch texts.

## Known limits

- Results are orders of magnitude, not official costings; see the gaps above.
- Behavioural parameters were tuned by hand on the HERMES variants; the literature references
  in `params/behaviour.yaml` and `params/dynamics.yaml` were written from memory.
- Jobs gained or lost through a measure carry a relative wage and productivity by origin
  (`dyn.marginal_wage`, `dyn.marginal_productivity`): judgement values, constant over time.
- Indirect-tax changes move consumer prices and indexation, but GDP and consumption stay
  valued at baseline indirect-tax rates.
- Regional results: net transfers between regions rest on allocation conventions (benefits
  by residence, indivisible spending by population); target-group subsidies change the
  budget but not hiring.
- `educ.cost_per_pupil` and `educ.real_cost_growth` change spending but not employment.
- The distribution view uses about 290 synthetic households (26 are shown). They carry about
  90% of aggregate income tax. Read changes rather than levels.
- Enacted but not modelled: capped indexation of high pensions, the new unemployment benefit
  schedule, the employer solidarity contribution on sickness, several smaller tax changes.
  Announced but not legislated measures are left out.
- Two baseline entries rest on decisions not yet final: the Walloon inheritance-tax cut from
  2028 (a cheaper replacement was announced in October 2026) and the Flemish 3% sole-home
  registration duty from 2027.
- After 2030 the baseline lets tax thresholds follow wages (`core.tax_threshold_wage_link`);
  under current law (prices only) revenue would be about 5.5 points of GDP higher by 2070.

## Licence

Free software under the GNU Affero General Public License, version 3 or later (`LICENSE`).
Third-party data and vendored files keep their own terms, listed in `NOTICE.md`. Source and
issue tracker: https://github.com/daliscia/belsim.

