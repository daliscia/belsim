# belsim — architecture and module contract

Annual, accounting-consistent simulator of the Belgian economy and public finances.
Base year 2024 (calibrated to actual accounts), simulated yearly to 2070.

Read `belsim/core/dims.py`, `state.py`, `params.py`, `engine.py` first: they are the contract.
**Do not edit `belsim/core/` or files owned by another module.** If the contract blocks you,
work around it inside your module (e.g. `state.extra["<module>"]`) and report it at the end.

## Units
- Persons: float headcounts. Population is at 1 January.
- Per-person amounts (wages, benefits, thresholds): EUR per year.
- Macro and fiscal aggregates: **EUR million, current prices**.
- Indices (`cpi`, `health_index`, `wage_index`): 1.0 in 2024. Rates are fractions (0.21), not percent.
- Accounting reference: ESA 2010 national accounts (Eurostat / NBB), general government S13.

## Yearly sequence (engine.run)
Every module function has the signature `f(state, prev, p, ctx) -> None` and fills its own
section of `state` (the `YearState` under construction). `prev` is last year's finished state
(`None` in 2024). `p` is a `ParamView` (`p["name"]`). `ctx` is `engine.Context`.

| # | call | owner | must fill |
|---|------|-------|-----------|
| 1 | `macro.prices` | labour+macro | `state.macro`: `inflation, cpi, health_index, wage_index, interest_rate_10y, interest_rate_short` (receives **un-uprated** params) |
| 2 | `demographics.run` | demographics | `state.pop [age,sex,region]`, `state.demo` |
| 3 | `behaviour.run` | behaviour | `state.adjust` (scenario runs only, year > 2024) |
| 4 | `labour.run` | labour+macro | `state.status, emp, wage, wage_dist, labour` |
| 5 | `macro.output` | labour+macro | `state.va`, rest of `state.macro` |
| 6 | `fiscal.expenditure.run` | expenditure | `state.expenditure` (line -> EUR mn), incl. interest |
| 7 | `fiscal.revenue.run` | revenue | `state.revenue` (line -> EUR mn) |
| 8 | `fiscal.balance.run` | expenditure | `state.fiscal` |
| 9 | `behaviour.demand_gap` | behaviour | returns output-gap fraction; if non-zero the engine sets `state.macro["demand_gap"]` and re-runs steps 4-8 once |

Rules:
- **Idempotent**: steps 4-8 can run twice on the same `state`; overwrite, never accumulate.
  Year-to-year stocks must be read from `prev`, not from module globals. No mutable module state
  (data tables loaded once and cached read-only are fine).
- **Base year = data.** In 2024 every module reproduces actual 2024 figures. Aggregates are
  computed as `base x rate x calibration factor`, the factor being fitted once so 2024 matches
  the accounts, then held constant.
- **Baseline neutrality**: a scenario with no measures must reproduce the baseline exactly
  (`tests/test_engine_smoke.py`). So behavioural effects are always functions of
  *differences* between scenario and baseline (`ctx.base_p`, `ctx.base_state`).
- Never hard-code legislation or assumptions: put them in your params file with `source:`.
- Deterministic; no randomness. A 2024-2070 run should take well under ~2 s.

## Required keys (other modules depend on these names)

`state.macro` (floats): `inflation, cpi, health_index, wage_index, interest_rate_10y,
interest_rate_short, demand_gap` (set by engine, default 0), `gdp_nominal, gdp_real,
gdp_real_growth, gdp_deflator, consumption_households, cons.<cat>` for each `CONS_CATS`
(household consumption **including** indirect taxes), `investment, investment_housing,
gov_consumption, exports, imports, compensation_employees` (D1), `wages_salaries` (D11),
`gross_operating_surplus, mixed_income, corporate_profits_taxable` (CIT base proxy),
`household_capital_income` (interest + dividends received by households),
`household_financial_wealth, property_transactions` (value of real-estate sales),
`inheritance_flow` (value of inheritances + gifts), `productivity_growth`.

`state.labour` (floats): `employment, unemployment_rate, employment_rate_20_64,
activity_rate_20_64, hours_per_worker, blue_collar_share` (of private employees),
`statutory_share` (of public employees), `flexi_jobs, student_jobs, service_voucher_workers,
cross_border_out, cross_border_in, avg_wage_private, avg_wage_public, avg_income_self_employed,
wage_bill_private, wage_bill_public` (gross wages D11, EUR mn), `self_employed_income` (EUR mn).

`state.status [status,age,sex,region]` sums over status to `state.pop` exactly.
`state.emp [sector,jobtype]` headcount; column sums equal the three employed status totals.
`state.wage [sector,jobtype]` average gross annual earnings per worker (EUR).
`state.wage_dist [jobtype,decile]` mean gross annual earnings of each tenth of workers (EUR);
its mean over deciles equals the jobtype average wage.

`state.demo`: `births, deaths, net_migration, immigration, emigration, tfr,
life_expectancy_M, life_expectancy_F, old_age_dependency` (67+/18-66).

`state.fiscal`: `revenue_total, expenditure_total, primary_expenditure, interest, balance,
primary_balance, debt, balance_pct_gdp, primary_balance_pct_gdp, debt_pct_gdp, revenue_pct_gdp,
expenditure_pct_gdp, balance.<entity>, revenue.<entity>, expenditure.<entity>, debt.<entity>`
for each `ENTITIES` (after inter-governmental transfers), `net_expenditure_growth`
(EU fiscal-rule indicator), `structural_balance_pct_gdp`.

`state.revenue` / `state.expenditure`: one float per line. Each package exports
`LINES: dict[str, LineMeta]` describing every line (label, group, ESA/COFOG code, entity
shares). Sum of all lines = total general-government revenue / expenditure (ESA, consolidated).

`state.adjust` (behaviour -> consumed by labour, macro, revenue; **missing key = neutral**):
- `emp_rate_delta`: ndarray `[age, sex]`, additive change in the employment rate of the
  population cell from labour-supply responses (people move employed <-> inactive/unemployed).
- `labour_demand_factor`: float, multiplicative on private employment (labour-cost response).
- `retirement_shift`: float, additional years of average effective retirement age from
  financial incentives (legal-age changes are handled mechanically by labour, not here).
- `cons_factor`: dict cat -> multiplicative factor on the *volume* consumed of a category.
- `cit_base_factor`, `capital_income_base_factor`, `self_employed_income_factor`,
  `property_transactions_factor`: floats, multiplicative on those tax bases.
- `productivity_factor`: float, multiplicative on labour productivity level (e.g. public
  investment, education).
`state.macro["demand_gap"]`: macro scales real GDP by `(1+gap)`; labour applies an Okun
response to private employment via its own parameter.

## Micro-calculators (pure functions, vectorised over numpy arrays, no state)
`belsim.fiscal.revenue.calc`:
- `tax_on_income(gross, p, kind) -> dict` with arrays `gross, employer_cost, ssc_employer,
  ssc_employee, pit, net`. `kind` in `emp_private, emp_public, self_employed, pension,
  unemployment, disability`. Single person, no dependants, average municipal surcharge.
- `consumption_tax_rates(p) -> dict cat -> rate`: VAT + excise as a fraction of the
  **pre-tax** price, per `CONS_CATS`.
- `cit_effective_rate(p) -> float`; `capital_income_tax_rate(p) -> float`.

`belsim.fiscal.expenditure.calc`:
- `replacement_income(status, prev_gross, p) -> array`: gross annual benefit of a new
  beneficiary with previous gross wage `prev_gross`; `status` in `unemployed, disabled, retired`.

## Parameters, measures, scenarios (`belsim/core/params.py`)
- One YAML file per module in `params/` (`demographics.yaml`, `labour.yaml`, `macro.yaml`,
  `revenue.yaml`, `expenditure.yaml`, `behaviour.yaml`). Names are globally unique and prefixed
  (`pit.`, `ssc.`, `vat.`, `pension.`, `demo.`, `macro.`, `labour.`, `beh.` ...).
- Each parameter: `description`, `unit`, `source`, optional `uprate` (`cpi|health_index|wages`),
  dated `values`. Already-legislated future changes are dated entries in the baseline.
- Design parameters so that realistic reforms are single `Change`s: rates, thresholds, ages,
  headcounts, growth norms as separate scalar (or short list/dict) parameters.
- A measure is `Measure(name, changes=[Change(param, value, start, op, end, phase_in_years, key)])`.

## Data
- Fetch scripts: `belsim/data/fetch_<module>.py` (run as `uv run python -m belsim.data.fetch_<module>`),
  raw downloads in `data/raw/<module>/`, cleaned calibration tables in `data/base/<module>_*.csv`.
  The model reads only `data/base/` at run time (never the network). Commit-size tables only.
- Preferred source: Eurostat dissemination API (open, no key):
  `https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/<dataset>?format=JSON&geo=BE&...`
  (JSON-stat). Useful datasets: `demo_pjan, demo_r_d2jan, demo_fasec, demo_mlifetable, demo_gind,
  proj_23np, nama_10_gdp, nama_10_a64, nama_10_a64_e, nama_10_co3_p3, nasa_10_nf_tr, lfsa_*,
  lfst_r_*, earn_ses_*, gov_10a_main, gov_10a_exp, gov_10a_taxag, gov_10dd_edpt1, spr_exp_*`.
  Also NBB.Stat, Statbel open data, Federal Planning Bureau.
- If 2024 is not yet published for a table, use the latest year and scale to 2024 totals.
- **Honesty about numbers**: anything not downloaded from an official source must be marked
  `source: "ASSUMPTION - <reason>"` in the params file. Never present a guess as data.
- Send no personal or project information in requests (generic User-Agent).

## Environment
- `uv run pytest -q` runs all tests. Python 3.12, numpy/pandas/scipy/pyyaml/requests available.
  Do not add dependencies without need; if needed, report it rather than editing `pyproject.toml`.
- Tests for your module: `tests/test_<module>*.py`. `tests/test_engine_smoke.py` must keep passing.
- Other agents are replacing the other stub modules **at the same time in the same directory**.
  The full engine may be temporarily broken by their work in progress: test your module mainly
  through unit tests that build the inputs you need, and do not "fix" other modules' files.
  Stubs of the other modules show the shape of what you will receive, with crude numbers.

## File ownership
| Owner | Files |
|---|---|
| core (integrator) | `belsim/core/`, `pyproject.toml`, `ARCHITECTURE.md`, `tests/test_engine_smoke.py` |
| demographics | `belsim/demographics/`, `params/demographics.yaml`, `belsim/data/fetch_demographics.py`, `data/*/demographics*`, `tests/test_demographics*.py` |
| labour+macro | `belsim/labour/`, `belsim/macro/`, `params/labour.yaml`, `params/macro.yaml`, `belsim/data/fetch_labour.py`, `fetch_macro.py`, `data/*/labour*`, `data/*/macro*`, `tests/test_labour*.py`, `tests/test_macro*.py` |
| revenue | `belsim/fiscal/revenue/`, `params/revenue.yaml`, `belsim/data/fetch_revenue.py`, `data/*/revenue*`, `tests/test_revenue*.py` |
| expenditure | `belsim/fiscal/expenditure/`, `belsim/fiscal/balance.py`, `params/expenditure.yaml`, `belsim/data/fetch_expenditure.py`, `data/*/expenditure*`, `tests/test_expenditure*.py`, `tests/test_balance*.py` |
| behaviour | `belsim/behaviour/`, `params/behaviour.yaml`, `tests/test_behaviour*.py` |
| api+ui | `belsim/api/`, `web/`, `tests/test_api*.py` |

## Addenda after module hand-over
- `Change(op="set")` on a parameter with `uprate` is an amount in start-year euros and keeps
  being indexed after `start` (core/params.py).
- `state.adjust` tax-base factors (`cit_base_factor`, `capital_income_base_factor`,
  `self_employed_income_factor`, `property_transactions_factor`, `cons_factor`,
  `productivity_factor`) are applied by labour/macro when they build the bases; revenue must
  not apply them again.
- Extra labour keys used by expenditure: `labour["rcc_beneficiaries"]`,
  `labour["unemp_excluded_to_inactive"]`.
- `fiscal["net_expenditure_growth"]` is a fraction. `debt.<entity>` are published subsector
  debts; they differ from `debt` by `fiscal["debt.intragov_consolidation"]`.
- Never replace `state.extra` wholesale; use `state.extra["<module>"]`.

## Regional dimension (added later)
- `state.regional`: flat dict `"<metric>.<REGION>" -> float`, REGION in `FLA, WAL, BRU`
  (exported as series `regional.<metric>.<REGION>`). Never replace the dict; write your own
  keys (idempotent). Owners: `belsim.regional` (economy, labour market, regional governments),
  `belsim.fiscal.revenue` (keys starting `revenue.`), `belsim.fiscal.expenditure` may add
  `expenditure.` keys.
- Engine step 8b: `regional.run(state, prev, p, ctx)` after `balance.run` (re-run on pass 2).
- Policy parameters that are a regional competence are dicts keyed by region
  (`{FLA: .., WAL: .., BRU: ..}`), so a measure for one region is a `Change` with `key="WAL"`.

## Tax calculator (added later)
- `revenue.calc.tax_on_income(gross, p, kind, household=None)` also returns `allowance`,
  `reduction`, `state_tax`, `child_credit`, `marital_quotient`; `tax_household` and `tax_base`
  carry the household rules. Aggregate income tax sums `tax_household` over a grid of tax units
  (`belsim/fiscal/revenue/taxunits.py`); `belsim/distribution/` uses the same functions.
- `pit.regional_surcharge` and `pit.municipal_surcharge` are dicts by region: read them with
  `calc.surcharge_rates(p, region)` or `calc.by_region`.
- Parameters with constant, unchanged legislation are handed out without copying
  (`ParamSet.value`): never mutate a parameter value.
