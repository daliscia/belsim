"""Fixed dimensions shared by every module. Do not change without updating docs/ARCHITECTURE.md."""

BASE_YEAR = 2024
END_YEAR = 2070

# Age: single years 0..99, last index (100) = 100 and over.
N_AGE = 101
AGES = list(range(N_AGE))

SEXES = ["M", "F"]
N_SEX = 2

# NUTS1: BE2 Flanders, BE3 Wallonia, BE1 Brussels.
REGIONS = ["FLA", "WAL", "BRU"]
REGION_NUTS1 = {"FLA": "BE2", "WAL": "BE3", "BRU": "BE1"}
N_REGION = 3

# Mutually exclusive socio-economic status; sums to population over the status axis.
STATUSES = [
    "child",          # 0-14
    "student",        # 15+ in education, not working
    "emp_private",    # private-sector employee (incl. cross-status flexi/student jobs counted by main status)
    "emp_public",     # general-government employee (statutory + contractual)
    "self_employed",  # self-employed as main activity (incl. helpers)
    "unemployed",     # receiving unemployment benefit / ILO unemployed
    "disabled",       # long-term sick / invalidity (> 1 year)
    "retired",        # pension or early-retirement (RCC/SWT) as main status
    "inactive",       # other inactive (home, social assistance, none)
]
N_STATUS = len(STATUSES)
S = {name: i for i, name in enumerate(STATUSES)}
EMPLOYED_STATUSES = ["emp_private", "emp_public", "self_employed"]

# Job types = the three employed statuses, same order.
JOBTYPES = ["emp_private", "emp_public", "self_employed"]
N_JOBTYPE = 3
J = {name: i for i, name in enumerate(JOBTYPES)}

# NACE Rev.2 aggregation (Eurostat nama_10_a64 codes in comments).
SECTORS = [
    "A",     # agriculture, forestry, fishing
    "BDE",   # mining, energy, water & waste (B, D, E)
    "C",     # manufacturing
    "F",     # construction
    "G",     # wholesale & retail trade
    "H",     # transport & storage
    "I",     # accommodation & food services
    "J",     # information & communication
    "K",     # finance & insurance
    "L",     # real estate
    "M",     # professional, scientific, technical
    "N",     # administrative & support services
    "O",     # public administration & defence
    "P",     # education
    "Q",     # health & social work
    "RSTU",  # arts, other services, households, extraterritorial
]
N_SECTOR = len(SECTORS)
SEC = {name: i for i, name in enumerate(SECTORS)}

N_DECILE = 10

# Household consumption categories (macro produces them, revenue taxes them).
CONS_CATS = [
    "food",            # food & non-alcoholic beverages
    "alcohol",
    "tobacco",
    "fuel",            # motor fuels
    "energy_home",     # electricity, gas, heating
    "housing",         # actual + imputed rents (mostly untaxed)
    "restaurants",     # restaurants & hotels
    "renovation",      # housing maintenance / construction services
    "other_goods",
    "other_services",
]

# General-government subsectors (ESA: S1311, S1314, S1312, S1313).
ENTITIES = ["federal", "social_security", "regions_communities", "local"]
ENTITY_ESA = {
    "federal": "S1311",
    "social_security": "S1314",
    "regions_communities": "S1312",
    "local": "S1313",
}
