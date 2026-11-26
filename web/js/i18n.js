// Interface language: catalogs web/i18n/<lang>.json (nested keys), interpolation, plurals.
//
//   t('budget.tree.title', { side, year })     "{side} in {year}"
//   tn('common.n_measures', n)                 {"one": "{n} measure", "other": "{n} measures"}
//
// A key missing in the active language falls back to English, then to the key itself, so a
// half-translated catalog never breaks a view. Text that comes from the model (line labels,
// parameter descriptions ...) is translated by the server: see `lang` in store.js `api()`.

export const LANGS = [
  { id: 'en', label: 'EN', name: 'English', locale: 'en-GB' },
  { id: 'fr', label: 'FR', name: 'Français', locale: 'fr-BE' },
  { id: 'nl', label: 'NL', name: 'Nederlands', locale: 'nl-BE' },
];
const LS = 'belsim.lang';
const catalogs = {};
let current = 'en';
let plural = null;

export const lang = () => current;
export const locale = () => LANGS.find(l => l.id === current).locale;

/** Stored choice, else the first browser language that is French or Dutch, else English. */
export function detectLang() {
  try {
    const saved = localStorage.getItem(LS);
    if (LANGS.some(l => l.id === saved)) return saved;
  } catch (e) { /* storage unavailable */ }
  const prefs = navigator.languages && navigator.languages.length ? navigator.languages : [navigator.language || 'en'];
  for (const p of prefs) {
    const code = String(p).toLowerCase().split('-')[0];
    if (code === 'fr' || code === 'nl') return code;
    if (code === 'en') return 'en';
  }
  return 'en';
}

async function load(id) {
  if (catalogs[id]) return;
  try {
    const resp = await fetch(`i18n/${id}.json`, { cache: 'no-cache' });
    catalogs[id] = resp.ok ? await resp.json() : {};
  } catch (e) {
    catalogs[id] = {};
  }
}

function apply(id) {
  current = id;
  plural = null;
  document.documentElement.lang = id;
}

export async function initI18n() {
  const id = detectLang();
  await Promise.all([load('en'), load(id)]);
  apply(id);
}

/** Switch language and remember it. The caller re-renders (and re-fetches model text). */
export async function setLang(id) {
  if (!LANGS.some(l => l.id === id)) return;
  await load(id);
  apply(id);
  try { localStorage.setItem(LS, id); } catch (e) { /* ignore */ }
}

function find(cat, key) {
  let node = cat;
  for (const part of key.split('.')) {
    if (node == null || typeof node !== 'object') return undefined;
    node = node[part];
  }
  return node;
}
const entry = key => {
  const v = find(catalogs[current], key);
  return v === undefined || v === '' ? find(catalogs.en, key) : v;
};
const fill = (text, vars) => (vars ? text.replace(/\{(\w+)\}/g, (m, k) => (vars[k] == null ? m : String(vars[k]))) : text);

/** True when the key exists (in the active language or in English). */
export const hasKey = key => entry(key) !== undefined;

export function t(key, vars) {
  let v = entry(key);
  if (v === undefined) return key;
  if (typeof v === 'object') v = v.other ?? Object.values(v)[0] ?? key;
  return fill(String(v), vars);
}

/** Plural form chosen by `n` (Intl.PluralRules); `{n}` is the count in the locale's format. */
export function tn(key, n, vars) {
  const v = entry(key);
  if (v === undefined) return key;
  let text = v;
  if (typeof v === 'object') {
    if (!plural) plural = new Intl.PluralRules(locale());
    text = v[n === 0 && v.zero != null ? 'zero' : plural.select(n)] ?? v.other ?? '';
  }
  return fill(String(text), { n: new Intl.NumberFormat(locale()).format(n), ...vars });
}

/** `t()` when the key exists, otherwise `fallback` (for labels keyed by a model code). */
export const tOr = (key, fallback, vars) => (hasKey(key) ? t(key, vars) : fallback);
