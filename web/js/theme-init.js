// Loaded in <head> before the stylesheet paints (a plain script, not a module, and not inline:
// the Content-Security-Policy of the public mode allows no inline script).
// Light unless the visitor chose dark; the language of a previous visit, or ?lang=.
(function () {
  var root = document.documentElement;
  var theme = 'light';
  try {
    if (localStorage.getItem('belsim.theme') === 'dark') theme = 'dark';
    var q = /[?&]lang=(en|fr|nl)\b/.exec(location.search);
    var l = q ? q[1] : localStorage.getItem('belsim.lang');
    if (l === 'en' || l === 'fr' || l === 'nl') root.lang = l;
  } catch (e) { /* storage unavailable: defaults */ }
  root.dataset.theme = theme;
  root.classList.add('js');
})();
