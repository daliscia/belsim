// Hand-rolled SVG/HTML charts. Colours are CSS variables applied through `style`, so the
// light/dark theme switch needs no redraw. Every chart redraws itself when its width changes.
import { h, s, $ } from './dom.js';
import { t, locale } from './i18n.js';

// ------------------------------------------------------------------ shared machinery
function mount(el, draw) {
  let last = 0;
  const ro = new ResizeObserver(() => {
    const w = Math.floor(el.clientWidth);
    if (w > 0 && w !== last) { last = w; draw(w); }
  });
  ro.observe(el);
  return el;
}

const tipEl = () => $('#tip');
export function showTip(evt, head, rows) {
  const tip = tipEl();
  tip.replaceChildren(
    head ? h('div', { class: 'tip-h' }, head) : null,
    ...rows.map(r => h('div', { class: 'tip-r' },
      r.color ? h('span', { class: 'key-line' + (r.dash ? ' dash' : ''), style: { color: r.color } }) : h('span'),
      h('b', null, r.value), h('span', { class: 'nm' }, r.name))),
  );
  tip.hidden = false;
  const pad = 14, w = tip.offsetWidth, hh = tip.offsetHeight;
  let x = evt.clientX + pad, y = evt.clientY + pad;
  if (x + w > innerWidth - 8) x = evt.clientX - w - pad;
  if (y + hh > innerHeight - 8) y = evt.clientY - hh - pad;
  tip.style.left = Math.max(4, x) + 'px';
  tip.style.top = Math.max(4, y) + 'px';
}
export const hideTip = () => { tipEl().hidden = true; };

function niceTicks(lo, hi, n = 5) {
  if (!(hi > lo)) { const d = Math.abs(lo) || 1; lo -= d * 0.1; hi += d * 0.1; }
  const raw = (hi - lo) / n, mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(st => st >= raw) || raw;
  const t0 = Math.floor(lo / step) * step, out = [];
  for (let v = t0; v <= hi + step * 0.999; v += step) out.push(+v.toPrecision(12));
  return out;
}
function xTicks(xs, width) {
  const n = xs.length, maxTicks = Math.max(2, Math.floor(width / 56));
  const span = xs[n - 1] - xs[0];
  const step = [1, 2, 5, 10, 20, 25, 50].find(st => span / st <= maxTicks) || 50;
  return xs.filter(x => x % step === 0);
}
function extent(arrs) {
  let lo = Infinity, hi = -Infinity;
  for (const a of arrs) for (const v of a) if (v != null && isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v; }
  return lo === Infinity ? null : [lo, hi];
}
function pathOf(xs, ys, X, Y) {
  let d = '', pen = false;
  ys.forEach((v, i) => {
    if (v == null || !isFinite(v)) { pen = false; return; }
    d += `${pen ? 'L' : 'M'}${X(xs[i]).toFixed(1)},${Y(v).toFixed(1)}`;
    pen = true;
  });
  return d;
}
function axisTickFormat(ticks, fmt) {
  // fewest decimals that keep the ticks distinct
  const step = ticks.length > 1 ? Math.abs(ticks[1] - ticks[0]) : 1;
  const dec = step >= 1 ? 0 : Math.min(4, Math.ceil(-Math.log10(step)) + (/(25|75)$/.test(String(step)) ? 1 : 0));
  const nf = new Intl.NumberFormat(locale(), { minimumFractionDigits: dec, maximumFractionDigits: dec });
  return v => nf.format(v);
}

export function legend(items) {
  return h('div', { class: 'legend' }, items.map(it => h('span', { class: 'item' },
    it.box ? h('span', { class: 'key-box', style: { background: it.color } })
           : h('span', { class: 'key-line' + (it.dash ? ' dash' : ''), style: { color: it.color } }),
    it.name)));
}

function frame(el, width, height, m, xs, yTicks, Y, X, opts) {
  const svg = s('svg', { viewBox: `0 0 ${width} ${height}`, height, role: 'img', 'aria-label': opts.title || t('chart.aria') });
  const tf = axisTickFormat(yTicks, opts.fmt);
  for (const tk of yTicks) {
    const y = Y(tk);
    if (y < m.t - 1 || y > height - m.b + 1) continue;
    svg.append(
      s('line', { x1: m.l, x2: width - m.r, y1: y, y2: y, style: `stroke:${tk === 0 ? 'var(--axis)' : 'var(--grid)'};stroke-width:1` }),
      s('text', { x: m.l - 6, y: y + 3.5, 'text-anchor': 'end' }, tf(tk)));
  }
  // `xFmt` marks a continuous x axis (e.g. a wage); otherwise x is a list of years
  const xt = opts.xFmt ? niceTicks(xs[0], xs[xs.length - 1], Math.max(2, Math.floor((width - m.l - m.r) / 90))).filter(v => v >= xs[0] && v <= xs[xs.length - 1])
    : xTicks(xs, width - m.l - m.r);
  for (const x of xt) {
    svg.append(s('text', { x: X(x), y: height - m.b + 15, 'text-anchor': 'middle' }, opts.xFmt ? opts.xFmt(x) : x));
  }
  svg.append(s('line', { x1: m.l, x2: width - m.r, y1: height - m.b, y2: height - m.b, style: 'stroke:var(--axis);stroke-width:1' }));
  return svg;
}

function hover(svg, m, width, height, xs, X, onIndex, onPick) {
  const cross = s('line', { y1: m.t, y2: height - m.b, style: 'stroke:var(--ink-3);stroke-width:1', visibility: 'hidden' });
  const dots = s('g');
  const hit = s('rect', { x: m.l, y: m.t, width: Math.max(0, width - m.l - m.r), height: Math.max(0, height - m.t - m.b), style: 'fill:transparent;cursor:crosshair' });
  const idx = evt => {
    const r = svg.getBoundingClientRect();
    const px = (evt.clientX - r.left) * (width / r.width);
    let best = 0, bd = Infinity;
    xs.forEach((x, i) => { const d = Math.abs(X(x) - px); if (d < bd) { bd = d; best = i; } });
    return best;
  };
  hit.addEventListener('pointermove', evt => {
    const i = idx(evt);
    cross.setAttribute('x1', X(xs[i])); cross.setAttribute('x2', X(xs[i]));
    cross.setAttribute('visibility', 'visible');
    dots.replaceChildren(...onIndex(i, evt));
  });
  hit.addEventListener('pointerleave', () => { cross.setAttribute('visibility', 'hidden'); dots.replaceChildren(); hideTip(); });
  if (onPick) hit.addEventListener('click', evt => onPick(xs[idx(evt)]));
  svg.append(cross, dots, hit);
}

function markX(svg, m, height, X, x) {
  svg.append(s('line', { x1: X(x), x2: X(x), y1: m.t, y2: height - m.b, style: 'stroke:var(--flag);stroke-width:2;opacity:.9' }));
}

// ------------------------------------------------------------------ line chart
/**
 * opts: { x:[...], series:[{name, values, color, dash, width}], refs:[{value,label}], fmt, height,
 *         mark, onPick, zero, title, xFmt, xPrefix, xTip: x => tooltip heading,
 *         band: {lo:[...], hi:[...], color, name}  shaded range drawn under the lines }
 */
export function lineChart(opts) {
  const series = opts.series.filter(se => se.values && se.values.some(v => v != null));
  const el = h('div', { class: 'chart' });
  if (!series.length) return h('div', { class: 'empty' }, opts.emptyText || t('chart.empty_series'));
  const shown = series.filter(se => !se.twin);
  const items = series.some(se => se.twin) ? [...shown, { name: t('chart.dashed_baseline'), color: 'var(--ink-2)', dash: true }] : shown.slice();
  const band = opts.band && opts.band.lo && opts.band.hi ? opts.band : null;
  if (band) items.push({ name: band.name, color: `color-mix(in srgb, ${band.color} 30%, var(--surface))`, box: true });
  const wrap = h('div', null, series.length > 1 || opts.forceLegend || band ? legend(items) : null, el);
  const height = opts.height || 210, fmt = opts.fmt || (v => String(v)), xs = opts.x;
  mount(el, width => {
    const m = { l: 46, r: 12, t: 8, b: 22 };
    let [lo, hi] = extent(series.map(se => se.values).concat(band ? [band.lo, band.hi] : []));
    for (const r of opts.refs || []) {
      // show a reference line only if it is within reach of the data (never let it flatten the series)
      const span = Math.max(hi - lo, Math.abs(hi) * 0.05, 1e-9);
      if (r.value > hi && r.value - hi < span * 1.5) hi = r.value;
      if (r.value < lo && lo - r.value < span * 1.5) lo = r.value;
    }
    if (opts.zero) { lo = Math.min(lo, 0); hi = Math.max(hi, 0); }
    const padY = (hi - lo || Math.abs(hi) || 1) * 0.06;
    // a zero-based axis starts or ends exactly at zero when the data stay on one side of it
    const ticks = niceTicks(opts.zero && lo >= 0 ? 0 : lo - padY, opts.zero && hi <= 0 ? 0 : hi + padY, Math.max(3, Math.round(height / 48)));
    const y0 = ticks[0], y1 = ticks[ticks.length - 1];
    const X = x => m.l + ((x - xs[0]) / (xs[xs.length - 1] - xs[0] || 1)) * (width - m.l - m.r);
    const Y = v => height - m.b - ((v - y0) / (y1 - y0 || 1)) * (height - m.t - m.b);
    const svg = frame(el, width, height, m, xs, ticks, Y, X, opts);
    for (const r of opts.refs || []) {
      if (r.value < y0 || r.value > y1) continue;
      svg.append(
        s('line', { x1: m.l, x2: width - m.r, y1: Y(r.value), y2: Y(r.value), style: 'stroke:var(--bad);stroke-width:1;opacity:.75' }),
        s('text', { class: 'ref-label', x: width - m.r - 2, y: Y(r.value) - 4, 'text-anchor': 'end' }, r.label));
    }
    if (band) {
      // one polygon per run of years where both edges exist
      let seg = [];
      const flush = () => {
        if (seg.length > 1) {
          const d = seg.map((i, j) => `${j ? 'L' : 'M'}${X(xs[i]).toFixed(1)},${Y(band.hi[i]).toFixed(1)}`).join('')
            + seg.slice().reverse().map(i => `L${X(xs[i]).toFixed(1)},${Y(band.lo[i]).toFixed(1)}`).join('') + 'Z';
          svg.append(s('path', { d, style: `fill:${band.color};opacity:.2;stroke:none` }));
        }
        seg = [];
      };
      xs.forEach((x, i) => { if (band.lo[i] != null && band.hi[i] != null) seg.push(i); else flush(); });
      flush();
    }
    if (opts.mark != null && xs.includes(opts.mark)) markX(svg, m, height, X, opts.mark);
    for (const se of series) {
      svg.append(s('path', {
        d: pathOf(xs, se.values, X, Y),
        style: `fill:none;stroke:${se.color};stroke-width:${se.width || 2};stroke-linejoin:round;stroke-linecap:round;${se.dash ? 'stroke-dasharray:5 4' : ''}`,
      }));
    }
    hover(svg, m, width, height, xs, X, (i, evt) => {
      const rows = series.filter(se => se.values[i] != null)
        .map(se => ({ color: se.color, dash: se.dash, value: fmt(se.values[i]), name: se.name }));
      if (band && band.lo[i] != null && band.hi[i] != null) {
        const a = Math.min(band.lo[i], band.hi[i]), z = Math.max(band.lo[i], band.hi[i]);
        rows.push({ value: t('chart.range', { lo: fmt(a), hi: fmt(z) }), name: band.name });
      }
      showTip(evt, opts.xTip ? opts.xTip(xs[i]) : (opts.xPrefix || '') + (opts.xFmt ? opts.xFmt(xs[i]) : String(xs[i])), rows);
      return series.filter(se => se.values[i] != null).map(se =>
        s('circle', { cx: X(xs[i]), cy: Y(se.values[i]), r: 4, style: `fill:${se.color};stroke:var(--surface);stroke-width:2` }));
    }, opts.onPick);
    el.replaceChildren(svg);
  });
  return wrap;
}

// ------------------------------------------------------------------ stacked area
/** opts: { x, layers:[{name, values, color}], fmt, height, mark, onPick, outline:{name, values}, totalName,
 *          xPrefix, xTip: x => tooltip heading } */
export function stackChart(opts) {
  const layers = opts.layers.filter(l => l.values && l.values.some(v => v != null && v !== 0));
  if (!layers.length) return h('div', { class: 'empty' }, opts.emptyText || t('chart.empty_series_plural'));
  const el = h('div', { class: 'chart' });
  const items = layers.map(l => ({ ...l, box: true }));
  if (opts.outline) items.push({ name: opts.outline.name, color: 'var(--baseline)', dash: true });
  const wrap = h('div', null, legend(items), el);
  const height = opts.height || 230, fmt = opts.fmt || (v => String(v)), xs = opts.x;
  mount(el, width => {
    const m = { l: 46, r: 12, t: 8, b: 22 };
    const n = xs.length, tops = [], base = new Array(n).fill(0);
    for (const l of layers) {
      const lo = base.slice();
      l.values.forEach((v, i) => { base[i] += Math.max(0, v || 0); });
      tops.push({ l, lo, hi: base.slice() });
    }
    let max = Math.max(...base, 1e-9);
    if (opts.outline) max = Math.max(max, ...opts.outline.values.map(v => v || 0));
    const ticks = niceTicks(0, max * 1.02, Math.max(3, Math.round(height / 48)));
    const y1 = ticks[ticks.length - 1];
    const X = x => m.l + ((x - xs[0]) / (xs[n - 1] - xs[0] || 1)) * (width - m.l - m.r);
    const Y = v => height - m.b - (v / y1) * (height - m.t - m.b);
    const svg = frame(el, width, height, m, xs, ticks, Y, X, opts);
    for (const { l, lo, hi } of tops) {
      let d = '';
      for (let i = 0; i < n; i++) d += `${i ? 'L' : 'M'}${X(xs[i]).toFixed(1)},${Y(hi[i]).toFixed(1)}`;
      for (let i = n - 1; i >= 0; i--) d += `L${X(xs[i]).toFixed(1)},${Y(lo[i]).toFixed(1)}`;
      svg.append(s('path', { d: d + 'Z', style: `fill:${l.color};stroke:var(--surface);stroke-width:1.5;stroke-linejoin:round` }));
    }
    if (opts.outline) {
      svg.append(s('path', { d: pathOf(xs, opts.outline.values, X, Y), style: 'fill:none;stroke:var(--baseline);stroke-width:1.5;stroke-dasharray:5 4' }));
    }
    if (opts.mark != null && xs.includes(opts.mark)) markX(svg, m, height, X, opts.mark);
    hover(svg, m, width, height, xs, X, (i, evt) => {
      const rows = [...tops].reverse().map(({ l }) => ({ color: l.color, value: fmt(l.values[i]), name: l.name }));
      rows.push({ value: fmt(base[i]), name: opts.totalName || t('common.total') });
      if (opts.outline) rows.push({ color: 'var(--baseline)', dash: true, value: fmt(opts.outline.values[i]), name: opts.outline.name });
      showTip(evt, opts.xTip ? opts.xTip(xs[i]) : (opts.xPrefix || '') + xs[i], rows);
      return [];
    }, opts.onPick);
    el.replaceChildren(svg);
  });
  return wrap;
}

// ------------------------------------------------------------------ horizontal bars (HTML)
/**
 * rows: [{label, value, base, parts:[{name,value,color}], title}]. `base` draws a baseline tick.
 * opts: { fmt, deltaFmt, plain }
 */
export function hbars(rows, opts = {}) {
  rows = rows.filter(r => r.value != null);
  if (!rows.length) return h('div', { class: 'empty' }, opts.emptyText || t('chart.empty_values'));
  const fmt = opts.fmt || (v => String(v));
  const vals = rows.flatMap(r => [r.value, r.base == null ? 0 : r.base]);
  const lo = Math.min(0, ...vals), hi = Math.max(0, ...vals, 1e-12), span = hi - lo;
  const pos = v => ((v - lo) / span) * 100;
  const el = h('div', { class: 'hbars' });
  for (const r of rows) {
    const track = h('div', { class: 'track' });
    if (lo < 0) track.append(h('div', { class: 'zero', style: { left: pos(0) + '%' } }));
    if (r.parts) {
      let acc = 0;
      r.parts.forEach((p, i) => {
        const w = (Math.max(0, p.value) / span) * 100;
        if (w > 0) {
          track.append(h('div', {
            class: 'seg-part',
            style: { left: `calc(${pos(acc)}% + ${i ? 2 : 0}px)`, width: `max(1px, calc(${w}% - ${i ? 2 : 0}px))`, background: p.color,
                     borderRadius: i === r.parts.length - 1 ? '0 3px 3px 0' : '0' },
          }));
        }
        acc += Math.max(0, p.value);
      });
    } else {
      const a = pos(Math.min(0, r.value)), b = pos(Math.max(0, r.value));
      track.append(h('div', {
        class: 'bar' + (r.value < 0 ? ' neg' : '') + (opts.plain ? ' plain' : ''),
        style: { left: a + '%', width: `max(1px, ${b - a}%)`, background: r.color || null },
      }));
    }
    if (r.base != null && Math.abs(r.base - r.value) > 0.002 * span) {
      track.append(h('div', { class: 'tick', style: { left: `calc(${pos(r.base)}% - 1px)` } }));
    }
    const tipRows = r.parts
      ? r.parts.map(p => ({ color: p.color, value: fmt(p.value), name: p.name })).concat([{ value: fmt(r.value), name: t('common.total') }])
      : [{ color: r.color || (opts.plain ? 'var(--baseline)' : 'var(--scenario)'), value: fmt(r.value), name: opts.valueName || t('common.value') }];
    if (r.base != null) tipRows.push({ color: 'var(--baseline)', value: fmt(r.base), name: t('common.baseline') });
    const move = evt => showTip(evt, r.title || r.label, tipRows);
    const lab = h('div', { class: 'lab', title: r.title || r.label }, r.label);
    const val = h('div', { class: 'val' }, fmt(r.value), r.delta || null);
    for (const node of [lab, track, val]) { node.addEventListener('pointermove', move); node.addEventListener('pointerleave', hideTip); }
    el.append(lab, track, val);
  }
  return el;
}

// ------------------------------------------------------------------ grouped columns
/** opts: { cats:[...], series:[{name, values, color}], fmt, height, base:[{values}], tip:(ci, si) => rows,
 *          catPrefix, catTip: cat => tooltip heading }
 *  Values may be negative: bars grow from the zero line in either direction. */
export function columnChart(opts) {
  const series = opts.series.filter(se => se.values && se.values.some(v => v != null));
  if (!series.length) return h('div', { class: 'empty' }, t('chart.empty_values'));
  const el = h('div', { class: 'chart' });
  const wrap = h('div', null, series.length > 1 ? legend(series.map(x => ({ ...x, box: true }))) : null, el);
  const height = opts.height || 220, fmt = opts.fmt || (v => String(v));
  mount(el, width => {
    const m = { l: 52, r: 8, t: 8, b: 22 };
    const all = [...series.flatMap(se => se.values.map(v => v || 0)), ...(opts.base || []).flatMap(b => b.values.map(v => v || 0))];
    const max = Math.max(...all, 0), min = Math.min(...all, 0);
    const pad = (max - min || 1) * 0.02;
    const ticks = niceTicks(min < 0 ? min - pad : 0, max > 0 ? max + pad : (min < 0 ? 0 : 1e-9), 4);
    const y0 = ticks[0], y1 = ticks[ticks.length - 1];
    const Y = v => height - m.b - ((v - y0) / (y1 - y0 || 1)) * (height - m.t - m.b);
    const svg = s('svg', { viewBox: `0 0 ${width} ${height}`, height, role: 'img', 'aria-label': opts.title || t('chart.aria') });
    const tf = axisTickFormat(ticks);
    for (const tk of ticks) {
      svg.append(s('line', { x1: m.l, x2: width - m.r, y1: Y(tk), y2: Y(tk), style: `stroke:${tk === 0 ? 'var(--axis)' : 'var(--grid)'}` }),
        s('text', { x: m.l - 6, y: Y(tk) + 3.5, 'text-anchor': 'end' }, tf(tk)));
    }
    const band = (width - m.l - m.r) / opts.cats.length;
    const bw = Math.max(2, Math.min(opts.barWidth || 24, (band - 8) / series.length - 2));
    opts.cats.forEach((c, ci) => {
      const x0 = m.l + ci * band + (band - series.length * (bw + 2) + 2) / 2;
      svg.append(s('text', { x: m.l + ci * band + band / 2, y: height - m.b + 15, 'text-anchor': 'middle' }, c));
      series.forEach((se, si) => {
        const v = se.values[ci];
        if (v == null) return;
        // rounded data-end, square at the zero line; `sg` flips the corner for negative values
        const x = x0 + si * (bw + 2), y = Y(v), sg = v < 0 ? -1 : 1, hh = Math.abs(Y(0) - y), r = Math.min(3, bw / 2, hh);
        const d = `M${x},${Y(0)}V${y + sg * r}Q${x},${y} ${x + r},${y}H${x + bw - r}Q${x + bw},${y} ${x + bw},${y + sg * r}V${Y(0)}Z`;
        const bar = s('path', { d, style: `fill:${se.color}` });
        const b = opts.base && opts.base[si] ? opts.base[si].values[ci] : null;
        const rows = opts.tip ? opts.tip(ci, si) : [{ color: se.color, value: fmt(v), name: se.name }];
        if (b != null) rows.push({ color: 'var(--baseline)', value: fmt(b), name: t('common.baseline') });
        // hit target wider and taller than a thin bar
        const hit = s('rect', { x: x - 1, y: Math.min(y, Y(0)) - 6, width: bw + 2, height: hh + 12, style: 'fill:transparent' });
        hit.addEventListener('pointermove', evt => showTip(evt, opts.catTip ? opts.catTip(c) : (opts.catPrefix || '') + c, rows));
        hit.addEventListener('pointerleave', hideTip);
        svg.append(bar, hit);
        if (b != null && Math.abs(b - v) > 1e-9 * Math.max(1, Math.abs(b))) {
          svg.append(s('line', { x1: x - 1, x2: x + bw + 1, y1: Y(b), y2: Y(b), style: 'stroke:var(--baseline);stroke-width:2' }));
        }
      });
    });
    el.replaceChildren(svg);
  });
  return wrap;
}

// ------------------------------------------------------------------ population pyramid
/** opts: { ages, male, female, baseMale, baseFemale, max, fmt, labels:{M,F} } */
export function pyramid(opts) {
  const el = h('div', { class: 'chart' });
  const hasBase = !!opts.baseMale;
  const wrap = h('div', null, legend([
    { name: opts.labels.M, color: 'var(--c1)', box: true },
    { name: opts.labels.F, color: 'var(--c2)', box: true },
    ...(hasBase ? [{ name: t('common.baseline'), color: 'var(--baseline)' }] : []),
  ]), el);
  const fmt = opts.fmt || (v => String(v));
  mount(el, width => {
    const n = opts.ages.length, rowH = 4, m = { l: 8, r: 8, t: 4, b: 22 }, mid = 30;
    const height = m.t + n * rowH + m.b;
    const half = (width - m.l - m.r - mid) / 2;
    const max = opts.max || Math.max(...opts.male, ...opts.female, 1);
    const ticks = niceTicks(0, max, 3).filter(tk => tk <= max * 1.001);
    const cx = m.l + half + mid / 2;
    const L = v => cx - mid / 2 - (v / max) * half, R = v => cx + mid / 2 + (v / max) * half;
    const yOf = a => m.t + (n - 1 - a) * rowH;
    const svg = s('svg', { viewBox: `0 0 ${width} ${height}`, height, role: 'img', 'aria-label': t('chart.pyramid_aria') });
    for (const tk of ticks) {
      for (const x of tk === 0 ? [L(0), R(0)] : [L(tk), R(tk)]) {
        svg.append(s('line', { x1: x, x2: x, y1: m.t, y2: height - m.b, style: `stroke:${tk === 0 ? 'var(--axis)' : 'var(--grid)'}` }),
          s('text', { x, y: height - m.b + 14, 'text-anchor': 'middle' }, fmt(tk)));
      }
    }
    for (let a = 0; a < n; a += 10) svg.append(s('text', { x: cx, y: yOf(a) + rowH / 2 + 3.5, 'text-anchor': 'middle' }, a === n - 1 ? a + '+' : a));
    let dm = '', df = '';
    for (let a = 0; a < n; a++) {
      const y = yOf(a), hh = rowH - 0.6;
      dm += `M${L(0)},${y}H${L(opts.male[a]).toFixed(1)}v${hh}H${L(0)}Z`;
      df += `M${R(0)},${y}H${R(opts.female[a]).toFixed(1)}v${hh}H${R(0)}Z`;
    }
    svg.append(s('path', { d: dm, style: 'fill:var(--c1)' }), s('path', { d: df, style: 'fill:var(--c2)' }));
    if (hasBase) {
      const step = (vals, F) => {
        let d = `M${F(vals[0]).toFixed(1)},${yOf(0) + rowH}`;
        for (let a = 0; a < n; a++) d += `V${yOf(a)}H${F(vals[Math.min(n - 1, a + 1)]).toFixed(1)}`;
        return d;
      };
      svg.append(s('path', { d: step(opts.baseMale, L), style: 'fill:none;stroke:var(--baseline);stroke-width:1.25' }),
        s('path', { d: step(opts.baseFemale, R), style: 'fill:none;stroke:var(--baseline);stroke-width:1.25' }));
    }
    const band = s('rect', { x: m.l, width: width - m.l - m.r, height: rowH, style: 'fill:var(--ink);opacity:.12', visibility: 'hidden' });
    const hit = s('rect', { x: m.l, y: m.t, width: width - m.l - m.r, height: n * rowH, style: 'fill:transparent' });
    hit.addEventListener('pointermove', evt => {
      const r = svg.getBoundingClientRect();
      const a = Math.max(0, Math.min(n - 1, n - 1 - Math.floor(((evt.clientY - r.top) * (height / r.height) - m.t) / rowH)));
      band.setAttribute('y', yOf(a)); band.setAttribute('visibility', 'visible');
      const rows = [
        { color: 'var(--c1)', value: fmt(opts.male[a]), name: opts.labels.M },
        { color: 'var(--c2)', value: fmt(opts.female[a]), name: opts.labels.F },
      ];
      if (hasBase) rows.push({ color: 'var(--baseline)', value: fmt(opts.baseMale[a]) + ' / ' + fmt(opts.baseFemale[a]), name: t('common.baseline') });
      showTip(evt, t('chart.age', { age: a === n - 1 ? a + '+' : a }), rows);
    });
    hit.addEventListener('pointerleave', () => { band.setAttribute('visibility', 'hidden'); hideTip(); });
    svg.append(band, hit);
    el.replaceChildren(svg);
  });
  return wrap;
}

// ------------------------------------------------------------------ sparkline (param browser)
export function spark(values, width = 120, height = 26) {
  const ext = extent([values]);
  if (!ext) return null;
  const [lo, hi] = ext, n = values.length;
  const X = i => 1 + (i / (n - 1 || 1)) * (width - 2);
  const Y = v => height - 2 - ((v - lo) / (hi - lo || 1)) * (height - 4);
  let d = '';
  values.forEach((v, i) => { if (v != null) d += `${d ? 'L' : 'M'}${X(i).toFixed(1)},${Y(v).toFixed(1)}`; });
  return s('svg', { viewBox: `0 0 ${width} ${height}`, width, height, 'aria-hidden': 'true' },
    s('path', { d, style: 'fill:none;stroke:var(--baseline);stroke-width:1.5' }));
}
