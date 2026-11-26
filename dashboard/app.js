'use strict';
/* VARUNA dashboard. Plain JavaScript and canvas: no libraries, no internet, no map tiles.
   Every map and number is drawn from a live model run made by dashboard/backend.py.
   Grid arrays keep the model's layout: index = lat_row * 36 + lon_col, row 0 is the south edge (6.5 N). */

const NC = 6, NLAT = 32, NLON = 36, CELLS = NLAT * NLON, STEP = NC * CELLS;
const $ = (id) => document.getElementById(id);
const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

// name, lat, lon, label side on the map ('' = used only for "near ..." text)
const CITIES = [
  ['Delhi', 28.6, 77.2, 'r'], ['Mumbai', 19.1, 72.9, 'l'], ['Kolkata', 22.6, 88.4, 'r'], ['Chennai', 13.1, 80.3, 'r'],
  ['Bengaluru', 13.0, 77.6, 'l'], ['Hyderabad', 17.4, 78.5, 'r'], ['Ahmedabad', 23.0, 72.6, 'l'], ['Guwahati', 26.1, 91.7, 'r'],
  ['Kochi', 10.0, 76.3, ''], ['Bhopal', 23.3, 77.4, ''], ['Lucknow', 26.8, 80.9, ''], ['Srinagar', 34.1, 74.8, ''],
  ['Nagpur', 21.1, 79.1, ''], ['Patna', 25.6, 85.1, ''], ['Jaipur', 26.9, 75.8, ''], ['Bhubaneswar', 20.3, 85.8, ''],
  ['Visakhapatnam', 17.7, 83.2, ''], ['Goa', 15.5, 73.8, ''], ['Thiruvananthapuram', 8.5, 76.9, ''], ['Leh', 34.2, 77.6, ''],
  ['Dhaka', 23.8, 90.4, ''], ['Karachi', 24.9, 67.0, ''], ['Colombo', 6.9, 79.9, ''], ['Port Blair', 11.6, 92.7, ''],
  ['Kathmandu', 27.7, 85.3, ''], ['Lhasa', 29.7, 91.1, ''], ['Yangon', 16.8, 96.2, ''],
];

/* Colour. Amount fields use one hue from dark (low) to light (high). Fields with a natural middle, and every
   difference map, use two hues around a neutral middle. Nine colours each, one per break. */
const PAL = {
  rain: ['#0e1726', '#10294a', '#123f73', '#17599c', '#2377c2', '#4697dd', '#7bb8ee', '#b8dafa', '#f1f8ff'],
  wind: ['#0e1726', '#0f3338', '#11504c', '#167063', '#22917a', '#43b391', '#7bd2ab', '#b9ecca', '#effcf2'],
  wet: ['#0e1726', '#221c4c', '#382a78', '#5140a6', '#6d5ecf', '#8e85e6', '#b2aff2', '#d6d5fa', '#f5f5ff'],
  temp: ['#1e3a8a', '#2f66c4', '#5d9be0', '#a5c8ea', '#e9e4d8', '#f2c38a', '#ee9150', '#dc5a33', '#a8202b'],
  pres: ['#5b1a8c', '#7f3fb5', '#a877d6', '#d0b8ea', '#e9e4d8', '#f0cf9e', '#e9a85a', '#d97b23', '#a85208'],
  dWarm: ['#9fd4ff', '#4f9ee6', '#2a69b3', '#1b3c6b', '#161d2b', '#6e2a2c', '#b53e3c', '#e6604f', '#ffb0a0'],
  dWet: ['#f6d49a', '#d39d4c', '#9c6c28', '#5c411c', '#161d2b', '#16493f', '#1f7c69', '#31b799', '#98f3da'],
};

function hex(h) { return [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)]; }
function makeScale(breaks, colors) { return { breaks, rgb: colors.map(hex) }; }
function diffScale(m, pal) { return makeScale([-1, -0.5, -0.2, -0.05, 0, 0.05, 0.2, 0.5, 1].map((x) => x * m), pal); }

// get(array, time offset, cell) returns the value in display units
const FIELDS = [
  { key: 'tp', name: 'Rain', unit: 'mm/day', breaks: [0, 1, 2.5, 5, 10, 20, 40, 80, 160], pal: 'rain', dpal: 'dWet', floor: 2, notice: 0.5,
    get: (a, o, p) => a[o + CELLS + p] },
  { key: 't2m', name: 'Temperature', unit: '°C', breaks: [-20, -5, 5, 14, 22, 28, 33, 38, 45], pal: 'temp', dpal: 'dWarm', floor: 1, notice: 0.1,
    get: (a, o, p) => a[o + p] - 273.15 },
  { key: 'wind', name: 'Wind speed', unit: 'm/s', breaks: [0, 2, 4, 6, 8, 10, 13, 16, 20], pal: 'wind', dpal: 'dWarm', floor: 1, notice: 0.2,
    get: (a, o, p) => Math.hypot(a[o + 2 * CELLS + p], a[o + 3 * CELLS + p]) },
  { key: 'msl', name: 'Pressure', unit: 'hPa', breaks: [984, 992, 998, 1003, 1008, 1012, 1016, 1022, 1030], pal: 'pres', dpal: 'dWarm', floor: 1, notice: 0.2,
    get: (a, o, p) => a[o + 4 * CELLS + p] / 100 },
  { key: 'tcwv', name: 'Moisture', unit: 'kg/m²', breaks: [0, 10, 20, 30, 40, 50, 60, 70, 80], pal: 'wet', dpal: 'dWet', floor: 2, notice: 0.5,
    get: (a, o, p) => a[o + 5 * CELLS + p] },
  { key: 'u10', name: 'Wind to the east', unit: 'm/s', breaks: [-16, -10, -6, -3, 0, 3, 6, 10, 16], pal: 'temp', dpal: 'dWarm', floor: 1, notice: 0.2,
    get: (a, o, p) => a[o + 2 * CELLS + p] },
  { key: 'v10', name: 'Wind to the north', unit: 'm/s', breaks: [-16, -10, -6, -3, 0, 3, 6, 10, 16], pal: 'temp', dpal: 'dWarm', floor: 1, notice: 0.2,
    get: (a, o, p) => a[o + 3 * CELLS + p] },
];
for (const f of FIELDS) f.scale = makeScale(f.breaks, PAL[f.pal]);

// The change the user makes: delta = region * (add + mul * today's value), per model field.
const KNOBS = [
  { id: 'kT', name: 'Air temperature', unit: '°C', min: -6, max: 6, step: 0.5, set: (v) => { S.add[0] = v; } },
  { id: 'kQ', name: 'Moisture in the air', unit: '%', min: -60, max: 60, step: 5, set: (v) => { S.mul[5] = v / 100; } },
  { id: 'kW', name: 'Wind strength', unit: '%', min: -100, max: 100, step: 5, set: (v) => { S.mul[2] = v / 100; S.mul[3] = v / 100; } },
  { id: 'kP', name: 'Air pressure', unit: 'hPa', min: -15, max: 15, step: 1, set: (v) => { S.add[4] = v * 100; } },
  { id: 'kR', name: 'Rain on the start day', unit: '%', min: -100, max: 200, step: 10, set: (v) => { S.mul[1] = v / 100; } },
];
const OFFICIAL = {
  warm: { box: [8, 20, 82, 94], knobs: { kT: 3, kQ: 15 } },       // same numbers as the organizers' warm_ocean shock
  stall: { box: [8, 20, 65, 75], knobs: { kW: -40 } },            // same numbers as the organizers' monsoon_stall shock
};

const S = {
  meta: null, date: null, day: 3, field: FIELDS[0], tab: 'forecast', run: null, seq: 0,
  mask: new Float32Array(CELLS), count: 0, dist: new Float32Array(CELLS).fill(99), maskSegs: [],
  add: [0, 0, 0, 0, 0, 0], mul: [0, 0, 0, 0, 0, 0], knob: {}, official: null,
  tool: 'look', brush: 2, stroke: null, smooth: true, arrows: false, sel: { i: 13, j: 7 }, playing: null, proof: null,
};
const PANELS = {};
let COAST = [], runTimer = null;

/* ---------- small helpers ---------- */
function dec(b64) {
  const s = atob(b64), u = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i);
  return new Float32Array(u.buffer);
}
function nice(x) { const e = Math.pow(10, Math.floor(Math.log10(x))), m = x / e; return (m <= 1 ? 1 : m <= 2 ? 2 : m <= 5 ? 5 : 10) * e; }
function fmt(v, d = 1) { const z = Math.abs(v) < Math.pow(10, -d) / 2 ? 0 : v; return z.toFixed(d).replace('-', '−'); }
function signed(v, d = 1) { const z = Math.abs(v) < Math.pow(10, -d) / 2 ? 0 : v; return (z > 0 ? '+' : z < 0 ? '−' : '') + Math.abs(z).toFixed(d); }
function tick(v) { const a = Math.abs(v); return (v < 0 ? '−' : '') + (a >= 10 ? a.toFixed(0) : a >= 1 ? String(+a.toFixed(1)) : String(+a.toFixed(2))); }
function signedTick(v) { return (v > 0 ? '+' : '') + tick(v); }
function dayDate(k) {
  const [y, m, d] = S.date.split('-').map(Number), t = new Date(Date.UTC(y, m - 1, d + k));
  return t.getUTCDate() + ' ' + MONTHS[t.getUTCMonth()] + ' ' + t.getUTCFullYear();
}
function dayText(k) { return k === 0 ? 'start day' : 'day +' + k; }
function latOf(i) { return 6.5 + i; }
function lonOf(j) { return 65.5 + j; }
function place(i, j) {
  const la = latOf(i), lo = lonOf(j);
  let best = null, bd = 2.6;
  for (const c of CITIES) { const d = Math.hypot(c[1] - la, (c[2] - lo) * Math.cos(la * Math.PI / 180)); if (d < bd) { bd = d; best = c[0]; } }
  return { coords: la.toFixed(1) + '°N ' + lo.toFixed(1) + '°E', near: best };
}
function placeText(i, j) { const p = place(i, j); return p.near ? 'near ' + p.near : p.coords; }
function banner(msg) { const b = $('banner'); b.hidden = !msg; b.textContent = msg || ''; }

/* ---------- contours (marching squares on a grid padded by one ring) ---------- */
function pad(g, zero) {
  const R = NLAT + 2, C = NLON + 2, o = new Float32Array(R * C);
  for (let r = 0; r < R; r++) for (let c = 0; c < C; c++) {
    const i = r - 1, j = c - 1, inside = i >= 0 && i < NLAT && j >= 0 && j < NLON;
    o[r * C + c] = inside ? g[i * NLON + j] : zero ? 0 : g[Math.min(NLAT - 1, Math.max(0, i)) * NLON + Math.min(NLON - 1, Math.max(0, j))];
  }
  return o;
}
function contour(g, lev) {
  const R = NLAT + 2, C = NLON + 2, out = [];
  for (let r = 0; r < R - 1; r++) for (let c = 0; c < C - 1; c++) {
    const a = g[r * C + c], b = g[r * C + c + 1], cc = g[(r + 1) * C + c + 1], d = g[(r + 1) * C + c];
    const k = (a > lev ? 1 : 0) | (b > lev ? 2 : 0) | (cc > lev ? 4 : 0) | (d > lev ? 8 : 0);
    if (k === 0 || k === 15) continue;
    const AB = () => [c + (lev - a) / (b - a), r], BC = () => [c + 1, r + (lev - b) / (cc - b)];
    const DC = () => [c + (lev - d) / (cc - d), r + 1], AD = () => [c, r + (lev - a) / (d - a)];
    const seg = (p, q) => out.push(p[0], p[1], q[0], q[1]);
    switch (k) {
      case 1: case 14: seg(AD(), AB()); break;
      case 2: case 13: seg(AB(), BC()); break;
      case 3: case 12: seg(AD(), BC()); break;
      case 4: case 11: seg(BC(), DC()); break;
      case 6: case 9: seg(AB(), DC()); break;
      case 7: case 8: seg(AD(), DC()); break;
      case 5: seg(AD(), AB()); seg(BC(), DC()); break;
      case 10: seg(AB(), BC()); seg(AD(), DC()); break;
    }
  }
  return out;
}
function pathSegs(ctx, segs, sx, sy) {
  ctx.beginPath();
  for (let k = 0; k < segs.length; k += 4) {
    ctx.moveTo((segs[k] - 0.5) * sx, (NLAT - segs[k + 1] + 0.5) * sy);
    ctx.lineTo((segs[k + 2] - 0.5) * sx, (NLAT - segs[k + 3] + 0.5) * sy);
  }
}
function haloLine(ctx, segs, sx, sy, color, width, dash) {
  if (!segs.length) return;
  pathSegs(ctx, segs, sx, sy);
  ctx.lineCap = 'round'; ctx.setLineDash([]);
  ctx.strokeStyle = 'rgba(6,10,18,0.7)'; ctx.lineWidth = width + 2; ctx.stroke();
  ctx.setLineDash(dash || []); ctx.strokeStyle = color; ctx.lineWidth = width; ctx.stroke();
  ctx.setLineDash([]);
}
function haloText(ctx, txt, x, y, align) {
  ctx.textAlign = align || 'left'; ctx.lineJoin = 'round';
  ctx.strokeStyle = 'rgba(6,10,18,0.85)'; ctx.lineWidth = 3; ctx.strokeText(txt, x, y);
  ctx.fillStyle = 'rgba(255,255,255,0.92)'; ctx.fillText(txt, x, y);
}

/* ---------- map drawing ---------- */
function paintPx(sc, v, d, k) {
  const b = sc.breaks, c = sc.rgb, n = b.length;
  let r = c[0];
  if (v >= b[n - 1]) r = c[n - 1];
  else if (v > b[0]) {
    let s = 0;
    while (v >= b[s + 1]) s++;
    const f = (v - b[s]) / (b[s + 1] - b[s]), p = c[s], q = c[s + 1];
    d[k] = p[0] + (q[0] - p[0]) * f; d[k + 1] = p[1] + (q[1] - p[1]) * f; d[k + 2] = p[2] + (q[2] - p[2]) * f; d[k + 3] = 255;
    return;
  }
  d[k] = r[0]; d[k + 1] = r[1]; d[k + 2] = r[2]; d[k + 3] = 255;
}

function buildPanels() {
  for (const id of ['mapA', 'mapB', 'mapC']) {
    const el = $(id);
    el.innerHTML = '<figcaption><span class="mt"></span><span class="ms"></span></figcaption>' +
      '<div class="cvwrap"><canvas class="mapcv"></canvas><div class="empty" hidden></div></div>' +
      '<div class="legend"><canvas class="lgcv" width="512" height="10"></canvas><div class="ticks"></div></div>' +
      '<p class="honest"></p>';
    const cv = el.querySelector('.mapcv'), off = document.createElement('canvas');
    off.width = NLON; off.height = NLAT;
    const P = { el, cv, ctx: cv.getContext('2d'), off, octx: off.getContext('2d'), mt: el.querySelector('.mt'), ms: el.querySelector('.ms'),
      empty: el.querySelector('.empty'), lg: el.querySelector('.lgcv'), ticks: el.querySelector('.ticks'), honest: el.querySelector('.honest') };
    P.img = P.octx.createImageData(NLON, NLAT);
    PANELS[id] = P;
    cv.addEventListener('pointerdown', (e) => onDown(e, cv));
    cv.addEventListener('pointermove', (e) => onMove(e, cv));
    cv.addEventListener('pointerup', (e) => onUp(e, cv));
    cv.addEventListener('pointercancel', (e) => onUp(e, cv));
    cv.addEventListener('pointerleave', () => { $('tip').hidden = true; });
  }
}

function drawMap(P, vals, sc, o) {
  const cv = P.cv, dpr = window.devicePixelRatio || 1, w = cv.clientWidth, h = cv.clientHeight;
  if (!w || !h) return;
  const W = Math.round(w * dpr), H = Math.round(h * dpr);
  if (cv.width !== W || cv.height !== H) { cv.width = W; cv.height = H; }
  const ctx = P.ctx, sx = w / NLON, sy = h / NLAT;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  if (vals) {
    const d = P.img.data;
    for (let i = 0; i < NLAT; i++) {
      const row = (NLAT - 1 - i) * NLON;
      for (let j = 0; j < NLON; j++) paintPx(sc, vals[i * NLON + j], d, (row + j) * 4);
    }
    P.octx.putImageData(P.img, 0, 0);
    ctx.imageSmoothingEnabled = S.smooth; ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(P.off, 0, 0, w, h);
  } else { ctx.fillStyle = '#0c1320'; ctx.fillRect(0, 0, w, h); }

  // latitude and longitude lines every 5 degrees
  ctx.lineWidth = 1; ctx.strokeStyle = 'rgba(8,12,20,0.22)'; ctx.beginPath();
  for (let lon = 70; lon <= 100; lon += 5) { const x = Math.round((lon - 65) * sx) + 0.5; ctx.moveTo(x, 0); ctx.lineTo(x, h); }
  for (let lat = 10; lat <= 35; lat += 5) { const y = Math.round((38 - lat) * sy) + 0.5; ctx.moveTo(0, y); ctx.lineTo(w, y); }
  ctx.stroke();

  if (o.heavy && vals) haloLine(ctx, contour(pad(vals, false), 10), sx, sy, '#ff7ab6', 1.3);
  haloLine(ctx, COAST, sx, sy, 'rgba(255,255,255,0.94)', 1.1);

  if (o.mask) {
    if (o.fill) {
      ctx.fillStyle = 'rgba(255,212,59,0.16)';
      for (let i = 0; i < NLAT; i++) for (let j = 0; j < NLON; j++) if (S.mask[i * NLON + j] > 0.5) ctx.fillRect(j * sx, (NLAT - 1 - i) * sy, sx + 0.5, sy + 0.5);
    }
    haloLine(ctx, S.maskSegs, sx, sy, '#ffd43b', 1.8);
  }
  if (o.reach > 0) haloLine(ctx, contour(pad(S.dist, false), o.reach + 0.5), sx, sy, 'rgba(255,255,255,0.9)', 1.2, [5, 5]);

  if (o.wind) {
    const st = w < 420 ? 3 : 2, a = o.wind.arr, of = o.wind.o, lines = [];
    for (let i = 1; i < NLAT; i += st) for (let j = 1; j < NLON; j += st) {
      const p = i * NLON + j, u = a[of + 2 * CELLS + p], v = a[of + 3 * CELLS + p], sp = Math.hypot(u, v);
      if (sp < 0.8) continue;
      const L = Math.min(sp, 14) / 14 * st * sx * 0.95, ux = u / sp, uy = -v / sp, cx = (j + 0.5) * sx, cy = (NLAT - i - 0.5) * sy;
      const x1 = cx + ux * L / 2, y1 = cy + uy * L / 2, hd = Math.min(4.5, L * 0.45), c = 0.9, s = 0.44;
      lines.push(cx - ux * L / 2, cy - uy * L / 2, x1, y1,
        x1, y1, x1 - hd * (ux * c - uy * s), y1 - hd * (uy * c + ux * s),
        x1, y1, x1 - hd * (ux * c + uy * s), y1 - hd * (uy * c - ux * s));
    }
    ctx.lineCap = 'round';
    for (const [col, lw] of [['rgba(6,10,18,0.55)', 2.8], ['rgba(255,255,255,0.9)', 1.1]]) {
      ctx.strokeStyle = col; ctx.lineWidth = lw; ctx.beginPath();
      for (let k = 0; k < lines.length; k += 4) { ctx.moveTo(lines[k], lines[k + 1]); ctx.lineTo(lines[k + 2], lines[k + 3]); }
      ctx.stroke();
    }
  }

  ctx.font = '10px "Segoe UI", system-ui, sans-serif'; ctx.textBaseline = 'middle';
  for (let lon = 70; lon <= 100; lon += 10) haloText(ctx, lon + '°E', (lon - 65) * sx + 3, h - 7, 'left');
  for (let lat = 10; lat <= 35; lat += 5) haloText(ctx, lat + '°N', 3, (38 - lat) * sy - 7, 'left');
  if (w >= 300) {
    ctx.font = '10.5px "Segoe UI", system-ui, sans-serif';
    for (const c of CITIES) {
      if (!c[3]) continue;
      const x = (c[2] - 65) * sx, y = (38 - c[1]) * sy;
      ctx.beginPath(); ctx.arc(x, y, 2.6, 0, 6.3); ctx.fillStyle = 'rgba(6,10,18,0.9)'; ctx.fill();
      ctx.beginPath(); ctx.arc(x, y, 1.6, 0, 6.3); ctx.fillStyle = '#fff'; ctx.fill();
      haloText(ctx, c[0], c[3] === 'l' ? x - 5 : x + 5, y, c[3] === 'l' ? 'right' : 'left');
    }
  }
  if (o.sel) {
    const x = o.sel.j * sx, y = (NLAT - 1 - o.sel.i) * sy;
    ctx.strokeStyle = 'rgba(6,10,18,0.9)'; ctx.lineWidth = 3.5; ctx.strokeRect(x, y, sx, sy);
    ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5; ctx.strokeRect(x, y, sx, sy);
  }
}

function drawLegend(P, sc, label) {
  const cv = P.lg, ctx = cv.getContext('2d'), w = cv.width, n = sc.breaks.length, img = ctx.createImageData(w, 1);
  for (let x = 0; x < w; x++) {
    const t = x / (w - 1) * (n - 1), s = Math.min(n - 2, Math.floor(t)), f = t - s, p = sc.rgb[s], q = sc.rgb[s + 1];
    img.data[x * 4] = p[0] + (q[0] - p[0]) * f; img.data[x * 4 + 1] = p[1] + (q[1] - p[1]) * f;
    img.data[x * 4 + 2] = p[2] + (q[2] - p[2]) * f; img.data[x * 4 + 3] = 255;
  }
  for (let y = 0; y < cv.height; y++) ctx.putImageData(img, 0, y);
  P.ticks.innerHTML = sc.breaks.map((b, k) => '<span style="left:' + (k / (n - 1) * 100) + '%">' + label(b) + '</span>').join('');
}

/* ---------- what the three maps show ---------- */
function view() {
  const R = S.run;
  if (S.tab === 'forecast') {
    return { A: R.A, B: S.day <= R.td ? R.T : null, sign: -1, last: R.td,
      titles: ['Model forecast', 'What really happened', 'Where the forecast missed'], short: ['Forecast', 'Recorded'],
      emptyB: 'No recorded weather for this day.' };
  }
  return { A: R.A, B: R.B, sign: 1, last: R.n,
    titles: ['Without your change', 'With your change', 'What your change did'], short: ['Without', 'With'],
    emptyB: 'Paint a region on a map, or press an official shock on the left.' };
}

// One fixed range per field for the difference map, so the change can be seen growing and fading across days.
function diffMax(f) {
  const R = S.run, key = S.tab + ':' + f.key;
  if (R.cache[key]) return R.cache[key];
  const fc = S.tab === 'forecast', B = fc ? R.T : R.B, last = fc ? R.td : R.n;
  let m = 0;
  if (B && last > 0) {                             // ignore the few most extreme squares so one of them does not flatten the map
    const all = new Float32Array(last * CELLS);
    for (let t = 1; t <= last; t++) for (let p = 0; p < CELLS; p++) all[(t - 1) * CELLS + p] = Math.abs(f.get(B, t * STEP, p) - f.get(R.A, t * STEP, p));
    all.sort(); m = all[Math.floor(all.length * (fc ? 0.99 : 0.997))];
  }
  return (R.cache[key] = nice(Math.max(m, f.floor)));
}

function render() {
  const R = S.run;
  if (!R) return;
  const f = S.field, o = S.day * STEP, V = view(), wet = f.dpal === 'dWet', fc = S.tab === 'forecast';
  const windOn = S.arrows || f.key === 'wind' || f.key === 'u10' || f.key === 'v10';
  const showMask = !fc && S.count > 0;
  const va = new Float32Array(CELLS);
  for (let p = 0; p < CELLS; p++) va[p] = f.get(V.A, o, p);
  let vb = null, vc = null;
  if (V.B) {
    vb = new Float32Array(CELLS); vc = new Float32Array(CELLS);
    for (let p = 0; p < CELLS; p++) { vb[p] = f.get(V.B, o, p); vc[p] = V.sign * (vb[p] - va[p]); }
  }
  const dsc = diffScale(diffMax(f), PAL[f.dpal]);
  const A = PANELS.mapA, B = PANELS.mapB, C = PANELS.mapC;
  drawMap(A, va, f.scale, { heavy: f.key === 'tp', wind: windOn ? { arr: V.A, o } : null, mask: showMask, fill: true, sel: S.sel });
  drawMap(B, vb, f.scale, { heavy: f.key === 'tp', wind: windOn && V.B ? { arr: V.B, o } : null, mask: showMask, sel: S.sel });
  drawMap(C, vc, dsc, { mask: showMask, sel: S.sel, reach: !fc && V.B && S.day > 0 ? S.meta.reach * S.day : 0 });

  const what = f.name + ', ' + f.unit;
  A.mt.textContent = V.titles[0]; B.mt.textContent = V.titles[1]; C.mt.textContent = V.titles[2];
  A.ms.textContent = what; B.ms.textContent = what; C.ms.textContent = 'difference, ' + f.unit;
  B.empty.hidden = C.empty.hidden = !!V.B;
  B.empty.textContent = V.emptyB;
  C.empty.textContent = fc ? 'Nothing to compare for this day.' : 'The difference shows up here once you make a change.';
  drawLegend(A, f.scale, tick); drawLegend(B, f.scale, tick); drawLegend(C, dsc, signedTick);

  const heavy = f.key === 'tp' ? ' Pink line: heavy rain, 10 mm a day or more.' : '';
  const seen = R.tag === 'unseen' ? 'The model never saw this year.' : R.tag === 'tune' ? 'This year was used for tuning.' : 'The model was trained on this year.';
  A.honest.textContent = 'Model forecast, about 100 km squares, daily.' + heavy;
  if (fc) {
    B.honest.textContent = 'Recorded weather from the organizers\' data. ' + seen;
    C.honest.textContent = 'Forecast minus recorded. ' + (wet ? 'Brown: forecast too dry. Green: too wet.' : 'Blue: forecast too low. Red: too high.');
  } else {
    B.honest.textContent = 'Modelled what-if, not proof of real-world effect. Yellow line: your region.';
    C.honest.textContent = 'With minus without. ' + (wet ? 'Brown: drier. Green: wetter.' : 'Blue: lower. Red: higher.') +
      ' Dashed line: the furthest the change can travel by this day.';
  }

  $('day').value = S.day;
  $('dayLabel').innerHTML = (S.day === 0 ? 'Start day' : 'Day +' + S.day) + '<small>' + dayDate(S.day) + '</small>';
  [...$('dticks').children].forEach((el, k) => el.classList.toggle('on', k === S.day));
  drawPlace();
  drawFacts(va, vb, vc);
}

/* ---------- "this place" chart ---------- */
const CHART = { L: 40, R: 12, T: 8, B: 22 };
function drawPlace() {
  const cv = $('placeCv'), dpr = window.devicePixelRatio || 1, w = cv.clientWidth, h = cv.clientHeight, R = S.run;
  if (!w || !R) return;
  cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
  const ctx = cv.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, w, h);
  const f = S.field, p = S.sel.i * NLON + S.sel.j, n = R.n, fc = S.tab === 'forecast';
  const src = fc ? R.T : R.B, bn = fc ? R.td : n, a = [], b = [];
  for (let t = 0; t <= n; t++) a.push(f.get(R.A, t * STEP, p));
  if (src) for (let t = 0; t <= bn; t++) b.push(f.get(src, t * STEP, p));
  let lo = Math.min(...a, ...b), hi = Math.max(...a, ...b);
  if (hi - lo < f.floor) { const mid = (hi + lo) / 2; lo = mid - f.floor / 2; hi = mid + f.floor / 2; }
  const gap = (hi - lo) * 0.12; hi += gap; lo -= gap;
  if (f.key === 'tp' || f.key === 'wind' || f.key === 'tcwv') lo = Math.max(0, lo);
  const pw = w - CHART.L - CHART.R, ph = h - CHART.T - CHART.B;
  const X = (t) => CHART.L + pw * t / n, Y = (v) => CHART.T + ph * (1 - (v - lo) / (hi - lo));

  ctx.font = '10.5px "Segoe UI", system-ui, sans-serif'; ctx.textBaseline = 'middle';
  for (let k = 0; k <= 3; k++) {
    const v = lo + (hi - lo) * k / 3, y = Math.round(Y(v)) + 0.5;
    ctx.strokeStyle = '#1e2a44'; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(CHART.L, y); ctx.lineTo(w - CHART.R, y); ctx.stroke();
    ctx.fillStyle = '#7a87a1'; ctx.textAlign = 'right'; ctx.fillText(fmt(v, hi - lo < 5 ? 1 : 0), CHART.L - 6, y);
  }
  ctx.textAlign = 'center';
  for (let t = 0; t <= n; t++) { ctx.fillStyle = t === S.day ? '#e9eef7' : '#7a87a1'; ctx.fillText(t === 0 ? 'Start' : '+' + t, X(t), h - 9); }
  ctx.strokeStyle = 'rgba(255,255,255,0.28)'; ctx.lineWidth = 1; ctx.beginPath();
  ctx.moveTo(Math.round(X(S.day)) + 0.5, CHART.T); ctx.lineTo(Math.round(X(S.day)) + 0.5, CHART.T + ph); ctx.stroke();

  const colB = fc ? '#e9eef7' : '#ffb454';
  const line = (s, col, dash) => {
    if (!s.length) return;
    ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.lineJoin = 'round'; ctx.setLineDash(dash || []); ctx.beginPath();
    s.forEach((v, t) => (t ? ctx.lineTo(X(t), Y(v)) : ctx.moveTo(X(t), Y(v))));
    ctx.stroke(); ctx.setLineDash([]);
    s.forEach((v, t) => {
      const r = t === S.day ? 5 : 3;
      ctx.beginPath(); ctx.arc(X(t), Y(v), r + 2, 0, 6.3); ctx.fillStyle = '#111a2a'; ctx.fill();
      ctx.beginPath(); ctx.arc(X(t), Y(v), r, 0, 6.3); ctx.fillStyle = col; ctx.fill();
    });
  };
  line(b, colB, fc ? [5, 4] : null);
  line(a, '#5ab0ff');

  const pl = place(S.sel.i, S.sel.j);
  $('placeTitle').textContent = (pl.near ? 'Near ' + pl.near + ' (' + pl.coords + ')' : pl.coords) + ': ' + f.name.toLowerCase() + ', ' + f.unit;
  const V = view();
  let html = '<span><i style="border-color:#5ab0ff"></i>' + V.titles[0] + ' <b>' + fmt(a[S.day]) + '</b></span>';
  if (b.length) html += '<span><i class="' + (fc ? 'dash' : '') + '" style="border-color:' + colB + '"></i>' + V.titles[1] + ' <b>' + (S.day < b.length ? fmt(b[S.day]) : 'none') + '</b></span>';
  $('placeLegend').innerHTML = html + '<span>on ' + dayText(S.day) + '</span>';
}

/* ---------- facts card ---------- */
function fact(label, value, small, cls) {
  return '<div class="fact"><span>' + label + '</span><b' + (cls ? ' class="' + cls + '"' : '') + '>' + value + (small ? '<small>' + small + '</small>' : '') + '</b></div>';
}
function drawFacts(va, vb, vc) {
  const R = S.run, f = S.field, o = S.day * STEP, fc = S.tab === 'forecast', body = $('changeBody'), hint = $('changeHint');
  $('changeTitle').textContent = fc ? 'How close was the forecast?' : 'What your change did';
  if (!vb) {
    body.innerHTML = fact(fc ? 'No recorded weather for this day' : 'No change yet', '');
    hint.textContent = fc ? '' : 'Paint a region, or press an official shock on the left.';
    return;
  }
  if (fc) {
    if (S.day === 0) {
      body.innerHTML = fact('This is the starting day', 'recorded');
      hint.textContent = 'Move the slider to see the forecast next to what really happened.';
      return;
    }
    let em = 0, ep = 0, worst = 0, wp = 0;
    for (let p = 0; p < CELLS; p++) {
      const t = vb[p], d = va[p] - t;
      em += Math.abs(d); ep += Math.abs(f.get(R.A, 0, p) - t);
      if (Math.abs(d) > Math.abs(worst)) { worst = d; wp = p; }
    }
    em /= CELLS; ep /= CELLS;
    const gain = (1 - em / ep) * 100;
    body.innerHTML = fact('Average miss, model', fmt(em, 2) + ' ' + f.unit) +
      fact('Average miss, "same as the start day" guess', fmt(ep, 2) + ' ' + f.unit) +
      fact('Model against that guess', Math.abs(gain).toFixed(0) + '% ' + (gain >= 0 ? 'closer' : 'further off'), '', gain >= 0 ? 'ok' : 'no') +
      fact('Worst miss', signed(worst) + ' ' + f.unit, placeText(Math.floor(wp / NLON), wp % NLON));
    hint.textContent = f.name + ', ' + dayText(S.day) + ', this one case only. The scoreboard in the proof panel covers two full years.';
    return;
  }
  if (S.day === 0) {
    let mx = 0;
    for (let p = 0; p < CELLS; p++) if (Math.abs(vc[p]) > Math.abs(mx)) mx = vc[p];
    body.innerHTML = fact('Change you made to ' + f.name.toLowerCase(), signed(mx) + ' ' + f.unit, 'largest value on the start day') +
      fact('Squares in your region', S.count + ' of ' + CELLS);
    hint.textContent = 'Move the slider to watch the change travel.';
    return;
  }
  let mx = -1e9, mn = 1e9, px = 0, pn = 0, cnt = 0, far = 0;
  for (let p = 0; p < CELLS; p++) {
    const d = vc[p];
    if (d > mx) { mx = d; px = p; }
    if (d < mn) { mn = d; pn = p; }
    if (Math.abs(d) > f.notice) { cnt++; if (S.dist[p] > far) far = S.dist[p]; }
  }
  const lim = S.meta.reach * S.day;
  body.innerHTML = fact('Biggest rise', signed(Math.max(mx, 0), 2) + ' ' + f.unit, mx > f.notice ? placeText(Math.floor(px / NLON), px % NLON) : 'none worth noting') +
    fact('Biggest fall', signed(Math.min(mn, 0), 2) + ' ' + f.unit, mn < -f.notice ? placeText(Math.floor(pn / NLON), pn % NLON) : 'none worth noting') +
    fact('Squares that changed by more than ' + f.notice + ' ' + f.unit, cnt + ' of ' + CELLS) +
    fact('Furthest such square from your region', far + ' squares', 'the model can carry it ' + lim + ' at most by ' + dayText(S.day));
  hint.textContent = f.name + ', ' + dayText(S.day) + '. Modelled what-if, not proof of real-world effect.';
}

/* ---------- proof card ---------- */
function drawProof() {
  const R = S.run, pr = S.proof, same = (x) => (x === 0 ? 'identical' : 'differs by ' + x.toExponential(1));
  let h = fact('10-day forecast on this processor', R ? R.baseMs.toFixed(0) + ' ms' : '...', 'the limit is 5000 ms');
  if (pr) {
    h += fact('Zero change in, zero change out', same(pr.zero_change_max_diff), '', pr.zero_change_max_diff === 0 ? 'ok' : 'no') +
      fact('Same input twice', same(pr.repeat_max_diff), '', pr.repeat_max_diff === 0 ? 'ok' : 'no') +
      fact('Output passes the validity rules', pr.valid_output ? 'yes' : 'no', '', pr.valid_output ? 'ok' : 'no');
    h += pr.graded.running
      ? fact('Graded server on port 8000', pr.graded.max_diff_vs_map === 0 ? 'same answer as this map' : 'differs by ' + pr.graded.max_diff_vs_map.toExponential(1),
        pr.graded.model_name + ', ' + pr.graded.round_trip_ms.toFixed(0) + ' ms per request', pr.graded.max_diff_vs_map === 0 ? 'ok' : 'no')
      : fact('Graded server on port 8000', 'not running', 'start it with: python serve.py --port 8000');
  } else h += fact('Live checks', 'not run yet');
  $('proofBody').innerHTML = h;
}
async function runProof() {
  $('proofGo').disabled = true;
  try {
    const r = await fetch('api/proof', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ date: S.date }) });
    if (r.ok) S.proof = await r.json();
  } catch (e) { /* the card keeps its last state */ }
  $('proofGo').disabled = false;
  drawProof();
}
function drawScores() {
  const s = S.meta.scores;
  if (!s) { $('scores').innerHTML = '<p class="hint">No saved scoreboard result. Run: python training/scoreboard.py --save dashboard/scores.json</p>'; return; }
  const row = (a, b, c) => '<tr' + (c ? ' class="total"' : '') + '><td>' + a + '</td><td>' + b + '</td></tr>';
  $('scores').innerHTML = '<table>' +
    row('Forecast error', s.forecast_error_pts_of_30 + ' of 30') + row('Heavy rain', s.heavy_rain_pts_of_15 + ' of 15') +
    row('Sharpness', s.sharpness_pts_of_10 + ' of 10') + row('Shock tests', s.shock_pts_of_15 + ' of 15') +
    row('Model total', s.model_pts_of_70 + ' of 70', true) + '</table>' +
    '<p class="hint">Scores are from our own scoreboard, run with the organizers\' scoring code on ' + s.years.join(' and ') +
    ', years the model never saw. Not the official grade.</p>';
}

/* ---------- the user's change ---------- */
function hasShock() { return S.count > 0 && KNOBS.some((k) => S.knob[k.id] !== 0); }
function maskChanged(run) {
  let n = 0, area = 0;
  const cells = [];
  for (let p = 0; p < CELLS; p++) if (S.mask[p] > 0.5) { n++; cells.push(p); area += 111.2 * 111.2 * Math.cos(latOf(Math.floor(p / NLON)) * Math.PI / 180); }
  S.count = n;
  S.maskSegs = contour(pad(S.mask, true), 0.5);
  for (let p = 0; p < CELLS; p++) {
    const i = Math.floor(p / NLON), j = p % NLON;
    let best = 99;
    for (const q of cells) { const d = Math.max(Math.abs(Math.floor(q / NLON) - i), Math.abs(q % NLON - j)); if (d < best) best = d; }
    S.dist[p] = best;
  }
  shockNote(area);
  render();
  if (run) scheduleRun(0);
}
function shockNote(area) {
  const el = $('shockNote');
  if (!S.count) { el.textContent = 'No region yet. Pick Brush or Box and draw on any map.'; return; }
  const a = area >= 1e6 ? (area / 1e6).toFixed(1) + ' million' : (Math.round(area / 1000) * 1000).toLocaleString('en-IN');
  el.textContent = 'Region: ' + S.count + ' squares, about ' + a + ' sq km.' + (KNOBS.some((k) => S.knob[k.id] !== 0) ? '' : ' Now move a slider to set the change.');
}
function setBox(la0, la1, lo0, lo1) {
  S.mask.fill(0);
  for (let i = 0; i < NLAT; i++) for (let j = 0; j < NLON; j++) {
    if (latOf(i) >= la0 && latOf(i) <= la1 && lonOf(j) >= lo0 && lonOf(j) <= lo1) S.mask[i * NLON + j] = 1;
  }
}
function setKnobs(vals) {
  S.add.fill(0); S.mul.fill(0);
  for (const k of KNOBS) {
    const v = vals[k.id] || 0;
    S.knob[k.id] = v; k.set(v);
    $(k.id).value = v; $(k.id + 'o').textContent = signed(v, k.step < 1 ? 1 : 0) + ' ' + k.unit;
    $(k.id).closest('.slider').classList.toggle('live', v !== 0);
  }
}
function markOfficial(name) {
  S.official = name;
  $('offWarm').classList.toggle('on', name === 'warm');
  $('offStall').classList.toggle('on', name === 'stall');
}
function official(name) {
  const o = OFFICIAL[name];
  setBox(...o.box); setKnobs(o.knobs); markOfficial(name); setTab('whatif', true);
  maskChanged(true);
}
function dab(i0, j0, val) {
  const r = S.brush - 0.5, r2 = r * r + 0.3, R = Math.ceil(r);
  for (let di = -R; di <= R; di++) for (let dj = -R; dj <= R; dj++) {
    const i = i0 + di, j = j0 + dj;
    if (di * di + dj * dj <= r2 && i >= 0 && i < NLAT && j >= 0 && j < NLON) S.mask[i * NLON + j] = val;
  }
}
function cellAt(e, cv) {
  const r = cv.getBoundingClientRect();
  const j = Math.floor((e.clientX - r.left) / r.width * NLON), row = Math.floor((e.clientY - r.top) / r.height * NLAT);
  return { i: Math.min(NLAT - 1, Math.max(0, NLAT - 1 - row)), j: Math.min(NLON - 1, Math.max(0, j)) };
}
function applyStroke(c) {
  const st = S.stroke;
  if (st.tool === 'box') {
    S.mask.set(st.base);
    for (let i = Math.min(st.start.i, c.i); i <= Math.max(st.start.i, c.i); i++)
      for (let j = Math.min(st.start.j, c.j); j <= Math.max(st.start.j, c.j); j++) S.mask[i * NLON + j] = 1;
  } else {
    const a = st.last || c, n = Math.max(Math.abs(c.i - a.i), Math.abs(c.j - a.j), 1);
    for (let k = 0; k <= n; k++) dab(Math.round(a.i + (c.i - a.i) * k / n), Math.round(a.j + (c.j - a.j) * k / n), st.tool === 'erase' ? 0 : 1);
    st.last = c;
  }
  maskChanged(false);
}
function onDown(e, cv) {
  if (!S.run) return;
  const c = cellAt(e, cv);
  if (S.tool === 'look') { S.sel = c; render(); return; }
  cv.setPointerCapture(e.pointerId);
  $('tip').hidden = true;
  markOfficial(null);
  S.stroke = { tool: S.tool, start: c, last: null, base: S.mask.slice() };
  applyStroke(c);
}
function onMove(e, cv) {
  if (S.stroke) { applyStroke(cellAt(e, cv)); return; }
  showTip(e, cellAt(e, cv));
}
function onUp(e, cv) {
  if (!S.stroke) return;
  S.stroke = null;
  maskChanged(true);
}
function showTip(e, c) {
  const R = S.run, tip = $('tip');
  if (!R) return;
  const V = view(), f = S.field, p = c.i * NLON + c.j, o = S.day * STEP, pl = place(c.i, c.j), a = f.get(V.A, o, p);
  let h = '<div class="th">' + pl.coords + (pl.near ? ', near ' + pl.near : '') + '<br>' + f.name + ', ' + dayText(S.day) + '</div>' +
    '<div class="tr"><span>' + V.short[0] + '</span><b>' + fmt(a) + ' ' + f.unit + '</b></div>';
  if (V.B) {
    const b = f.get(V.B, o, p);
    h += '<div class="tr"><span>' + V.short[1] + '</span><b>' + fmt(b) + ' ' + f.unit + '</b></div>' +
      '<div class="tr"><span>' + (S.tab === 'forecast' ? 'Forecast miss' : 'Change') + '</span><b>' + signed(V.sign * (b - a), 2) + '</b></div>';
  }
  tip.innerHTML = h; tip.hidden = false;
  const tw = tip.offsetWidth, th = tip.offsetHeight;
  tip.style.left = Math.min(window.innerWidth - tw - 8, e.clientX + 16) + 'px';
  tip.style.top = Math.max(8, Math.min(window.innerHeight - th - 8, e.clientY + 16)) + 'px';
}

/* ---------- running the model ---------- */
function scheduleRun(ms) { clearTimeout(runTimer); runTimer = setTimeout(runModel, ms || 0); }
async function runModel() {
  const my = ++S.seq, on = hasShock();
  const body = { date: S.date, steps: S.meta.max_steps, shock: on ? { mask: Array.from(S.mask), add: S.add, mul: S.mul } : null };
  let j;
  try {
    const r = await fetch('api/run', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    j = await r.json();
    if (!r.ok) throw new Error(j.error || 'the run failed');
  } catch (e) { if (my === S.seq) banner('Could not run the model: ' + e.message); return; }
  if (my !== S.seq) return;
  banner('');
  const n = j.steps, init = dec(j.init), A = new Float32Array((n + 1) * STEP);
  A.set(init, 0); A.set(dec(j.base), STEP);
  let B = null;
  if (j.shocked) {
    const dl = dec(j.delta);
    B = new Float32Array((n + 1) * STEP);
    for (let k = 0; k < STEP; k++) B[k] = init[k] + dl[k];
    B.set(dec(j.shocked), STEP);
  }
  const T = new Float32Array((j.truth_days + 1) * STEP);
  T.set(init, 0);
  if (j.truth_days) T.set(dec(j.truth), STEP);
  S.run = { n, A, B, T, td: j.truth_days, baseMs: j.base_ms, shockMs: j.shock_ms, tag: j.tag, cache: {} };
  if (S.day > n) S.day = n;
  $('chipSpeed').innerHTML = '<b>' + j.base_ms.toFixed(0) + ' ms</b> for a ' + n + '-day forecast' + (j.shock_ms ? ', <b>' + j.shock_ms.toFixed(0) + ' ms</b> for the what-if' : '');
  const yc = $('chipYear');
  yc.className = 'chip ' + (j.tag === 'unseen' ? 'good' : 'warn');
  yc.textContent = j.tag === 'unseen' ? 'The model never saw this year' : j.tag === 'tune' ? 'This year was used for tuning' : 'The model was trained on this year';
  render(); drawProof();
}

/* ---------- controls ---------- */
function setTab(tab, quiet) {
  S.tab = tab;
  [...$('tabs').children].forEach((b) => b.classList.toggle('on', b.dataset.tab === tab));
  if (!quiet) render();
}
function setTool(tool) {
  S.tool = tool;
  [...$('tools').children].forEach((b) => b.classList.toggle('on', b.dataset.tool === tool));
  for (const id in PANELS) PANELS[id].cv.style.cursor = tool === 'look' ? 'crosshair' : 'cell';
  if (tool !== 'look' && S.tab !== 'whatif') setTab('whatif');
}
function setDay(d) { S.day = Math.max(0, Math.min(S.run ? S.run.n : 10, d)); render(); }
function setDate(d) {
  S.date = d; $('date').value = d;
  [...$('presets').children].forEach((b) => b.classList.toggle('on', b.dataset.date === d));
  scheduleRun(0);
}
function togglePlay() {
  if (S.playing) { clearInterval(S.playing); S.playing = null; }
  else S.playing = setInterval(() => { if (S.run) setDay((S.day + 1) % (S.run.n + 1)); }, 750);
  $('play').innerHTML = S.playing ? '&#10074;&#10074;' : '&#9654;';
}
function buildControls() {
  $('fields').innerHTML = FIELDS.map((f, k) => '<button class="field' + (k ? '' : ' on') + '" data-k="' + k + '">' + f.name + '<small>' + f.unit + '</small></button>').join('');
  $('fields').addEventListener('click', (e) => {
    const b = e.target.closest('.field');
    if (!b) return;
    S.field = FIELDS[+b.dataset.k];
    [...$('fields').children].forEach((x) => x.classList.toggle('on', x === b));
    render();
  });
  $('sliders').innerHTML = KNOBS.map((k) => '<label class="slider"><span>' + k.name + '</span><output id="' + k.id + 'o"></output>' +
    '<input type="range" id="' + k.id + '" min="' + k.min + '" max="' + k.max + '" step="' + k.step + '" value="0"></label>').join('');
  for (const k of KNOBS) {
    S.knob[k.id] = 0;
    $(k.id).addEventListener('input', (e) => {
      const v = +e.target.value;
      S.knob[k.id] = v; k.set(v); markOfficial(null);
      $(k.id + 'o').textContent = signed(v, k.step < 1 ? 1 : 0) + ' ' + k.unit;
      e.target.closest('.slider').classList.toggle('live', v !== 0);
      if (S.tab !== 'whatif') setTab('whatif');
      maskChanged(false); scheduleRun(120);
    });
  }
  setKnobs({});
  $('tabs').addEventListener('click', (e) => { const b = e.target.closest('button'); if (b) setTab(b.dataset.tab); });
  $('tools').addEventListener('click', (e) => { const b = e.target.closest('button'); if (b) setTool(b.dataset.tool); });
  $('brush').addEventListener('input', (e) => { S.brush = +e.target.value; $('brushOut').textContent = S.brush; });
  $('offWarm').addEventListener('click', () => official('warm'));
  $('offStall').addEventListener('click', () => official('stall'));
  $('boxGo').addEventListener('click', () => {
    const v = ['la0', 'la1', 'lo0', 'lo1'].map((id) => +$(id).value);
    setBox(Math.min(v[0], v[1]), Math.max(v[0], v[1]), Math.min(v[2], v[3]), Math.max(v[2], v[3]));
    markOfficial(null); setTab('whatif', true); maskChanged(true);
  });
  $('clearRegion').addEventListener('click', () => { S.mask.fill(0); markOfficial(null); maskChanged(true); });
  $('resetShock').addEventListener('click', () => { S.mask.fill(0); setKnobs({}); markOfficial(null); setTool('look'); maskChanged(true); });
  $('smooth').addEventListener('change', (e) => { S.smooth = e.target.checked; render(); });
  $('arrows').addEventListener('change', (e) => { S.arrows = e.target.checked; render(); });
  $('day').addEventListener('input', (e) => setDay(+e.target.value));
  $('play').addEventListener('click', togglePlay);
  $('date').addEventListener('change', (e) => { if (e.target.value) setDate(e.target.value); });
  $('proofGo').addEventListener('click', runProof);
  $('placeCv').addEventListener('click', (e) => {
    if (!S.run) return;
    const r = e.target.getBoundingClientRect(), pw = r.width - CHART.L - CHART.R;
    setDay(Math.round((e.clientX - r.left - CHART.L) / pw * S.run.n));
  });
  window.addEventListener('resize', render);
  window.addEventListener('keydown', (e) => {
    if (/^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName) && e.target.type !== 'range' && e.target.type !== 'checkbox') return;
    if (e.key === 'ArrowRight') { e.preventDefault(); setDay(S.day + 1); }
    else if (e.key === 'ArrowLeft') { e.preventDefault(); setDay(S.day - 1); }
    else if (e.key === ' ') { e.preventDefault(); togglePlay(); }
  });
}

async function start() {
  buildPanels(); buildControls();
  let m;
  try { m = await (await fetch('api/meta')).json(); }
  catch (e) { banner('The dashboard backend is not answering. Start it with: python dashboard/backend.py'); return; }
  S.meta = m;
  COAST = contour(pad(dec(m.mask), false), 0.5);
  $('chipModel').innerHTML = 'Local-view model, <b>' + m.params.toLocaleString('en-IN') + '</b> weights, reach <b>' + m.reach + '</b> squares a day';
  drawScores();
  if (m.error) { banner(m.error); return; }
  $('dticks').innerHTML = Array.from({ length: m.max_steps + 1 }, (_, k) => '<span>' + (k ? '+' + k : 'Start') + '</span>').join('');
  $('day').max = m.max_steps;
  $('date').min = m.date_min; $('date').max = m.date_max;
  $('presets').innerHTML = m.presets.map((p) => '<button class="preset" data-date="' + p.date + '">' + p.name + '<small>' + p.note + '</small></button>').join('');
  $('presets').addEventListener('click', (e) => { const b = e.target.closest('.preset'); if (b) setDate(b.dataset.date); });
  shockNote(0);
  S.date = m.presets[0].date; $('date').value = S.date;
  $('presets').children[0].classList.add('on');
  await runModel();
  runProof();
}
start();
