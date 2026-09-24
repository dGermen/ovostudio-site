/* Ovo Studio, sand edition.
 *
 * Every product on this page is a sculpture of grains: N points sampled off the
 * surface of its real CAD assembly (the same web meshes the immersive edition
 * uses), simulated on the GPU. Scrolling decides which sculpture is standing
 * and how much of it:
 *
 *   form    grains bind to the sculpture bottom-up, with a ragged front, and fly
 *           out of the heap to their place on a damped spring;
 *   erode   the wind takes them off the windward face first; once released a
 *           grain falls under gravity, drifts downwind and settles on the heap.
 *
 * The same grain moves from sculpture to sculpture, so scrolling back simply
 * reverses it. The cursor is a blower.
 *
 * Each grain carries two thresholds per shape (rForm in target.w, rErode in
 * meta.w) and is bound while form > rForm and erode < rErode. Everything else
 * is two small fragment shaders over 512x512 floats.
 */
import * as THREE from "three";
import { GPUComputationRenderer } from "three/addons/misc/GPUComputationRenderer.js";

// ------------------------------------------------------------------ helpers
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
const clamp01 = (v) => clamp(v, 0, 1);
const smooth = (a, b, x) => { const t = clamp01((x - a) / (b - a)); return t * t * (3 - 2 * t); };
const lerp = (a, b, t) => a + (b - a) * t;
const DEG = Math.PI / 180;
const REDUCED = matchMedia("(prefers-reduced-motion: reduce)").matches;
const fmtInt = (n) => n.toLocaleString("en-US");

function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function makeNoise3(seed) {
  const rnd = mulberry32(seed);
  const perm = Array.from({ length: 256 }, (_, i) => i);
  for (let i = 255; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [perm[i], perm[j]] = [perm[j], perm[i]]; }
  const p = new Uint8Array(512);
  for (let i = 0; i < 512; i++) p[i] = perm[i & 255];
  const G = [1,1,0, -1,1,0, 1,-1,0, -1,-1,0, 1,0,1, -1,0,1, 1,0,-1, -1,0,-1, 0,1,1, 0,-1,1, 0,1,-1, 0,-1,-1];
  const fade = (t) => t * t * t * (t * (t * 6 - 15) + 10);
  const lerp = (a, b, t) => a + (b - a) * t;
  const grad = (h, x, y, z) => { const k = (h % 12) * 3; return G[k] * x + G[k + 1] * y + G[k + 2] * z; };
  return function (x, y, z) {
    const X = Math.floor(x), Y = Math.floor(y), Z = Math.floor(z);
    x -= X; y -= Y; z -= Z;
    const xi = X & 255, yi = Y & 255, zi = Z & 255;
    const u = fade(x), v = fade(y), w = fade(z);
    const A = p[xi] + yi, AA = p[A] + zi, AB = p[A + 1] + zi;
    const B = p[xi + 1] + yi, BA = p[B] + zi, BB = p[B + 1] + zi;
    return lerp(
      lerp(lerp(grad(p[AA], x, y, z), grad(p[BA], x - 1, y, z), u),
           lerp(grad(p[AB], x, y - 1, z), grad(p[BB], x - 1, y - 1, z), u), v),
      lerp(lerp(grad(p[AA + 1], x, y, z - 1), grad(p[BA + 1], x - 1, y, z - 1), u),
           lerp(grad(p[AB + 1], x, y - 1, z - 1), grad(p[BB + 1], x - 1, y - 1, z - 1), u), v), w);
  };
}
const noise = makeNoise3(17);

// ------------------------------------------------------------------ config
const SMALL = innerWidth < 860 || (navigator.hardwareConcurrency || 8) <= 4;
const SIDE = SMALL ? 256 : 512;
const N = SIDE * SIDE;
const FLOOR_N = SMALL ? 30000 : 90000;
const T_END = 6.5;
const SECTION_SHAPE = [0, 1, 2, 3, 4, 5, 0];     // wordmark, Cascade, Drift, Stem, Skincare, Axis, wordmark
const SHAPE_NAMES = ["Wordmark", "Cascade", "Drift", "Stem", "Skincare set", "Axis"];
const MODELS = ["drift", "stem", "skincare", "axis"];
const YOLK = "#F4A300";
const NOTES = [880, 987.77, 1108.73, 1318.51, 1479.98];

// ground and heap, identical in JS and GLSL
function groundJS(x, z) {
  const r = Math.hypot(x, z);
  const d = (Math.sin(x * 1.1 + Math.sin(z * 0.6) * 1.8) * 0.5 + 0.5) * 0.16
          + (Math.sin(z * 1.9 + x * 0.5 + 1.3) * 0.5 + 0.5) * 0.06;
  return d * smooth(1.7, 3.8, r) - 0.012;
}
const restJS = (x, z, seed) => groundJS(x, z) + 0.36 * Math.exp(-(x * x + z * z) / 1.4) * seed;

const GLSL_COMMON = /* glsl */ `
float ground(vec2 xz){
  float r = length(xz);
  float d = (sin(xz.x * 1.1 + sin(xz.y * 0.6) * 1.8) * 0.5 + 0.5) * 0.16
          + (sin(xz.y * 1.9 + xz.x * 0.5 + 1.3) * 0.5 + 0.5) * 0.06;
  return d * smoothstep(1.7, 3.8, r) - 0.012;
}
float restH(vec2 xz, float seed){ return ground(xz) + 0.36 * exp(-dot(xz, xz) / 1.4) * seed; }
vec3 swirl(vec3 p, float t){
  return vec3(
    sin(p.y * 1.7 + t) + sin(p.z * 2.3 - t * 0.7) * 0.5,
    sin(p.z * 1.5 + t * 0.8) + sin(p.x * 2.1 + t * 0.5) * 0.5,
    sin(p.x * 1.9 - t * 0.6) + sin(p.y * 2.7 + t * 0.9) * 0.5);
}
`;

// ------------------------------------------------------------------ DOM
const body = document.body;
const canvas = $("#gl");
const loader = $("#loader");
const ldPct = $("#ldPct"), ldBar = $("#ldBar"), ldLine = $("#ldLine");
const hud = $("#hud"), hint = $("#hint");
const soundBtn = $("#soundBtn");
const indexItems = $$("#index li");
const scroller = $("#scroller");
$("#grainCount").textContent = fmtInt(N);
if ("scrollRestoration" in history) history.scrollRestoration = "manual";
window.scrollTo(0, 0);

// ---------------------------------------------------------- text effects
const GLYPHS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/=+-<>*";
function scramble(el, text, dur = 700) {
  if (text == null) text = el.dataset.text || el.textContent;
  el.dataset.text = text;
  if (REDUCED) { el.textContent = text; return; }
  cancelAnimationFrame(el._scr || 0);
  const t0 = performance.now();
  const order = [...text].map((_, i) => (i / Math.max(text.length, 1)) * 0.65 + Math.random() * 0.35);
  const step = (now) => {
    const p = (now - t0) / dur;
    let out = "";
    for (let i = 0; i < text.length; i++) out += text[i] === " " || p >= order[i] ? text[i] : GLYPHS[(Math.random() * GLYPHS.length) | 0];
    el.textContent = out;
    if (p < 1) el._scr = requestAnimationFrame(step); else el.textContent = text;
  };
  el._scr = requestAnimationFrame(step);
}
function splitWords(el) {
  let n = 0;
  const walk = (node) => {
    for (const child of [...node.childNodes]) {
      if (child.nodeType === 3) {
        const frag = document.createDocumentFragment();
        for (const part of child.textContent.split(/([ \t\r\n]+)/)) {
          if (!part) continue;
          if (/^[ \t\r\n]+$/.test(part)) { frag.appendChild(document.createTextNode(part)); continue; }
          const w = document.createElement("span"); w.className = "w";
          const i = document.createElement("span"); i.textContent = part;
          i.style.setProperty("--d", (n++ * 0.026).toFixed(3) + "s");
          w.appendChild(i); frag.appendChild(w);
        }
        child.replaceWith(frag);
      } else if (child.nodeType === 1) walk(child);
    }
  };
  walk(el);
}
$$(".title, .sub, .line, .name").forEach(splitWords);

// ------------------------------------------------------------------ sound
// Synthesised: a low wind bed, a sand hiss that follows how much is moving,
// and chimes on the partials of a free bar.
class Sound {
  constructor() { this.on = false; this.ctx = null; }
  ensure() {
    if (this.ctx) return;
    const AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) return;
    const ctx = (this.ctx = new AC());
    this.master = ctx.createGain(); this.master.gain.value = 0; this.master.connect(ctx.destination);
    const len = Math.floor(ctx.sampleRate * 2.5);
    const ir = ctx.createBuffer(2, len, ctx.sampleRate);
    for (let ch = 0; ch < 2; ch++) { const d = ir.getChannelData(ch); for (let i = 0; i < len; i++) d[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / len, 3); }
    const verb = ctx.createConvolver(); verb.buffer = ir;
    const vg = ctx.createGain(); vg.gain.value = 0.35; verb.connect(vg).connect(this.master);
    this.dry = ctx.createGain(); this.dry.connect(this.master); this.dry.connect(verb);
    const brown = ctx.createBuffer(1, ctx.sampleRate * 4, ctx.sampleRate);
    const bd = brown.getChannelData(0); let last = 0;
    for (let i = 0; i < bd.length; i++) { last = (last + 0.02 * (Math.random() * 2 - 1)) / 1.02; bd[i] = last * 3.5; }
    const white = ctx.createBuffer(1, ctx.sampleRate * 4, ctx.sampleRate);
    const wd = white.getChannelData(0); for (let i = 0; i < wd.length; i++) wd[i] = Math.random() * 2 - 1;
    const loop = (buf) => { const s = ctx.createBufferSource(); s.buffer = buf; s.loop = true; s.start(); return s; };
    const wlp = ctx.createBiquadFilter(); wlp.type = "lowpass"; wlp.frequency.value = 420;
    const wg = ctx.createGain(); wg.gain.value = 0.06;
    loop(brown).connect(wlp).connect(wg).connect(this.master);
    const hbp = ctx.createBiquadFilter(); hbp.type = "bandpass"; hbp.frequency.value = 3800; hbp.Q.value = 0.5;
    this.hiss = ctx.createGain(); this.hiss.gain.value = 0;
    loop(white).connect(hbp).connect(this.hiss).connect(this.master);
  }
  set(on) {
    this.on = on;
    soundBtn.setAttribute("aria-pressed", String(on));
    soundBtn.querySelector("b").textContent = on ? "on" : "off";
    if (on) { this.ensure(); if (this.ctx) this.ctx.resume(); }
    if (!this.ctx) return;
    this.master.gain.setTargetAtTime(on ? 1 : 0, this.ctx.currentTime, 0.35);
  }
  activity(level) {
    if (!this.ctx) return;
    this.hiss.gain.setTargetAtTime(0.004 + level * 0.05, this.ctx.currentTime, 0.12);
  }
  chime(f, vel = 0.07) {
    if (!this.on || !this.ctx) return;
    const ctx = this.ctx, t = ctx.currentTime;
    for (const [r, a, d] of [[1, 1, 2.4], [2.756, 0.45, 1.3], [5.404, 0.2, 0.6]]) {
      const o = ctx.createOscillator(); o.frequency.value = f * r;
      const g = ctx.createGain();
      g.gain.setValueAtTime(0, t); g.gain.linearRampToValueAtTime(vel * a, t + 0.004); g.gain.exponentialRampToValueAtTime(0.0001, t + d);
      o.connect(g).connect(this.dry); o.start(t); o.stop(t + d + 0.05);
    }
  }
}
const sound = new Sound();

// --------------------------------------------------------------- renderer
let renderer = null;
try { renderer = new THREE.WebGLRenderer({ canvas, antialias: false, powerPreference: "high-performance" }); } catch (e) { renderer = null; }
if (!renderer || !renderer.capabilities.isWebGL2) {
  $("#noGL").hidden = false;
  loader.remove();
  throw new Error("WebGL 2 unavailable");
}
renderer.outputColorSpace = THREE.SRGBColorSpace;

const scene = new THREE.Scene();
scene.background = (() => {
  const c = document.createElement("canvas");
  c.width = 32; c.height = 512;
  const g = c.getContext("2d");
  const grd = g.createLinearGradient(0, 0, 0, 512);
  grd.addColorStop(0, "#060607");
  grd.addColorStop(0.52, "#101013");
  grd.addColorStop(0.6, "#131316");
  grd.addColorStop(1, "#070708");
  g.fillStyle = grd; g.fillRect(0, 0, 32, 512);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
})();
const camera = new THREE.PerspectiveCamera(32, 1, 0.05, 60);
const view = { w: 1, h: 1, pr: 1, aspect: 1, narrow: false };

// ------------------------------------------------------------------ shapes
// One part of a quantised web mesh: int16 xyz against the part's centre and
// scale, int8 normal, a pad byte; then uint16 triangles.
function partTriangles(buf, p) {
  return {
    pos: new Int16Array(buf, p.vertex_at, p.vertices * 5),
    occ: new Uint8Array(buf, p.vertex_at, p.vertices * 10),   // byte 9 of each vertex: baked AO
    idx: new Uint16Array(buf, p.index_at, p.triangles * 3),
    org: p.org, scale: p.scale,
  };
}

// Share of grains that trace the sharp edges instead of covering the faces.
// The web meshes split their vertices at every crease over 32 degrees, so an
// edge used by only one triangle IS a feature edge: the outline a designer
// would draw. A few grains there make a sculpture read as a form, not a haze.
const EDGE_FRAC = 0.16;

// Area-weighted surface samples plus length-weighted edge samples, each with a
// face normal, baked AO and an accent flag. Rhino is Z-up; (x, y, z) -> (x, z, -y).
function sampleAssembly(parts, n) {
  let triCount = 0;
  for (const p of parts) triCount += p.idx.length / 3;
  const cum = new Float64Array(triCount);
  const owner = new Uint32Array(triCount);
  const worlds = [];
  const edges = [];              // [part, vertex a, vertex b, triangle]
  const edgeCum = [];
  let total = 0, edgeTotal = 0, k = 0;
  parts.forEach((p, pi) => {
    const nv = p.pos.length / 5;
    const w = new Float32Array(nv * 3);
    for (let i = 0; i < nv; i++) {
      const x = p.org[0] + p.pos[i * 5] * p.scale;
      const y = p.org[1] + p.pos[i * 5 + 1] * p.scale;
      const z = p.org[2] + p.pos[i * 5 + 2] * p.scale;
      w[i * 3] = x; w[i * 3 + 1] = z; w[i * 3 + 2] = -y;
    }
    worlds.push(w);
    const idx = p.idx;
    const occ = (i) => p.occ[i * 10 + 9] / 255;
    const seen = new Map();      // edge key -> first triangle that used it, or -1 once shared
    for (let t = 0; t < idx.length; t += 3) {
      const a = idx[t] * 3, b = idx[t + 1] * 3, c = idx[t + 2] * 3;
      const ux = w[b] - w[a], uy = w[b + 1] - w[a + 1], uz = w[b + 2] - w[a + 2];
      const vx = w[c] - w[a], vy = w[c + 1] - w[a + 1], vz = w[c + 2] - w[a + 2];
      const cx = uy * vz - uz * vy, cy = uz * vx - ux * vz, cz = ux * vy - uy * vx;
      // grains go where they can be seen: an enclosed face (a driver inside the
      // housing, a bolt inside its knuckle) gets a sixth of its share
      const open = (occ(idx[t]) + occ(idx[t + 1]) + occ(idx[t + 2])) / 3;
      total += Math.sqrt(cx * cx + cy * cy + cz * cz) * 0.5 * (0.16 + 0.84 * open);
      cum[k] = total; owner[k] = pi; k++;
      for (let e = 0; e < 3; e++) {
        const i0 = idx[t + e], i1 = idx[t + (e + 1) % 3];
        const key = i0 < i1 ? i0 * 65536 + i1 : i1 * 65536 + i0;
        seen.set(key, seen.has(key) ? -1 : t);
      }
    }
    for (const [key, t] of seen) {
      if (t < 0) continue;
      const i0 = Math.floor(key / 65536), i1 = key % 65536;
      const len = Math.hypot(w[i1 * 3] - w[i0 * 3], w[i1 * 3 + 1] - w[i0 * 3 + 1], w[i1 * 3 + 2] - w[i0 * 3 + 2])
        * (0.08 + 0.92 * (occ(i0) + occ(i1)) / 2);
      if (len <= 0) continue;
      edgeTotal += len;
      edges.push(pi, i0, i1, t);
      edgeCum.push(edgeTotal);
    }
  });
  const firstTri = []; let off = 0;
  for (const p of parts) { firstTri.push(off); off += p.idx.length / 3; }

  const pos = new Float32Array(n * 3), nor = new Float32Array(n * 3);
  const acc = new Float32Array(n), ao = new Float32Array(n), edge = new Float32Array(n);
  const aoOf = (p, i) => p.occ[i * 10 + 9] / 255;
  const faceNormal = (w, a, b, c, s) => {
    const ux = w[b] - w[a], uy = w[b + 1] - w[a + 1], uz = w[b + 2] - w[a + 2];
    const vx = w[c] - w[a], vy = w[c + 1] - w[a + 1], vz = w[c + 2] - w[a + 2];
    const cx = uy * vz - uz * vy, cy = uz * vx - ux * vz, cz = ux * vy - uy * vx;
    const l = Math.hypot(cx, cy, cz) || 1;
    nor[s * 3] = cx / l; nor[s * 3 + 1] = cy / l; nor[s * 3 + 2] = cz / l;
  };
  const nEdge = edges.length ? Math.round(n * EDGE_FRAC) : 0;
  for (let s = 0; s < n; s++) {
    if (s < n - nEdge) {
      const x = Math.random() * total;
      let lo = 0, hi = triCount - 1;
      while (lo < hi) { const m = (lo + hi) >> 1; if (cum[m] < x) lo = m + 1; else hi = m; }
      const pi = owner[lo], p = parts[pi], w = worlds[pi];
      const t = (lo - firstTri[pi]) * 3;
      const ia = p.idx[t], ib = p.idx[t + 1], ic = p.idx[t + 2];
      const a = ia * 3, b = ib * 3, c = ic * 3;
      let u = Math.random(), v = Math.random();
      if (u + v > 1) { u = 1 - u; v = 1 - v; }
      for (let d = 0; d < 3; d++) pos[s * 3 + d] = w[a + d] + (w[b + d] - w[a + d]) * u + (w[c + d] - w[a + d]) * v;
      faceNormal(w, a, b, c, s);
      ao[s] = aoOf(p, ia) * (1 - u - v) + aoOf(p, ib) * u + aoOf(p, ic) * v;
      acc[s] = p.accent ? 1 : 0;
    } else {
      const x = Math.random() * edgeTotal;
      let lo = 0, hi = edgeCum.length - 1;
      while (lo < hi) { const m = (lo + hi) >> 1; if (edgeCum[m] < x) lo = m + 1; else hi = m; }
      const pi = edges[lo * 4], i0 = edges[lo * 4 + 1], i1 = edges[lo * 4 + 2], t = edges[lo * 4 + 3];
      const p = parts[pi], w = worlds[pi], f = Math.random();
      for (let d = 0; d < 3; d++) pos[s * 3 + d] = w[i0 * 3 + d] + (w[i1 * 3 + d] - w[i0 * 3 + d]) * f;
      faceNormal(w, p.idx[t] * 3, p.idx[t + 1] * 3, p.idx[t + 2] * 3, s);
      ao[s] = aoOf(p, i0) * (1 - f) + aoOf(p, i1) * f;
      acc[s] = p.accent ? 1 : 0;
      edge[s] = 1;
    }
  }
  return { pos, nor, acc, ao, edge, area: total };
}

function sampleWordmark(n) {
  const W = 640, H = 240, S = 2;
  const c = document.createElement("canvas");
  c.width = W; c.height = H;
  const g = c.getContext("2d", { willReadFrequently: true });
  g.setTransform(S, 0, 0, S, 10, 16);
  g.fillStyle = "#0f0";
  g.fill(new Path2D("M0 50 a51.5 51.5 0 1 0 103 0 a51.5 51.5 0 1 0 -103 0 Z M17 50 a34.5 34.5 0 1 0 69 0 a34.5 34.5 0 1 0 -69 0 Z"), "evenodd");
  g.fill(new Path2D("M207 50 a51.5 51.5 0 1 0 103 0 a51.5 51.5 0 1 0 -103 0 Z M224 50 a34.5 34.5 0 1 0 69 0 a34.5 34.5 0 1 0 -69 0 Z"), "evenodd");
  g.fill(new Path2D("M41.5 50 a10 10 0 1 0 20 0 a10 10 0 1 0 -20 0 Z"));
  g.fill(new Path2D("M248.5 50 a10 10 0 1 0 20 0 a10 10 0 1 0 -20 0 Z"));
  g.fillStyle = "#f00";
  g.fill(new Path2D("M109 0 L132.85 85.29 A23 23 0 0 0 177.15 85.29 L201 0 L182.31 0 L159.82 80.45 A5 5 0 0 1 150.18 80.45 L127.69 0 Z"));
  const data = g.getImageData(0, 0, W, H).data;
  const filled = (x, y) => x >= 0 && y >= 0 && x < W && y < H && data[(y * W + x) * 4 + 3] > 140;
  const cells = [], rims = [];
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    if (!filled(x, y)) continue;
    const k = (y * W + x) * 4;
    const accent = data[k] > data[k + 1] ? 1 : 0;
    cells.push(x, y, accent);
    // a filled pixel with an empty neighbour is on the outline
    if (!filled(x - 1, y) || !filled(x + 1, y) || !filled(x, y - 1) || !filled(x, y + 1)) rims.push(x, y, accent);
  }
  const count = cells.length / 3, rimCount = rims.length / 3;
  const pos = new Float32Array(n * 3), nor = new Float32Array(n * 3);
  const acc = new Float32Array(n), ao = new Float32Array(n).fill(1), edge = new Float32Array(n);
  const depth = 14;                      // mark units, front to back
  const nEdge = Math.round(n * EDGE_FRAC);
  for (let i = 0; i < n; i++) {
    const onRim = i >= n - nEdge;
    const src = onRim ? rims : cells;
    const j = Math.floor(Math.random() * (onRim ? rimCount : count)) * 3;
    const front = onRim || Math.random() < 0.62;   // the camera mostly sees the front
    pos[i * 3] = (src[j] + Math.random() - 10) / S - 155;
    pos[i * 3 + 1] = 104 - ((src[j + 1] + Math.random() - 16) / S);
    pos[i * 3 + 2] = (front ? 0.5 : -0.5) * depth + (Math.random() - 0.5) * 1.5;
    nor[i * 3 + 2] = front ? 1 : -1;
    acc[i] = src[j + 2];
    edge[i] = onRim ? 1 : 0;
  }
  // area of both faces, in mark units
  const area = (count / (S * S)) * 2;
  return { pos, nor, acc, ao, edge, area };
}

// normalise, set it on the ground, and give every grain its two thresholds
function finishShape(sample, size, erodeSeed) {
  const { pos, nor, acc, ao, edge } = sample;
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < N; i++) for (let d = 0; d < 3; d++) {
    const v = pos[i * 3 + d];
    if (v < lo[d]) lo[d] = v;
    if (v > hi[d]) hi[d] = v;
  }
  const k = size / Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
  const cx = (lo[0] + hi[0]) / 2, cz = (lo[2] + hi[2]) / 2;
  const height = (hi[1] - lo[1]) * k;
  const width = (hi[0] - lo[0]) * k;
  const target = new Float32Array(N * 4), meta = new Float32Array(N * 4);
  for (let i = 0; i < N; i++) {
    const x = (pos[i * 3] - cx) * k, y = (pos[i * 3 + 1] - lo[1]) * k + 0.004, z = (pos[i * 3 + 2] - cz) * k;
    const yn = height > 0 ? y / height : 0;
    const xn = width > 0 ? x / width + 0.5 : 0.5;
    // bottom-up, ragged: the sculpture grows out of the heap
    const rF = clamp01(0.8 * yn + 0.2 * (0.5 + 0.5 * noise(x * 2.4, y * 2.4, z * 2.4)));
    // windward face and top go first
    const rE = clamp01(0.62 * xn + 0.16 * (1 - yn) + 0.22 * (0.5 + 0.5 * noise(x * 3.1 + erodeSeed, y * 3.1, z * 3.1)));
    target[i * 4] = x; target[i * 4 + 1] = y; target[i * 4 + 2] = z; target[i * 4 + 3] = rF;
    meta[i * 4] = nor[i * 3]; meta[i * 4 + 1] = nor[i * 3 + 1]; meta[i * 4 + 2] = nor[i * 3 + 2];
    // meta.w packs four things: rErode in [0,1), +2 yolk grain, +4 edge grain,
    // +8 x AO quantised to 0..31. Every flag is a multiple of 2, so the
    // simulation can still read rErode as mod(w, 2).
    const aoq = Math.round(clamp01(ao[i]) * 31);
    meta[i * 4 + 3] = Math.min(rE, 0.999) + 2 * acc[i] + 4 * edge[i] + 8 * aoq;
  }
  const tex = (arr) => {
    const t = new THREE.DataTexture(arr, SIDE, SIDE, THREE.RGBAFormat, THREE.FloatType);
    t.minFilter = t.magFilter = THREE.NearestFilter;
    t.needsUpdate = true;
    return t;
  };
  const area = sample.area * k * k;
  return { target: tex(target), meta: tex(meta), height, gain: clamp(Math.sqrt(area / 7), 0.6, 1.45) };
}

// ------------------------------------------------------------------ loading
let ldShown = 0, ldTarget = 0, ldDone = false;
(function paintLoader() {
  ldShown += (ldTarget - ldShown) * 0.1;
  if (ldDone && ldShown > 0.995) ldShown = 1;
  ldPct.textContent = String(Math.round(ldShown * 100)).padStart(3, "0");
  ldBar.style.transform = `scaleX(${ldShown.toFixed(4)})`;
  if (ldShown < 1) requestAnimationFrame(paintLoader);
})();

async function fetchText(url, onProgress, expected) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(url + " " + res.status);
  if (!res.body || !res.body.getReader) { const t = await res.text(); onProgress(1); return t; }
  const reader = res.body.getReader();
  const chunks = []; let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value); got += value.length;
    onProgress(Math.min(got / expected, 1));
  }
  const all = new Uint8Array(got); let o = 0;
  for (const c of chunks) { all.set(c, o); o += c.length; }
  return new TextDecoder().decode(all);
}
const decode64 = (t) => Uint8Array.from(atob(t.trim()), (ch) => ch.charCodeAt(0)).buffer;
const tick = () => new Promise((r) => setTimeout(r, 0));

let shapes = [];
let gpu, posVar, velVar, grains, floorPts;

async function load() {
  const names = ["cascade", ...MODELS];
  const metas = await Promise.all(names.map((m, i) =>
    fetch(i === 0 ? "assets/cascade.json" : `assets/products/${m}.json`).then((r) => r.json())));
  const sizes = metas.map((m) => Math.ceil(m.bytes / 3) * 4);
  const totalBytes = sizes.reduce((s, v) => s + v, 0);
  const got = sizes.map(() => 0);
  const texts = await Promise.all(names.map((m, i) =>
    fetchText(i === 0 ? "assets/cascade.b64.txt" : `assets/products/${m}.b64.txt`,
      (p) => { got[i] = p * sizes[i]; ldTarget = 0.5 * got.reduce((s, v) => s + v, 0) / totalBytes; }, sizes[i])));
  const bufs = texts.map(decode64);

  // grains, sampled shape by shape so the loader keeps moving
  ldLine.textContent = `Sampling ${fmtInt(N * 6)} grains from 5 assemblies`;
  await tick();
  shapes.push(finishShape(sampleWordmark(N), 2.8, 0));
  ldTarget = 0.56; await tick();

  const cascadeParts = metas[0].parts
    .filter((p) => !p.name.startsWith("marbles"))
    .map((p) => Object.assign(partTriangles(bufs[0], p), { accent: p.name.startsWith("crank_knob") || p.name.startsWith("flip_flop_gate") }));
  shapes.push(finishShape(sampleAssembly(cascadeParts, N), 2.45, 11));
  ldTarget = 0.64; await tick();

  const sizesByModel = { drift: 2.3, stem: 2.4, skincare: 2.3, axis: 2.3 };
  for (let m = 0; m < MODELS.length; m++) {
    const meta = metas[m + 1], buf = bufs[m + 1];
    const parts = meta.parts.map((p) => Object.assign(partTriangles(buf, p), { accent: p.color.toUpperCase() === YOLK }));
    shapes.push(finishShape(sampleAssembly(parts, N), sizesByModel[MODELS[m]], 23 + m * 7));
    ldTarget = 0.64 + 0.06 * (m + 1); await tick();
  }

  initSimulation();
  ldTarget = 0.94; await tick();
  resize();
  try { if (renderer.compileAsync) await renderer.compileAsync(scene, camera); } catch (e) { /* compile on first frame */ }
  ldLine.textContent = `${fmtInt(N)} grains · 6 sculptures · ready`;
  ldTarget = 1; ldDone = true;
  loader.classList.add("ready");
  requestAnimationFrame(frame);

  // ?t=2.5 enters silently and jumps to that point of the scroll, for stills
  const pin = new URLSearchParams(location.search).get("t");
  if (pin !== null) {
    $("[data-enter='0']").click();
    const t = clamp(parseFloat(pin) || 0, 0, T_END);
    window.scrollTo(0, t * innerHeight);
    T = t;
    enterTime = performance.now() - 4000;     // skip the intro rise
  }
}

// ------------------------------------------------------------ simulation
const VEL_FRAG = /* glsl */ `
uniform float uDt, uTime, uForm, uErode, uK, uPush, uPushR, uGust;
uniform vec3 uRayO, uRayD, uWind;
uniform sampler2D tTarget, tMeta;
${GLSL_COMMON}
void main(){
  vec2 uv = gl_FragCoord.xy / resolution.xy;
  vec4 P = texture2D(texturePosition, uv);
  vec4 V = texture2D(textureVelocity, uv);
  vec4 T = texture2D(tTarget, uv);
  vec4 M = texture2D(tMeta, uv);
  float rE = mod(M.w, 2.0);
  float bound = step(T.w, uForm) * step(uErode, rE);
  vec3 p = P.xyz, v = V.xyz;
  float seed = P.w;
  float grounded = step(p.y, restH(p.xz, seed) + 0.004);
  vec3 acc = vec3(0.0);
  if (bound > 0.5) {
    vec3 d = T.xyz - p;
    float dist = length(d);
    acc += d * uK - v * (1.6 * sqrt(uK));
    acc += swirl(p * 1.6, uTime * 0.6) * min(dist, 1.2) * 5.0;   // a sand storm while travelling
    acc.y += grounded * 6.0;                                     // unstick from the heap
  } else {
    float air = 1.0 - grounded;
    acc.y -= 7.5;
    acc += uWind * (0.3 + uGust) * air;
    acc += swirl(p * 1.3, uTime * 0.5) * (0.6 + uGust * 3.0) * air;
    v *= exp(-0.35 * uDt);
  }
  // the cursor is a blower pointed along the view ray
  vec3 o = p - uRayO;
  float along = dot(o, uRayD);
  vec3 q = o - uRayD * along;
  float dq = length(q);
  float f = uPush * smoothstep(uPushR, 0.0, dq) * step(0.0, along);
  acc += (q / max(dq, 1e-4)) * f * 26.0 + uRayD * f * 6.0 + vec3(0.0, f * 5.0, 0.0);
  v += acc * uDt;
  if (bound < 0.5) {
    vec3 np = p + v * uDt;
    if (np.y <= restH(np.xz, seed)) { v.y = max(v.y, 0.0); v.xz *= exp(-14.0 * uDt); }
  }
  float sp = length(v);
  if (sp > 9.0) v *= 9.0 / sp;
  gl_FragColor = vec4(v, mix(V.w, bound, 1.0 - exp(-uDt * 8.0)));
}`;

const POS_FRAG = /* glsl */ `
uniform float uDt;
${GLSL_COMMON}
void main(){
  vec2 uv = gl_FragCoord.xy / resolution.xy;
  vec4 P = texture2D(texturePosition, uv);
  vec4 V = texture2D(textureVelocity, uv);
  vec3 p = P.xyz + V.xyz * uDt;
  float floorY = V.w > 0.5 ? ground(p.xz) : restH(p.xz, P.w);
  p.y = max(p.y, floorY);
  float r = length(p.xz);
  if (r > 5.5) p.xz *= 5.5 / r;
  gl_FragColor = vec4(p, P.w);
}`;

const GRAIN_VERT = /* glsl */ `
uniform sampler2D tPos, tVel, tMeta;
uniform float uPR, uSize, uGain;
uniform vec3 uLight;
attribute vec2 ref;
attribute float seed;
varying vec3 vCol;
void main(){
  vec4 P = texture2D(tPos, ref);
  vec4 V = texture2D(tVel, ref);
  vec4 M = texture2D(tMeta, ref);
  vec3 p = P.xyz;
  float bound = clamp(V.w, 0.0, 1.0);
  // meta.w: rErode + 2 yolk + 4 edge + 8 * AO(0..31)
  float occ = floor(M.w / 8.0) / 31.0;
  float flags = mod(M.w, 8.0);
  float edge = step(3.99, flags) * bound;
  float accent = step(1.99, mod(flags, 4.0)) * bound;
  vec3 n = normalize(mix(vec3(0.0, 1.0, 0.0), M.xyz, bound));
  vec3 toCam = normalize(cameraPosition - p);
  float nl = dot(n, uLight);
  float lit = 0.2 + 0.95 * max(nl, 0.0) + 0.15 * (nl * 0.5 + 0.5);
  // crevices and contacts go dark, as they do on the real part
  lit *= mix(1.0, 0.08 + 0.92 * smoothstep(0.0, 0.85, occ), bound);
  // grains on the far side of a sculpture dim hard, so it reads as a solid, not an x-ray
  lit *= mix(1.0, mix(0.14, 1.0, smoothstep(-0.2, 0.3, dot(n, toCam))), bound);
  // the drawn edges carry the form
  lit *= 1.0 + edge * 0.6 * smoothstep(0.2, 0.8, occ);
  float speed = length(V.xyz);
  float hot = smoothstep(0.6, 3.5, speed);
  // a heap is thousands of grains stacked on one spot: resting grains dim, or it burns out
  lit *= mix(0.32, 1.0, max(bound, hot));
  vec3 sand = mix(vec3(0.95, 0.91, 0.84), vec3(0.62, 0.58, 0.52), seed * seed);
  vec3 yolk = vec3(1.0, 0.62, 0.05);
  vec3 c = mix(sand, yolk, max(accent, hot * 0.9)) * lit + yolk * hot * 0.5;
  vCol = c * uGain * 0.5;
  vec4 mv = viewMatrix * vec4(p, 1.0);
  gl_Position = projectionMatrix * mv;
  gl_PointSize = uSize * uPR * (0.7 + seed * 0.6) * (1.0 + hot * 0.5) * (1.0 - edge * 0.15) / max(-mv.z, 0.1);
}`;

const FLOOR_VERT = /* glsl */ `
uniform float uPR, uSize;
uniform vec3 uLight;
attribute vec2 xz;
attribute float seed;
varying vec3 vCol;
${GLSL_COMMON}
void main(){
  vec3 p = vec3(xz.x, ground(xz) + seed * 0.008, xz.y);
  // slope from the ground's own gradient, so dunes catch the light
  float e = 0.02;
  vec3 n = normalize(vec3(ground(xz - vec2(e, 0.0)) - ground(xz + vec2(e, 0.0)), 2.0 * e, ground(xz - vec2(0.0, e)) - ground(xz + vec2(0.0, e))));
  float lit = 0.25 + 0.9 * max(dot(n, uLight), 0.0);
  vec4 mv = viewMatrix * vec4(p, 1.0);
  float fade = 1.0 - smoothstep(3.2, 5.8, length(xz));
  vCol = mix(vec3(0.9, 0.86, 0.79), vec3(0.6, 0.56, 0.5), seed) * lit * 0.2 * fade;
  gl_Position = projectionMatrix * mv;
  gl_PointSize = uSize * uPR * (0.6 + seed * 0.6) / max(-mv.z, 0.1);
}`;

const DOT_FRAG = /* glsl */ `
varying vec3 vCol;
void main(){
  vec2 c = gl_PointCoord - 0.5;
  float r2 = dot(c, c);
  if (r2 > 0.25) discard;
  gl_FragColor = vec4(vCol * smoothstep(0.25, 0.04, r2), 1.0);
}`;

const LIGHT = new THREE.Vector3(-0.55, 0.78, 0.45).normalize();

function initSimulation() {
  gpu = new GPUComputationRenderer(SIDE, SIDE, renderer);
  if (!renderer.extensions.has("EXT_color_buffer_float")) gpu.setDataType(THREE.HalfFloatType);
  const pos0 = gpu.createTexture(), vel0 = gpu.createTexture();
  const pa = pos0.image.data, va = vel0.image.data;
  const rnd = mulberry32(5);
  for (let i = 0; i < N; i++) {
    const r = 2.8 * Math.sqrt(rnd()), a = rnd() * Math.PI * 2, s = rnd();
    const x = Math.cos(a) * r, z = Math.sin(a) * r;
    pa[i * 4] = x; pa[i * 4 + 1] = restJS(x, z, s) * rnd(); pa[i * 4 + 2] = z; pa[i * 4 + 3] = s;
    va[i * 4 + 3] = 0;
  }
  posVar = gpu.addVariable("texturePosition", POS_FRAG, pos0);
  velVar = gpu.addVariable("textureVelocity", VEL_FRAG, vel0);
  gpu.setVariableDependencies(posVar, [posVar, velVar]);
  gpu.setVariableDependencies(velVar, [posVar, velVar]);
  Object.assign(posVar.material.uniforms, { uDt: { value: 0.016 } });
  Object.assign(velVar.material.uniforms, {
    uDt: { value: 0.016 }, uTime: { value: 0 }, uForm: { value: 0 }, uErode: { value: 0 },
    uK: { value: 16 }, uPush: { value: 0 }, uPushR: { value: 0.32 }, uGust: { value: 0 },
    uRayO: { value: new THREE.Vector3() }, uRayD: { value: new THREE.Vector3(0, 0, -1) },
    uWind: { value: new THREE.Vector3(2.4, 0.5, 0.6) },
    tTarget: { value: shapes[0].target }, tMeta: { value: shapes[0].meta },
  });
  const err = gpu.init();
  if (err) throw new Error(err);

  // the grains
  const g = new THREE.BufferGeometry();
  const ref = new Float32Array(N * 2), seed = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    ref[i * 2] = ((i % SIDE) + 0.5) / SIDE;
    ref[i * 2 + 1] = (Math.floor(i / SIDE) + 0.5) / SIDE;
    seed[i] = pa[i * 4 + 3];
  }
  g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(N * 3), 3));
  g.setAttribute("ref", new THREE.BufferAttribute(ref, 2));
  g.setAttribute("seed", new THREE.BufferAttribute(seed, 1));
  grains = new THREE.Points(g, new THREE.ShaderMaterial({
    vertexShader: GRAIN_VERT, fragmentShader: DOT_FRAG,
    transparent: true, depthWrite: false, depthTest: false, blending: THREE.AdditiveBlending,
    uniforms: {
      tPos: { value: null }, tVel: { value: null }, tMeta: { value: shapes[0].meta },
      uPR: { value: 1 }, uSize: { value: SMALL ? 13 : 10 }, uGain: { value: 1 }, uLight: { value: LIGHT },
    },
  }));
  grains.frustumCulled = false;
  scene.add(grains);

  // a still bed of sand, so the ground reads even when every grain is up in a sculpture
  const fg = new THREE.BufferGeometry();
  const xz = new Float32Array(FLOOR_N * 2), fs = new Float32Array(FLOOR_N);
  for (let i = 0; i < FLOOR_N; i++) {
    const r = 6 * Math.pow(rnd(), 0.62), a = rnd() * Math.PI * 2;
    xz[i * 2] = Math.cos(a) * r; xz[i * 2 + 1] = Math.sin(a) * r; fs[i] = rnd();
  }
  fg.setAttribute("position", new THREE.BufferAttribute(new Float32Array(FLOOR_N * 3), 3));
  fg.setAttribute("xz", new THREE.BufferAttribute(xz, 2));
  fg.setAttribute("seed", new THREE.BufferAttribute(fs, 1));
  floorPts = new THREE.Points(fg, new THREE.ShaderMaterial({
    vertexShader: FLOOR_VERT, fragmentShader: DOT_FRAG,
    transparent: true, depthWrite: false, depthTest: false, blending: THREE.AdditiveBlending,
    uniforms: { uPR: { value: 1 }, uSize: { value: SMALL ? 12 : 9 }, uLight: { value: LIGHT } },
  }));
  floorPts.frustumCulled = false;
  floorPts.renderOrder = -1;
  scene.add(floorPts);
}

// ================================================================== interaction
const ptr = { x: 0, y: 0, tx: 0, ty: 0, ndc: new THREE.Vector2(), inside: false, lastMove: 0, down: false, burst: 0 };
window.addEventListener("pointermove", (e) => {
  ptr.tx = (e.clientX / innerWidth) * 2 - 1;
  ptr.ty = -((e.clientY / innerHeight) * 2 - 1);
  ptr.ndc.set(ptr.tx, ptr.ty);
  ptr.inside = true;
  ptr.lastMove = performance.now();
}, { passive: true });
window.addEventListener("mouseout", (e) => { if (!e.relatedTarget) ptr.inside = false; });
canvas.addEventListener("pointerdown", (e) => {
  ptr.ndc.set((e.clientX / innerWidth) * 2 - 1, -((e.clientY / innerHeight) * 2 - 1));
  ptr.inside = true;
  ptr.lastMove = performance.now();
  if (e.pointerType === "mouse") { ptr.down = true; body.classList.add("blowing"); }
  else ptr.burst = 0.45;                 // a tap is a short gust
});
window.addEventListener("pointerup", () => { ptr.down = false; body.classList.remove("blowing"); });

function jumpTo(k) {
  const t = k === 0 ? 0 : k === 6 ? T_END : k + 0.5;
  window.scrollTo({ top: t * innerHeight, behavior: REDUCED ? "auto" : "smooth" });
}
$$("[data-jump]").forEach((el) => el.addEventListener("click", (e) => { e.preventDefault(); jumpTo(+el.dataset.jump); }));

let contactShape = 0, lastHover = 0;
$$("[data-shape]").forEach((el) => {
  el.addEventListener("pointerenter", () => { contactShape = +el.dataset.shape; lastHover = performance.now(); });
  el.addEventListener("focus", () => { contactShape = +el.dataset.shape; lastHover = performance.now(); });
  el.addEventListener("pointerleave", () => { lastHover = performance.now(); });
  // Behance and LinkedIn have no address yet (site.yaml still says TODO)
  if (el.dataset.todo) el.addEventListener("click", (e) => e.preventDefault());
});

let entered = false, enterTime = 0;
soundBtn.addEventListener("click", () => sound.set(!sound.on));
$$("[data-enter]").forEach((b) => b.addEventListener("click", () => {
  body.classList.add("is-in");
  body.classList.remove("is-loading");
  entered = true;
  enterTime = performance.now();
  if (b.dataset.enter === "1") sound.set(true);
  sound.chime(NOTES[0]);
}));
window.addEventListener("keydown", (e) => { if (!entered && ldDone && e.key === "Enter") $("[data-enter='0']").click(); });

// ================================================================== panels
const panels = $$(".panel").map((el) => ({ el, sec: +el.dataset.sec, live: false, in: false, scr: $$("[data-scramble]", el) }));
function panelVis(sec, T) {
  if (sec === 0) return 1 - smooth(0.6, 0.74, T);
  if (sec === 6) return smooth(6.24, 6.4, T);
  return smooth(sec + 0.28, sec + 0.38, T) * (1 - smooth(sec + 0.74, sec + 0.84, T));
}
function updatePanels(T) {
  for (const p of panels) {
    const v = entered ? panelVis(p.sec, T) : 0;
    p.el.style.opacity = v.toFixed(3);
    p.el.style.transform = `translate3d(0, ${((1 - v) * 18).toFixed(1)}px, 0)`;
    const live = v > 0.01;
    if (live !== p.live) { p.live = live; p.el.classList.toggle("live", live); }
    if (v > 0.5 && !p.in) { p.in = true; p.el.classList.add("in"); p.scr.forEach((e) => scramble(e, null, 650)); }
    else if (v < 0.1 && p.in) { p.in = false; p.el.classList.remove("in"); }
  }
}

// ================================================================== frame
let T = 0, Tv = 0, last = performance.now();
let boundShape = 0, lastSec = -1, lastErode = 0, gust = 0, targetY = 0.5, push = 0;
const ray = new THREE.Raycaster();
const camTarget = new THREE.Vector3();

function resize() {
  // a hidden or collapsed frame reports 0 x 0, which would turn the scroll into NaN
  view.w = Math.max(innerWidth, 1); view.h = Math.max(innerHeight, 1);
  view.pr = Math.min(devicePixelRatio || 1, 2);
  view.aspect = view.w / view.h;
  view.narrow = view.w < 860;
  renderer.setPixelRatio(view.pr);
  renderer.setSize(view.w, view.h, false);
  scroller.style.height = Math.round(view.h * (T_END + 1)) + "px";
}
window.addEventListener("resize", resize);

// One viewing angle per sculpture (degrees from the front, toward +x). The camera
// swings between them only across a section boundary, while the grains are down
// in the heap, so it never shows a finished sculpture - or the wordmark - from behind.
const AZ = [-14, -36, -28, -42, -24, -34, -12];
function azAt(T) {
  const i = clamp(Math.floor(T - 0.75), -1, 5);
  if (i < 0) return AZ[0];
  return lerp(AZ[i], AZ[i + 1], smooth(i + 0.75, i + 1.25, T));
}

function setShape(k) {
  if (k === boundShape) return;
  boundShape = k;
  velVar.material.uniforms.tTarget.value = shapes[k].target;
  velVar.material.uniforms.tMeta.value = shapes[k].meta;
  grains.material.uniforms.tMeta.value = shapes[k].meta;
}

function frame(now) {
  requestAnimationFrame(frame);
  const dt = Math.min((now - last) / 1000, 1 / 30);
  last = now;
  const time = now / 1000;

  const target = entered ? window.scrollY / view.h : 0;
  const prevT = T;
  T += (target - T) * (1 - Math.exp(-dt * 5));
  if (Math.abs(target - T) < 1e-4) T = target;
  Tv += ((T - prevT) / Math.max(dt, 1e-3) - Tv) * 0.2;
  ptr.x += (ptr.tx - ptr.x) * (1 - Math.exp(-dt * 3));
  ptr.y += (ptr.ty - ptr.y) * (1 - Math.exp(-dt * 3));

  // which sculpture, and how much of it
  const sec = clamp(Math.floor(T), 0, 6);
  const u = T - sec;
  let form = smooth(0, 0.34, u) * 1.06;
  let erode = smooth(0.7, 1, u) * 1.06;
  if (sec === 0) form = entered ? smooth(0.2, 3.2, (now - enterTime) / 1000) * 1.06 : 0;
  if (sec === 6) erode = 0;
  // in the contact chapter the links pick the shape; left alone, it cycles
  if (sec === 6 && u > 0.3 && now - lastHover > 5500 && !$(".links a:hover")) {
    contactShape = (contactShape + 1) % shapes.length;
    lastHover = now;
  }
  const shapeIdx = sec === 6 ? (u > 0.3 ? contactShape : 0) : SECTION_SHAPE[sec];
  if (shapeIdx !== boundShape) { setShape(shapeIdx); if (sec === 6) sound.chime(NOTES[(shapeIdx + 1) % NOTES.length], 0.05); }
  if (sec !== lastSec) {
    if (lastSec >= 0) sound.chime(NOTES[sec % NOTES.length], 0.06);
    lastSec = sec;
    indexItems.forEach((li, i) => li.classList.toggle("on", i === sec));
  }
  gust += (clamp(Math.abs(erode - lastErode) / Math.max(dt, 1e-3) * 0.8, 0, 1.6) + (erode > 0.02 && erode < 1.05 ? 0.35 : 0) - gust) * (1 - Math.exp(-dt * 4));
  lastErode = erode;

  // camera: a slow orbit, framed to the sculpture's height
  const sh = shapes[shapeIdx];
  targetY += (Math.max(sh.height * 0.46, 0.45) - targetY) * (1 - Math.exp(-dt * 1.5));
  const rk = view.aspect < 1.25 ? clamp(1.3 / view.aspect, 1, 2.4) : 1;
  const dist = 6.4 * rk;
  const az = (azAt(T) + (REDUCED ? 0 : Math.sin(time * 0.12) * 9)) * DEG + ptr.x * 0.12;
  const el = (9 + ptr.y * 3) * DEG;
  camTarget.set(0, targetY, 0);
  camera.position.set(Math.sin(az) * Math.cos(el) * dist, targetY + Math.sin(el) * dist, Math.cos(az) * Math.cos(el) * dist);
  camera.lookAt(camTarget);
  camera.aspect = view.aspect;
  const sx = view.narrow ? 0 : view.aspect < 1 ? 0.07 : 0.16;
  const sy = view.narrow || view.aspect < 1 ? 0.17 : 0.03;
  camera.setViewOffset(view.w, view.h, -sx * view.w, sy * view.h, view.w, view.h);
  camera.updateMatrixWorld();

  // blower
  ptr.burst = Math.max(ptr.burst - dt, 0);
  const moving = ptr.inside && now - ptr.lastMove < 250;
  const wantPush = ptr.down ? 2.4 : ptr.burst > 0 ? 2.0 : moving ? 0.9 : 0;
  push += (wantPush - push) * (1 - Math.exp(-dt * 10));
  ray.setFromCamera(ptr.ndc, camera);

  const vu = velVar.material.uniforms;
  vu.uDt.value = dt; vu.uTime.value = time;
  vu.uForm.value = form; vu.uErode.value = erode;
  vu.uGust.value = REDUCED ? gust * 0.3 : gust;
  vu.uPush.value = entered ? push : 0;
  vu.uRayO.value.copy(ray.ray.origin); vu.uRayD.value.copy(ray.ray.direction);
  posVar.material.uniforms.uDt.value = dt;
  if (entered) gpu.compute();

  const gu = grains.material.uniforms;
  gu.tPos.value = gpu.getCurrentRenderTarget(posVar).texture;
  gu.tVel.value = gpu.getCurrentRenderTarget(velVar).texture;
  gu.uPR.value = view.pr;
  gu.uGain.value += (sh.gain - gu.uGain.value) * (1 - Math.exp(-dt * 2));
  floorPts.material.uniforms.uPR.value = view.pr;
  renderer.render(scene, camera);

  updatePanels(T);
  hint.classList.toggle("gone", T > 0.1);
  const formed = Math.round(100 * clamp01(form) * (1 - clamp01(erode)));
  hud.textContent = `${fmtInt(N)} grains · ${SHAPE_NAMES[shapeIdx]} · formed ${String(formed).padStart(3, "0")}%`;
  sound.activity(clamp01(gust * 0.5 + Math.abs(Tv) * 0.4 + push * 0.25));
}

load().catch((err) => {
  console.error(err);
  ldLine.textContent = "Could not load the models";
  ldPct.textContent = "ERR";
});
