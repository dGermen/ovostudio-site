/* Ovo Studio, immersive edition.
 *
 * One fixed canvas, three scenes, and the page scroll as a timeline:
 *
 *   A  Cascade on an ice slab in a snowfield: the real web mesh of
 *      projects/06-cascade-machine. In the studio chapter it opens up along the
 *      same approach vectors the assembly rail on the main site uses. The marbles
 *      are left out: they would only hang in mid-air, and nothing here rolls them.
 *   B  The four portfolio products, one per screen, floating in a dark studio.
 *      Each is its own CAD assembly (tools/web_products.py) and explodes along
 *      its designed explode vectors when hovered.
 *   C  A particle field that redraws itself as the wordmark, Cascade or one of
 *      the products, depending on which contact link is hovered.
 *
 * Every scene renders into a half-float target; one composite pass mixes two of
 * them through a frost dissolve, adds chromatic aberration (more of it the
 * faster you scroll), grain and vignette, and only then tone maps.
 *
 * T is the timeline in screens: T = scrollY / innerHeight, damped.
 */
import * as THREE from "three";

// ------------------------------------------------------------------ helpers
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
const clamp01 = (v) => clamp(v, 0, 1);
const lerp = (a, b, t) => a + (b - a) * t;
const smooth = (a, b, x) => { const t = clamp01((x - a) / (b - a)); return t * t * (3 - 2 * t); };
const DEG = Math.PI / 180;
const REDUCED = matchMedia("(prefers-reduced-motion: reduce)").matches;
const HOVER = matchMedia("(hover: hover)").matches;
const fmtInt = (n) => n.toLocaleString("en-US");

function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// improved gradient noise, seeded, for the snow dunes
function makeNoise3(seed) {
  const rnd = mulberry32(seed);
  const perm = Array.from({ length: 256 }, (_, i) => i);
  for (let i = 255; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [perm[i], perm[j]] = [perm[j], perm[i]]; }
  const p = new Uint8Array(512);
  for (let i = 0; i < 512; i++) p[i] = perm[i & 255];
  const G = [1,1,0, -1,1,0, 1,-1,0, -1,-1,0, 1,0,1, -1,0,1, 1,0,-1, -1,0,-1, 0,1,1, 0,-1,1, 0,1,-1, 0,-1,-1];
  const fade = (t) => t * t * t * (t * (t * 6 - 15) + 10);
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

// uniform Catmull-Rom through keyframes {t, ...channels}
const cr = (p0, p1, p2, p3, u) =>
  0.5 * (2 * p1 + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u * u + (-p0 + 3 * p1 - 3 * p2 + p3) * u * u * u);
function track(keys, t) {
  const n = keys.length;
  if (t <= keys[0].t) return keys[0];
  if (t >= keys[n - 1].t) return keys[n - 1];
  let i = 0;
  while (t > keys[i + 1].t) i++;
  const k0 = keys[Math.max(i - 1, 0)], k1 = keys[i], k2 = keys[i + 1], k3 = keys[Math.min(i + 2, n - 1)];
  const u = (t - k1.t) / (k2.t - k1.t);
  const out = {};
  for (const c in k1) if (c !== "t") out[c] = cr(k0[c], k1[c], k2[c], k3[c], u);
  return out;
}

const GLSL_NOISE = /* glsl */ `
float h3(vec3 p){ p = fract(p * 0.3183099 + vec3(0.1, 0.2, 0.3)); p *= 17.0; return fract(p.x * p.y * p.z * (p.x + p.y + p.z)); }
`;

// One part of a quantised web mesh (Cascade and the products share the format):
// 10 bytes a vertex, int16 x,y,z against the part's own centre and scale, then
// int8 nx,ny,nz and one byte of baked ambient occlusion (255 = open), read as
// three views over the same bytes.
function partGeometry(buf, p) {
  const g = new THREE.BufferGeometry();
  const n = p.vertices;
  const pos = new THREE.InterleavedBuffer(new Int16Array(buf, p.vertex_at, n * 5), 5);
  const nor = new THREE.InterleavedBuffer(new Int8Array(buf, p.vertex_at, n * 10), 10);
  const occ = new THREE.InterleavedBuffer(new Uint8Array(buf, p.vertex_at, n * 10), 10);
  g.setAttribute("position", new THREE.InterleavedBufferAttribute(pos, 3, 0, false));
  g.setAttribute("normal", new THREE.InterleavedBufferAttribute(nor, 3, 6, true));
  g.setAttribute("ao", new THREE.InterleavedBufferAttribute(occ, 1, 9, true));
  g.setIndex(new THREE.BufferAttribute(new Uint16Array(buf, p.index_at, p.triangles * 3), 1));
  return g;
}

// Baked occlusion into a standard or physical material, on top of whatever
// onBeforeCompile it already has. It mostly takes out ambient and reflected
// light, which is where occlusion lives, and only a little of the direct light.
function withAO(mat, strength) {
  const prev = mat.onBeforeCompile;
  const prevKey = mat.customProgramCacheKey.bind(mat);
  const k = { value: strength };
  mat.onBeforeCompile = (sh, r) => {
    prev.call(mat, sh, r);
    sh.uniforms.uAOK = k;
    sh.vertexShader = sh.vertexShader
      .replace("#include <common>", "#include <common>\nattribute float ao;\nvarying float vAO;")
      .replace("#include <begin_vertex>", "#include <begin_vertex>\nvAO = ao;");
    sh.fragmentShader = sh.fragmentShader
      .replace("#include <common>", "#include <common>\nvarying float vAO;\nuniform float uAOK;")
      .replace("#include <aomap_fragment>", `#include <aomap_fragment>
        {
          float aoF = mix(1.0, 0.18 + 0.82 * smoothstep(0.0, 1.0, vAO), uAOK);
          reflectedLight.indirectDiffuse *= aoF;
          reflectedLight.indirectSpecular *= mix(1.0, aoF, 0.85);
          reflectedLight.directDiffuse *= mix(1.0, aoF, 0.45);
          reflectedLight.directSpecular *= mix(1.0, aoF, 0.35);
        }`);
  };
  mat.customProgramCacheKey = () => prevKey() + "|ao";
  return mat;
}

// ------------------------------------------------------------------ content
const T_END = 10;
const A_TO_B = [3.5, 4.1];
const B_TO_C = [8.1, 8.7];
const WORK_START = 4.1;
const JUMP = { hero: 0, studio: 1.4, work: 4.35, contact: T_END };
const CHAPTERS = ["Cascade", "Studio", "Work", "Contact"];

const WORK = [
  {
    model: "drift", name: "Drift", type: "Portable speaker",
    line: "Two ABS housings on a 0.2 mm shut line. 1.5° of draft off the parting plane, a 103-hole grille and not one undercut. Ready to tool.",
    specs: [["Process", "Injection moulding"], ["Parts", "17"], ["Size", "186 × 70 × 62 mm"]],
    href: "en/work/drift-speaker/",
  },
  {
    model: "stem", name: "Stem", type: "Desk lamp",
    line: "Prints without supports, goes together without tools. Three M4 joints, tightened by hand with printed knurled knobs.",
    specs: [["Process", "FDM · PETG"], ["Supports", "None"], ["Reach", "403 mm"]],
    href: "en/work/stem-lamp/",
  },
  {
    model: "skincare", name: "Skincare set", type: "Render pack",
    line: "Five products modelled for the camera. Real glass wall thickness, so the refraction is honest, not faked in post.",
    specs: [["Products", "5"], ["Glass wall", "3.5–4 mm"], ["Output", "Marketplace pack"]],
    href: "en/work/skincare-set/",
  },
  {
    model: "axis", name: "Axis", type: "Desktop robot arm",
    line: "Four axes, printed and bolted. A servo drives each joint from one side, an M8 bolt turns in a 608 bearing on the other. 36 brass inserts, so it comes apart and goes back together.",
    specs: [["Axes", "4 + gripper"], ["Reach", "276 mm"], ["Bearings", "3 × 608"]],
    href: null,
  },
];
const SHAPES = ["Wordmark", "Cascade", "Drift", "Stem", "Axis"];

// ------------------------------------------------------------------ DOM
const body = document.body;
const canvas = $("#gl");
const loader = $("#loader");
const ldPct = $("#ldPct");
const ldBar = $("#ldBar");
const ldFile = $("#ldFile");
const soundBtn = $("#soundBtn");
const telemetry = $("#telemetry");
const scrollHint = $("#scrollHint");
const chapNum = $("#chapNum");
const chapName = $("#chapName");
const railFill = $("#railFill");
const railItems = $$(".rail li");
const navLinks = $$(".hud-nav a");
const reticle = $("#reticle");
const retTag = $("#retTag");
const scroller = $("#scroller");
const wNote = $("#wNote");

if ("scrollRestoration" in history) history.scrollRestoration = "manual";
window.scrollTo(0, 0);
wNote.textContent = HOVER ? "Hover to explode · drag to turn" : "Tap the model to explode it";

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
    for (let i = 0; i < text.length; i++) {
      const ch = text[i];
      out += ch === " " || p >= order[i] ? ch : GLYPHS[(Math.random() * GLYPHS.length) | 0];
    }
    el.textContent = out;
    if (p < 1) el._scr = requestAnimationFrame(step);
    else el.textContent = text;
  };
  el._scr = requestAnimationFrame(step);
}

// split paragraphs into word masks so a panel can lift its lines in
function splitWords(el) {
  let n = 0;
  const walk = (node) => {
    for (const child of [...node.childNodes]) {
      if (child.nodeType === 3) {
        const frag = document.createDocumentFragment();
        for (const part of child.textContent.split(/([ \t\r\n]+)/)) {
          if (!part) continue;
          if (/^[ \t\r\n]+$/.test(part)) { frag.appendChild(document.createTextNode(part)); continue; }
          const w = document.createElement("span");
          w.className = "w";
          const i = document.createElement("span");
          i.textContent = part;
          i.style.setProperty("--d", (n++ * 0.028).toFixed(3) + "s");
          w.appendChild(i);
          frag.appendChild(w);
        }
        child.replaceWith(frag);
      } else if (child.nodeType === 1) walk(child);
    }
  };
  walk(el);
}
$$(".big, .lede").forEach(splitWords);

function countUp(el) {
  const to = +el.dataset.count;
  if (REDUCED || to === 0) { el.textContent = String(to); return; }
  const t0 = performance.now();
  const step = (now) => {
    const p = clamp01((now - t0) / 1100);
    el.textContent = String(Math.round(to * (1 - Math.pow(1 - p, 3))));
    if (p < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

// ------------------------------------------------------------------ sound
// Everything is synthesised: wind through bandpassed brown noise, a low drone,
// and chimes built from the partials of a free bar (1 : 2.76 : 5.40 : 8.93),
// which is what Cascade's own chime bars ring at.
class Sound {
  constructor() { this.on = false; this.ctx = null; this.sceneK = 0; }
  ensure() {
    if (this.ctx) return;
    const AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) return;
    const ctx = (this.ctx = new AC());
    this.master = ctx.createGain(); this.master.gain.value = 0; this.master.connect(ctx.destination);

    const len = Math.floor(ctx.sampleRate * 2.8);
    const ir = ctx.createBuffer(2, len, ctx.sampleRate);
    for (let ch = 0; ch < 2; ch++) {
      const d = ir.getChannelData(ch);
      for (let i = 0; i < len; i++) d[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / len, 3.2);
    }
    const verb = ctx.createConvolver(); verb.buffer = ir;
    const verbGain = ctx.createGain(); verbGain.gain.value = 0.4;
    verb.connect(verbGain).connect(this.master);
    this.dry = ctx.createGain(); this.dry.connect(this.master); this.dry.connect(verb);

    const brown = ctx.createBuffer(1, ctx.sampleRate * 4, ctx.sampleRate);
    const bd = brown.getChannelData(0);
    let last = 0;
    for (let i = 0; i < bd.length; i++) { last = (last + 0.02 * (Math.random() * 2 - 1)) / 1.02; bd[i] = last * 3.5; }
    this.brown = brown;
    const white = ctx.createBuffer(1, ctx.sampleRate * 0.5, ctx.sampleRate);
    const wd = white.getChannelData(0);
    for (let i = 0; i < wd.length; i++) wd[i] = Math.random() * 2 - 1;
    this.white = white;

    const wind = ctx.createBufferSource(); wind.buffer = brown; wind.loop = true;
    const bp = ctx.createBiquadFilter(); bp.type = "bandpass"; bp.frequency.value = 420; bp.Q.value = 0.6;
    const lfo = ctx.createOscillator(); lfo.frequency.value = 0.07;
    const lfoG = ctx.createGain(); lfoG.gain.value = 260;
    lfo.connect(lfoG).connect(bp.frequency);
    this.windGain = ctx.createGain(); this.windGain.gain.value = 0;
    wind.connect(bp).connect(this.windGain).connect(this.master);
    wind.start(); lfo.start();

    this.droneGain = ctx.createGain(); this.droneGain.gain.value = 0; this.droneGain.connect(this.dry);
    for (const [f, g] of [[55, 0.5], [82.41, 0.25], [110.3, 0.18]]) {
      const o = ctx.createOscillator(); o.frequency.value = f;
      const og = ctx.createGain(); og.gain.value = g;
      o.connect(og).connect(this.droneGain); o.start();
    }
    this.scene(this.sceneK, true);
  }
  set(on) {
    this.on = on;
    soundBtn.setAttribute("aria-pressed", String(on));
    soundBtn.querySelector("b").textContent = on ? "on" : "off";
    if (on) { this.ensure(); if (this.ctx) this.ctx.resume(); }
    if (!this.ctx) return;
    const t = this.ctx.currentTime;
    this.master.gain.cancelScheduledValues(t);
    this.master.gain.setTargetAtTime(on ? 1 : 0, t, 0.35);
  }
  scene(k, force) {
    if (k === this.sceneK && !force) return;
    this.sceneK = k;
    if (!this.ctx) return;
    const t = this.ctx.currentTime;
    this.windGain.gain.setTargetAtTime([0.075, 0.022, 0.016][k], t, 1.2);
    this.droneGain.gain.setTargetAtTime([0.0, 0.035, 0.022][k], t, 1.6);
  }
  chime(f, vel = 0.1) {
    if (!this.on || !this.ctx) return;
    const ctx = this.ctx, t = ctx.currentTime;
    for (const [r, a, d] of [[1, 1, 2.6], [2.756, 0.45, 1.4], [5.404, 0.2, 0.7], [8.933, 0.08, 0.35]]) {
      const o = ctx.createOscillator(); o.frequency.value = f * r;
      const g = ctx.createGain();
      g.gain.setValueAtTime(0, t);
      g.gain.linearRampToValueAtTime(vel * a, t + 0.004);
      g.gain.exponentialRampToValueAtTime(0.0001, t + d);
      o.connect(g).connect(this.dry); o.start(t); o.stop(t + d + 0.05);
    }
  }
  tick(v = 0.035) {
    if (!this.on || !this.ctx) return;
    const ctx = this.ctx, t = ctx.currentTime;
    const s = ctx.createBufferSource(); s.buffer = this.white;
    const hp = ctx.createBiquadFilter(); hp.type = "highpass"; hp.frequency.value = 3200;
    const g = ctx.createGain(); g.gain.setValueAtTime(v, t); g.gain.exponentialRampToValueAtTime(0.0001, t + 0.05);
    s.connect(hp).connect(g).connect(this.dry); s.start(t, Math.random() * 0.4, 0.06);
  }
  whoosh(v = 0.14) {
    if (!this.on || !this.ctx) return;
    const ctx = this.ctx, t = ctx.currentTime;
    const s = ctx.createBufferSource(); s.buffer = this.brown;
    const bp = ctx.createBiquadFilter(); bp.type = "bandpass"; bp.Q.value = 1.1;
    bp.frequency.setValueAtTime(220, t); bp.frequency.exponentialRampToValueAtTime(1900, t + 0.7);
    const g = ctx.createGain();
    g.gain.setValueAtTime(0.0001, t); g.gain.exponentialRampToValueAtTime(v, t + 0.25); g.gain.exponentialRampToValueAtTime(0.0001, t + 1.0);
    s.connect(bp).connect(g).connect(this.dry); s.start(t, Math.random() * 2, 1.1);
  }
}
const sound = new Sound();
const NOTES = [880, 987.77, 1108.73, 1318.51, 1479.98]; // A major pentatonic, octave 5

// --------------------------------------------------------------- renderer
let renderer;
try {
  renderer = new THREE.WebGLRenderer({ canvas, antialias: false, powerPreference: "high-performance" });
} catch (e) {
  renderer = null;
}
if (!renderer) {
  $("#noGL").hidden = false;
  loader.remove();
  throw new Error("WebGL unavailable");
}
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

const isGL2 = renderer.capabilities.isWebGL2;
const canHalf = isGL2 && (renderer.extensions.has("EXT_color_buffer_float") || renderer.extensions.has("EXT_color_buffer_half_float"));
const rtOpts = { type: canHalf ? THREE.HalfFloatType : THREE.UnsignedByteType, samples: isGL2 ? 4 : 0 };
const rtA = new THREE.WebGLRenderTarget(4, 4, rtOpts);
const rtB = new THREE.WebGLRenderTarget(4, 4, rtOpts);

const view = { w: 1, h: 1, pr: 1, aspect: 1, narrow: false };

// a cool, soft studio for reflections, shared by A and B
function makeEnvironment() {
  const env = new THREE.Scene();
  const sky = new THREE.Mesh(
    new THREE.SphereGeometry(50, 32, 16),
    new THREE.ShaderMaterial({
      side: THREE.BackSide,
      uniforms: { uTop: { value: new THREE.Color(0x8e98a8) }, uBot: { value: new THREE.Color(0x3a3f48) } },
      vertexShader: "varying vec3 vP; void main(){ vP = position; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }",
      fragmentShader: "uniform vec3 uTop, uBot; varying vec3 vP; void main(){ float h = normalize(vP).y; gl_FragColor = vec4(mix(uBot, uTop, smoothstep(-0.4, 0.6, h)), 1.0); }",
    }));
  env.add(sky);
  const panel = (w, h, x, y, z, k) => {
    const m = new THREE.Mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshBasicMaterial({ color: new THREE.Color(1, 1, 1).multiplyScalar(k), side: THREE.DoubleSide }));
    m.position.set(x, y, z); m.lookAt(0, 0, 0); env.add(m);
  };
  panel(30, 18, -25, 30, 25, 6);   // key softbox, upper front left
  panel(12, 40, 35, 8, -20, 2.2);  // strip rim behind right
  panel(40, 8, 0, 45, 0, 1.4);     // weak top
  const pmrem = new THREE.PMREMGenerator(renderer);
  const tex = pmrem.fromScene(env, 0.02, 0.1, 100).texture;
  pmrem.dispose();
  return tex;
}

function skyDome(top, mid, bot, sunDir, sunCol) {
  const mat = new THREE.ShaderMaterial({
    side: THREE.BackSide, depthWrite: false,
    uniforms: {
      uTop: { value: new THREE.Color(top) }, uMid: { value: new THREE.Color(mid) }, uBot: { value: new THREE.Color(bot) },
      uSun: { value: sunDir.clone().normalize() }, uSunCol: { value: new THREE.Color(sunCol) },
    },
    vertexShader: /* glsl */ `
      varying vec3 vDir;
      void main(){ vDir = position; vec4 p = projectionMatrix * modelViewMatrix * vec4(position, 1.0); gl_Position = p.xyww; }`,
    fragmentShader: /* glsl */ `
      uniform vec3 uTop, uMid, uBot, uSun, uSunCol; varying vec3 vDir;
      void main(){
        vec3 d = normalize(vDir); float h = d.y;
        vec3 c = mix(uMid, uTop, smoothstep(0.0, 0.6, h));
        c = mix(c, uBot, smoothstep(0.0, -0.3, h));
        float s = max(dot(d, uSun), 0.0);
        c += uSunCol * (pow(s, 6.0) * 0.35 + pow(s, 60.0) * 0.8);
        gl_FragColor = vec4(c, 1.0);
      }`,
  });
  const m = new THREE.Mesh(new THREE.SphereGeometry(1000, 48, 24), mat);
  m.frustumCulled = false;
  m.renderOrder = -10;
  return m;
}

// fresnel rim as emission: what makes the pale plastic read as frosted ice
function rimify(mat, color, strength, power = 3) {
  mat.userData.rim = { value: strength };
  mat.onBeforeCompile = (sh) => {
    sh.uniforms.uRimCol = { value: new THREE.Color(color) };
    sh.uniforms.uRimK = mat.userData.rim;
    sh.fragmentShader = sh.fragmentShader
      .replace("#include <common>", "#include <common>\nuniform vec3 uRimCol; uniform float uRimK;")
      .replace("#include <emissivemap_fragment>", `#include <emissivemap_fragment>
        float rimF = 1.0 - clamp(dot(normal, normalize(vViewPosition)), 0.0, 1.0);
        totalEmissiveRadiance += uRimCol * uRimK * pow(rimF, ${power.toFixed(1)});`);
  };
  mat.customProgramCacheKey = () => "rim" + power;
  return mat;
}

// Area-weighted points on a set of meshes. Each entry is a geometry plus the
// matrix that takes its raw (quantised) positions into the frame wanted.
function surfaceSample(entries, n) {
  const refs = [];
  const areas = [];
  let total = 0;
  const a = new THREE.Vector3(), b = new THREE.Vector3(), c = new THREE.Vector3();
  const ab = new THREE.Vector3(), ac = new THREE.Vector3();
  for (const { geometry, matrix } of entries) {
    const pos = geometry.attributes.position, idx = geometry.index.array;
    const w = new Float32Array(pos.count * 3);
    for (let i = 0; i < pos.count; i++) {
      a.set(pos.getX(i), pos.getY(i), pos.getZ(i)).applyMatrix4(matrix);
      w[i * 3] = a.x; w[i * 3 + 1] = a.y; w[i * 3 + 2] = a.z;
    }
    for (let t = 0; t < idx.length; t += 3) {
      const i0 = idx[t] * 3, i1 = idx[t + 1] * 3, i2 = idx[t + 2] * 3;
      a.fromArray(w, i0); b.fromArray(w, i1); c.fromArray(w, i2);
      total += ab.subVectors(b, a).cross(ac.subVectors(c, a)).length() * 0.5;
      areas.push(total);
      refs.push(w, i0, i1, i2);
    }
  }
  const out = new Float32Array(n * 3);
  for (let s = 0; s < n; s++) {
    const x = Math.random() * total;
    let lo = 0, hi = areas.length - 1;
    while (lo < hi) { const m = (lo + hi) >> 1; if (areas[m] < x) lo = m + 1; else hi = m; }
    const w = refs[lo * 4], i0 = refs[lo * 4 + 1], i1 = refs[lo * 4 + 2], i2 = refs[lo * 4 + 3];
    let u = Math.random(), v = Math.random();
    if (u + v > 1) { u = 1 - u; v = 1 - v; }
    for (let d = 0; d < 3; d++) out[s * 3 + d] = w[i0 + d] + (w[i1 + d] - w[i0 + d]) * u + (w[i2 + d] - w[i0 + d]) * v;
  }
  return out;
}

// centre a point set on the origin and scale its largest extent to `size`
function normalizePoints(pts, size) {
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < pts.length; i += 3) for (let d = 0; d < 3; d++) {
    if (pts[i + d] < lo[d]) lo[d] = pts[i + d];
    if (pts[i + d] > hi[d]) hi[d] = pts[i + d];
  }
  const c = [0, 1, 2].map((d) => (lo[d] + hi[d]) / 2);
  const k = size / Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
  for (let i = 0; i < pts.length; i += 3) for (let d = 0; d < 3; d++) pts[i + d] = (pts[i + d] - c[d]) * k;
  return pts;
}

// ================================================================ SCENE A
function buildSceneA(meta, buf, envTex) {
  const scene = new THREE.Scene();
  scene.environment = envTex;
  const FOG = 0xa4adbb;
  scene.fog = new THREE.FogExp2(FOG, 0.00016);
  const sunDir = new THREE.Vector3(-0.55, 0.62, 0.55);
  const sky = skyDome(0x5d6878, FOG, 0x9aa3b0, sunDir, 0xffe9cc);
  scene.add(sky);

  const camera = new THREE.PerspectiveCamera(30, 1, 5, 30000);

  // ---- light
  scene.add(new THREE.HemisphereLight(0xd4def0, 0x7d8798, 0.62));
  const sun = new THREE.DirectionalLight(0xfff0de, 3.1);
  sun.position.copy(sunDir).multiplyScalar(2200).add(new THREE.Vector3(0, 250, 0));
  sun.target.position.set(0, 250, 0);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  const sc = sun.shadow.camera;
  sc.left = -760; sc.right = 760; sc.top = 760; sc.bottom = -760; sc.near = 200; sc.far = 5200;
  sun.shadow.bias = -0.0004;
  sun.shadow.normalBias = 1.2;
  scene.add(sun, sun.target);
  const fill = new THREE.DirectionalLight(0xc3d2ff, 0.7);
  fill.position.set(1100, 420, -700);
  scene.add(fill);

  // ---- snowfield: flat under the machine, dunes further out
  const noise = makeNoise3(7);
  const gGeo = new THREE.PlaneGeometry(36000, 36000, 300, 300);
  gGeo.rotateX(-Math.PI / 2);
  const gp = gGeo.attributes.position;
  for (let i = 0; i < gp.count; i++) {
    const x = gp.getX(i), z = gp.getZ(i);
    const d = Math.hypot(x, z);
    const k = smooth(900, 4200, d);
    const h = (noise(x * 0.00032, 0.5, z * 0.00032) * 0.7 + noise(x * 0.0011, 3.1, z * 0.0011) * 0.3) * 520 * k
      + noise(x * 0.004, 7.7, z * 0.004) * 16 * smooth(500, 1400, d);
    gp.setY(i, h - 6);
  }
  gGeo.computeVertexNormals();
  const glint = { value: 1 };
  const groundMat = new THREE.MeshStandardMaterial({ color: 0xdfe6ef, roughness: 0.93, metalness: 0 });
  groundMat.onBeforeCompile = (sh) => {
    sh.uniforms.uGlint = glint;
    sh.vertexShader = sh.vertexShader
      .replace("#include <common>", "#include <common>\nvarying vec3 vWPos;")
      .replace("#include <project_vertex>", "#include <project_vertex>\nvWPos = (modelMatrix * vec4(transformed, 1.0)).xyz;");
    sh.fragmentShader = sh.fragmentShader
      .replace("#include <common>", "#include <common>\nvarying vec3 vWPos; uniform float uGlint;\n" + GLSL_NOISE)
      .replace("#include <emissivemap_fragment>", `#include <emissivemap_fragment>
        {
          vec3 cell = floor(vWPos * 0.16);
          float pick = step(0.982, h3(cell));
          vec3 V = normalize(cameraPosition - vWPos);
          float tw = pow(max(0.0, sin(h3(cell + 7.0) * 40.0 + dot(V, vec3(31.0, 23.0, 17.0)))), 16.0);
          float near = 1.0 - smoothstep(900.0, 3200.0, length(cameraPosition - vWPos));
          totalEmissiveRadiance += vec3(0.95, 0.97, 1.0) * pick * tw * near * 3.0 * uGlint;
          // soft blue in the hollows, like snow under an overcast sky
          diffuseColor.rgb *= mix(vec3(0.74, 0.8, 0.92), vec3(1.0), smoothstep(-80.0, 160.0, vWPos.y));
        }`);
  };
  const ground = new THREE.Mesh(gGeo, groundMat);
  ground.receiveShadow = true;
  scene.add(ground);

  // ---- ice slab the machine stands on
  const slabMat = rimify(new THREE.MeshPhysicalMaterial({
    color: 0x9fb3c8, roughness: 0.12, metalness: 0, clearcoat: 1, clearcoatRoughness: 0.06,
  }), 0xd8e8ff, 0.7, 2.5);
  const slab = new THREE.Mesh(new THREE.CylinderGeometry(390, 410, 30, 120, 1), slabMat);
  slab.position.y = -15;
  slab.receiveShadow = true;
  const lip = new THREE.Mesh(new THREE.CylinderGeometry(418, 440, 12, 120, 1), slabMat);
  lip.position.y = -30;
  lip.receiveShadow = true;
  scene.add(slab, lip);

  // ---- Cascade
  const M = {
    ice: rimify(new THREE.MeshPhysicalMaterial({
      color: 0xc9d6e4, roughness: 0.34, metalness: 0, clearcoat: 0.7, clearcoatRoughness: 0.22,
      sheen: 0.5, sheenColor: new THREE.Color(0xb5ccf5), sheenRoughness: 0.5,
    }), 0xd6e6ff, 0.55, 3),
    steel: new THREE.MeshStandardMaterial({ color: 0xc8cdd3, metalness: 1, roughness: 0.27 }),
    yolk: rimify(new THREE.MeshStandardMaterial({ color: 0xf4a300, roughness: 0.42 }), 0xffd98a, 0.3, 3),
  };
  // frosted ice wants its crevices soft, not sooty
  withAO(M.ice, 0.7); withAO(M.steel, 0.8); withAO(M.yolk, 0.8);
  const matFor = (n) => {
    if (n.startsWith("crank_knob") || n.startsWith("flip_flop_gate")) return M.yolk;
    if (/^(screws|nuts|inserts|bearing|shaft_collar|flywheel_weights|chime_bar)/.test(n) || /_shaft$/.test(n)) return M.steel;
    return M.ice;
  };

  const [lo, hi] = meta.bbox;
  const mid = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2];
  const machine = new THREE.Group();            // lifted as it opens
  const root = new THREE.Group();
  root.rotation.x = -Math.PI / 2;               // Rhino is Z-up
  machine.add(root);
  scene.add(machine);

  const rnd = mulberry32(99);
  const parts = meta.parts
    .filter((p) => !p.name.startsWith("marbles"))  // nothing rolls them, so they would only float
    .map((p) => {
      const mesh = new THREE.Mesh(partGeometry(buf, p), matFor(p.name));
      mesh.scale.setScalar(p.scale);
      mesh.position.set(p.org[0] - mid[0], p.org[1] - mid[1], p.org[2]);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      root.add(mesh);
      return {
        mesh, name: p.name, home: mesh.position.clone(), from: new THREE.Vector3(...p.from),
        delay: (p.stage / 5) * 0.3 + rnd() * 0.08, phase: rnd() * Math.PI * 2,
      };
    });
  const triangles = meta.parts.filter((p) => !p.name.startsWith("marbles")).reduce((s, p) => s + p.triangles, 0);

  // ---- snowfall, wrapped around the camera so it never runs out
  const SNOW = 5200;
  const sGeo = new THREE.BufferGeometry();
  const sPos = new Float32Array(SNOW * 3), sSeed = new Float32Array(SNOW * 4);
  for (let i = 0; i < SNOW; i++) {
    sPos[i * 3] = (Math.random() - 0.5) * 5200;
    sPos[i * 3 + 1] = Math.random() * 2600;
    sPos[i * 3 + 2] = (Math.random() - 0.5) * 5200;
    sSeed[i * 4] = 0.5 + Math.random();          // size
    sSeed[i * 4 + 1] = Math.random();
    sSeed[i * 4 + 2] = 45 + Math.random() * 70;  // fall speed, mm/s
    sSeed[i * 4 + 3] = Math.random() * 6.28;
  }
  sGeo.setAttribute("position", new THREE.BufferAttribute(sPos, 3));
  sGeo.setAttribute("aSeed", new THREE.BufferAttribute(sSeed, 4));
  const snowMat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false,
    uniforms: { uTime: { value: 0 }, uCenter: { value: new THREE.Vector3() }, uPR: { value: 1 }, uSize: { value: 9 } },
    vertexShader: /* glsl */ `
      uniform float uTime, uPR, uSize; uniform vec3 uCenter;
      attribute vec4 aSeed; varying float vA;
      void main(){
        vec3 p = position;
        p.y = mod(p.y - uTime * aSeed.z, 2600.0) - 200.0;
        p.x += sin(uTime * 0.7 + aSeed.w + p.y * 0.004) * 42.0 + uTime * 38.0;
        p.z += cos(uTime * 0.5 + aSeed.w * 1.3 + p.y * 0.003) * 42.0;
        p.x = mod(p.x - uCenter.x + 2600.0, 5200.0) - 2600.0 + uCenter.x;
        p.z = mod(p.z - uCenter.z + 2600.0, 5200.0) - 2600.0 + uCenter.z;
        vec4 mv = modelViewMatrix * vec4(p, 1.0);
        gl_Position = projectionMatrix * mv;
        gl_PointSize = uSize * aSeed.x * uPR * (900.0 / max(-mv.z, 1.0));
        float d = -mv.z;
        vA = smoothstep(40.0, 260.0, d) * (1.0 - smoothstep(2200.0, 4800.0, d));
      }`,
    fragmentShader: /* glsl */ `
      varying float vA;
      void main(){ float r = length(gl_PointCoord - 0.5); float a = smoothstep(0.5, 0.05, r); gl_FragColor = vec4(vec3(1.0), a * vA * 0.9); }`,
  });
  const snow = new THREE.Points(sGeo, snowMat);
  snow.frustumCulled = false;
  scene.add(snow);

  // camera keyframes; az is measured from the front (+Z) toward +X
  const KEYS = [
    { t: 0.0, az: -35, el: 11, r: 1480, ty: 205, fov: 30 },
    { t: 1.0, az: -18, el: 17, r: 1500, ty: 220, fov: 30 },
    { t: 1.95, az: 32, el: 21, r: 1900, ty: 250, fov: 30 },
    { t: 2.85, az: 92, el: 13, r: 1950, ty: 260, fov: 30 },
    { t: 3.6, az: 150, el: 5, r: 900, ty: 300, fov: 36 },
    { t: 4.2, az: 175, el: 1, r: 430, ty: 320, fov: 42 },
  ];
  // Sideways screen offset, held flat while each panel is readable and moved
  // only while the text is faded out, so the machine never slides under copy.
  const SX = [[-1, 0.2], [0.55, 0.2], [1.0, 0.25], [1.8, 0.25], [2.0, -0.22], [2.68, -0.22], [2.9, 0.23], [3.45, 0.23], [3.9, 0]];
  const sxAt = (t) => {
    if (t <= SX[0][0]) return SX[0][1];
    for (let i = 1; i < SX.length; i++) {
      if (t <= SX[i][0]) return lerp(SX[i - 1][1], SX[i][1], smooth(SX[i - 1][0], SX[i][0], t));
    }
    return SX[SX.length - 1][1];
  };

  const tgt = new THREE.Vector3();
  const state = { az: 0, el: 0, r: 0, ex: 0 };
  function update(T, dt, time, ptr) {
    const k = track(KEYS, T);
    const ex = smooth(1.75, 3.1, T);
    const lift = ex * 0.85 * 240;
    machine.position.y = lift;
    for (const p of parts) {
      const e = smooth(0, 1, (ex - p.delay) / 0.62);
      p.mesh.position.set(
        p.home.x + p.from.x * e * 0.85,
        p.home.y + p.from.y * e * 0.85,
        p.home.z + p.from.z * e * 0.85 + (REDUCED ? 0 : Math.sin(time * 0.9 + p.phase) * 5 * e));
    }
    M.ice.userData.rim.value = 0.5 + ex * 0.25;

    const drift = REDUCED ? 0 : Math.sin(time * 0.11) * 2.5;
    const az = (k.az + ptr.x * 3.5 + drift) * DEG;
    const el = clamp(k.el + ptr.y * 2, -2, 60) * DEG;
    const rk = view.aspect < 1 ? clamp(0.82 / view.aspect, 1, 1.9) : 1;
    const r = k.r * rk;
    tgt.set(0, k.ty + lift * 0.55, 0);
    camera.position.set(tgt.x + Math.sin(az) * Math.cos(el) * r, tgt.y + Math.sin(el) * r, tgt.z + Math.cos(az) * Math.cos(el) * r);
    camera.lookAt(tgt);
    camera.fov = k.fov;
    camera.aspect = view.aspect;
    const sx = view.narrow ? 0 : sxAt(T);
    const sy = view.narrow ? 0.14 : 0;
    camera.setViewOffset(view.w, view.h, -sx * view.w, sy * view.h, view.w, view.h);
    sky.position.copy(camera.position);
    snowMat.uniforms.uTime.value = time * (REDUCED ? 0.3 : 1);
    snowMat.uniforms.uCenter.value.copy(camera.position);
    snowMat.uniforms.uPR.value = view.pr;
    state.az = ((k.az % 360) + 360) % 360; state.el = k.el; state.r = r; state.ex = ex;
  }

  // the assembled machine as mesh entries, for the particle field
  function entries() {
    machine.position.y = 0;
    for (const p of parts) p.mesh.position.copy(p.home);
    machine.updateMatrixWorld(true);
    return parts.map((p) => ({ geometry: p.mesh.geometry, matrix: p.mesh.matrixWorld.clone() }));
  }

  return { scene, camera, update, entries, state, triangles };
}

// ================================================================ SCENE B
const SPACING = 1700;
const PRODUCT_R = 300;      // every product is scaled to the same bounding radius
const FLOOR_Y = -720;
const EXPLODE_K = 0.8;      // of the designed explode shift

const matCache = new Map();
function productMaterial(kind, hex) {
  const key = kind + hex;
  if (matCache.has(key)) return matCache.get(key);
  const color = new THREE.Color(hex);
  let m;
  switch (kind) {
    case "aluminium": m = new THREE.MeshStandardMaterial({ color, metalness: 1, roughness: 0.3 }); break;
    case "steel": m = new THREE.MeshStandardMaterial({ color, metalness: 1, roughness: 0.42 }); break;
    case "brass": m = new THREE.MeshStandardMaterial({ color, metalness: 1, roughness: 0.3 }); break;
    // Full transmission is right on a white sweep and wrong here: on a dark set
    // the backdrop shows straight through and glass reads as black plastic. Part
    // transmission plus a fresnel edge keeps it reading as glass.
    case "glass": m = rimify(new THREE.MeshPhysicalMaterial({ color, roughness: 0.05, transmission: 0.9, thickness: 4, ior: 1.5, clearcoat: 1, clearcoatRoughness: 0.03 }), 0xe2ebff, 0.45, 2.5); break;
    case "glass_frosted": m = rimify(new THREE.MeshPhysicalMaterial({ color, roughness: 0.45, transmission: 0.5, thickness: 4, ior: 1.5 }), 0xeef2fa, 0.3, 2); break;
    case "glow": m = new THREE.MeshStandardMaterial({ color: 0x2a2a2a, emissive: color, emissiveIntensity: 0.75, roughness: 0.6 }); break;
    case "abs_gloss": m = new THREE.MeshPhysicalMaterial({ color, roughness: 0.2, clearcoat: 0.8, clearcoatRoughness: 0.1 }); break;
    case "abs_hard": m = new THREE.MeshPhysicalMaterial({ color, roughness: 0.38, clearcoat: 0.2, clearcoatRoughness: 0.4 }); break;
    case "tpe": m = new THREE.MeshPhysicalMaterial({ color, roughness: 0.82, sheen: 0.4, sheenColor: new THREE.Color(0x60646c), sheenRoughness: 0.6 }); break;
    case "rubber": m = new THREE.MeshStandardMaterial({ color, roughness: 0.9 }); break;
    case "paper": m = new THREE.MeshStandardMaterial({ color, roughness: 0.95 }); break;
    case "silicone": m = new THREE.MeshPhysicalMaterial({ color, roughness: 0.5, sheen: 0.3, sheenColor: new THREE.Color(0xffd27a) }); break;
    case "pcb": m = new THREE.MeshStandardMaterial({ color, roughness: 0.45 }); break;
    case "fdm": m = new THREE.MeshStandardMaterial({ color, roughness: 0.66 }); break;
    default: m = new THREE.MeshPhysicalMaterial({ color, roughness: 0.48, clearcoat: 0.15, clearcoatRoughness: 0.5 });
  }
  if (!("glow" === kind)) m.envMapIntensity = kind.startsWith("glass") ? 1.8 : 0.85;
  if (kind !== "glow") withAO(m, kind.startsWith("glass") ? 0.5 : 1.0);
  matCache.set(key, m);
  return m;
}

function blobTexture() {
  const c = document.createElement("canvas");
  c.width = c.height = 128;
  const g = c.getContext("2d");
  const grd = g.createRadialGradient(64, 64, 0, 64, 64, 64);
  grd.addColorStop(0, "rgba(0,0,0,0.85)");
  grd.addColorStop(0.45, "rgba(0,0,0,0.35)");
  grd.addColorStop(1, "rgba(0,0,0,0)");
  g.fillStyle = grd;
  g.fillRect(0, 0, 128, 128);
  return new THREE.CanvasTexture(c);
}

function buildProduct(meta, buf, i) {
  const group = new THREE.Group();        // its slot on the line
  group.position.x = i * SPACING;
  const float = new THREE.Group();        // bob, spin and drag
  group.add(float);
  const pivot = new THREE.Group();        // Z-up to Y-up, centred, fitted
  pivot.rotation.x = -Math.PI / 2;
  float.add(pivot);
  const [lo, hi] = meta.bbox;
  const c = [0, 1, 2].map((d) => (lo[d] + hi[d]) / 2);
  const diag = Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
  pivot.scale.setScalar((PRODUCT_R * 2) / diag);

  const parts = meta.parts.map((p) => {
    const mesh = new THREE.Mesh(partGeometry(buf, p), productMaterial(p.mat, p.color));
    mesh.scale.setScalar(p.scale);
    mesh.position.set(p.org[0] - c[0], p.org[1] - c[1], p.org[2] - c[2]);
    pivot.add(mesh);
    return { mesh, home: mesh.position.clone(), ex: new THREE.Vector3(...p.explode) };
  });
  float.rotation.y = -0.55;               // start three-quarter from the front left
  // the designed shifts are sized for a drawing sheet; on screen, cap the
  // longest one at 40% of the product's own diagonal so it stays off the copy
  const longest = Math.max(1e-6, ...parts.map((q) => q.ex.length()));
  const explodeK = Math.min(EXPLODE_K, (0.4 * diag) / longest);
  return {
    meta, group, float, pivot, parts, seed: i * 1.7, explodeK,
    vel: 0, velX: 0, explode: 0, shown: -1, tap: false,
    triangles: meta.triangles,
  };
}

function buildSceneB(envTex, productData) {
  const scene = new THREE.Scene();
  scene.environment = envTex;
  const FOG = 0x242931;
  scene.fog = new THREE.FogExp2(FOG, 0.0001);
  const sky = skyDome(0x0b0d10, 0x2b313a, 0x14171b, new THREE.Vector3(-0.4, 0.35, 0.6), 0x323b4a);
  scene.add(sky);
  const camera = new THREE.PerspectiveCamera(32, 1, 5, 30000);

  scene.add(new THREE.HemisphereLight(0x9aa6ba, 0x101216, 0.75));
  const key = new THREE.DirectionalLight(0xfff0dc, 3.4);
  key.position.set(-1, 1.25, 0.9);
  const rim = new THREE.DirectionalLight(0x9cbcff, 3.2);
  rim.position.set(1.3, 0.55, -1);
  const top = new THREE.DirectionalLight(0xffffff, 1.0);
  top.position.set(0.1, 1, 0.1);
  scene.add(key, rim, top);

  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(40000, 40000),
    new THREE.MeshStandardMaterial({ color: 0x15181c, roughness: 0.38, metalness: 0.45, envMapIntensity: 0.5 }));
  floor.rotation.x = -Math.PI / 2;
  floor.position.y = FLOOR_Y;
  scene.add(floor);

  const blobTex = blobTexture();
  const products = productData.map(({ meta, buf }, i) => {
    const p = buildProduct(meta, buf, i);
    scene.add(p.group);
    p.blob = new THREE.Mesh(new THREE.PlaneGeometry(PRODUCT_R * 3, PRODUCT_R * 3),
      new THREE.MeshBasicMaterial({ map: blobTex, transparent: true, depthWrite: false, color: 0x000000 }));
    p.blob.rotation.x = -Math.PI / 2;
    p.blob.position.set(i * SPACING, FLOOR_Y + 1, 0);
    scene.add(p.blob);
    return p;
  });

  // dust hanging in the light
  const DUST = 1400;
  const dGeo = new THREE.BufferGeometry();
  const dPos = new Float32Array(DUST * 3);
  for (let i = 0; i < DUST; i++) {
    dPos[i * 3] = -1800 + Math.random() * (SPACING * 3 + 3600);
    dPos[i * 3 + 1] = FLOOR_Y + Math.random() * 1900;
    dPos[i * 3 + 2] = -1800 + Math.random() * 3000;
  }
  dGeo.setAttribute("position", new THREE.BufferAttribute(dPos, 3));
  const dustMat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    uniforms: { uTime: { value: 0 }, uPR: { value: 1 } },
    vertexShader: /* glsl */ `
      uniform float uTime, uPR; varying float vA;
      void main(){
        vec3 p = position;
        float s = fract(sin(dot(position, vec3(12.9, 78.2, 37.7))) * 43758.5);
        p.y += sin(uTime * 0.2 + s * 30.0) * 40.0;
        p.x += cos(uTime * 0.15 + s * 20.0) * 30.0;
        vec4 mv = modelViewMatrix * vec4(p, 1.0);
        gl_Position = projectionMatrix * mv;
        gl_PointSize = (2.0 + s * 3.0) * uPR * (900.0 / max(-mv.z, 1.0));
        vA = (0.25 + 0.75 * s) * (1.0 - smoothstep(1500.0, 4200.0, -mv.z));
      }`,
    fragmentShader: /* glsl */ `
      varying float vA;
      void main(){ float r = length(gl_PointCoord - 0.5); gl_FragColor = vec4(vec3(0.75, 0.82, 1.0), smoothstep(0.5, 0.0, r) * vA * 0.55); }`,
  });
  const dust = new THREE.Points(dGeo, dustMat);
  dust.frustumCulled = false;
  scene.add(dust);

  const tgt = new THREE.Vector3();
  const ray = new THREE.Raycaster();
  const sphere = new THREE.Sphere();
  const state = { stoneF: 0, active: 0, hover: false };

  function update(T, dt, time, ptr) {
    const u = T - WORK_START;
    const x = clamp(u - 0.15, 0, 3);
    const fi = Math.floor(x);
    const stoneF = Math.min(fi + smooth(0.45, 1, x - fi), 3);
    state.stoneF = stoneF;
    const active = Math.round(stoneF);
    if (active !== state.active) products[state.active].tap = false;
    state.active = active;

    const approach = 1 - smooth(A_TO_B[0], A_TO_B[1] + 0.25, T);
    const leave = smooth(B_TO_C[0] - 0.1, B_TO_C[1], T);
    const rk = view.aspect < 1 ? clamp(0.9 / view.aspect, 1, 2.0) : 1;
    const dist = (1700 + approach * 1500 + leave * 900) * rk;
    const az = (-10 + approach * -28 + ptr.x * 4 + Math.sin(stoneF * Math.PI) * 6) * DEG;
    const el = (7 + ptr.y * 2.5 + leave * 16) * DEG;
    tgt.set(stoneF * SPACING, 10 + leave * 120, 0);
    camera.position.set(tgt.x + Math.sin(az) * Math.cos(el) * dist, tgt.y + Math.sin(el) * dist, tgt.z + Math.cos(az) * Math.cos(el) * dist);
    camera.lookAt(tgt);
    camera.aspect = view.aspect;
    const sx = view.narrow ? 0 : 0.2;
    const sy = view.narrow ? 0.2 : 0;
    camera.setViewOffset(view.w, view.h, -sx * view.w, sy * view.h, view.w, view.h);
    sky.position.copy(camera.position);

    products.forEach((p, i) => {
      const bob = REDUCED ? 0 : Math.sin(time * 0.55 + p.seed) * 22;
      p.float.position.y = bob;
      p.vel *= Math.exp(-dt * 1.6);
      p.velX *= Math.exp(-dt * 2.2);
      p.float.rotation.y += ((REDUCED ? 0.02 : 0.11) + p.vel) * dt;
      p.float.rotation.x += p.velX * dt;
      p.float.rotation.x *= Math.exp(-dt * 0.9);
      p.float.rotation.z = Math.sin(time * 0.3 + p.seed) * 0.03;

      // explode along the designed vectors while hovered (or tapped, on touch)
      const want = i === active && (state.hover || p.tap) ? 1 : 0;
      p.explode += (want - p.explode) * (1 - Math.exp(-dt * 3.4));
      if (Math.abs(p.explode - p.shown) > 1e-4) {
        p.shown = p.explode;
        const e = p.explode * p.explode * (3 - 2 * p.explode) * p.explodeK;
        for (const q of p.parts) q.mesh.position.copy(q.home).addScaledVector(q.ex, e);
      }
      p.blob.material.opacity = 0.6 - bob / 400;
      p.blob.scale.setScalar(1 + bob / 700 + p.explode * 0.25);
    });
    dustMat.uniforms.uTime.value = time;
    dustMat.uniforms.uPR.value = view.pr;

    // hover test against the active product's bounding sphere
    const p = products[active];
    ray.setFromCamera(ptr.ndc, camera);
    sphere.center.setFromMatrixPosition(p.float.matrixWorld);
    sphere.radius = PRODUCT_R * 1.05;
    state.hover = HOVER && ptr.inside && ray.ray.intersectsSphere(sphere);
    state.hit = ray.ray.intersectsSphere(sphere);
  }

  const v3 = new THREE.Vector3();
  function productScreen(i) {
    const p = products[i];
    p.float.getWorldPosition(v3);
    const c = v3.clone().project(camera);
    const up = v3.clone().addScaledVector(camera.up, PRODUCT_R * 1.05).project(camera);
    const x = (c.x * 0.5 + 0.5) * view.w, y = (-c.y * 0.5 + 0.5) * view.h;
    const r = Math.abs((up.y - c.y) * 0.5 * view.h);
    return { x, y, r };
  }

  // one product, assembled and unrotated, as mesh entries for the particle field
  function entries(i) {
    const p = products[i];
    const rot = p.float.rotation.clone();
    p.float.rotation.set(0, -0.55, 0);
    for (const q of p.parts) q.mesh.position.copy(q.home);
    p.group.updateMatrixWorld(true);
    const inv = p.group.matrixWorld.clone().invert();
    const out = p.parts.map((q) => ({ geometry: q.mesh.geometry, matrix: inv.clone().multiply(q.mesh.matrixWorld) }));
    p.float.rotation.copy(rot);
    p.shown = -1;
    return out;
  }

  return { scene, camera, update, state, products, productScreen, entries };
}

// ================================================================ SCENE C
const PCOUNT = 22000;

function wordmarkPoints(n) {
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
  const cells = [];
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const k = (y * W + x) * 4;
    if (data[k + 3] > 140) cells.push(x, y, data[k] > data[k + 1] ? 1 : 0);
  }
  const pos = new Float32Array(n * 3), acc = new Float32Array(n);
  const count = cells.length / 3;
  const scale = 2.35;
  for (let i = 0; i < n; i++) {
    const j = Math.floor(Math.random() * count) * 3;
    pos[i * 3] = ((cells[j] + Math.random() - 10) / S - 155) * scale;
    pos[i * 3 + 1] = -((cells[j + 1] + Math.random() - 16) / S - 50) * scale;
    pos[i * 3 + 2] = (Math.random() - 0.5) * 70;
    acc[i] = cells[j + 2];
  }
  return { pos, acc };
}

function buildSceneC(shapes) {
  const scene = new THREE.Scene();
  scene.add(skyDome(0x040506, 0x0c0f13, 0x060708, new THREE.Vector3(0.3, 0.2, 1), 0x151a22));
  const camera = new THREE.PerspectiveCamera(35, 1, 5, 30000);
  camera.position.set(0, 0, 2400);

  const group = new THREE.Group();
  scene.add(group);

  const pos = new Float32Array(PCOUNT * 3);
  const vel = new Float32Array(PCOUNT * 3);
  const col = new Float32Array(PCOUNT * 3);
  const size = new Float32Array(PCOUNT);
  const acc = new Float32Array(PCOUNT);
  const switchAt = new Float32Array(PCOUNT);
  for (let i = 0; i < PCOUNT; i++) {
    const d = new THREE.Vector3().randomDirection().multiplyScalar(1400 + Math.random() * 1600);
    pos.set([d.x, d.y, d.z], i * 3);
    size[i] = 0.55 + Math.random() * 0.9;
  }
  const geo = new THREE.BufferGeometry();
  const pAttr = new THREE.BufferAttribute(pos, 3).setUsage(THREE.DynamicDrawUsage);
  const cAttr = new THREE.BufferAttribute(col, 3).setUsage(THREE.DynamicDrawUsage);
  geo.setAttribute("position", pAttr);
  geo.setAttribute("color", cAttr);
  geo.setAttribute("aSize", new THREE.BufferAttribute(size, 1));
  const mat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    uniforms: { uPR: { value: 1 }, uSize: { value: 7 } },
    vertexShader: /* glsl */ `
      uniform float uPR, uSize; attribute vec3 color; attribute float aSize; varying vec3 vCol;
      void main(){ vec4 mv = modelViewMatrix * vec4(position, 1.0); gl_Position = projectionMatrix * mv;
        gl_PointSize = uSize * aSize * uPR * (900.0 / max(-mv.z, 1.0)); vCol = color; }`,
    fragmentShader: /* glsl */ `
      varying vec3 vCol;
      void main(){ float r = length(gl_PointCoord - 0.5); float a = smoothstep(0.5, 0.08, r); gl_FragColor = vec4(vCol, a); }`,
  });
  const points = new THREE.Points(geo, mat);
  points.frustumCulled = false;
  group.add(points);

  let cur = 0, prev = 0, clock = 0;
  const base = new THREE.Color(0xc9d3e6), yolk = new THREE.Color(0xf4a300);
  const ray = new THREE.Raycaster();
  const inv = new THREE.Matrix4();
  const ro = new THREE.Vector3(), rd = new THREE.Vector3();

  function setShape(k) {
    if (k === cur) return;
    prev = cur; cur = k;
    for (let i = 0; i < PCOUNT; i++) switchAt[i] = clock + Math.random() * 0.45;
  }

  function update(T, dt, time, ptr, active) {
    clock = time;
    const rk = view.aspect < 1.1 ? clamp(1.05 / view.aspect, 1, 1.8) : 1;
    camera.position.set(ptr.x * 90, ptr.y * 60, 2750 * rk);
    camera.lookAt(0, 0, 0);
    camera.aspect = view.aspect;
    const sx = view.narrow ? 0 : 0.3;
    const sy = view.narrow ? 0.27 : 0;
    camera.setViewOffset(view.w, view.h, -sx * view.w, sy * view.h, view.w, view.h);
    group.rotation.y = Math.sin(time * 0.18) * 0.35 + ptr.x * 0.25;
    group.rotation.x = ptr.y * -0.12;
    group.updateMatrixWorld();
    mat.uniforms.uPR.value = view.pr;
    if (!active) return;

    // pointer ray in the group's own space, for the push
    ray.setFromCamera(ptr.ndc, camera);
    inv.copy(group.matrixWorld).invert();
    ro.copy(ray.ray.origin).applyMatrix4(inv);
    rd.copy(ray.ray.direction).transformDirection(inv);
    const push = ptr.inside && ptr.moving;

    // a damped spring toward each point's target, integrated in 60 Hz frames:
    // stiffness 0.02 and damping 0.18 per frame is a damping ratio near 0.64,
    // so a shape lands with one soft overshoot instead of wobbling
    const f = Math.min(dt * 60, 3);
    const stiff = 0.02 * f;
    const damp = Math.exp(-0.18 * f);
    const accK = 1 - Math.exp(-dt * 4);
    const sCur = shapes[cur], sPrev = shapes[prev];
    for (let i = 0; i < PCOUNT; i++) {
      const s = time >= switchAt[i] ? sCur : sPrev;
      const i3 = i * 3;
      let px = pos[i3], py = pos[i3 + 1], pz = pos[i3 + 2];
      let vx = vel[i3], vy = vel[i3 + 1], vz = vel[i3 + 2];
      vx += (s.pos[i3] - px) * stiff;
      vy += (s.pos[i3 + 1] - py) * stiff;
      vz += (s.pos[i3 + 2] - pz) * stiff;
      // a slow curl so a finished shape still breathes
      vx += Math.sin(py * 0.006 + time * 0.8) * 0.22 * f;
      vy += Math.cos(pz * 0.006 + time * 0.7) * 0.22 * f;
      vz += Math.sin(px * 0.006 + time * 0.9) * 0.22 * f;
      if (push) {
        const ox = px - ro.x, oy = py - ro.y, oz = pz - ro.z;
        const t = ox * rd.x + oy * rd.y + oz * rd.z;
        const qx = ox - rd.x * t, qy = oy - rd.y * t, qz = oz - rd.z * t;
        const d2 = qx * qx + qy * qy + qz * qz;
        if (d2 < 160 * 160) {
          const d = Math.sqrt(d2) + 1e-3, g = (1 - d / 160) * 7 * f;
          vx += (qx / d) * g; vy += (qy / d) * g; vz += (qz / d) * g;
        }
      }
      vx *= damp; vy *= damp; vz *= damp;
      px += vx * f; py += vy * f; pz += vz * f;
      pos[i3] = px; pos[i3 + 1] = py; pos[i3 + 2] = pz;
      vel[i3] = vx; vel[i3 + 1] = vy; vel[i3 + 2] = vz;
      acc[i] += (s.acc[i] - acc[i]) * accK;
      const sp = Math.min(Math.sqrt(vx * vx + vy * vy + vz * vz) / 38, 1);
      const a = Math.max(acc[i], sp);
      const glow = 0.78 + sp * 1.6;
      col[i3] = lerp(base.r, yolk.r, a) * glow;
      col[i3 + 1] = lerp(base.g, yolk.g, a) * glow;
      col[i3 + 2] = lerp(base.b, yolk.b, a) * glow;
    }
    pAttr.needsUpdate = true;
    cAttr.needsUpdate = true;
  }

  return { scene, camera, update, setShape, get shape() { return cur; } };
}

// ============================================================ COMPOSITE
const post = new THREE.ShaderMaterial({
  depthTest: false, depthWrite: false,
  uniforms: {
    tA: { value: null }, tB: { value: null }, uMix: { value: 0 }, uTime: { value: 0 },
    uCA: { value: 0.0015 }, uGlitch: { value: 0 }, uAspect: { value: 1 }, uRes: { value: new THREE.Vector2(1, 1) },
    uExpA: { value: 1 }, uExpB: { value: 1 }, uGrain: { value: 0.035 }, uVig: { value: 0.35 },
  },
  vertexShader: "varying vec2 vUv; void main(){ vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }",
  fragmentShader: /* glsl */ `
    uniform sampler2D tA, tB; uniform float uMix, uTime, uCA, uGlitch, uAspect, uExpA, uExpB, uGrain, uVig; uniform vec2 uRes;
    varying vec2 vUv;
    float hash(vec2 p){ p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); }
    float vnoise(vec2 p){ vec2 i = floor(p), f = fract(p); vec2 u = f * f * (3.0 - 2.0 * f);
      return mix(mix(hash(i), hash(i + vec2(1, 0)), u.x), mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), u.x), u.y); }
    float fbm(vec2 p){ float s = 0.0, a = 0.5; for (int i = 0; i < 5; i++){ s += a * vnoise(p); p = p * 2.03 + 17.1; a *= 0.5; } return s; }
    float cells(vec2 p){ vec2 i = floor(p), f = fract(p); float d = 1.0;
      for (int y = -1; y <= 1; y++) for (int x = -1; x <= 1; x++){ vec2 g = vec2(float(x), float(y)); vec2 o = vec2(hash(i + g), hash(i + g + 19.7)); d = min(d, length(g + o - f)); }
      return d; }
    vec3 sampleCA(sampler2D t, vec2 uv, vec2 dir){ return vec3(texture2D(t, uv + dir).r, texture2D(t, uv).g, texture2D(t, uv - dir).b); }
    void main(){
      vec2 uv = vUv;
      vec2 p = (uv - 0.5) * vec2(uAspect, 1.0);

      // frost: the new scene crystallises in through noise plus cells
      float n = fbm(p * 1.8 + 3.0) * 0.62 + fbm(p * 9.0 - 1.3) * 0.24 + (1.0 - cells(p * 14.0)) * 0.14;
      n = clamp((n - 0.24) / 0.46, 0.0, 1.0);   // value-noise fbm lives in about 0.24..0.70
      float edge = uMix * 1.24 - 0.12;
      float m = smoothstep(edge + 0.012, edge - 0.012, n);
      float band = smoothstep(0.05, 0.0, abs(n - edge)) * step(0.0005, uMix) * step(uMix, 0.9995);
      float halo = smoothstep(0.14, 0.0, abs(n - edge)) * step(0.0005, uMix) * step(uMix, 0.9995);

      // tech displacement: a few rows jump sideways while a transition runs
      float tick = floor(uTime * 16.0);
      float row = floor(uv.y * 54.0);
      float jump = step(0.88, hash(vec2(row, tick))) * (hash(vec2(row * 1.7, tick + 3.0)) - 0.5);
      uv.x += jump * 0.06 * uGlitch;

      // frosted glass just ahead of the edge: the old scene refracts and blurs a little
      vec2 grad = vec2(fbm(p * 7.0 + uTime * 0.25), fbm(p * 7.0 + 9.1 - uTime * 0.25)) - 0.5;
      uv += grad * halo * 0.035;

      vec2 dir = (uv - 0.5) * (uCA + halo * 0.012 + uGlitch * 0.012);
      vec3 a = sampleCA(tA, uv, dir) * uExpA;
      vec3 b = sampleCA(tB, uv, dir) * uExpB;
      vec3 col = mix(a, b, m);

      float crystal = pow(1.0 - cells(p * 38.0 + 4.0), 9.0);
      col = mix(col, col * 0.85 + vec3(0.62, 0.72, 0.86) * 0.35, halo * 0.5);
      col += band * (vec3(0.7, 0.82, 1.0) * 0.45 + crystal * 1.4);

      float v = smoothstep(1.3, 0.3, length((vUv - 0.5) * vec2(uAspect * 0.85, 1.0)));
      col *= mix(1.0 - uVig, 1.0, v);

      gl_FragColor = vec4(col, 1.0);
      #include <tonemapping_fragment>
      #include <colorspace_fragment>
      gl_FragColor.rgb += (hash(vUv * uRes + fract(uTime * 7.0) * 100.0) - 0.5) * uGrain;
    }`,
});
const postScene = new THREE.Scene();
const postQuad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), post);
postQuad.frustumCulled = false;
postScene.add(postQuad);
const postCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

// ============================================================== LOADING
let ldShown = 0, ldTarget = 0, ldDone = false;
function paintLoader() {
  ldShown += (ldTarget - ldShown) * 0.1;
  if (ldDone && ldShown > 0.995) ldShown = 1;
  ldPct.textContent = String(Math.round(ldShown * 100)).padStart(3, "0");
  ldBar.style.transform = `scaleX(${ldShown.toFixed(4)})`;
  if (ldShown < 1) requestAnimationFrame(paintLoader);
}
requestAnimationFrame(paintLoader);

async function fetchText(url, onProgress, expected) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(url + " " + res.status);
  if (!res.body || !res.body.getReader) { const t = await res.text(); onProgress(1); return t; }
  const reader = res.body.getReader();
  const chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value); got += value.length;
    onProgress(Math.min(got / expected, 1));
  }
  const all = new Uint8Array(got);
  let o = 0;
  for (const c of chunks) { all.set(c, o); o += c.length; }
  return new TextDecoder().decode(all);
}
const decode64 = (t) => Uint8Array.from(atob(t.trim()), (ch) => ch.charCodeAt(0)).buffer;
const tick = () => new Promise((r) => setTimeout(r, 0));

let A, B, C;
async function load() {
  const metas = await Promise.all([
    fetch("assets/cascade.json").then((r) => r.json()),
    ...WORK.map((w) => fetch(`assets/products/${w.model}.json`).then((r) => r.json())),
  ]);
  const urls = ["assets/cascade.b64.txt", ...WORK.map((w) => `assets/products/${w.model}.b64.txt`)];
  const labels = ["Cascade", ...WORK.map((w) => w.name)];
  const sizes = metas.map((m) => Math.ceil(m.bytes / 3) * 4);
  const totalBytes = sizes.reduce((s, v) => s + v, 0);
  const got = sizes.map(() => 0);
  let done = 0;
  const texts = await Promise.all(urls.map((url, k) =>
    fetchText(url, (p) => { got[k] = p * sizes[k]; ldTarget = 0.72 * got.reduce((s, v) => s + v, 0) / totalBytes; }, sizes[k])
      .then((t) => {
        done++;
        ldFile.textContent = `${labels[k]} · ${fmtInt(metas[k].parts.length)} parts · ${done}/${urls.length}`;
        return t;
      })));
  const bufs = texts.map(decode64);
  ldTarget = 0.76; await tick();

  const env = makeEnvironment();
  A = buildSceneA(metas[0], bufs[0], env);
  ldTarget = 0.82; await tick();
  B = buildSceneB(env, WORK.map((_, i) => ({ meta: metas[i + 1], buf: bufs[i + 1] })));
  ldTarget = 0.88; await tick();

  const plain = (pts) => ({ pos: pts, acc: new Float32Array(PCOUNT) });
  const shapes = [
    wordmarkPoints(PCOUNT),
    plain(normalizePoints(surfaceSample(A.entries(), PCOUNT), 760)),
    plain(normalizePoints(surfaceSample(B.entries(0), PCOUNT), 820)),
    plain(normalizePoints(surfaceSample(B.entries(1), PCOUNT), 860)),
    plain(normalizePoints(surfaceSample(B.entries(3), PCOUNT), 860)),
  ];
  C = buildSceneC(shapes);
  ldTarget = 0.93; await tick();

  resize();
  // warm every program and the shadow map now, not at the first transition
  const warm = [[A.scene, A.camera], [B.scene, B.camera], [C.scene, C.camera]];
  for (const [sc, cam] of warm) {
    try { if (renderer.compileAsync) await renderer.compileAsync(sc, cam); else renderer.compile(sc, cam); } catch (e) { /* compile on first use instead */ }
  }
  const ptr0 = { x: 0, y: 0, ndc: new THREE.Vector2(), inside: false, moving: false };
  A.update(0, 0.016, 0, ptr0); B.update(6.2, 0.016, 0, ptr0); C.update(9.5, 0.016, 0, ptr0, false);
  for (const [sc, cam] of warm) { renderer.setRenderTarget(rtA); renderer.render(sc, cam); }
  renderer.setRenderTarget(null);

  const tris = A.triangles + B.products.reduce((s, p) => s + p.triangles, 0);
  ldFile.textContent = `5 assemblies · ${fmtInt(tris)} triangles`;
  ldTarget = 1;
  ldDone = true;
  loader.classList.add("ready");
  requestAnimationFrame(frame);

  // ?t=4.5 enters silently and jumps straight to that point of the timeline,
  // for stills and for checking one chapter without scrolling to it
  const pin = new URLSearchParams(location.search).get("t");
  if (pin !== null) {
    $("[data-enter='0']").click();
    const t = clamp(parseFloat(pin) || 0, 0, T_END);
    window.scrollTo(0, t * innerHeight);
    T = t;
  }
}

// ============================================================ INTERACTION
const ptr = { x: 0, y: 0, tx: 0, ty: 0, ndc: new THREE.Vector2(), inside: false, moving: false, lastMove: 0 };
window.addEventListener("pointermove", (e) => {
  ptr.tx = (e.clientX / innerWidth) * 2 - 1;
  ptr.ty = -((e.clientY / innerHeight) * 2 - 1);
  ptr.ndc.set(ptr.tx, ptr.ty);
  ptr.inside = true;
  ptr.lastMove = performance.now();
  if (drag && B) {
    const p = B.products[B.state.active];
    p.vel += (e.clientX - drag.x) * 0.011;
    p.velX += (e.clientY - drag.y) * 0.008;
    drag.x = e.clientX; drag.y = e.clientY;
  }
}, { passive: true });
window.addEventListener("mouseout", (e) => { if (!e.relatedTarget) ptr.inside = false; });

let drag = null;
canvas.addEventListener("pointerdown", (e) => {
  if (!B || !inWork) return;
  if (e.pointerType === "mouse") {
    if (!B.state.hover) return;
    drag = { x: e.clientX, y: e.clientY };
    body.classList.add("grabbing");
    sound.tick(0.05);
    return;
  }
  // touch: a tap on the model toggles its exploded view
  ptr.ndc.set((e.clientX / innerWidth) * 2 - 1, -((e.clientY / innerHeight) * 2 - 1));
  tapCheck = { x: e.clientX, y: e.clientY, t: performance.now() };
});
let tapCheck = null;
canvas.addEventListener("pointerup", (e) => {
  if (!tapCheck || !B) return;
  const still = Math.hypot(e.clientX - tapCheck.x, e.clientY - tapCheck.y) < 10 && performance.now() - tapCheck.t < 400;
  tapCheck = null;
  if (still && B.state.hit) {
    const p = B.products[B.state.active];
    p.tap = !p.tap;
    sound.chime(NOTES[p.tap ? 3 : 1], 0.06);
  }
});
window.addEventListener("pointerup", () => { drag = null; body.classList.remove("grabbing"); });

function jumpTo(key) {
  const t = JUMP[key];
  if (t == null) return;
  window.scrollTo({ top: t * innerHeight, behavior: REDUCED ? "auto" : "smooth" });
}
$$("[data-jump]").forEach((el) => el.addEventListener("click", (e) => { e.preventDefault(); jumpTo(el.dataset.jump); sound.tick(); }));

let lastShapeHover = 0;
$$("[data-shape]").forEach((el) => {
  el.addEventListener("pointerenter", () => { if (C) { C.setShape(+el.dataset.shape); lastShapeHover = performance.now(); sound.whoosh(); } });
  el.addEventListener("focus", () => { if (C) C.setShape(+el.dataset.shape); });
  el.addEventListener("pointerleave", () => { lastShapeHover = performance.now(); });
  // Behance and LinkedIn have no address yet (site.yaml still says TODO)
  if (el.dataset.todo) el.addEventListener("click", (e) => { e.preventDefault(); });
});

soundBtn.addEventListener("click", () => sound.set(!sound.on));
$$("[data-enter]").forEach((b) => b.addEventListener("click", () => {
  const withSound = b.dataset.enter === "1";
  body.classList.add("is-in");
  body.classList.remove("is-loading");
  entered = true;
  if (withSound) sound.set(true);
  sound.chime(NOTES[0], 0.08);
  enterTime = performance.now();
}));
window.addEventListener("keydown", (e) => {
  if (!entered && ldDone && e.key === "Enter") $("[data-enter='0']").click();
});

// ================================================================= PANELS
const panels = $$(".panel").map((el) => ({
  el, from: +el.dataset.from, to: +el.dataset.to, live: false, in: false,
  scr: $$("[data-scramble]", el), counts: $$("[data-count]", el),
}));
function updatePanels(T) {
  for (const p of panels) {
    const v = smooth(p.from, p.from + 0.16, T) * (1 - smooth(p.to - 0.16, p.to, T));
    p.el.style.opacity = v.toFixed(3);
    const dir = T < (p.from + p.to) / 2 ? 1 : -1;
    p.el.style.transform = `translate3d(0, ${((1 - v) * 26 * dir).toFixed(1)}px, 0)`;
    const live = v > 0.01;
    if (live !== p.live) { p.live = live; p.el.classList.toggle("live", live); }
    if (!entered) continue;
    if (v > 0.55 && !p.in) {
      p.in = true;
      p.el.classList.add("in");
      p.scr.forEach((e, i) => scramble(e, null, 620 + i * 140));
      p.counts.forEach(countUp);
    } else if (v < 0.15 && p.in) {
      p.in = false;
      p.el.classList.remove("in");
    }
  }
}

// work panel: swap the copy when the camera settles on the next product
const wNum = $("#wNum"), wTitle = $("#wTitle"), wType = $("#wType"), wLine = $("#wLine"), wSpecs = $("#wSpecs"), wLink = $("#wLink");
const workPanel = $(".p-work");
let shownWork = -1;
function showWork(i) {
  if (i === shownWork) return;
  const first = shownWork < 0;
  shownWork = i;
  const w = WORK[i];
  const apply = () => {
    wNum.textContent = String(i + 1).padStart(2, "0");
    scramble(wTitle, w.name, 600);
    scramble(wType, w.type, 500);
    wLine.textContent = w.line;
    wSpecs.innerHTML = w.specs.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
    if (w.href) { wLink.href = w.href; wLink.classList.remove("soon"); wLink.querySelector("span").textContent = "View project"; }
    else { wLink.removeAttribute("href"); wLink.classList.add("soon"); wLink.querySelector("span").textContent = "Case study soon"; }
    const p = B && B.products[i];
    retTag.textContent = p ? `${w.name} · ${p.parts.length} parts · ${fmtInt(p.triangles)} tris` : w.name;
    workPanel.classList.remove("swapping");
  };
  if (first) { apply(); return; }
  workPanel.classList.add("swapping");
  setTimeout(apply, 220);
  sound.chime(NOTES[(i + 1) % NOTES.length], 0.09);
}

// ================================================================== FRAME
let entered = false, enterTime = 0;
let T = 0, Tv = 0, inWork = false;
let last = performance.now();
let chapter = -1;
let hoverWas = false;

function resize() {
  // a hidden or collapsed frame reports 0 x 0, which would turn the scroll into NaN
  view.w = Math.max(innerWidth, 1); view.h = Math.max(innerHeight, 1);
  view.pr = Math.min(devicePixelRatio || 1, 1.75);
  view.aspect = view.w / view.h;
  view.narrow = view.w < 860;
  renderer.setPixelRatio(view.pr);
  renderer.setSize(view.w, view.h, false);
  rtA.setSize(Math.floor(view.w * view.pr), Math.floor(view.h * view.pr));
  rtB.setSize(Math.floor(view.w * view.pr), Math.floor(view.h * view.pr));
  post.uniforms.uAspect.value = view.aspect;
  post.uniforms.uRes.value.set(view.w * view.pr, view.h * view.pr);
  scroller.style.height = Math.round(view.h * (T_END + 1)) + "px";
}
window.addEventListener("resize", resize);

const fmt = (v, n = 3) => String(Math.round(Math.abs(v))).padStart(n, "0");

function frame(now) {
  requestAnimationFrame(frame);
  const dt = Math.min((now - last) / 1000, 0.05);
  last = now;
  const time = now / 1000;

  // damped scroll: a flick of the wheel becomes a glide
  const target = entered ? window.scrollY / view.h : 0;
  const prevT = T;
  T += (target - T) * (1 - Math.exp(-dt * 5.5));
  if (Math.abs(target - T) < 1e-4) T = target;
  const speed = (T - prevT) / Math.max(dt, 1e-3);
  Tv += (speed - Tv) * 0.2;

  ptr.x += (ptr.tx - ptr.x) * (1 - Math.exp(-dt * 3));
  ptr.y += (ptr.ty - ptr.y) * (1 - Math.exp(-dt * 3));
  ptr.moving = now - ptr.lastMove < 1200;

  updatePanels(T);
  inWork = T > WORK_START - 0.05 && T < B_TO_C[0];

  // which scenes are on screen
  const mAB = smooth(A_TO_B[0], A_TO_B[1], T);
  const mBC = smooth(B_TO_C[0], B_TO_C[1], T);
  let first, second = null, mix = 0;
  if (T < A_TO_B[0]) first = A;
  else if (T < A_TO_B[1]) { first = A; second = B; mix = mAB; }
  else if (T < B_TO_C[0]) first = B;
  else if (T < B_TO_C[1]) { first = B; second = C; mix = mBC; }
  else first = C;

  const needA = first === A || second === A;
  const needB = first === B || second === B;
  const needC = first === C || second === C;
  if (needA) A.update(T, dt, time, ptr);
  if (needB) B.update(T, dt, time, ptr);
  // the particle sim runs whenever the contact chapter is near, so the field
  // has already gathered into the wordmark by the time it is uncovered
  if (needC || T > B_TO_C[0] - 0.6) C.update(T, dt, time, ptr, true);

  // idle: with nothing hovered for a while, the field cycles its shapes
  if (T > B_TO_C[1] && now - lastShapeHover > 5200 && !$(".links a:hover")) {
    C.setShape((C.shape + 1) % SHAPES.length);
    lastShapeHover = now;
  }

  // hover over the active product
  const hov = needB && inWork && B.state.hover;
  if (hov !== hoverWas) {
    hoverWas = hov;
    body.classList.toggle("grab", hov);
    if (hov) sound.chime(NOTES[(B.state.active + 2) % NOTES.length], 0.05);
  }
  if (needB) showWork(B.state.active);

  // render
  const expOf = (s) => (s === A ? 0.86 : s === B ? 1.2 : 1.0);
  renderer.setRenderTarget(rtA);
  renderer.render(first.scene, first.camera);
  if (second) { renderer.setRenderTarget(rtB); renderer.render(second.scene, second.camera); }
  renderer.setRenderTarget(null);
  const u = post.uniforms;
  u.tA.value = rtA.texture;
  u.tB.value = second ? rtB.texture : rtA.texture;
  u.uMix.value = second ? mix : 0;
  u.uExpA.value = expOf(first);
  u.uExpB.value = expOf(second || first);
  u.uTime.value = time;
  const trans = second ? Math.sin(Math.PI * mix) : 0;
  const intro = entered ? 1 - smooth(0, 1.6, (now - enterTime) / 1000) : 1;
  u.uGlitch.value = REDUCED ? 0 : Math.max(trans, intro * 0.8);
  u.uCA.value = REDUCED ? 0.0008 : 0.0012 + Math.min(Math.abs(Tv) * 0.006, 0.012);
  u.uVig.value = first === A && !second ? 0.28 : 0.45;
  renderer.render(postScene, postCam);

  // HUD
  const ch = T < 1 ? 0 : T < 3.8 ? 1 : T < 8.4 ? 2 : 3;
  if (ch !== chapter) {
    chapter = ch;
    chapNum.textContent = String(ch).padStart(2, "0");
    scramble(chapName, CHAPTERS[ch], 500);
    railItems.forEach((li, i) => li.classList.toggle("on", i === ch));
    navLinks.forEach((a) => a.classList.toggle("on", a.dataset.jump === ["hero", "studio", "work", "contact"][ch]));
    sound.scene(ch === 0 || ch === 1 ? 0 : ch === 2 ? 1 : 2);
    if (entered) sound.tick(0.03);
  }
  railFill.style.transform = `scaleY(${(T / T_END).toFixed(4)})`;
  scrollHint.classList.toggle("gone", T > 0.12);

  if (needA && first === A) {
    telemetry.textContent = `AZ ${fmt(A.state.az)}° · EL ${fmt(A.state.el, 2)}° · R ${fmt(A.state.r, 4)} MM · EXPLODE ${fmt(A.state.ex * 100)}%`;
  } else if (needB && first === B) {
    const p = B.products[B.state.active];
    const deg = ((p.float.rotation.y / DEG) % 360 + 360) % 360;
    telemetry.textContent = `${WORK[B.state.active].name.toUpperCase()} · ROT ${fmt(deg)}° · EXPLODE ${fmt(p.explode * 100)}%`;
  } else if (first === C) {
    telemetry.textContent = `POINTS 22,000 · SHAPE ${SHAPES[C.shape].toUpperCase()}`;
  }

  // reticle on the active product
  const showRet = needB && first === B && !second && inWork && !view.narrow && T > WORK_START + 0.1;
  reticle.classList.toggle("on", showRet);
  if (showRet) {
    const sc = B.productScreen(B.state.active);
    const r = sc.r;
    reticle.style.width = reticle.style.height = (r * 2).toFixed(0) + "px";
    reticle.style.transform = `translate3d(${(sc.x - r).toFixed(1)}px, ${(sc.y - r).toFixed(1)}px, 0)`;
  }
}

load().catch((err) => {
  console.error(err);
  ldFile.textContent = "Could not load the models";
  ldPct.textContent = "ERR";
});
