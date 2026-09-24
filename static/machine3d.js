/* Cascade assembling in 3D behind the left of the home page.
 *
 * 92 parts, decimated from the Rhino assembly, each flying in along the explode
 * vector build.py already gives it. The parts, their positions and the direction
 * they arrive from all come from the model; the palette, the camera and the
 * pacing are chosen here.
 *
 * Two things make the assembly read cleanly:
 *
 *  - Every approach is solved offline, by assembly_paths.py, which sweeps each
 *    part's own vertices along candidate directions and keeps the first that
 *    clears everything already standing. 74 of the 92 can use the explode
 *    vector the exploded view gives them; the other 18 would have driven
 *    straight through the column or the posts, and come in radially or from
 *    above instead. An exploded view only promises that the exploded POSITIONS
 *    do not overlap - never that the line between there and home is clear.
 *  - Inside a stage they arrive strictly one at a time, in the order that solve
 *    produced. The solver only ever proved a part clear against parts already
 *    STANDING, so any two in flight at once could still meet in mid-air - the
 *    one case it cannot speak for. Sequencing them removes it entirely.
 *
 * Scroll drives it, damped: the displayed progress eases toward the scroll
 * position instead of snapping to it, so a flick of the wheel is a glide.
 * Nothing is simulated - scroll back and it comes apart again.
 */
import * as THREE from "three";

// Blueprint: the machine is drawn, not shaded. Each part is a pale flat fill
// that hides whatever is behind it, plus its real feature edges as lines - which
// is what separates a blueprint from a wireframe. The fill is almost the colour
// of the paper, so the drawing is carried by the linework.
const PAPER_FILL = 0xf2f0ea;
const LINE_INK = 0x16355f;          // drawing blue
const LINE_ACCENT = 0xb06f00;       // the driven parts, still readable as a line

const rail = document.querySelector(".machine");
const canvas = rail && rail.querySelector(".mc-canvas");
if (canvas) init();

// Rhino hands every part back in black (the colour lives on the layer, not the
// object), so the palette is assigned here, by what each part is and does.
// only two pens: the drive train is what you turn, everything else is structure
function penFor(name) {
  if (name.startsWith("crank_") || name.startsWith("knob_sleeve") ||
      name.startsWith("flywheel") || name.startsWith("helical_gear") ||
      name.startsWith("bevel_") || name.startsWith("marbles")) return LINE_ACCENT;
  return LINE_INK;
}

async function init() {
  const base = canvas.getAttribute("data-base") || "";
  const still = window.matchMedia("(prefers-reduced-motion: reduce)");

  const [meta, b64] = await Promise.all([
    fetch(base + "static/machine/cascade.json").then((r) => r.json()),
    fetch(base + "static/machine/cascade.b64.txt").then((r) => r.text()),
  ]);
  // base64 rather than a bare .bin: a static artifact host will not serve an
  // unknown extension, and one code path beats two
  const buf = Uint8Array.from(atob(b64), (ch) => ch.charCodeAt(0)).buffer;

  const sections = meta.stages.map((_, i) =>
    document.querySelector(canvas.dataset["s" + i]));
  const caps = [].slice.call(rail.querySelectorAll(".mc-cap"));

  // one fill material for every part: it exists only to occlude
  const fillMat = new THREE.MeshBasicMaterial({
    color: PAPER_FILL, polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1,
  });
  const pens = new Map();
  const penMat = (c) => {
    if (!pens.has(c)) pens.set(c, new THREE.LineBasicMaterial({ color: c }));
    return pens.get(c);
  };
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  const scene = new THREE.Scene();

  const [lo, hi] = meta.bbox;
  const mid = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2];
  const height = hi[2] - lo[2];

  const root = new THREE.Group();
  root.rotation.x = -Math.PI / 2;          // model is Z-up; stand it up on screen
  const spin = new THREE.Group();
  spin.add(root);
  scene.add(spin);

  // rank within each stage by the solved arrival order, not document order
  const byStage = {};
  for (const p of meta.parts) (byStage[p.stage] = byStage[p.stage] || []).push(p);
  const rank = new Map();
  for (const k of Object.keys(byStage)) {
    const list = byStage[k].slice().sort((a, b) => a.arrive - b.arrive);
    list.forEach((p, i) => rank.set(p, { i: i, n: list.length }));
  }

  const parts = meta.parts.map((p) => {
    const g = new THREE.BufferGeometry();
    // 10 bytes a vertex: int16 x,y,z against this part's own origin and scale,
    // then int8 nx,ny,nz and a pad byte, read as two views over the same bytes
    const n = p.vertices;
    const pos = new THREE.InterleavedBuffer(new Int16Array(buf, p.vertex_at, n * 5), 5);
    const nor = new THREE.InterleavedBuffer(new Int8Array(buf, p.vertex_at, n * 10), 10);
    g.setAttribute("position", new THREE.InterleavedBufferAttribute(pos, 3, 0, false));
    g.setAttribute("normal", new THREE.InterleavedBufferAttribute(nor, 3, 6, true));
    g.setIndex(new THREE.BufferAttribute(
      new Uint16Array(buf, p.index_at, p.triangles * 3), 1));

    const mesh = new THREE.Mesh(g, fillMat);
    // real feature edges, not a wireframe: 24 degrees keeps the silhouette and
    // the machined steps and drops the triangulation
    const edge = new THREE.LineSegments(
      new THREE.EdgesGeometry(g, 24), penMat(penFor(p.name)));
    mesh.add(edge);
    mesh.scale.setScalar(p.scale);
    mesh.position.set(p.org[0] - mid[0], p.org[1] - mid[1], p.org[2] - mid[2]);
    root.add(mesh);

    return {
      mesh, stage: p.stage, home: mesh.position.clone(), qscale: p.scale,
      from: p.from,                                  // the solved approach vector
      slot: rank.get(p),                             // its turn, and how many share the stage
    };
  });

  scene.add(new THREE.HemisphereLight(0xfff6e6, 0x8a857c, 1.25));
  const key = new THREE.DirectionalLight(0xfff3df, 2.4);
  key.position.set(-300, 460, 340);
  scene.add(key);
  const fill = new THREE.DirectionalLight(0xdfe6ff, 0.85);
  fill.position.set(340, 140, -260);
  scene.add(fill);
  const rim = new THREE.DirectionalLight(0xffffff, 0.5);
  rim.position.set(60, -180, -320);
  scene.add(rim);

  const camera = new THREE.PerspectiveCamera(26, 1, 10, 4000);

  function resize() {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    // Large, and deliberately cropped. The machine is pushed left so its body
    // sits in the part of the canvas the mask keeps, and runs off the edge
    // rather than shrinking to fit - which is how a background reads as big.
    const dist = (height * 0.74) / Math.tan((camera.fov * Math.PI) / 360);
    camera.position.set(0, height * 0.05, dist);
    camera.lookAt(0, height * 0.02, 0);
    // sit in the left of the frame, where the mask keeps the drawing
    spin.position.x = -(2 * height * 0.74) * (w / h) * 0.13;
    camera.updateProjectionMatrix();
  }

  let breaks = [];
  function measure() {
    const vh = window.innerHeight;
    const end = Math.max(document.documentElement.scrollHeight - vh, 1);
    breaks = sections.map((s, i) => {
      if (i === 0 || !s) return 0;
      return Math.min(Math.max(s.getBoundingClientRect().top + window.scrollY - vh * 0.5, 0), end);
    });
    breaks.push(end);
    for (let i = 1; i < breaks.length; i++) {
      if (breaks[i] <= breaks[i - 1]) breaks[i] = breaks[i - 1] + 1;
    }
  }

  const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);
  // smootherstep: no kick at either end, which is what makes a part settle
  const ease = (t) => t * t * t * (t * (t * 6 - 15) + 10);
  // each part owns one slice of its stage and has it to itself

  function paint(scrollY) {
    let current = 0;
    const t = [];
    for (let i = 0; i < meta.stages.length; i++) {
      const b0 = breaks[i], b1 = breaks[i + 1];
      const a = i === 0 && scrollY <= 0 ? 1 :
        clamp01((scrollY - b0) / Math.max(b1 - b0, 1));
      t.push(a);
      if (a > 0.35) current = i;
    }
    for (const p of parts) {
      const a = clamp01(t[p.stage] * p.slot.n - p.slot.i);
      p.mesh.visible = a > 0.0005;
      if (!p.mesh.visible) continue;
      const e = ease(a);
      p.mesh.position.set(
        p.home.x + p.from[0] * (1 - e),
        p.home.y + p.from[1] * (1 - e),
        p.home.z + p.from[2] * (1 - e));
    }
    caps.forEach((c, i) => c.classList.toggle("on", i === current));
    const whole = clamp01(scrollY / breaks[breaks.length - 1]);
    spin.rotation.y = -0.30 + whole * 0.58;      // a slow turn down the page
    renderer.render(scene, camera);
  }

  function settled() {
    for (const p of parts) {
      p.mesh.visible = true;
      p.mesh.position.copy(p.home);
    }
    caps.forEach((c, i) => c.classList.toggle("on", i === caps.length - 1));
    spin.rotation.y = 0.28;
    renderer.render(scene, camera);
  }

  // the displayed progress chases the scroll rather than tracking it exactly,
  // so a flick of the wheel becomes a glide instead of a jump
  let target = 0, shown = 0, running = false;
  function loop() {
    shown += (target - shown) * 0.16;
    if (Math.abs(target - shown) < 0.4) { shown = target; running = false; }
    paint(shown);
    if (running) requestAnimationFrame(loop);
  }
  function onScroll() {
    if (still.matches) return;
    target = window.scrollY;
    if (!running) { running = true; requestAnimationFrame(loop); }
  }

  function start() {
    resize();
    if (still.matches) { settled(); return; }
    measure();
    const pin = new URLSearchParams(location.search).get("mc");
    target = shown = pin !== null
      ? parseFloat(pin) * breaks[breaks.length - 1] : window.scrollY;
    paint(shown);
  }

  window.addEventListener("scroll", onScroll, { passive: true });
  window.addEventListener("resize", () => {
    // a resize is a jump, not a glide: snap the damped value to where the page
    // actually is rather than easing toward it from a stale position
    resize(); measure();
    target = shown = window.scrollY;
    paint(shown);
  });
  if (still.addEventListener) still.addEventListener("change", start);
  rail.classList.add("is-3d");
  start();
}
