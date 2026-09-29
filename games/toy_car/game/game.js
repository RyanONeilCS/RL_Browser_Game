// Toy Car: a tiny top-down racing game built as an RL test bed.
//
// Python (Playwright) talks to it through these globals:
//   window.gameState                 latest state object (see getState())
//   window.resetGame(seed?)          reset; a numeric seed gives a random start checkpoint + heading noise
//   window.stepGame(keys, frames=1)  stepped mode only: hold `keys` for `frames` physics frames, return state
//   window.setMode('stepped'|'realtime')
//   window.setDebug(bool), window.setHud(bool)
//   window.getTrack()                centerline, half width, checkpoints (for plotting/analysis)
//   window.gameConfig                { lapsToFinish, maxFrames, difficulty: 'easy'|'hard' }
//                                    (difficulty takes effect on the next resetGame)
//
// Manual play: index.html?difficulty=hard&debug
//
// Keys are the same names Playwright uses: "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"
// (w/a/s/d also work).
(() => {
  'use strict';

  const canvas = document.getElementById('game');
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;

  // ---- constants -------------------------------------------------------
  const DT = 1 / 60;            // fixed physics timestep (s)
  const RAY_ANGLES = [-90, -60, -30, 0, 30, 60, 90].map(d => d * Math.PI / 180);
  const RAY_MAX = 200;          // px
  const RAY_STEP = 3;           // px
  const N_POINTS = 64;
  const CP_EVERY = 4;

  // easy: steering works at any speed, so flat out is optimal and the brake is useless.
  // hard: grip limits how sharply the car can turn at speed (max turn rate = grip / speed),
  //       coasting barely slows the car, and the track has tight corners -> brake before them.
  const DIFFICULTIES = {
    easy: {
      maxSpeed: 260, accel: 320, brake: 520, drag: 0.9, turnRate: 3.2, grip: Infinity, halfWidth: 34,
      shape: t => 180 + 45 * Math.sin(2 * t) + 30 * Math.cos(3 * t + 0.5),
    },
    hard: {
      maxSpeed: 360, accel: 300, brake: 650, drag: 0.15, turnRate: 3.2, grip: 520, halfWidth: 26,
      shape: t => 175 + 55 * Math.sin(2 * t) + 35 * Math.cos(3 * t + 0.5) + 22 * Math.sin(5 * t + 1),
    },
  };

  const gameConfig = { lapsToFinish: 1, maxFrames: 60 * 90, difficulty: 'easy' };

  // ---- track -----------------------------------------------------------
  // Rebuilt by buildWorld() whenever the difficulty changes.
  let P, MAX_SPEED, HALF_WIDTH, centerline, checkpoints, N_CP, gates, builtFor = null;

  function buildWorld(difficulty) {
    P = DIFFICULTIES[difficulty];
    if (!P) throw new Error(`unknown difficulty ${difficulty}`);
    MAX_SPEED = P.maxSpeed;
    HALF_WIDTH = P.halfWidth;

    centerline = [];
    for (let i = 0; i < N_POINTS; i++) {
      const t = (i / N_POINTS) * Math.PI * 2;
      const r = P.shape(t);
      centerline.push({ x: W / 2 + r * Math.cos(t), y: H / 2 + r * Math.sin(t) * 0.9 });
    }
    checkpoints = centerline.filter((_, i) => i % CP_EVERY === 0);
    N_CP = checkpoints.length;

    // Each checkpoint is a gate line across the track (perpendicular to the centerline).
    // It only counts when the car's movement crosses the line, so the car has to
    // actually be on the road at that spot.
    gates = checkpoints.map((p, k) => {
      const i = k * CP_EVERY;
      const prev = centerline[(i - 1 + N_POINTS) % N_POINTS], next = centerline[(i + 1) % N_POINTS];
      const len = Math.hypot(next.x - prev.x, next.y - prev.y);
      const nx = -(next.y - prev.y) / len, ny = (next.x - prev.x) / len;
      return {
        a: { x: p.x + nx * HALF_WIDTH, y: p.y + ny * HALF_WIDTH },
        b: { x: p.x - nx * HALF_WIDTH, y: p.y - ny * HALF_WIDTH },
      };
    });
    builtFor = difficulty;
  }

  function segmentsCross(p1, p2, q1, q2) {
    const cross = (o, a, b) => (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
    const d1 = cross(q1, q2, p1), d2 = cross(q1, q2, p2);
    const d3 = cross(p1, p2, q1), d4 = cross(p1, p2, q2);
    return d1 * d2 < 0 && d3 * d4 < 0;
  }

  function distToSegment(px, py, a, b) {
    const dx = b.x - a.x, dy = b.y - a.y;
    const len2 = dx * dx + dy * dy;
    let t = len2 ? ((px - a.x) * dx + (py - a.y) * dy) / len2 : 0;
    t = Math.max(0, Math.min(1, t));
    const cx = a.x + t * dx - px, cy = a.y + t * dy - py;
    return Math.sqrt(cx * cx + cy * cy);
  }

  function distToCenterline(px, py) {
    let best = Infinity;
    for (let i = 0; i < N_POINTS; i++) {
      const d = distToSegment(px, py, centerline[i], centerline[(i + 1) % N_POINTS]);
      if (d < best) best = d;
    }
    return best;
  }

  const onTrack = (x, y) => distToCenterline(x, y) <= HALF_WIDTH;

  function wrapAngle(a) {
    while (a > Math.PI) a -= 2 * Math.PI;
    while (a < -Math.PI) a += 2 * Math.PI;
    return a;
  }

  // Small seeded RNG so resets are reproducible.
  function mulberry32(seed) {
    return () => {
      seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // ---- state -----------------------------------------------------------
  let car, mode = 'realtime', debug = false, hud = true;
  const held = new Set();

  function resetGame(seed) {
    if (builtFor !== gameConfig.difficulty) buildWorld(gameConfig.difficulty);
    let startCp = 0, headingNoise = 0;
    if (typeof seed === 'number') {
      const rng = mulberry32(seed);
      startCp = Math.floor(rng() * N_CP);
      headingNoise = (rng() - 0.5) * 0.4;
    }
    const p = checkpoints[startCp];
    const q = checkpoints[(startCp + 1) % N_CP];
    car = {
      x: p.x, y: p.y,
      angle: Math.atan2(q.y - p.y, q.x - p.x) + headingNoise,
      speed: 0,
      nextCheckpoint: (startCp + 1) % N_CP,
      checkpointsPassed: 0,
      crashed: false,
      frame: 0,
    };
    held.clear();
    return publish();
  }

  function controlsFrom(keys) {
    const k = new Set(keys);
    return {
      gas: k.has('ArrowUp') || k.has('w'),
      brake: k.has('ArrowDown') || k.has('s'),
      left: k.has('ArrowLeft') || k.has('a'),
      right: k.has('ArrowRight') || k.has('d'),
    };
  }

  function isDone() {
    const lap = Math.floor(car.checkpointsPassed / N_CP);
    return car.crashed || lap >= gameConfig.lapsToFinish;
  }
  const isTruncated = () => !isDone() && car.frame >= gameConfig.maxFrames;

  function physicsStep(c) {
    if (isDone() || isTruncated()) return;

    if (c.gas) car.speed += P.accel * DT;
    if (c.brake) car.speed -= P.brake * DT;
    if (!c.gas) car.speed -= P.drag * car.speed * DT;
    car.speed = Math.max(0, Math.min(MAX_SPEED, car.speed));

    // Steering needs some speed (full authority from ~80 px/s), and grip caps the
    // turn rate at high speed, so fast cars run wide in tight corners.
    const authority = Math.min(1, car.speed / 80);
    const turnRate = Math.min(P.turnRate, P.grip / Math.max(car.speed, 1));
    const steer = (c.right ? 1 : 0) - (c.left ? 1 : 0);
    car.angle = wrapAngle(car.angle + steer * turnRate * authority * DT);

    const from = { x: car.x, y: car.y };
    car.x += Math.cos(car.angle) * car.speed * DT;
    car.y += Math.sin(car.angle) * car.speed * DT;
    car.frame++;

    // Crash is checked first: leaving the road on the same frame as crossing a gate is a crash.
    if (!onTrack(car.x, car.y)) { car.crashed = true; return; }

    const gate = gates[car.nextCheckpoint];
    if (segmentsCross(from, car, gate.a, gate.b)) {
      car.checkpointsPassed++;
      car.nextCheckpoint = (car.nextCheckpoint + 1) % N_CP;
    }
  }

  function castRay(offset) {
    const a = car.angle + offset;
    const dx = Math.cos(a), dy = Math.sin(a);
    for (let d = RAY_STEP; d <= RAY_MAX; d += RAY_STEP) {
      if (!onTrack(car.x + dx * d, car.y + dy * d)) return d;
    }
    return RAY_MAX;
  }

  function getState() {
    const cp = checkpoints[car.nextCheckpoint];
    const done = isDone();
    return {
      x: car.x, y: car.y, angle: car.angle,
      speed: car.speed, maxSpeed: MAX_SPEED,
      nextCheckpoint: car.nextCheckpoint, numCheckpoints: N_CP,
      checkpointsPassed: car.checkpointsPassed,
      lap: Math.floor(car.checkpointsPassed / N_CP),
      angleToCheckpoint: wrapAngle(Math.atan2(cp.y - car.y, cp.x - car.x) - car.angle),
      distToCheckpoint: Math.hypot(cp.x - car.x, cp.y - car.y),
      distFromCenter: distToCenterline(car.x, car.y), halfWidth: HALF_WIDTH,
      rays: RAY_ANGLES.map(castRay), rayMax: RAY_MAX,
      crashed: car.crashed,
      done,                      // crashed or finished the required laps
      truncated: isTruncated(),  // ran out of time
      frame: car.frame, dt: DT, difficulty: builtFor,
    };
  }

  function publish() {
    window.gameState = getState();
    render();
    return window.gameState;
  }

  function stepGame(keys = [], frames = 1) {
    if (mode !== 'stepped') throw new Error("stepGame() needs setMode('stepped')");
    const c = controlsFrom(keys);
    for (let i = 0; i < frames; i++) physicsStep(c);
    return publish();
  }

  // ---- rendering -------------------------------------------------------
  function tracePath() {
    ctx.beginPath();
    centerline.forEach((p, i) => (i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y)));
    ctx.closePath();
  }

  function render() {
    ctx.fillStyle = '#3a7d3a';
    ctx.fillRect(0, 0, W, H);

    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    tracePath();
    ctx.strokeStyle = '#eee';
    ctx.lineWidth = HALF_WIDTH * 2 + 6;
    ctx.stroke();
    ctx.strokeStyle = '#555';
    ctx.lineWidth = HALF_WIDTH * 2;
    ctx.stroke();

    // start/finish line at checkpoint 0
    const s0 = checkpoints[0], s1 = centerline[1];
    const n = Math.atan2(s1.y - s0.y, s1.x - s0.x) + Math.PI / 2;
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.moveTo(s0.x + Math.cos(n) * HALF_WIDTH, s0.y + Math.sin(n) * HALF_WIDTH);
    ctx.lineTo(s0.x - Math.cos(n) * HALF_WIDTH, s0.y - Math.sin(n) * HALF_WIDTH);
    ctx.stroke();

    if (debug) {
      gates.forEach((g, i) => {
        ctx.strokeStyle = i === car.nextCheckpoint ? '#ff0' : 'rgba(255,255,255,0.3)';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(g.a.x, g.a.y);
        ctx.lineTo(g.b.x, g.b.y);
        ctx.stroke();
      });
      const rays = window.gameState ? window.gameState.rays : RAY_ANGLES.map(castRay);
      ctx.strokeStyle = 'rgba(255,80,80,0.9)';
      ctx.lineWidth = 1;
      RAY_ANGLES.forEach((off, i) => {
        ctx.beginPath();
        ctx.moveTo(car.x, car.y);
        ctx.lineTo(car.x + Math.cos(car.angle + off) * rays[i], car.y + Math.sin(car.angle + off) * rays[i]);
        ctx.stroke();
      });
    }

    ctx.save();
    ctx.translate(car.x, car.y);
    ctx.rotate(car.angle);
    ctx.fillStyle = car.crashed ? '#c33' : '#2a6cf0';
    ctx.fillRect(-11, -6, 22, 12);
    ctx.fillStyle = '#9cf';
    ctx.fillRect(3, -5, 5, 10);
    ctx.restore();

    if (hud) {
      const lap = Math.floor(car.checkpointsPassed / N_CP);
      ctx.fillStyle = 'rgba(0,0,0,0.55)';
      ctx.fillRect(6, 6, 170, 58);
      ctx.fillStyle = '#fff';
      ctx.font = '13px monospace';
      ctx.fillText(`speed ${car.speed.toFixed(0).padStart(3)} / ${MAX_SPEED}`, 14, 24);
      ctx.fillText(`cp ${car.checkpointsPassed % N_CP}/${N_CP}  lap ${lap}`, 14, 40);
      ctx.fillText(`frame ${car.frame}  ${builtFor}`, 14, 56);
    }

    if (mode === 'realtime' && (isDone() || isTruncated())) {
      ctx.fillStyle = 'rgba(0,0,0,0.6)';
      ctx.fillRect(0, H / 2 - 30, W, 60);
      ctx.fillStyle = '#fff';
      ctx.font = 'bold 22px system-ui, sans-serif';
      ctx.textAlign = 'center';
      const msg = car.crashed ? 'Crashed' : isTruncated() ? 'Out of time' : 'Finished!';
      ctx.fillText(`${msg}  (press R)`, W / 2, H / 2 + 8);
      ctx.textAlign = 'left';
    }
  }

  // ---- real-time loop + keyboard ----------------------------------------
  let last = performance.now(), acc = 0;
  function loop(now) {
    if (mode === 'realtime') {
      acc += Math.min(0.25, (now - last) / 1000);
      const c = controlsFrom([...held]);
      let stepped = false;
      while (acc >= DT) { physicsStep(c); acc -= DT; stepped = true; }
      if (stepped) publish();
    } else {
      acc = 0;
    }
    last = now;
    requestAnimationFrame(loop);
  }

  const DRIVE_KEYS = new Set(['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'w', 'a', 's', 'd']);
  window.addEventListener('keydown', e => {
    if (DRIVE_KEYS.has(e.key)) { held.add(e.key); e.preventDefault(); }
    else if (e.key === 'r' || e.key === 'R') resetGame();
    else if (e.key === 'D') setDebug(!debug);
    else if (e.key === 'h' || e.key === 'H') setHud(!hud);
  });
  window.addEventListener('keyup', e => held.delete(e.key));
  window.addEventListener('blur', () => held.clear());

  function setMode(m) {
    if (m !== 'stepped' && m !== 'realtime') throw new Error(`unknown mode ${m}`);
    mode = m;
    held.clear();
    document.getElementById('mode').textContent = m;
    render();
  }
  function setDebug(on) { debug = !!on; render(); }
  function setHud(on) { hud = !!on; render(); }

  Object.assign(window, {
    gameConfig, resetGame, stepGame, setMode, setDebug, setHud,
    getTrack: () => ({ centerline, halfWidth: HALF_WIDTH, checkpoints, gates }),
  });

  const params = new URLSearchParams(location.search);
  if (params.has('debug')) debug = true;
  if (params.has('difficulty')) gameConfig.difficulty = params.get('difficulty');
  resetGame();
  setMode(params.get('mode') === 'stepped' ? 'stepped' : 'realtime');
  window.gameReady = true;
  requestAnimationFrame(loop);
})();
