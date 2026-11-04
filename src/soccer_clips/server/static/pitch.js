// The field map (Phase 5), shared by the tagger and the Player view. A calibrated game has, per
// frame, a homography from pitch metres (origin at the centre spot, x toward the right-hand goal, y
// toward the far touchline) to frame pixels; it comes from the server in chunks like the player
// boxes. The map is a top-down canvas: the field, the part the camera shows, players and the ball.

const PITCH_CHUNK = 450;  // frames per request (15 s)
const PITCH_STRIDE = 10;  // frame, then the 3x3 row-major

function pitchFrames(gameId) {
  // per-frame homographies around the playhead; at(f) = Float32Array(9) or null (loading / not calibrated)
  const chunks = new Map();
  function rows(ci) {
    const c = chunks.get(ci);
    if (c instanceof Float32Array) return c;
    if (c === "loading" || ci < 0) return null;
    chunks.set(ci, "loading");
    fetch(`/api/games/${encodeURIComponent(gameId)}/pitch/frames?start=${ci * PITCH_CHUNK}&end=${(ci + 1) * PITCH_CHUNK}`)
      .then((r) => (r.ok ? r.arrayBuffer() : Promise.reject(r.status)))
      .then((buf) => chunks.set(ci, new Float32Array(buf)))
      .catch(() => chunks.delete(ci));
    for (const k of chunks.keys()) if (k < ci - 2 || k > ci + 3) chunks.delete(k);
    return null;
  }
  return {
    gameId,
    at(f) {
      const ci = Math.floor(f / PITCH_CHUNK), r = rows(ci);
      rows(ci + 1);  // ahead of playback
      const i = f - ci * PITCH_CHUNK;
      if (!r || (i + 1) * PITCH_STRIDE > r.length) return null;
      const G = r.subarray(i * PITCH_STRIDE + 1, (i + 1) * PITCH_STRIDE);
      return Number.isFinite(G[0]) ? G : null;
    },
  };
}

function projectH(G, x, y) {
  // a point through a row-major 3x3; null if it lands behind the view
  const w = G[6] * x + G[7] * y + G[8];
  if (!(w > 1e-9)) return null;
  return [(G[0] * x + G[1] * y + G[2]) / w, (G[3] * x + G[4] * y + G[5]) / w];
}

function inv3(m) {
  const [a, b, c, d, e, f, g, h, i] = m;
  const A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
  const det = a * A + b * B + c * C;
  return [A / det, -(b * i - c * h) / det, (b * f - c * e) / det,
          B / det, (a * i - c * g) / det, -(a * f - c * d) / det,
          C / det, -(a * h - b * g) / det, (a * e - b * d) / det];
}

function pixelToPitch(Ginv, px, py) {
  // a frame pixel on the ground, in pitch metres; null above the horizon
  return projectH(Ginv, px, py);
}

function pitchFootprint(Ginv, W, H) {
  // the ground the frame shows, as a polygon in pitch metres (cut off far away, where the view meets the horizon)
  const pts = [], n = 16, FAR = 120;
  const edge = [];
  for (let k = 0; k < n; k++) edge.push([W * k / n, H]);  // bottom, left to right... then up the right side, etc.
  for (let k = 0; k < n; k++) edge.push([W, H - H * k / n]);
  for (let k = 0; k < n; k++) edge.push([W - W * k / n, 0]);
  for (let k = 0; k < n; k++) edge.push([0, H * k / n]);
  for (const [x, y] of edge) {
    const p = pixelToPitch(Ginv, x, y);
    if (p && Math.hypot(p[0], p[1]) < FAR) pts.push(p);
  }
  return pts;
}

function insidePolygon(p, poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j];
    if ((yi > p[1]) !== (yj > p[1]) && p[0] < ((xj - xi) * (p[1] - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

function mapMemory(fadeS = 6) {
  // players who leave the camera's view stay on the map where they were last seen, fading out over fadeS
  // seconds. Only those whose last place is now out of view: a tracklet that just ended in view (the
  // tracker swapped ids) shouldn't leave a ghost next to the new one. A jump in time starts over.
  let seen = new Map(), lastF = null;
  return (f, fps, players, footprint) => {
    if (lastF == null || f < lastF || f - lastF > 2 * fps) seen = new Map();
    lastF = f;
    const out = [...players], here = new Set(players.map((p) => p.track));
    for (const p of players) if (p.track != null) seen.set(p.track, { ...p, f });
    for (const [t, p] of seen) {
      if (here.has(t)) continue;
      const age = (f - p.f) / fps;
      if (age > fadeS || (footprint.length > 2 && insidePolygon([p.x, p.y], footprint))) { seen.delete(t); continue; }
      out.push({ ...p, alpha: 0.7 * (1 - age / fadeS) });
    }
    return out;
  };
}

function segmentDistance(p, a, b) {
  const dx = b[0] - a[0], dy = b[1] - a[1], L = dx * dx + dy * dy;
  const t = L ? Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / L)) : 0;
  return Math.hypot(p[0] - a[0] - t * dx, p[1] - a[1] - t * dy);
}

function nearestTarget(drawing, p, perMetre) {
  // what a click at pitch point p means: a landmark within ~7 map pixels, else a line or the circle within
  // ~7, else a landmark within ~12; null if nothing is close. (Landmarks first only when close: a goal
  // area's side is ~20 map pixels long, and its middle has to be the side, not a corner.)
  const near = (r) => {
    let best = null, bd = r / perMetre;
    for (const [id, q] of Object.entries(drawing.landmarks)) {
      const d = Math.hypot(p[0] - q[0], p[1] - q[1]);
      if (d < bd) { bd = d; best = id; }
    }
    return best;
  };
  let best = near(7), bd = 7 / perMetre;
  if (best) return best;
  for (const [id, [a, b]] of Object.entries(drawing.segments)) {
    const d = segmentDistance(p, a, b);
    if (d < bd) { bd = d; best = id; }
  }
  const dc = Math.abs(Math.hypot(p[0], p[1]) - drawing.circle);
  if (dc < bd) best = "circle";
  return best || near(12);
}

function mapFrame(canvas, drawing) {
  // fit the field (and a margin) in the canvas, camera side at the bottom; returns the transforms
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.clientWidth, H = canvas.clientHeight;
  if (canvas.width !== Math.round(W * dpr) || canvas.height !== Math.round(H * dpr)) {
    canvas.width = Math.round(W * dpr);
    canvas.height = Math.round(H * dpr);
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const M = 5, s = Math.min(W / (drawing.length + 2 * M), H / (drawing.width + 2 * M));
  const cx = W / 2, cy = H / 2;
  return {
    ctx, W, H, s,
    to: (x, y) => [cx + x * s, cy - y * s],
    from: (X, Y) => [(X - cx) / s, (cy - Y) / s],
  };
}

const MAP_TEAM = { 0: "#ffffff", 1: "#5ab0ff", 2: "#9a9a9a" };
const MAP_GOALIE = { 1: "#7dff7a", 2: "#c792ff" };  // our goalie green, theirs violet (match.py; the tagger's GK tags too)

function drawPitchMap(canvas, o) {
  // o: {drawing, footprint (the camera's view on the pitch, pitchFootprint), players: [{x, y, team, ring, num, guessed, alpha, goalie}]
  //     (num: their number where it's known, drawn instead of a dot; guessed: by appearance, drawn fainter; goalie: 1 ours, 2 theirs),
  //     ball: {x, y, stale} or null (stale: not seen just now, where it was last), targets: bool (show landmarks), hover: target id or null, clicks: [{pitch, bad}], note: text over the map,
  //     attack: +1 / -1 the goal we attack (right / left), 0 or missing: not shown,
  //     possession: 1 ours / 2 theirs have the ball, 3 contested, 4 loose (ball_state.py), 0 or missing: not shown}
  const m = mapFrame(canvas, o.drawing), { ctx, to } = m;
  ctx.clearRect(0, 0, m.W, m.H);
  ctx.fillStyle = "#1f3b26";
  ctx.fillRect(0, 0, m.W, m.H);
  {  // what the camera shows
    const fp = o.footprint || [];
    if (fp.length > 2) {
      ctx.save();
      ctx.beginPath();
      fp.forEach((p, i) => (i ? ctx.lineTo(...to(...p)) : ctx.moveTo(...to(...p))));
      ctx.closePath();
      ctx.fillStyle = "rgba(255,255,255,0.10)";
      ctx.fill();
      ctx.strokeStyle = "rgba(255,255,255,0.35)";
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.restore();
    }
  }
  ctx.strokeStyle = "rgba(255,255,255,0.75)";
  ctx.lineWidth = 1.2;
  for (const line of o.drawing.lines) {
    ctx.beginPath();
    line.forEach((p, i) => (i ? ctx.lineTo(...to(...p)) : ctx.moveTo(...to(...p))));
    ctx.stroke();
  }
  ctx.fillStyle = "rgba(255,255,255,0.75)";
  for (const p of [...o.drawing.spots, ...o.drawing.posts]) {
    const [X, Y] = to(...p);
    ctx.beginPath(); ctx.arc(X, Y, 1.8, 0, 2 * Math.PI); ctx.fill();
  }
  if (o.targets) {  // calibrating: where you can click
    for (const [id, p] of Object.entries(o.drawing.landmarks)) {
      const [X, Y] = to(...p);
      ctx.fillStyle = id === o.hover ? "#ff3df5" : "rgba(255,61,245,0.55)";
      ctx.beginPath(); ctx.arc(X, Y, id === o.hover ? 5 : 2.6, 0, 2 * Math.PI); ctx.fill();
    }
    const seg = o.hover && (o.drawing.segments[o.hover] || (o.hover === "circle" && "circle"));
    if (seg) {
      ctx.strokeStyle = "#ff3df5";
      ctx.lineWidth = 3;
      ctx.beginPath();
      if (seg === "circle") { const [X, Y] = to(0, 0); ctx.arc(X, Y, o.drawing.circle * m.s, 0, 2 * Math.PI); }
      else { ctx.moveTo(...to(...seg[0])); ctx.lineTo(...to(...seg[1])); }
      ctx.stroke();
    }
  }
  for (const c of o.clicks || []) {  // calibration clicks: where each lands on the field
    if (!c.pitch) continue;
    const [X, Y] = to(...c.pitch);
    ctx.strokeStyle = c.bad ? "#ff4d4d" : "#4dfcff";
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(X - 3, Y - 3); ctx.lineTo(X + 3, Y + 3); ctx.moveTo(X + 3, Y - 3); ctx.lineTo(X - 3, Y + 3); ctx.stroke();
  }
  for (const p of o.players || []) {
    const [X, Y] = to(p.x, p.y);
    const a = p.alpha ?? 1;
    if (p.num != null) {  // a known player: their number instead of a dot
      ctx.save();
      ctx.font = "bold 11px system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.globalAlpha = (p.guessed ? 0.7 : 1) * a;
      ctx.lineWidth = 3;
      ctx.strokeStyle = "rgba(0,0,0,0.75)";
      ctx.strokeText(String(p.num), X, Y);
      ctx.fillStyle = p.ring ? "#ff9f40" : MAP_GOALIE[p.goalie] || MAP_TEAM[0];
      ctx.fillText(String(p.num), X, Y);
      ctx.restore();
    } else {
      const gk = MAP_GOALIE[p.goalie];
      ctx.fillStyle = gk || MAP_TEAM[p.team] || MAP_TEAM[0];
      ctx.globalAlpha = (p.team === 2 && !gk ? 0.6 : 0.95) * a;
      ctx.beginPath(); ctx.arc(X, Y, gk ? 4.2 : p.team === 2 ? 2.5 : 3.6, 0, 2 * Math.PI); ctx.fill();
    }
    ctx.globalAlpha = a;
    if (p.ring) {
      ctx.strokeStyle = "#ff9f40";
      ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(X, Y, p.num != null ? 10 : 7, 0, 2 * Math.PI); ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }
  if (o.ball) {  // on top of everyone: a yellow ball with a dark edge, hollow and fainter where it was last seen
    const [X, Y] = to(o.ball.x, o.ball.y);
    ctx.save();
    ctx.globalAlpha = o.ball.stale ? 0.55 : 1;
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = "#000";
    ctx.fillStyle = "#ffe14d";
    ctx.beginPath(); ctx.arc(X, Y, 5, 0, 2 * Math.PI);
    if (o.ball.stale) { ctx.stroke(); ctx.strokeStyle = "#ffe14d"; ctx.beginPath(); ctx.arc(X, Y, 3.5, 0, 2 * Math.PI); ctx.stroke(); }
    else { ctx.fill(); ctx.stroke(); }
    ctx.restore();
  }
  if (o.attack) {  // which way we attack, along the bottom (the camera's side)
    ctx.save();
    ctx.font = "11px system-ui, sans-serif";
    ctx.fillStyle = "rgba(255,255,255,0.8)";
    ctx.textBaseline = "bottom";
    ctx.textAlign = o.attack > 0 ? "right" : "left";
    ctx.fillText(o.attack > 0 ? "we attack →" : "← we attack", o.attack > 0 ? m.W - 6 : 6, m.H - 4);
    ctx.restore();
  }
  if (o.possession) {  // who has the ball, top left, in the Analytics colours (index.html POSS_COLOR)
    ctx.save();
    ctx.font = "11px system-ui, sans-serif";
    ctx.fillStyle = { 1: "#4fc472", 2: "#d93f3f", 3: "#9d7ae8", 4: "#aaaaaa" }[o.possession] || "#aaaaaa";
    ctx.textBaseline = "top";
    ctx.textAlign = "left";
    ctx.fillText({ 1: "we have the ball", 2: "they have the ball", 3: "contested ball", 4: "loose ball" }[o.possession] || "", 6, 4);
    ctx.restore();
  }
  if (o.note) {
    ctx.font = "12px system-ui, sans-serif";
    ctx.fillStyle = "rgba(0,0,0,0.6)";
    const w = ctx.measureText(o.note).width + 12;
    ctx.fillRect(6, 6, w, 20);
    ctx.fillStyle = "#fff";
    ctx.fillText(o.note, 12, 20);
  }
  return m;
}

const BALL_HOLD_S = 1.5;  // a ball not seen for this long drops off the map
const BALL_OFF_M = 10;  // a ball further outside the lines than this is in the air (projected onto the ground far beyond)

function pitchBall(ball, f, Ginv, frameW, frameH, fps, size) {
  // the ball on the field at frame f, {x, y, stale}, or null. ball: the /ball path ({frame (sorted), x, y}
  // in fractions of the frame). Where it isn't seen, its last position for up to BALL_HOLD_S, stale.
  const a = ball?.frame;
  if (!a?.length || !Ginv) return null;
  let lo = 0, hi = a.length - 1, i = -1;  // the last path frame at or before f
  while (lo <= hi) { const m = (lo + hi) >> 1; if (a[m] <= f) { i = m; lo = m + 1; } else hi = m - 1; }
  if (i < 0 || f - a[i] > BALL_HOLD_S * fps) return null;
  const p = pixelToPitch(Ginv, ball.x[i] * frameW, ball.y[i] * frameH);
  if (!p || Math.abs(p[0]) > size.length / 2 + BALL_OFF_M || Math.abs(p[1]) > size.width / 2 + BALL_OFF_M) return null;
  return { x: p[0], y: p[1], stale: a[i] !== f };
}

function pitchPlayers(G, frameW, frameH, rows, i0, i1, stride, opts = {}) {
  // map dots from box rows (frame, track, x1, y1, x2, y2 as fractions, team, pitch x, y, off the field).
  // opts.ring(track); opts.num(track) = [number, guessed] for our players whose number is known (else a dot).
  // Where the server has placed the game (positions.py: smoothed, the bench and parents flagged off the field)
  // its positions are used; until then the feet (bottom centre) are placed here, and people more than 3 m
  // outside the lines are left off
  const out = [];
  if (!G) return out;
  const Gi = inv3(G), L = opts.length ?? 70, Wd = opts.width ?? 46;
  for (let i = i0; i < i1; i++) {
    const o = i * stride;
    let p;
    if (stride >= 10 && Number.isFinite(rows[o + 7])) {
      if (rows[o + 9]) continue;
      p = [rows[o + 7], rows[o + 8]];
    } else {
      p = pixelToPitch(Gi, (rows[o + 2] + rows[o + 4]) / 2 * frameW, rows[o + 5] * frameH);
      if (!p || Math.abs(p[0]) > L / 2 + 3 || Math.abs(p[1]) > Wd / 2 + 3) continue;
    }
    const track = rows[o + 1];
    // numbers for our team, and for anyone labelled or guessed as one of ours whose kit didn't read as ours
    // (half hidden behind the referee); never on the opponents
    const gk = stride >= 11 ? rows[o + 10] : 0;  // a goalie (1 ours, 2 theirs) takes their team's colour
    const kit = gk ? gk - 1 : rows[o + 6], n = kit === 1 ? null : opts.num?.(track), team = n ? 0 : kit;
    out.push({ x: p[0], y: p[1], track, team, ring: opts.ring?.(track), num: n?.[0], guessed: n?.[1], goalie: gk || 0 });
  }
  return out;
}

function boxFrames(gameId) {
  // every tracked box (people/boxes: rows of frame, track, x1, y1, x2, y2, team, pitch x, y, off, goalie), fetched
  // in chunks around the playhead; at(f) = [i0, i1, rows] or null while loading
  const STRIDE = 11, chunks = new Map();
  function rows(ci) {
    const c = chunks.get(ci);
    if (c instanceof Float32Array) return c;
    if (c === "loading" || ci < 0) return null;
    chunks.set(ci, "loading");
    fetch(`/api/games/${encodeURIComponent(gameId)}/people/boxes?start=${ci * PITCH_CHUNK}&end=${(ci + 1) * PITCH_CHUNK}`)
      .then((r) => (r.ok ? r.arrayBuffer() : Promise.reject(r.status)))
      .then((buf) => chunks.set(ci, new Float32Array(buf)))
      .catch(() => chunks.delete(ci));
    for (const k of chunks.keys()) if (k < ci - 2 || k > ci + 3) chunks.delete(k);
    return null;
  }
  return {
    stride: STRIDE,
    at(f) {
      const ci = Math.floor(f / PITCH_CHUNK), r = rows(ci);
      rows(ci + 1);
      if (!r) return null;
      const n = r.length / STRIDE;
      let lo = 0, hi = n;
      while (lo < hi) { const mid = (lo + hi) >> 1; if (r[mid * STRIDE] < f) lo = mid + 1; else hi = mid; }
      let i1 = lo;
      while (i1 < n && r[i1 * STRIDE] === f) i1++;
      return [lo, i1, r];
    },
  };
}

function trackNumbers(segments, autoByPlayer) {
  // (track, frame) -> [number, guessed] or null, from labelled stretches ({track, start, end, player}) and the
  // automatic ones by appearance ({player: [{track, start, end}]}); a labelled stretch wins over a guess
  const by = new Map();
  const add = (track, start, end, player, guessed) => {
    if (!by.has(track)) by.set(track, []);
    by.get(track).push([start, end, player, guessed]);
  };
  for (const s of segments || []) add(s.track, s.start, s.end, s.player, false);
  for (const [p, list] of Object.entries(autoByPlayer || {})) for (const s of list) add(s.track, s.start, s.end, +p, true);
  return (track, f) => {
    let best = null;
    for (const [s, e, p, g] of by.get(track) || []) if (f >= s && f < e && (!best || best[1])) best = [p, g];
    return best;
  };
}
