// Shared by every page: the tagger, the reel page and the ball labelling pages.

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);

function fmt(t) {
  const m = Math.floor(t / 60), s = t - m * 60;
  return `${m}:${s.toFixed(1).padStart(4, "0")}`;
}

// "23'" / "2H 7'" from a game's kickoff times; "" if no kickoff is set.
function matchClockFor(t, k1, k2) {
  if (k2 != null && t >= k2) return `2H ${Math.floor((t - k2) / 60) + 1}'`;
  if (k1 != null && t >= k1) return `${Math.floor((t - k1) / 60) + 1}'`;
  return k1 != null ? "pre" : "";
}

// "gu11-vs-saa-2026-10-10" -> "gu11-vs-saa"
// Who a clip is noteworthy for: "#38 #7 team", the default player highlighted.
function whoBadges(c, defaultPlayer) {
  const nums = (c.players || []).map((n) => `<span class="${n === defaultPlayer ? "her" : "player"}">#${n}</span>`);
  if (c.team) nums.push('<span class="team">team</span>');
  return nums.length ? nums.join(" ") + " " : "";
}
const clipColour = (c, defaultPlayer) =>
  c.players?.includes(defaultPlayer) ? "#ff8a3d" : c.team ? "#b48cff" : "#4ea1ff";
const shortGame = (id) => id.replace(/-\d{4}-\d{2}-\d{2}$/, "");

function flash(msg, err = false) {
  const el = $("flash");
  el.textContent = msg;
  el.className = err ? "err" : "";
  clearTimeout(flash.t);
  flash.t = setTimeout(() => (el.textContent = ""), err ? 8000 : 2500);
}

async function api(method, url, body) {
  const r = await fetch(url, {
    method,
    headers: body ? { "Content-Type": "application/json" } : {},
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!r.ok) {
    let msg = await r.text();
    try { msg = JSON.parse(msg).detail; } catch {}
    msg = typeof msg === "string" ? msg : JSON.stringify(msg);
    flash(`${method} failed: ${msg}`, true);
    throw new Error(msg);
  }
  return r.status === 204 ? null : r.json();
}
const getJson = (url) => api("GET", url);

// Location hash as an object: "#game=x&t=1.5" -> {game: "x", t: "1.5"}
const readHash = () => Object.fromEntries(new URLSearchParams(location.hash.slice(1)));
const makeHash = (obj) => "#" + new URLSearchParams(Object.entries(obj).filter(([, v]) => v != null)).toString();

// Keyboard handlers should ignore keys typed into form fields (Esc leaves the field).
function typingInField(e) {
  const tag = e.target.tagName;
  if (tag === "TEXTAREA" || tag === "INPUT" || tag === "SELECT") {
    if (e.key === "Escape") e.target.blur();
    return true;
  }
  return false;
}

// Sound starts off; the #sound button in the header and the M key toggle it.
function setupMute(video) {
  const button = $("sound");
  const show = () => {
    button.textContent = video.muted ? "🔇 sound off" : "🔊 sound on";
    button.classList.toggle("on", !video.muted);
  };
  video.muted = true;
  video.addEventListener("volumechange", show);
  button.addEventListener("click", () => { button.blur(); video.muted = !video.muted; });
  show();
  return () => { video.muted = !video.muted; };
}

// Tooltips: anything with a title gets a styled one, shown sooner than the browser's own
// (which waits about a second). Line breaks in a title (&#10; in HTML) are kept.
(() => {
  const tip = document.createElement("div");
  tip.id = "tip";
  tip.hidden = true;
  let target = null, timer = null;
  function hide() {
    clearTimeout(timer);
    tip.hidden = true;
    if (target && target.dataset.tip != null) {  // put the title back (re-renders may read it)
      target.title = target.dataset.tip;
      delete target.dataset.tip;
    }
    target = null;
  }
  function show() {
    if (!target?.isConnected) return hide();
    if (!tip.isConnected) document.body.append(tip);
    tip.textContent = target.dataset.tip;
    tip.hidden = false;
    const b = target.getBoundingClientRect(), t = tip.getBoundingClientRect();
    const left = Math.max(6, Math.min(innerWidth - t.width - 6, b.left + b.width / 2 - t.width / 2));
    const below = b.bottom + 6 + t.height < innerHeight;
    tip.style.left = `${left}px`;
    tip.style.top = `${below ? b.bottom + 6 : Math.max(6, b.top - t.height - 6)}px`;
  }
  document.addEventListener("pointerover", (e) => {
    const el = e.target.closest?.("[title], [data-tip]");
    if (el === target) return;
    hide();
    if (!el?.title) return;
    target = el;
    el.dataset.tip = el.title;
    el.title = "";  // so the browser doesn't show its own as well
    timer = setTimeout(show, 350);
  });
  document.addEventListener("pointerout", (e) => {
    if (target && !target.contains(e.relatedTarget)) hide();
  });
  document.addEventListener("pointerdown", hide, true);
  document.addEventListener("keydown", hide, true);
  addEventListener("scroll", hide, true);
})();

// The links between the pages, in a row at the top of every page (this script loads at the end of <body>).
// A page can point a link at what it's showing: navLink("/watch", makeHash({player: 38, game: id})).
const NAV = [
  ["/", "Tagger", "Tag clips in a game, label players and fix the ball while watching"],
  ["/watch", "Player view", "Watch one player's games with everything else skipped: only the stretches where they're on screen"],
  ["/roster", "Roster", "Names for numbers, each player's pictures from every game, and the unassigned ones to sort"],
  ["/reel", "Reels", "Build highlight reels from tagged clips, across games"],
  ["/label-ball", "Ball labels", "Label the ball in sampled frames: the training and benchmark data for the ball detector"],
  ["/review-balls", "Other balls", "Check the model's finds that aren't the game ball: real spare balls or junk"],
];
(function siteNav() {
  const here = location.pathname.replace(/\/$/, "") || "/";
  const nav = document.createElement("nav");
  nav.className = "site";
  nav.innerHTML = `<span class="brand">⚽ Soccer Clips</span>` + NAV.map(([href, name, tip]) =>
    `<a href="${href}" data-nav="${href}" title="${esc(tip)}"${href === here ? ' class="on" aria-current="page"' : ""}>${name}</a>`).join("");
  document.body.prepend(nav);
})();
function navLink(path, hash = "") {
  const a = document.querySelector(`nav.site a[data-nav="${path}"]`);
  if (a) a.href = path + hash;
}
