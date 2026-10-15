// Shared by the tagger (index.html) and the reel page (reel.html).

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
