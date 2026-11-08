// Today tab: the daily checklist for one local date, with a date switcher.
import {api, escapeHtml, localIsoDate, registerTab} from '/static/api.js';

const esc = escapeHtml;

const SLOT_LABELS = {
  morning: 'Morning',
  midday: 'Midday',
  evening: 'Evening',
  bedtime: 'Bedtime',
  anytime: 'Any time today',
};

// UI state that survives re-renders and tab switches.
const view = {
  docId: null,
  date: null,       // 'YYYY-MM-DD' (local)
  data: null,       // last GET /api/checklist response for view.date
  loading: false,
  error: null,      // load error text
  itemError: null,  // {id, text} for a failed toggle
  open: new Set(),  // ids whose source quote is expanded
  request: 0,       // guards against out-of-order responses
  today: null,      // todayIso() when the view was last shown, to notice midnight
};

let panelRef = null;
let ctxRef = null;

// ---------- dates (all local, never UTC) ----------

const isoDate = localIsoDate;

function parseIso(iso) {
  const [y, m, d] = iso.split('-').map(Number);
  return new Date(y, m - 1, d);
}

function addDays(iso, days) {
  const d = parseIso(iso);
  d.setDate(d.getDate() + days);
  return isoDate(d);
}

const todayIso = () => isoDate(new Date());

function longDate(iso) {
  return parseIso(iso).toLocaleDateString('en-US', {weekday: 'long', month: 'long', day: 'numeric'});
}

function shortDate(iso) {
  return parseIso(iso).toLocaleDateString('en-US', {month: 'short', day: 'numeric'});
}

function daysFromToday(iso) {
  return Math.round((parseIso(iso) - parseIso(todayIso())) / 86400000);
}

// Heading and subline for a day: relative days get a word ("Today"), the
// rest get the date with the weekday and distance underneath.
function dayHeading(iso) {
  const diff = daysFromToday(iso);
  const words = {0: 'Today', [-1]: 'Yesterday', 1: 'Tomorrow'};
  if (words[diff]) return {title: words[diff], lead: longDate(iso)};
  const weekday = parseIso(iso).toLocaleDateString('en-US', {weekday: 'long'});
  const title = parseIso(iso).toLocaleDateString('en-US', {month: 'long', day: 'numeric'});
  const distance = diff > 0 ? `in ${diff} days` : `${-diff} days ago`;
  return {title, lead: `${weekday}, ${distance}`};
}

// ---------- data ----------

function show(panel, ctx) {
  panelRef = panel;
  ctxRef = ctx;
  const doc = ctx.document;
  const today = todayIso();
  if (doc.id !== view.docId) {
    view.docId = doc.id;
    view.date = today;
    view.data = null;
    view.open.clear();
  } else if (view.date === view.today && view.today !== today) {
    view.date = today;  // the day changed while the patient was away: follow "Today"
  }
  view.today = today;
  load(panel, ctx, view.date);
}

registerTab('today', {show});

// A phone tab left open overnight must not keep checking off yesterday. When the
// page comes back (or once a minute) after midnight, a view that showed "Today"
// moves to the new day; any other date stays but its heading is recomputed.
function refreshAfterMidnight() {
  const today = todayIso();
  if (view.today === null || view.today === today) return;
  if (!panelRef || panelRef.hidden || !panelRef.querySelector('.today')) return;
  const followToday = view.date === view.today;
  view.today = today;
  load(panelRef, ctxRef, followToday ? today : view.date);
}

document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'visible') refreshAfterMidnight();
});
window.addEventListener('focus', refreshAfterMidnight);
setInterval(refreshAfterMidnight, 60 * 1000);

async function load(panel, ctx, date) {
  const request = ++view.request;
  view.date = date;
  view.loading = true;
  view.error = null;
  view.itemError = null;
  if (!view.data || view.data.date !== date) view.data = null;
  render(panel, ctx);
  try {
    const data = await api('/api/checklist?date=' + encodeURIComponent(date));
    if (request !== view.request) return;
    view.data = data;
  } catch (error) {
    if (request !== view.request) return;
    view.error = error.message;
  }
  view.loading = false;
  render(panel, ctx);
}

async function toggle(panel, ctx, id, done) {
  const item = findItem(id);
  if (!item) return;
  const date = view.date;
  item.done = done;
  view.itemError = null;
  render(panel, ctx);
  try {
    await api(`/api/checklist/${id}/completion`, {method: 'PUT', json: {date, done}});
  } catch (error) {
    if (view.date !== date) return;
    const current = findItem(id);
    if (current) current.done = !done;
    view.itemError = {id, text: 'Could not save this change. ' + error.message};
    render(panel, ctx);
  }
}

function findItem(id) {
  if (!view.data) return null;
  for (const slot of view.data.slots) {
    const item = slot.items.find(i => i.id === id);
    if (item) return item;
  }
  return null;
}

// ---------- rendering ----------

const CHECK_ICON = '<svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true"><path d="M3.5 8.5l3 3 6-7" fill="none" stroke="currentColor" stroke-width="2.25" stroke-linecap="round" stroke-linejoin="round"/></svg>';
const PREV_ICON = '<svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true"><path d="M10 3.5L5.5 8l4.5 4.5" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>';
const NEXT_ICON = '<svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true"><path d="M6 3.5L10.5 8 6 12.5" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>';

function sourceHtml(id, quote) {
  if (!quote) return '';
  return `
    <details class="source" data-source="${id}" ${view.open.has(id) ? 'open' : ''}>
      <summary><span class="when-closed">Show source</span><span class="when-open">Hide source</span></summary>
      <blockquote class="quote">${esc(quote)}</blockquote>
    </details>`;
}

function checkItemHtml(item, future) {
  const inputId = 'check-' + item.id;
  const error = view.itemError && view.itemError.id === item.id
    ? `<p class="form-error check-error" role="alert">${esc(view.itemError.text)}</p>` : '';
  return `
    <li class="item check-item${item.done ? ' is-done' : ''}">
      <div class="check-row">
        <input type="checkbox" class="check-input" id="${inputId}" data-id="${item.id}" ${item.done ? 'checked' : ''} ${future ? 'disabled' : ''}>
        <label class="check-label" for="${inputId}">
          <span class="check-box" aria-hidden="true">${CHECK_ICON}</span>
          <span class="check-body">
            <span class="check-text">${esc(item.label)}</span>
            ${item.approximate ? `<span class="badge badge-around" title="Your document gives this as a number of days after discharge, so the date is approximate.">Around ${esc(shortDate(view.date))}</span>` : ''}
          </span>
        </label>
      </div>
      <div class="check-source">${sourceHtml(item.id, item.source_quote)}</div>
      ${error}
    </li>`;
}

function slotsHtml(data, future) {
  if (!data.slots.length) {
    return `<div class="card empty-inline today-empty"><p>Nothing on your checklist for ${esc(longDate(data.date))}.</p></div>`;
  }
  return data.slots.map(slot => `
    <section class="group" aria-labelledby="slot-${slot.slot}">
      <h2 class="group-title" id="slot-${slot.slot}">${esc(SLOT_LABELS[slot.slot] || slot.slot)}</h2>
      <ul class="item-list">${slot.items.map(item => checkItemHtml(item, future)).join('')}</ul>
    </section>`).join('');
}

function undatedHtml(data) {
  if (!data.undated.length) return '';
  return `
    <section class="group to-schedule" aria-labelledby="slot-undated">
      <h2 class="group-title" id="slot-undated">To schedule</h2>
      <p class="section-text">Your document does not give a date for these. Call to set them up.</p>
      <ul class="item-list">
        ${data.undated.map(item => `
          <li class="item item-static">
            <p class="item-text">${esc(item.label)}</p>
            ${sourceHtml('u' + item.id, item.source_quote)}
          </li>`).join('')}
      </ul>
    </section>`;
}

function progressHtml(data) {
  const items = data.slots.flatMap(s => s.items);
  if (!items.length) return '';
  const done = items.filter(i => i.done).length;
  const pct = Math.round((done / items.length) * 100);
  return `
    <div class="progress" role="status">
      <span class="progress-text">${done} of ${items.length} done</span>
      <span class="progress-bar" aria-hidden="true"><span style="width: ${pct}%"></span></span>
    </div>`;
}

function render(panel, ctx) {
  if (panel.hidden) return;
  const today = todayIso();
  const date = view.date;
  const anchor = ctx.document.anchor_date;
  const atStart = anchor && date <= anchor;
  const future = date > today;
  const data = view.data && view.data.date === date ? view.data : null;
  const heading = dayHeading(date);
  const focusedId = document.activeElement && panel.contains(document.activeElement) ? document.activeElement.id : null;

  let body;
  if (view.error && !data) {
    body = `<p class="form-error" role="alert">${esc(view.error)}</p>`;
  } else if (!data) {
    body = `<p class="busy-note today-loading" role="status"><span class="spinner" aria-hidden="true"></span>Loading your checklist...</p>`;
  } else {
    body = `
      ${future && data.slots.length ? '<p class="note-inline">This day has not come yet. You can check these off on the day.</p>' : ''}
      ${slotsHtml(data, future)}
      ${undatedHtml(data)}`;
  }

  panel.innerHTML = `
    <div class="today">
      <div class="page-head">
        <h1 class="page-title">${esc(heading.title)}</h1>
        <p class="lead">${esc(heading.lead)}</p>
      </div>
      <div class="day-bar">
        <nav class="day-switch" aria-label="Choose a day">
          <button type="button" class="btn btn-secondary btn-small day-step" id="prevDay" aria-label="Previous day" ${atStart ? 'disabled' : ''}>${PREV_ICON}</button>
          <button type="button" class="btn btn-secondary btn-small" id="todayBtn" ${date === today ? 'disabled' : ''}>Today</button>
          <button type="button" class="btn btn-secondary btn-small day-step" id="nextDay" aria-label="Next day">${NEXT_ICON}</button>
        </nav>
        ${data && !future ? progressHtml(data) : ''}
      </div>
      ${atStart && anchor === date ? '<p class="note-inline">Your checklist starts on this day, your discharge date.</p>' : ''}
      <div class="today-body${view.loading && data ? ' is-refreshing' : ''}">${body}</div>
    </div>`;

  panel.querySelector('#prevDay').addEventListener('click', () => load(panel, ctx, addDays(view.date, -1)));
  panel.querySelector('#nextDay').addEventListener('click', () => load(panel, ctx, addDays(view.date, 1)));
  panel.querySelector('#todayBtn').addEventListener('click', () => load(panel, ctx, todayIso()));
  panel.querySelectorAll('.check-input').forEach(input => {
    input.addEventListener('change', () => toggle(panel, ctx, Number(input.dataset.id), input.checked));
  });
  panel.querySelectorAll('details.source').forEach(details => {
    details.addEventListener('toggle', () => {
      const raw = details.dataset.source;
      const id = raw.startsWith('u') ? raw : Number(raw);
      if (details.open) view.open.add(id);
      else view.open.delete(id);
    });
  });
  if (focusedId) {
    const again = panel.querySelector('#' + CSS.escape(focusedId));
    if (again && !again.disabled) again.focus();
  }
}
