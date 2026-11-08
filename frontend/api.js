// Shared frontend helpers: the API client, HTML escaping, a confirm dialog,
// and the tab registry used by the app shell (app.html).

const TOKEN_KEY = 'aftercare_token';
const USERNAME_KEY = 'aftercare_username';

export function getToken() {
  return localStorage.getItem(TOKEN_KEY) || '';
}

export function saveSession(token, username) {
  localStorage.setItem(TOKEN_KEY, token);
  localStorage.setItem(USERNAME_KEY, username);
}

export function clearSession() {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(USERNAME_KEY);
}

export function savedUsername() {
  return localStorage.getItem(USERNAME_KEY) || '';
}

function errorDetail(body, status) {
  const detail = body && body.detail;
  if (Array.isArray(detail)) return detail.map(d => d.msg || String(d)).join('; ');
  if (typeof detail === 'string' && detail) return detail;
  return 'Something went wrong (error ' + status + '). Please try again.';
}

// api(path, {method, json, form}) -> parsed JSON body.
// Sends the saved Bearer token, returns to the login page on 401 (when a token
// was sent), and throws Error(detail) with a `.status` for other failures.
export async function api(path, {method, json, form} = {}) {
  const token = getToken();
  const headers = {};
  let body;
  if (token) headers.Authorization = 'Bearer ' + token;
  if (json !== undefined) {
    headers['Content-Type'] = 'application/json';
    body = JSON.stringify(json);
  } else if (form !== undefined) {
    body = form;  // the browser sets the multipart boundary
  }
  let res;
  try {
    res = await fetch(path, {method: method || (body ? 'POST' : 'GET'), headers, body});
  } catch (_) {
    const error = new Error('Cannot reach AfterCare. Check your connection and try again.');
    error.status = 0;
    throw error;
  }
  let data = null;
  try { data = await res.json(); } catch (_) { data = null; }
  if (res.status === 401 && token) {
    clearSession();
    window.location.href = '/';
  }
  if (!res.ok) {
    const error = new Error(errorDetail(data, res.status));
    error.status = res.status;
    throw error;
  }
  return data;
}

const ESCAPES = {'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'};

// The patient's local calendar date as YYYY-MM-DD (never UTC).
export function localIsoDate(d = new Date()) {
  const pad = n => String(n).padStart(2, '0');
  return d.getFullYear() + '-' + pad(d.getMonth() + 1) + '-' + pad(d.getDate());
}

export function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, ch => ESCAPES[ch]);
}

// A styled replacement for window.confirm(). Resolves true or false.
export function confirmDialog({title, body, confirmLabel = 'Continue', cancelLabel = 'Cancel', danger = false}) {
  return new Promise(resolve => {
    const dialog = document.createElement('dialog');
    dialog.className = 'dialog';
    dialog.setAttribute('aria-labelledby', 'dialogTitle');
    dialog.innerHTML = `
      <form method="dialog" class="dialog-body">
        <h2 id="dialogTitle" class="dialog-title">${escapeHtml(title)}</h2>
        <p class="dialog-text">${escapeHtml(body)}</p>
        <div class="dialog-actions">
          <button value="cancel" class="btn btn-secondary">${escapeHtml(cancelLabel)}</button>
          <button value="ok" class="btn ${danger ? 'btn-danger' : 'btn-primary'}">${escapeHtml(confirmLabel)}</button>
        </div>
      </form>`;
    document.body.append(dialog);
    dialog.addEventListener('close', () => {
      resolve(dialog.returnValue === 'ok');
      dialog.remove();
    });
    dialog.showModal();
  });
}

// Tab registry. Each tab module (review.js, today.js, ...) calls
// registerTab(id, {show(panel, ctx)}); the shell calls show() whenever the tab
// becomes visible or the active document changes while it is visible.
// ctx = {document, username, setDocument(doc), reloadDocument(), go(tabId)}.
const tabs = new Map();

export function registerTab(id, module) {
  tabs.set(id, module);
}

export function getTab(id) {
  return tabs.get(id);
}
