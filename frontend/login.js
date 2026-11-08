import {api, clearSession, getToken, saveSession} from '/static/api.js';

const $ = id => document.getElementById(id);
const form = $('authForm');
const message = $('authMsg');
let busy = false;

function showError(text) {
  message.textContent = text;
  message.hidden = !text;
}

function validate() {
  const username = $('username').value.trim();
  const password = $('password').value;
  if (username.length < 3) return ['username', 'Enter a username of at least 3 characters.'];
  if (password.length < 8) return ['password', 'Enter a password of at least 8 characters.'];
  return null;
}

async function authenticate(signup) {
  if (busy) return;
  const problem = validate();
  if (problem) {
    showError(problem[1]);
    $(problem[0]).focus();
    return;
  }
  showError('');
  busy = true;
  const button = signup ? $('signupBtn') : $('loginBtn');
  const label = button.textContent;
  $('loginBtn').disabled = $('signupBtn').disabled = true;
  button.textContent = signup ? 'Creating account...' : 'Logging in...';
  clearSession();
  try {
    const data = await api('/api/' + (signup ? 'signup' : 'login'), {
      method: 'POST',
      json: {username: $('username').value.trim(), password: $('password').value},
    });
    saveSession(data.token, data.username);
    window.location.href = '/app';
  } catch (error) {
    showError(error.message);
    $('password').value = '';
    $('password').focus();
  } finally {
    busy = false;
    button.textContent = label;
    $('loginBtn').disabled = $('signupBtn').disabled = false;
  }
}

form.addEventListener('submit', event => {
  event.preventDefault();
  authenticate(false);
});
$('signupBtn').addEventListener('click', () => authenticate(true));

// Already logged in with a valid token: go straight to the app. Only a rejected
// token logs out; a network error or a server error keeps the session.
if (getToken()) {
  api('/api/me')
    .then(() => { window.location.href = '/app'; })
    .catch(error => { if (error.status === 401) clearSession(); });
}
$('username').focus();
