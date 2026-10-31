const $ = id => document.getElementById(id);
let token = localStorage.getItem('token') || '';
let history = [];
let busy = false;
let requestGeneration = 0;

function signedIn() {
  $('authPanel').classList.toggle('hidden', !!token);
  $('profilePanel').classList.toggle('hidden', !token);
  $('curatorPanel').classList.toggle('hidden', !token);
  $('question').disabled = !token || busy;
  $('sendBtn').disabled = !token || busy;
  $('clearBtn').disabled = !token || busy;
  $('question').placeholder = token ? 'Ask about F-1 status, CPT, OPT or STEM OPT…' : 'Log in to ask your question…';
}

async function api(path, options = {}) {
  const res = await fetch(path, {...options, headers: {
    ...(token ? {Authorization: 'Bearer ' + token} : {}), ...options.headers
  }});
  if (!res.ok) {
    let detail;
    try { detail = (await res.json()).detail; } catch (_) { detail = 'Request failed (' + res.status + ').'; }
    if (res.status === 401) logout();
    throw new Error(Array.isArray(detail) ? detail.map(d => d.msg).join('; ') : String(detail));
  }
  return res.json();
}

async function authenticate(signup) {
  if (!$('authForm').reportValidity()) return;
  $('authMsg').textContent = '';
  $('loginBtn').disabled = $('signupBtn').disabled = true;
  try {
    const data = await api('/api/' + (signup ? 'signup' : 'login'), {method: 'POST',
      headers: {'Content-Type': 'application/json'}, body: JSON.stringify({
        username: $('username').value.trim(), password: $('password').value
      })});
    token = data.token;
    localStorage.setItem('token', token);
    $('password').value = '';
    signedIn();
    $('question').focus();
  } catch (error) { $('authMsg').textContent = error.message; }
  finally { $('loginBtn').disabled = $('signupBtn').disabled = false; }
}

function clearChat() {
  requestGeneration++;
  history = [];
  document.querySelectorAll('.message').forEach(node => node.remove());
  $('welcome').classList.remove('hidden');
  $('questionStatus').textContent = '';
  $('question').value = '';
}

function logout() {
  token = '';
  localStorage.removeItem('token');
  clearChat();
  busy = false;
  $('adminKey').value = '';
  $('degree').selectedIndex = $('stage').selectedIndex = 0;
  $('school').value = '';
  signedIn();
}

function addMessage(role, text, sources = [], needsOis = false) {
  $('welcome').classList.add('hidden');
  const article = document.createElement('article');
  article.className = 'message ' + (role === 'user' ? 'user' : 'bot');
  const label = document.createElement('div');
  label.className = 'message-label';
  label.textContent = role === 'user' ? 'YOU' : 'STUDENT GUIDE';
  const body = document.createElement('div');
  body.className = 'message-body';
  body.textContent = text;
  article.append(label, body);
  for (const source of sources) {
    const detail = document.createElement('details');
    detail.className = 'evidence';
    const summary = document.createElement('summary');
    summary.textContent = `[${source.number}] ${source.title}`;
    const link = document.createElement('a');
    link.href = source.url;
    link.target = '_blank';
    link.rel = 'noopener noreferrer';
    link.textContent = 'Open official source ↗';
    const excerpt = document.createElement('p');
    excerpt.textContent = source.content;
    const metadata = document.createElement('small');
    metadata.textContent = 'Collected ' + new Date(source.collected_at).toLocaleDateString() +
      (source.page ? ' · Page ' + source.page : '') + ' · Check the source for updates.';
    detail.append(summary, link, excerpt, metadata);
    article.append(detail);
  }
  if (needsOis) {
    const link = document.createElement('a');
    link.href = 'https://ois.usc.edu/';
    link.target = '_blank'; link.rel = 'noopener noreferrer';
    link.textContent = 'Check with USC OIS ↗';
    link.className = 'ois-link';
    article.append(link);
  }
  $('messages').append(article);
  $('messages').scrollTop = $('messages').scrollHeight;
}

async function askQuestion(event) {
  event.preventDefault();
  const question = $('question').value.trim();
  if (!question || busy || !token) return;
  const generation = requestGeneration;
  const priorHistory = history.slice(-6);
  addMessage('user', question);
  busy = true; signedIn();
  $('questionStatus').textContent = 'Looking through the collected official guidance…';
  try {
    const data = await api('/api/student/ask', {method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({question, history: priorHistory, profile: {
        degree: $('degree').value, stage: $('stage').value, school: $('school').value.trim()
      }})});
    if (generation !== requestGeneration) return;
    addMessage('assistant', data.answer, data.sources, data.needs_ois);
    history.push({role: 'user', content: question}, {role: 'assistant', content: data.answer.slice(0,8000)});
    history = history.slice(-6);
    $('question').value = '';
    $('questionStatus').textContent = '';
  } catch (error) {
    if (generation === requestGeneration) $('questionStatus').textContent = error.message;
  } finally {
    busy = false; signedIn();
    if (token) $('question').focus();
  }
}

async function loadKnowledge() {
  try {
    const data = await api('/api/student/knowledge');
    $('knowledgeStatus').textContent = data.ready ?
      `${data.documents.length} indexed sources · ${data.chunks} passages` : data.message;
    $('readyDot').classList.toggle('ready', data.ready);
    $('sourceList').replaceChildren();
    for (const doc of data.documents) {
      const item = document.createElement('li');
      const a = document.createElement('a');
      a.href = doc.url; a.target = '_blank'; a.rel = 'noopener noreferrer'; a.textContent = doc.title;
      const date = document.createElement('small');
      date.textContent = 'Collected ' + new Date(doc.collected_at).toLocaleDateString();
      item.append(a, date); $('sourceList').append(item);
    }
  } catch (_) { $('knowledgeStatus').textContent = 'Could not check the knowledge base. Is the backend running?'; }
}

$('uploadForm').onsubmit = async event => {
  event.preventDefault();
  const file = $('documentFile').files[0];
  if (!file) return;
  if (file.size > 10 * 1024 * 1024) { $('uploadStatus').textContent = 'Use a file smaller than 10 MB.'; return; }
  $('uploadBtn').disabled = true;
  $('uploadStatus').textContent = 'Reading and embedding the document…';
  const form = new FormData();
  form.set('file', file); form.set('title', $('documentTitle').value.trim());
  form.set('url', $('documentUrl').value); form.set('topic', $('documentTopic').value);
  try {
    const data = await api('/api/student/documents', {method: 'POST',
      headers: {'X-Knowledge-Admin-Key': $('adminKey').value}, body: form});
    $('uploadStatus').textContent = `Added ${data.chunks} passages.`;
    $('adminKey').value = '';
    await loadKnowledge();
  } catch (error) { $('uploadStatus').textContent = error.message; }
  finally { $('uploadBtn').disabled = false; }
};
$('authForm').onsubmit = event => { event.preventDefault(); authenticate(false); };
$('signupBtn').onclick = () => authenticate(true);
$('logoutBtn').onclick = logout;
$('clearBtn').onclick = clearChat;
$('questionForm').onsubmit = askQuestion;
$('question').onkeydown = event => {
  if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); $('questionForm').requestSubmit(); }
};
document.querySelectorAll('[data-question]').forEach(button => button.onclick = () => {
  if (!token) { $('username').focus(); $('authMsg').textContent = 'Log in or create an account to ask a question.'; return; }
  $('question').value = button.dataset.question;
  $('question').focus();
});
signedIn();
loadKnowledge();
