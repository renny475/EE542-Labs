const $ = (id) => document.getElementById(id);

const authPanel = $("auth");
const chatPanel = $("chat");
const messagesDiv = $("messages");
const welcomeCard = $("welcome");
const typingBar = $("typing");
const sendError = $("sendError");
const corpusDate = $("corpusDate");
const usernameInput = $("username");
const passwordInput = $("password");
const authMsg = $("authMsg");

const signupBtn = $("signupBtn");
const loginBtn = $("loginBtn");
const logoutBtn = $("logoutBtn");
const chatInput = $("chatInput");
const sendBtn = $("sendBtn");

const API = location.origin + "/api";
const WS_UNAUTHORIZED = 4401;
const TYPING_TIMEOUT_MS = 120000;
const RECONNECT_MS = 2000;

let token = localStorage.getItem("token") || "";
let ws;
let typingTimer;
let errorTimer;

function showAuth() {
  authPanel.classList.remove("hidden");
  chatPanel.classList.add("hidden");
}

function showChat() {
  authPanel.classList.add("hidden");
  chatPanel.classList.remove("hidden");
  chatInput.focus();
}

async function callAPI(path, method = "GET", body) {
  const headers = {"Content-Type": "application/json"};
  if (token) headers["Authorization"] = "Bearer " + token;
  const res = await fetch(API + path, {
    method, headers, body: body ? JSON.stringify(body) : undefined
  });
  if (!res.ok) {
    const err = new Error(await errorMessage(res));
    err.status = res.status;
    throw err;
  }
  return res.json();
}

// Turn any API error body into a sentence a person can act on.
// FastAPI validation errors (422) arrive as a list of {loc, msg} objects.
async function errorMessage(res) {
  let detail;
  try { detail = (await res.json()).detail; } catch (e) { detail = null; }
  if (Array.isArray(detail)) {
    return detail.map(d => {
      const field = Array.isArray(d.loc) ? d.loc[d.loc.length - 1] : "";
      return field ? `${field}: ${d.msg}` : d.msg;
    }).join("; ");
  }
  if (typeof detail === "string" && detail) return detail;
  return "HTTP " + res.status;
}

async function loadMeta() {
  try {
    const meta = await callAPI("/meta");
    corpusDate.textContent = meta.corpus_as_of || "an unknown date";
  } catch (e) {
    corpusDate.textContent = "an unknown date";
  }
}

function setTyping(on) {
  clearTimeout(typingTimer);
  typingBar.classList.toggle("hidden", !on);
  if (on) typingTimer = setTimeout(() => setTyping(false), TYPING_TIMEOUT_MS);
}

function showSendError(text) {
  clearTimeout(errorTimer);
  sendError.textContent = text;
  sendError.classList.remove("hidden");
  errorTimer = setTimeout(() => sendError.classList.add("hidden"), 6000);
}

function sourcesElement(sources) {
  const details = document.createElement("details");
  details.className = "sources";
  const summary = document.createElement("summary");
  summary.textContent = `Sources (${sources.length})`;
  details.appendChild(summary);
  const list = document.createElement("ol");
  for (const s of sources) {
    const item = document.createElement("li");
    item.appendChild(document.createTextNode(`[${s.n}] `));
    const link = document.createElement("a");
    link.href = s.url;
    link.target = "_blank";
    link.rel = "noopener";
    link.textContent = s.title;
    item.appendChild(link);
    const extra = [];
    if (s.page) extra.push(`p.${s.page}`);
    if (s.published) extra.push(s.published);
    if (extra.length) item.appendChild(document.createTextNode(" · " + extra.join(" · ")));
    if (s.excerpt) {
      const quote = document.createElement("blockquote");
      quote.textContent = s.excerpt;
      item.appendChild(quote);
    }
    list.appendChild(item);
  }
  details.appendChild(list);
  return details;
}

function addMessage(m) {
  welcomeCard.classList.add("hidden");
  const el = document.createElement("div");
  el.className = "message" + (m.is_bot ? " bot" : "");
  const meta = document.createElement("div");
  meta.className = "meta";
  meta.textContent = `${m.username || "unknown"} • ${new Date(m.created_at).toLocaleString()}`;
  const body = document.createElement("div");
  body.className = "body";
  body.textContent = m.content;
  el.appendChild(meta);
  el.appendChild(body);
  if (m.is_bot && Array.isArray(m.sources) && m.sources.length) {
    el.appendChild(sourcesElement(m.sources));
  }
  if (m.is_bot) setTyping(false);
  messagesDiv.appendChild(el);
  messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

function clearMessages() {
  for (const el of Array.from(messagesDiv.querySelectorAll(".message"))) el.remove();
  welcomeCard.classList.remove("hidden");
}

async function loadMessages() {
  const data = await callAPI("/messages");
  clearMessages();
  for (const m of data.messages) addMessage(m);
}

function connectWS() {
  // Idempotent: never hold more than one live socket per page, otherwise every
  // push is rendered once per socket.
  if (ws && (ws.readyState === WebSocket.CONNECTING || ws.readyState === WebSocket.OPEN)) return;
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${proto}://${location.host}/ws?token=${encodeURIComponent(token)}`);
  ws = socket;
  socket.onmessage = (ev) => {
    try {
      const data = JSON.parse(ev.data);
      if (data.type === "message") addMessage(data.message);
    } catch (e) {}
  };
  socket.onclose = (ev) => {
    if (ws !== socket) return;
    if (ev.code === WS_UNAUTHORIZED) {
      // The token is expired or forged: back to login, and no reconnect storm.
      ws = null;
      token = "";
      localStorage.removeItem("token");
      clearMessages();
      authMsg.textContent = "Your session has expired. Please log in again.";
      showAuth();
      return;
    }
    if (token) setTimeout(connectWS, RECONNECT_MS);
  };
}

function disconnectWS() {
  if (!ws) return;
  const socket = ws;
  ws = null;
  socket.onclose = null; // prevent the reconnect timer from firing
  socket.close();
}

async function authenticate(path) {
  authMsg.textContent = "";
  try {
    const out = await callAPI(path, "POST", {
      username: usernameInput.value.trim(),
      password: passwordInput.value
    });
    token = out.token;
    localStorage.setItem("token", token);
    passwordInput.value = "";
    await loadMessages();
    connectWS();
    showChat();
  } catch (e) {
    authMsg.textContent = e.message;
  }
}

signupBtn.onclick = () => authenticate("/signup");
loginBtn.onclick = () => authenticate("/login");
passwordInput.addEventListener("keydown", (ev) => {
  if (ev.key === "Enter") authenticate("/login");
});

logoutBtn.onclick = () => {
  token = "";
  localStorage.removeItem("token");
  disconnectWS();
  setTyping(false);
  clearMessages();
  showAuth();
};

async function sendQuestion(text) {
  text = text.trim();
  if (!text) return;
  chatInput.value = "";
  sendError.classList.add("hidden");
  setTyping(true);
  try {
    await callAPI("/messages", "POST", {content: text});
  } catch (e) {
    setTyping(false);
    chatInput.value = text;
    showSendError(e.message);
  }
}

sendBtn.onclick = () => sendQuestion(chatInput.value);
chatInput.addEventListener("keydown", (ev) => {
  if (ev.key === "Enter" && !ev.shiftKey) {
    ev.preventDefault();
    sendQuestion(chatInput.value);
  }
});
for (const button of document.querySelectorAll(".example")) {
  button.onclick = () => sendQuestion(button.textContent);
}

loadMeta();
if (token) {
  loadMessages().then(() => {
    connectWS();
    showChat();
  }).catch((e) => {
    if (e.status === 401) {
      // Only a rejected token ends the session; a server that is down keeps it.
      token = "";
      localStorage.removeItem("token");
      authMsg.textContent = "Your session has expired. Please log in again.";
    } else {
      authMsg.textContent = "Could not reach the server. Please reload to try again.";
    }
    showAuth();
  });
} else {
  showAuth();
}
