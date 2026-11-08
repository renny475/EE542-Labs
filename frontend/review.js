// Instructions tab: upload, review (pending_review) and read-only (confirmed) views.
import {api, confirmDialog, escapeHtml, localIsoDate, registerTab} from '/static/api.js';

const CATEGORIES = [
  ['medication', 'Medications'],
  ['diet', 'Diet'],
  ['activity', 'Activity'],
  ['wound_or_device_care', 'Wound and device care'],
  ['monitoring', 'Monitoring'],
  ['follow_up', 'Follow-up'],
  ['warning_signs', 'Warning signs'],
  ['other', 'Other'],
];
const CATEGORY_LABEL = Object.fromEntries(CATEGORIES);
const MAX_UPLOAD_BYTES = 10 * 1024 * 1024;
const WIDE = window.matchMedia('(min-width: 720px)');

// UI state that survives re-renders of the same document.
const view = {
  docKey: null,       // document id + status; a change resets the rest
  replacing: false,   // showing the upload form while a document exists
  selectedId: null,   // item whose quote is highlighted
  busy: null,         // 'upload' | 're-extract' | 'confirm' | item id being rejected
  error: null,        // {where: 'upload' | 'actions' | item id, text}
};

const esc = escapeHtml;

function show(panel, ctx) {
  const doc = ctx.document;
  const key = doc ? doc.id + ':' + doc.status : 'none';
  if (key !== view.docKey) {
    view.docKey = key;
    view.replacing = false;
    view.selectedId = null;
    if (view.busy !== 'upload') view.error = null;
  }
  if (!doc || view.replacing || view.busy === 'upload') renderUpload(panel, ctx);
  else if (doc.status === 'pending_review') renderReview(panel, ctx, doc);
  else renderConfirmed(panel, ctx, doc);
}

registerTab('instructions', {show});

// ---------- helpers ----------

function formatDate(iso) {
  if (!iso) return '';
  const [y, m, d] = iso.split('-').map(Number);
  return new Date(y, m - 1, d).toLocaleDateString('en-US', {month: 'long', day: 'numeric', year: 'numeric'});
}

function errorHtml(where) {
  if (!view.error || view.error.where !== where) return '';
  return `<p class="form-error" role="alert">${esc(view.error.text)}</p>`;
}

function grouped(items) {
  return CATEGORIES
    .map(([key, label]) => ({key, label, items: items.filter(i => (CATEGORY_LABEL[i.category] ? i.category : 'other') === key)}))
    .filter(group => group.items.length);
}

function plural(n, word) {
  return n + ' ' + word + (n === 1 ? '' : 's');
}

const SPINNER = '<span class="spinner" aria-hidden="true"></span>';

// Run an action that returns a new document view; errors show at `where`.
async function act(panel, ctx, busy, where, request) {
  view.busy = busy;
  view.error = null;
  show(panel, ctx);
  try {
    const doc = await request();
    view.busy = null;
    ctx.setDocument(doc);
  } catch (error) {
    view.busy = null;
    view.error = {where, text: error.message};
    show(panel, ctx);
  }
}

// ---------- quote highlighting ----------
// Mirrors backend extraction.normalize(): case, quote/dash variants and
// whitespace runs are ignored, so a verified quote is always found again.

function foldChar(ch) {
  return ch.toLowerCase()
    .replace(/[\u2018\u2019\u201a\u201b\u2032`\u00b4]/g, "'")
    .replace(/[\u201c\u201d\u201e\u201f\u2033]/g, '"')
    .replace(/[\u2010\u2011\u2012\u2013\u2014\u2015\u2212]/g, '-');
}

function normalize(text) {
  return Array.from(text, foldChar).join('').replace(/\s+/g, ' ').trim();
}

// Returns [{p, start, end}] ranges (per paragraph) covering the quote, or [].
function findQuote(paragraphs, quote) {
  const needle = normalize(quote || '');
  if (!needle) return [];
  let hay = '';
  const map = [];  // map[k] = {p, i}: source of normalized char k
  let gap = null;
  paragraphs.forEach((para, p) => {
    for (let i = 0; i < para.length; i++) {
      const ch = para[i];
      if (/\s/.test(ch)) {
        if (hay) gap = gap || {p, i};
        continue;
      }
      if (gap) { hay += ' '; map.push(gap); gap = null; }
      for (const c of foldChar(ch)) { hay += c; map.push({p, i}); }
    }
    if (hay) gap = gap || {p, i: para.length};
  });
  const at = hay.indexOf(needle);
  if (at < 0) return [];
  const ranges = new Map();
  for (let k = at; k < at + needle.length; k++) {
    const {p, i} = map[k];
    if (i >= paragraphs[p].length || /\s/.test(paragraphs[p][i])) continue;
    const r = ranges.get(p) || {p, start: i, end: i + 1};
    r.start = Math.min(r.start, i);
    r.end = Math.max(r.end, i + 1);
    ranges.set(p, r);
  }
  return [...ranges.values()];
}

function paragraphHtml(text, range) {
  if (!range) return esc(text);
  return esc(text.slice(0, range.start)) +
    '<mark class="quote-mark">' + esc(text.slice(range.start, range.end)) + '</mark>' +
    esc(text.slice(range.end));
}

function renderDocBody(panel, doc) {
  const body = panel.querySelector('#docBody');
  if (!body) return null;
  const item = doc.items.find(i => i.id === view.selectedId);
  const ranges = item ? findQuote(doc.paragraphs, item.source_quote) : [];
  const byPara = new Map(ranges.map(r => [r.p, r]));
  body.innerHTML = doc.paragraphs
    .map((text, p) => `<p class="doc-para${byPara.has(p) ? ' has-mark' : ''}">${paragraphHtml(text, byPara.get(p))}</p>`)
    .join('');
  return body.querySelector('mark');
}

function applySelection(panel, doc, {scroll}) {
  panel.querySelectorAll('.item').forEach(li => {
    const selected = Number(li.dataset.id) === view.selectedId;
    li.classList.toggle('is-selected', selected);
    const button = li.querySelector('.item-main');
    if (button) button.setAttribute('aria-pressed', String(selected));
    const quote = li.querySelector('.item-quote');
    if (quote) quote.hidden = !selected;
  });
  const mark = renderDocBody(panel, doc);
  if (mark && scroll && WIDE.matches) {
    const body = panel.querySelector('#docBody');
    body.scrollTo({top: Math.max(0, mark.offsetTop - body.clientHeight / 3), behavior: 'smooth'});
  }
}

// ---------- upload ----------

function renderUpload(panel, ctx) {
  const doc = ctx.document;
  const busy = view.busy === 'upload';
  panel.innerHTML = `
    <div class="page-head">
      <h1 class="page-title">${doc ? 'Upload a new document' : 'Upload your discharge document'}</h1>
      <p class="lead">AfterCare reads the instructions your doctor wrote and lists them for you to check against your document. Nothing is used until you confirm.</p>
    </div>
    <form class="card upload-card" id="uploadForm" novalidate>
      <fieldset class="upload-fields" ${busy ? 'disabled' : ''}>
        <legend class="visually-hidden">Your discharge document</legend>
        <div class="dropzone" id="dropzone">
          <input type="file" id="fileInput" class="file-input" accept=".pdf,.txt,.md,application/pdf,text/plain,text/markdown" aria-describedby="fileHint">
          <label for="fileInput" class="dropzone-label">
            <svg class="dropzone-icon" viewBox="0 0 24 24" width="28" height="28" aria-hidden="true"><path d="M12 15V4m0 0L7.5 8.5M12 4l4.5 4.5M5 15v3a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-3" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"/></svg>
            <span class="dropzone-title">Choose a file</span>
            <span class="dropzone-hint" id="fileHint">PDF, TXT, or Markdown, up to 10 MB</span>
          </label>
          <div class="file-chosen" id="fileChosen" hidden>
            <span class="file-name" id="fileName"></span>
            <button type="button" class="btn-link" id="clearFile">Remove</button>
          </div>
        </div>
        <div class="divider" aria-hidden="true"><span>or paste text</span></div>
        <div class="field">
          <label for="pasteText">Paste the text of your document</label>
          <textarea id="pasteText" rows="7" placeholder="Paste your discharge instructions here"></textarea>
        </div>
      </fieldset>
      ${errorHtml('upload')}
      <div class="form-actions">
        <p class="busy-note" role="status" id="uploadBusy" ${busy ? '' : 'hidden'}>${SPINNER}Reading your document. This can take up to a minute.</p>
        ${doc && view.replacing ? `<button type="button" class="btn btn-secondary" id="cancelUpload" ${busy ? 'disabled' : ''}>Cancel</button>` : ''}
        <button type="submit" class="btn btn-primary" id="uploadBtn" ${busy ? 'disabled' : ''}>${busy ? 'Reading...' : 'Read my document'}</button>
      </div>
    </form>`;

  const fileInput = panel.querySelector('#fileInput');
  const dropzone = panel.querySelector('#dropzone');
  const syncFile = () => {
    const file = fileInput.files[0];
    panel.querySelector('#fileChosen').hidden = !file;
    dropzone.classList.toggle('has-file', !!file);
    panel.querySelector('#fileName').textContent = file ? file.name : '';
  };
  fileInput.addEventListener('change', syncFile);
  panel.querySelector('#clearFile').addEventListener('click', () => {
    fileInput.value = '';
    syncFile();
    fileInput.focus();
  });
  ['dragenter', 'dragover'].forEach(type => dropzone.addEventListener(type, event => {
    event.preventDefault();
    dropzone.classList.add('is-dragging');
  }));
  ['dragleave', 'drop'].forEach(type => dropzone.addEventListener(type, () => dropzone.classList.remove('is-dragging')));
  dropzone.addEventListener('drop', event => {
    event.preventDefault();
    if (event.dataTransfer.files.length && !busy) {
      fileInput.files = event.dataTransfer.files;
      syncFile();
    }
  });
  const cancel = panel.querySelector('#cancelUpload');
  if (cancel) cancel.addEventListener('click', () => {
    view.replacing = false;
    view.error = null;
    show(panel, ctx);
  });
  panel.querySelector('#uploadForm').addEventListener('submit', event => {
    event.preventDefault();
    if (view.busy) return;
    const file = fileInput.files[0];
    const text = panel.querySelector('#pasteText').value;
    const form = panel.querySelector('#uploadForm');
    const fail = message => {
      view.error = {where: 'upload', text: message};
      const old = form.querySelector(':scope > .form-error');
      if (old) old.remove();
      form.querySelector('.form-actions').insertAdjacentHTML('beforebegin', errorHtml('upload'));
    };
    if (file && text.trim()) return fail('Use either a file or pasted text, not both.');
    if (!file && !text.trim()) return fail('Choose a file or paste the text of your document.');
    if (file && file.size > MAX_UPLOAD_BYTES) return fail('Use a file smaller than 10 MB.');
    const body = new FormData();
    if (file) body.append('file', file);
    else body.append('text', text);
    upload(panel, ctx, form, body, fail);
  });
}

// Upload keeps the form in place while busy, so a failed attempt can be retried
// without choosing the file again.
async function upload(panel, ctx, form, body, fail) {
  const setBusy = busy => {
    form.querySelector('.upload-fields').disabled = busy;
    form.querySelector('#uploadBusy').hidden = !busy;
    const button = form.querySelector('#uploadBtn');
    button.disabled = busy;
    button.textContent = busy ? 'Reading...' : 'Read my document';
    const cancel = form.querySelector('#cancelUpload');
    if (cancel) cancel.disabled = busy;
  };
  view.busy = 'upload';
  view.error = null;
  const old = form.querySelector(':scope > .form-error');
  if (old) old.remove();
  setBusy(true);
  try {
    const doc = await api('/api/documents', {method: 'POST', form: body});
    view.busy = null;
    ctx.setDocument(doc);
  } catch (error) {
    view.busy = null;
    if (form.isConnected) {
      setBusy(false);
      fail(error.message);
    } else {
      view.error = {where: 'upload', text: error.message};
      show(panel, ctx);
    }
  }
}

// ---------- pending review ----------

function dateLine(doc) {
  if (doc.discharge_date) {
    return `<p class="meta-line"><span class="meta-label">Discharge date</span> ${esc(formatDate(doc.discharge_date))}, from your document</p>`;
  }
  return `<p class="meta-line"><span class="meta-label">Discharge date</span> No date found in your document. Today will be used when you confirm.</p>`;
}

function reviewItemHtml(item) {
  const rejected = item.status === 'rejected';
  const busy = view.busy === item.id;
  return `
    <li class="item${rejected ? ' is-rejected' : ''}" data-id="${item.id}">
      <button type="button" class="item-main" aria-pressed="false" aria-controls="docBody">
        <span class="item-text">${esc(item.text)}</span>
        ${rejected ? '<span class="badge">Not matching</span>' : ''}
      </button>
      <div class="item-quote" hidden>
        <span class="quote-label">From your document</span>
        <blockquote class="quote">${esc(item.source_quote)}</blockquote>
        <button type="button" class="btn-link show-in-doc">Show in document</button>
      </div>
      ${rejected
        ? '<p class="item-note">Left out and added to your care-team questions.</p>'
        : `<div class="item-actions">
             <button type="button" class="btn-link btn-link-muted reject-btn" ${view.busy ? 'disabled' : ''}>${busy ? 'Saving...' : "Doesn't match my document"}</button>
           </div>`}
      ${errorHtml(item.id)}
    </li>`;
}

function unverifiedHtml(doc) {
  if (!doc.unverified.length) return '';
  return `
    <section class="unverified" aria-labelledby="unverifiedTitle">
      <h2 class="section-title" id="unverifiedTitle">Could not be verified from your document</h2>
      <p class="section-text">These were read from your document, but their wording could not be found in it, so they are left out. Ask your care team if any of them apply to you.</p>
      <ul class="unverified-list">
        ${doc.unverified.map(u => `<li><span class="unverified-cat">${esc(CATEGORY_LABEL[u.category] || 'Other')}</span>${esc(u.text)}</li>`).join('')}
      </ul>
    </section>`;
}

function renderReview(panel, ctx, doc) {
  const active = doc.items.filter(i => i.status === 'active').length;
  const disabled = view.busy ? 'disabled' : '';
  panel.innerHTML = `
    <div class="page-head">
      <h1 class="page-title">Review your instructions</h1>
      <p class="lead">Check each instruction against your document. Select one to see the words it came from.</p>
    </div>
    <div class="meta card">
      <p class="meta-line"><span class="meta-label">Document</span> ${esc(doc.filename || 'Pasted text')}</p>
      ${dateLine(doc)}
      <p class="meta-line"><span class="meta-label">Found</span> ${plural(doc.items.length, 'instruction')}${doc.unverified.length ? `, ${doc.unverified.length} could not be verified` : ''}</p>
    </div>
    <div class="review-grid">
      <div class="review-items">
        ${doc.items.length ? '' : '<div class="card empty-inline"><p>No instructions could be verified from this document. You can read it again or upload a different one.</p></div>'}
        ${grouped(doc.items).map(group => `
          <section class="group" aria-labelledby="g-${group.key}">
            <h2 class="group-title" id="g-${group.key}">${esc(group.label)}</h2>
            <ul class="item-list">${group.items.map(reviewItemHtml).join('')}</ul>
          </section>`).join('')}
        ${unverifiedHtml(doc)}
      </div>
      <aside class="doc-pane" aria-labelledby="docTitle">
        <div class="doc-head">
          <h2 class="doc-title" id="docTitle">Your discharge document</h2>
        </div>
        <div class="doc-body" id="docBody" tabindex="0" aria-label="Document text"></div>
      </aside>
      <div class="review-actions card">
        <p class="section-text">Confirming builds your daily checklist from the ${plural(active, 'instruction')} you kept and opens Today and Ask.</p>
        ${errorHtml('actions')}
        <div class="form-actions">
          ${view.busy === 're-extract' || view.busy === 'confirm'
            ? `<p class="busy-note" role="status">${SPINNER}${view.busy === 'confirm' ? 'Building your checklist...' : 'Reading your document again. This can take up to a minute.'}</p>` : ''}
          <button type="button" class="btn btn-secondary" id="reExtractBtn" ${disabled}>Re-extract</button>
          <button type="button" class="btn btn-primary" id="confirmBtn" ${disabled || (active ? '' : 'disabled')}>Confirm instructions</button>
        </div>
        <p class="actions-foot">Wrong document? <button type="button" class="btn-link" id="replaceBtn" ${disabled}>Upload a different document</button></p>
      </div>
    </div>`;

  applySelection(panel, doc, {scroll: false});

  panel.querySelectorAll('.item').forEach(li => {
    const id = Number(li.dataset.id);
    const item = doc.items.find(i => i.id === id);
    li.querySelector('.item-main').addEventListener('click', () => {
      view.selectedId = view.selectedId === id ? null : id;
      applySelection(panel, doc, {scroll: true});
    });
    li.querySelector('.show-in-doc').addEventListener('click', () => {
      const mark = panel.querySelector('#docBody mark');
      if (mark) mark.scrollIntoView({behavior: 'smooth', block: 'center'});
    });
    const reject = li.querySelector('.reject-btn');
    if (reject) reject.addEventListener('click', async () => {
      const ok = await confirmDialog({
        title: 'Mark as not matching?',
        body: `"${item.text}" will be left out of your checklist and added to your questions for your care team. This cannot be undone.`,
        confirmLabel: 'Mark as not matching',
      });
      if (!ok) return;
      act(panel, ctx, id, id, () => api(`/api/documents/active/items/${id}/reject`, {method: 'POST'}));
    });
  });

  panel.querySelector('#replaceBtn').addEventListener('click', () => {
    view.replacing = true;
    view.error = null;
    show(panel, ctx);
  });
  panel.querySelector('#reExtractBtn').addEventListener('click', async () => {
    const ok = await confirmDialog({
      title: 'Read the document again?',
      body: 'The list will be replaced with a fresh reading of your document, including items you marked as not matching. Questions already added for your care team are kept.',
      confirmLabel: 'Re-extract',
    });
    if (!ok) return;
    view.selectedId = null;
    act(panel, ctx, 're-extract', 'actions', () => api('/api/documents/active/re-extract', {method: 'POST'}));
  });
  panel.querySelector('#confirmBtn').addEventListener('click', async () => {
    const ok = await confirmDialog({
      title: 'Confirm these instructions?',
      body: `AfterCare will build your daily checklist from the ${plural(active, 'instruction')} you kept. To change them later, upload your document again.`,
      confirmLabel: 'Confirm',
    });
    if (!ok) return;
    // Without a discharge date in the document, the checklist starts on the
    // patient's own date, not the server's UTC date.
    act(panel, ctx, 'confirm', 'actions', () =>
      api('/api/documents/active/confirm', {method: 'POST', json: {local_date: localIsoDate()}}));
  });
}

// ---------- confirmed ----------

function anchorLine(doc) {
  const date = esc(formatDate(doc.anchor_date));
  if (doc.date_source === 'document') return `Checklist dates count from ${date}, the discharge date in your document.`;
  return `Checklist dates count from ${date}, the day you confirmed, because your document has no discharge date.`;
}

function renderConfirmed(panel, ctx, doc) {
  const active = doc.items.filter(i => i.status === 'active');
  const rejected = doc.items.filter(i => i.status === 'rejected');
  panel.innerHTML = `
    <div class="confirmed">
    <div class="page-head page-head-row">
      <div>
        <h1 class="page-title">Your instructions</h1>
        <p class="lead">Confirmed from ${esc(doc.filename || 'pasted text')}. ${anchorLine(doc)}</p>
      </div>
      <a class="btn btn-primary" href="#today">Open Today</a>
    </div>
    <div class="confirmed-list">
      ${grouped(active).map(group => `
        <section class="group" aria-labelledby="g-${group.key}">
          <h2 class="group-title" id="g-${group.key}">${esc(group.label)}</h2>
          <ul class="item-list">
            ${group.items.map(item => `
              <li class="item item-static">
                <p class="item-text">${esc(item.text)}</p>
                <details class="source">
                  <summary><span class="when-closed">Show source</span><span class="when-open">Hide source</span></summary>
                  <blockquote class="quote">${esc(item.source_quote)}</blockquote>
                </details>
              </li>`).join('')}
          </ul>
        </section>`).join('')}
      ${rejected.length ? `
        <section class="group group-muted" aria-labelledby="g-left-out">
          <h2 class="group-title" id="g-left-out">Left out</h2>
          <p class="section-text">You marked these as not matching your document. They are on your care-team list.</p>
          <ul class="item-list">
            ${rejected.map(item => `<li class="item item-static is-rejected"><p class="item-text">${esc(item.text)}</p></li>`).join('')}
          </ul>
        </section>` : ''}
      <div class="card replace-card">
        <div>
          <h2 class="section-title">Have a new discharge document?</h2>
          <p class="section-text">Uploading a new one replaces your current checklist and conversation. Your care-team questions are kept.</p>
        </div>
        <button type="button" class="btn btn-secondary" id="newUploadBtn">Upload a new document</button>
      </div>
    </div>
    </div>`;

  panel.querySelector('#newUploadBtn').addEventListener('click', async () => {
    const ok = await confirmDialog({
      title: 'Upload a new document?',
      body: 'This replaces your current checklist and conversation once the new document is read. Your care-team questions are kept.',
      confirmLabel: 'Upload a new document',
    });
    if (!ok) return;
    view.replacing = true;
    view.error = null;
    show(panel, ctx);
    panel.querySelector('#fileInput').focus();
  });
}
