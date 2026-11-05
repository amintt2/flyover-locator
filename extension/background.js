// Talks to the local helper (yt-dlp on this PC) and to the Flyover server for the page.
const HELPER = 'http://127.0.0.1:8765';
const SERVER = 'https://flyover.mciut.fr';

async function call(base, path, body) {
  const r = await fetch(base + path, body ? {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  } : {});
  const d = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(d.error || d.detail || `HTTP ${r.status}`);
  return d;
}

function explain(m, type) {
  if (m === 'Failed to fetch' && type !== 'results')
    return "Assistant local injoignable : lance « python -m dopplergeo.helper » sur ton PC.";
  if (/not a bot|Sign in to confirm/i.test(m))
    return "YouTube bloque les téléchargements anonymes depuis ton adresse. Relance l'assistant avec " +
      "« --cookies-from-browser chrome » (il utilise ta session YouTube), ou réessaie plus tard.";
  return m;
}

chrome.runtime.onMessage.addListener((msg, _sender, reply) => {
  const run = {
    health: () => call(HELPER, '/health'),
    probe: () => call(HELPER, '/probe', { url: msg.url }),
    send: () => call(HELPER, '/send', { url: msg.url }),
    jobs: () => call(HELPER, '/jobs'),
    results: () => call(SERVER, '/api/survey'),
  }[msg.type];
  if (!run) return false;
  run().then(data => reply({ ok: true, data }), e => reply({ ok: false, error: explain(e.message, msg.type) }));
  return true; // async reply
});

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: 'flyover', title: 'Flyover : analyser cette vidéo', contexts: ['link'],
    targetUrlPatterns: ['https://www.youtube.com/watch*', 'https://youtu.be/*', 'https://www.youtube.com/live/*'],
  });
});

chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === 'flyover' && tab?.id) chrome.tabs.sendMessage(tab.id, { type: 'open', url: info.linkUrl });
});
