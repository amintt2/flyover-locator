// Toolbar popup: starts the analysis on the YouTube tab (the panel itself lives in the page).
const m = document.getElementById('m');
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const isVideo = url => /^https:\/\/www\.youtube\.com\/(watch\?|live\/)/.test(url || '');

async function ready(tab) {
  // tabs opened before the extension was installed have no content script yet
  try {
    await chrome.tabs.sendMessage(tab.id, { type: 'ping' });
  } catch {
    await chrome.scripting.insertCSS({ target: { tabId: tab.id }, files: ['panel.css'] });
    await chrome.scripting.executeScript({ target: { tabId: tab.id }, files: ['content.js'] });
  }
}

async function start(tab, type) {
  try {
    await ready(tab);
    await chrome.tabs.sendMessage(tab.id, { type });
    window.close();
  } catch (e) {
    m.insertAdjacentHTML('beforeend', `<div class="err">${esc(e.message)}</div>`);
  }
}

(async () => {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!isVideo(tab?.url)) {
    m.innerHTML = `<div class="title">Ouvre une vidéo YouTube</div>
      <div class="sub">Puis clique à nouveau sur l'icône (ou sur le bouton ✈ en bas à droite de la vidéo).
      L'extension écoute la vidéo et donne ses chances de contenir un passage d'avion.</div>
      <button class="ghost" id="site">Ouvrir le site Flyover</button>`;
    document.getElementById('site').onclick = () => chrome.tabs.create({ url: 'https://flyover.mciut.fr/' });
    return;
  }
  m.innerHTML = `<div class="title">${esc(tab.title.replace(/^\(\d+\)\s*/, '').replace(/ - YouTube$/, ''))}</div>
    <div class="sub">Le score écoute 4 extraits de 30 s (environ 30 s, sans son pour toi).
    L'analyse complète écoute toute la vidéo en accéléré ×4 puis l'envoie au serveur.</div>
    <button class="primary" id="probe">Score rapide</button>
    <button class="ghost" id="full">Analyse complète</button>`;
  document.getElementById('probe').onclick = () => start(tab, 'open');
  document.getElementById('full').onclick = () => start(tab, 'full');
})();
