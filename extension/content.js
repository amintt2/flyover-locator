// Flyover button and panel on YouTube pages.
(() => {
  const SERVER = 'https://flyover.mciut.fr';
  const ask = (type, extra = {}) => new Promise(res => chrome.runtime.sendMessage({ type, ...extra }, res));
  const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const videoId = url => (url.match(/(?:v=|youtu\.be\/|\/live\/|\/shorts\/)([\w-]{11})/) || [])[1];
  const canonical = url => `https://www.youtube.com/watch?v=${videoId(url)}`;

  let panel, current, poll;

  const fab = document.createElement('button');
  fab.className = 'flyover-fab';
  fab.title = 'Flyover : chances de capter un avion';
  fab.textContent = '✈';
  fab.onclick = () => open(location.href);
  document.documentElement.appendChild(fab);

  function syncFab() { fab.style.display = videoId(location.href) && location.pathname !== '/' ? '' : 'none'; }
  window.addEventListener('yt-navigate-finish', syncFab);
  syncFab();

  chrome.runtime.onMessage.addListener(msg => { if (msg.type === 'open') open(msg.url); });

  function open(url) {
    if (!videoId(url)) return;
    current = canonical(url);
    clearInterval(poll);
    if (!panel) {
      panel = document.createElement('div');
      panel.className = 'flyover-panel';
      document.documentElement.appendChild(panel);
    }
    panel.innerHTML = `<div class="fo-head"><b>Flyover</b><button class="fo-x" title="Fermer">×</button></div>
      <div class="fo-body"><div class="fo-wait">Extraits de 30 s en cours de téléchargement, puis mesure sur le serveur…</div></div>`;
    panel.querySelector('.fo-x').onclick = () => { panel.remove(); panel = null; clearInterval(poll); };
    probe(current);
  }

  async function probe(url) {
    const r = await ask('probe', { url });
    if (url !== current || !panel) return;
    const body = panel.querySelector('.fo-body');
    if (!r?.ok) { body.innerHTML = `<div class="fo-err">${esc(r?.error || 'Erreur')}</div>`; return; }
    const d = r.data, info = d.info || {};
    const tone = d.score >= 65 ? 'good' : d.score >= 40 ? 'mid' : 'bad';
    const ex = d.excerpts || [];
    const med = k => ex.length ? [...ex.map(e => e[k])].sort((a, b) => a - b)[Math.floor(ex.length / 2)] : null;
    const dur = info.duration ? `${Math.floor(info.duration / 3600) ? Math.floor(info.duration / 3600) + ' h ' : ''}${Math.round(info.duration % 3600 / 60)} min` : 'live';
    body.innerHTML = `
      <div class="fo-title">${esc(info.title)}</div>
      <div class="fo-sub">${esc(info.channel)} · ${dur}${info.live_status === 'was_live' ? ' · rediffusion de live' : ''}</div>
      <div class="fo-score ${tone}"><span>${d.score}</span><small>/100</small></div>
      <div class="fo-note">Score indicatif (règles tirées de quelques vidéos vérifiées), pas une probabilité.</div>
      ${ex.length ? `<div class="fo-metrics">
        <div><b>${med('floor_50_300_db')} dB</b><small>fond 50–300 Hz</small></div>
        <div><b>${Math.round(100 * med('silent_share'))} %</b><small>silence numérique</small></div>
        <div><b>${med('lines')}</b><small>sons continus</small></div>
      </div>` : ''}
      <ul class="fo-why">${(d.reasons || []).map(w => `<li class="${w.sign === '+' ? 'pos' : 'neg'}">${esc(w.text)}</li>`).join('')}</ul>
      <button class="fo-send">Envoyer au serveur pour l'analyse complète</button>
      <div class="fo-status"></div>`;
    body.querySelector('.fo-send').onclick = () => send(url, info);
  }

  async function send(url, info) {
    const btn = panel.querySelector('.fo-send'), st = panel.querySelector('.fo-status');
    btn.disabled = true;
    const r = await ask('send', { url });
    if (!r?.ok) { st.innerHTML = `<div class="fo-err">${esc(r?.error)}</div>`; btn.disabled = false; return; }
    const job = r.data.job;
    poll = setInterval(async () => {
      if (!panel || url !== current) return clearInterval(poll);
      const j = (await ask('jobs'))?.data?.[job];
      if (!j) return;
      if (j.stage === 'error') { clearInterval(poll); st.innerHTML = `<div class="fo-err">${esc(j.error)}</div>`; btn.disabled = false; return; }
      if (j.stage !== 'sent') {
        st.textContent = { queued: 'Préparation…', metadata: 'Lecture des infos…', upload: `Envoi au serveur (${j.mb} Mo)…` }[j.stage]
          || `Téléchargement de l'audio sur ton PC : ${Math.round(j.progress || 0)} %`;
        return;
      }
      const res = (await ask('results'))?.data;
      const id = j.server_id;
      const row = res?.results?.filter(x => x.id === id && x.ts * 1000 >= j.started * 1000 - 5000).pop();
      if (!row) {
        const waiting = (res?.waiting || []).filter(x => x !== id).length;
        st.textContent = res?.waiting?.includes(id) && waiting ? `Sur le serveur : en attente (${waiting} avant)…` : 'Analyse sur le serveur (environ 5 min par heure de vidéo)…';
        return;
      }
      clearInterval(poll);
      if (row.error) { st.innerHTML = `<div class="fo-err">Serveur : ${esc(row.error)}</div>`; return; }
      const fits = row.fits || [];
      const at = s => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
      st.innerHTML = `<div class="fo-done">${fits.length ? `${fits.length} passage(s) possible(s) — à vérifier à l'œil :` : 'Aucun passage détecté.'}</div>` +
        fits.map(f => `<div class="fo-fit"><a href="${canonical(url)}&t=${Math.floor(f.tc)}s">${at(f.tc)}</a>
          ~${Math.round(f.v)} m/s, ~${Math.round(f.d)} m${f.band === 'wide' ? ' (aigu)' : ''}
          <img src="${SERVER}/api/survey/img/${encodeURIComponent(id)}/${f.tc.toFixed(1).padStart(7, '0')}.png"></div>`).join('');
    }, 3000);
  }
})();
