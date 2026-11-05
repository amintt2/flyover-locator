// Flyover on YouTube: listens to the video playing in the page (no download), scores it
// from a few excerpts, and can send its whole soundtrack to the Flyover server.
//
// Capture runs at 4x speed without pitch correction: the samples are then exactly the
// original sound, slowed back down by declaring a sample rate 4 times lower.
(() => {
  if (window.__flyover) return;  // injected again by the popup into an older tab
  window.__flyover = true;
  const SERVER = 'https://flyover.mciut.fr';
  const RATE = 4, OUT_SR = 8000;  // effective sample rate sent to the server
  const EXCERPTS = 4, EXCERPT_S = 30;
  const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const videoId = () => new URLSearchParams(location.search).get('v') || (location.pathname.match(/\/live\/([\w-]{11})/) || [])[1];
  const player = () => document.querySelector('#movie_player');
  const video = () => document.querySelector('#movie_player video');
  const adShowing = () => player()?.classList.contains('ad-showing');
  const sleep = ms => new Promise(r => setTimeout(r, ms));

  let panel, busy = false, cancel = false;

  const fab = document.createElement('button');
  fab.className = 'flyover-fab';
  fab.title = 'Flyover : chances de capter un avion';
  fab.textContent = '✈';
  fab.onclick = () => open();
  document.documentElement.appendChild(fab);
  const syncFab = () => { fab.style.display = videoId() ? '' : 'none'; };
  window.addEventListener('yt-navigate-finish', () => { syncFab(); if (panel && !busy) panel.remove(), panel = null; });
  syncFab();

  // the toolbar popup starts things from here
  globalThis.chrome?.runtime?.onMessage?.addListener((msg, _sender, reply) => {
    if (msg.type === 'open' || msg.type === 'full') open(msg.type);
    reply(true);
  });

  // ---------------------------------------------------------------- capture

  let graph;  // one MediaElementSource per <video>, for the element's whole life
  function tap(v) {
    if (graph?.video !== v) {
      const ctx = new AudioContext({ sampleRate: OUT_SR * RATE });
      const src = ctx.createMediaElementSource(v);
      src.connect(ctx.destination);
      graph = { video: v, ctx, src };
    }
    return graph;
  }

  // Record [t0, t1) of the video; onProgress(fraction). Returns Int16Array at OUT_SR.
  async function record(t0, t1, onProgress) {
    const v = video();
    if (!v) throw new Error('lecteur introuvable');
    const g = tap(v);
    await g.ctx.resume();
    const blocked = new Error('Chrome bloque le son tant que tu n\'as pas touché la page : clique une fois sur la vidéo, puis relance.');
    if (g.ctx.state !== 'running') throw blocked;
    const keep = { t: v.currentTime, paused: v.paused, rate: v.playbackRate, pitch: v.preservesPitch };
    const need = Math.round((t1 - t0) * OUT_SR);
    const out = new Int16Array(need);
    let n = 0, quiet = 0;
    const proc = g.ctx.createScriptProcessor(4096, 2, 1);
    const mute = g.ctx.createGain();
    mute.gain.value = 0;
    proc.onaudioprocess = e => {
      // keep only what the element really played: not ads, stalls or seeks (those read as silence)
      if (n >= need || v.paused || v.seeking || v.readyState < 3 || adShowing()) return;
      const b = e.inputBuffer, a = b.getChannelData(0), c = b.numberOfChannels > 1 ? b.getChannelData(1) : a;
      for (let i = 0; i < a.length && n < need; i++) {
        const x = (a[i] + c[i]) / 2;
        if (x === 0) quiet++;
        out[n++] = Math.max(-32768, Math.min(32767, Math.round(x * 32767)));
      }
      onProgress?.(n / need);
    };
    g.src.disconnect();
    g.src.connect(proc).connect(mute).connect(g.ctx.destination);  // silent for the viewer
    try {
      v.currentTime = t0;
      await new Promise(r => v.addEventListener('seeked', r, { once: true }));
      v.preservesPitch = false;
      v.playbackRate = RATE;
      await v.play().catch(e => { throw e.name === 'NotAllowedError' ? blocked : e; });
      while (n < need && !cancel) {
        await sleep(250);
        if (v.ended) break;
        if (v.paused && !adShowing()) await v.play().catch(() => {});
      }
    } finally {
      proc.onaudioprocess = null;
      g.src.disconnect(); proc.disconnect(); mute.disconnect();
      g.src.connect(g.ctx.destination);
      v.playbackRate = keep.rate; v.preservesPitch = keep.pitch;
      v.currentTime = keep.t;
      if (keep.paused) v.pause();
    }
    if (cancel) throw new Error('annulé');
    if (n && quiet > 0.98 * n) throw new Error('aucun son capté : le lecteur YouTube est-il en sourdine (volume à 0) ?');
    return out.subarray(0, n);
  }

  function wav(samples) {
    const b = new ArrayBuffer(44 + samples.length * 2), d = new DataView(b);
    const s = (o, t) => [...t].forEach((ch, i) => d.setUint8(o + i, ch.charCodeAt(0)));
    s(0, 'RIFF'); d.setUint32(4, 36 + samples.length * 2, true); s(8, 'WAVE'); s(12, 'fmt ');
    d.setUint32(16, 16, true); d.setUint16(20, 1, true); d.setUint16(22, 1, true);
    d.setUint32(24, OUT_SR, true); d.setUint32(28, OUT_SR * 2, true); d.setUint16(32, 2, true); d.setUint16(34, 16, true);
    s(36, 'data'); d.setUint32(40, samples.length * 2, true);
    new Int16Array(b, 44).set(samples);
    return new Blob([b], { type: 'audio/wav' });
  }

  function pageInfo() {
    const v = video(), live = !!document.querySelector('.ytp-live');
    const desc = document.querySelector('#description, ytd-watch-metadata')?.innerText || '';
    return {
      id: videoId(),
      title: document.querySelector('ytd-watch-metadata h1, h1.title')?.innerText?.trim() || document.title.replace(/ - YouTube$/, ''),
      channel: document.querySelector('ytd-watch-metadata #channel-name a, #owner #channel-name a')?.innerText?.trim(),
      duration: live ? null : Math.round(v?.duration || 0),
      live_status: live ? 'is_live' : /Streamed live|Diffusé en direct|Live diffusé/i.test(desc) ? 'was_live' : null,
      webpage_url: `https://www.youtube.com/watch?v=${videoId()}`,
    };
  }

  async function post(path, form) {
    const r = await fetch(SERVER + path, { method: 'POST', body: form });
    const d = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(d.detail || `serveur : HTTP ${r.status}`);
    return d;
  }

  // ---------------------------------------------------------------- panel

  function open(mode = 'open') {
    if (!videoId()) return;
    if (!panel) {
      panel = document.createElement('div');
      panel.className = 'flyover-panel';
      document.documentElement.appendChild(panel);
    }
    if (busy) return;
    panel.innerHTML = `<div class="fo-head"><b>Flyover</b><button class="fo-x" title="Fermer">×</button></div><div class="fo-body"></div>`;
    panel.querySelector('.fo-x').onclick = () => { cancel = busy; panel.remove(); panel = null; };
    if (mode !== 'full') return probe();
    const info = pageInfo();
    if (!info.duration) { body().innerHTML = '<div class="fo-err">Live en cours : l\'analyse complète marche sur une vidéo terminée.</div>'; return; }
    body().innerHTML = `<div class="fo-title">${esc(info.title)}</div><div class="fo-sub">${fmt(info.duration)}, écoute en ×${RATE} : environ ${fmt(info.duration / RATE)}</div>
      <button class="fo-send" disabled>Analyse complète en cours</button><div class="fo-status"></div>`;
    full(info);
  }

  const body = () => panel?.querySelector('.fo-body');
  const fmt = s => `${Math.floor(s / 3600) ? Math.floor(s / 3600) + ' h ' : ''}${Math.round(s % 3600 / 60)} min`;

  async function probe() {
    busy = true; cancel = false;
    const info = pageInfo();
    try {
      const files = [];
      if (info.duration) {
        const n = info.duration >= EXCERPTS * EXCERPT_S * 2 ? EXCERPTS : 1;
        for (let k = 0; k < n; k++) {
          const a = Math.max(0, info.duration * (k + 0.5) / n - EXCERPT_S / 2);
          const b = Math.min(info.duration, a + EXCERPT_S);
          const pcm = await record(a, b, f => {
            if (body()) body().innerHTML = `<div class="fo-wait">Écoute de l'extrait ${k + 1}/${n} (${Math.round(100 * f)} %)…<br>
              La vidéo est jouée en accéléré, sans son pour toi, puis revient où elle était.</div>`;
          });
          files.push(wav(pcm));
        }
      }
      const fd = new FormData();
      files.forEach((f, i) => fd.append('files', f, `excerpt${i}.wav`));
      fd.append('info', JSON.stringify(info));
      if (body()) body().innerHTML = '<div class="fo-wait">Mesure sur le serveur…</div>';
      show(await post('/api/probe', fd), info);
    } catch (e) {
      if (body()) body().innerHTML = `<div class="fo-err">${esc(e.message)}</div>`;
    } finally {
      busy = false;
    }
  }

  function show(d, info) {
    if (!body()) return;
    const tone = d.score >= 65 ? 'good' : d.score >= 40 ? 'mid' : 'bad';
    const ex = d.excerpts || [];
    const med = k => ex.length ? ex.map(e => e[k]).sort((a, b) => a - b)[Math.floor(ex.length / 2)] : null;
    body().innerHTML = `
      <div class="fo-title">${esc(info.title)}</div>
      <div class="fo-sub">${esc(info.channel)} · ${info.duration ? fmt(info.duration) : 'live en cours'}${info.live_status === 'was_live' ? ' · rediffusion de live' : ''}</div>
      <div class="fo-score ${tone}"><span>${d.score}</span><small>/100</small></div>
      <div class="fo-note">Score indicatif (règles tirées de quelques vidéos vérifiées), pas une probabilité.</div>
      ${ex.length ? `<div class="fo-metrics">
        <div><b>${med('floor_50_300_db')} dB</b><small>fond 50–300 Hz</small></div>
        <div><b>${Math.round(100 * med('silent_share'))} %</b><small>silence numérique</small></div>
        <div><b>${med('lines')}</b><small>sons continus</small></div>
      </div>` : ''}
      <ul class="fo-why">${(d.reasons || []).map(w => `<li class="${w.sign === '+' ? 'pos' : 'neg'}">${esc(w.text)}</li>`).join('')}</ul>
      ${info.duration ? `<button class="fo-send">Analyse complète sur le serveur</button>
      <div class="fo-note">La vidéo entière est écoutée en accéléré ×${RATE} : environ ${fmt(info.duration / RATE)}, onglet ouvert.</div>` : ''}
      <div class="fo-status"></div>`;
    const btn = body().querySelector('.fo-send');
    if (btn) btn.onclick = () => full(info);
  }

  async function full(info) {
    const st = () => panel?.querySelector('.fo-status');
    const btn = panel.querySelector('.fo-send');
    btn.disabled = true; busy = true; cancel = false;
    try {
      const pcm = await record(0, info.duration, f => { if (st()) st().textContent = `Écoute de la vidéo : ${Math.round(100 * f)} %`; });
      if (st()) st().textContent = `Envoi au serveur (${(pcm.length * 2 / 1e6).toFixed(0)} Mo)…`;
      const fd = new FormData();
      fd.append('file', wav(pcm), `${info.id}.wav`);
      fd.append('info', JSON.stringify(info));
      const sent = await post('/api/survey/file', fd);
      const t0 = Date.now() / 1000 - 5;
      while (panel) {
        const r = await fetch(SERVER + '/api/survey').then(x => x.json());
        const row = r.results.filter(x => x.id === sent.id && x.ts >= t0).pop();
        if (row) return result(row, info);
        const ahead = (r.waiting || []).filter(x => x !== sent.id).length;
        if (st()) st().textContent = r.waiting?.includes(sent.id) && ahead
          ? `Sur le serveur : en attente (${ahead} avant)…` : 'Analyse sur le serveur (environ 5 min par heure de vidéo)…';
        await sleep(5000);
      }
    } catch (e) {
      if (st()) st().innerHTML = `<div class="fo-err">${esc(e.message)}</div>`;
      btn.disabled = false;
    } finally {
      busy = false;
    }
  }

  function result(row, info) {
    const st = panel?.querySelector('.fo-status');
    if (!st) return;
    if (row.error) { st.innerHTML = `<div class="fo-err">Serveur : ${esc(row.error)}</div>`; return; }
    const fits = row.fits || [];
    const at = s => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
    st.innerHTML = `<div class="fo-done">${fits.length ? `${fits.length} passage(s) possible(s), à vérifier à l'œil :` : 'Aucun passage détecté.'}</div>` +
      fits.map(f => `<div class="fo-fit"><a href="${info.webpage_url}&t=${Math.floor(f.tc)}s">${at(f.tc)}</a>
        ~${Math.round(f.v)} m/s, ~${Math.round(f.d)} m${f.band === 'wide' ? ' (aigu)' : ''}
        <img src="${SERVER}/api/survey/img/${encodeURIComponent(row.id)}/${f.tc.toFixed(1).padStart(7, '0')}.png"></div>`).join('');
  }
})();
