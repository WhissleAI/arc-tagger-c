// Arc Tagger demo, Live tab (schema v2). Tab / mic -> AudioWorklet 16 kHz PCM16 -> WS -> server -> Deepgram nova-3 -> arctag C library.
// The Live tab works only when the server has DEEPGRAM_API_KEY set (hello.live.enabled); the Type tab (type.js) needs no key.
// Domain-driven: the chosen vertical (or the server's detected domain in auto) decides which roles, acts, intents, events,
// safety categories and side panels are shown. Everything rendered is a model read-out (no generated text).
// Reads update.enrich (SCHEMA.md v2) when present and falls back to the v1 keys (turns / readouts / utt_done) otherwise.
// Plain ES2019 (no optional chaining) so the V8 shim test (tests/shim_ui_next.py, node 12) runs the same file.
'use strict';
const $ = (id) => document.getElementById(id);
const BASE = location.pathname.replace(/[^/]*$/, '');                       // relative to the page: sub-paths / reverse proxies
const WS_URL = (location.protocol === 'https:' ? 'wss' : 'ws') + '://' + location.host + BASE + 'ws';
const nz = (v, d) => (v === undefined || v === null ? d : v);
function esc(s) { return String(nz(s, '')).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c])); }
function fmtT(t) { if (t == null) return ''; const m = Math.floor(t / 60), s = Math.floor(t % 60); return m + ':' + String(s).padStart(2, '0'); }
const pct = (p) => (p == null ? '' : Math.round(100 * p) + '%');
const last = (tag) => String(tag).split('.').slice(-1)[0];
function pretty(tag) {
  const t = String(tag || '').replace(/\.(recent|imminent|at)$/, '');
  const s = t.replace(/^(entity\.mention|entity\.number|safety\.[a-z]+|discourse\.intent|discourse\.act|discourse\.assistant|speaker\.role|derived\.role|content\.[a-z_]+|event\.[a-z]+|intent|discourse)\./, '').replace(/_/g, ' ');
  return s.charAt(0).toUpperCase() + s.slice(1);
}

// ------------------------------------------------------------------ verticals and profiles
// Client defaults per vertical (tag sets, panels); hello.profiles (the server's configs/*.json) adds the redaction policy and any
// extra vertical. Panels are this UI's layout per vertical.
const VERTS = [
  ['auto', 'Auto (detect domain)', 'auto'], ['contact_center', 'Contact center', 'talk'], ['legal_court', 'Legal / court', 'court'],
  ['sports_media', 'Sports / media', 'sports'], ['meetings', 'Meetings', 'meeting'], ['voice_assistant', 'Voice assistant', 'assistant'],
  ['trust_safety_gaming', 'Trust & safety / gaming', 'talk'], ['news_broadcast', 'News / broadcast', 'news'],
];
const SCEN_VERT = { contact: 'contact_center', legal: 'legal_court', sports: 'sports_media', meeting: 'meetings', assistant: 'voice_assistant',
  safety: 'trust_safety_gaming', news: 'news_broadcast' };
const LEGACY_VERT = { auto: 'auto', contact: 'contact_center', legal: 'legal_court', sports: 'sports_media', meeting: 'meetings', assistant: 'voice_assistant', safety: 'trust_safety_gaming', news: 'news_broadcast' };
const R = (ids) => ids.map((x) => ({ id: 'speaker.role.' + x, label: pretty('speaker.role.' + x) }));
const SV = (id, label, severity, kind) => ({ id, label, severity, kind: kind || 'span' });
const PII_ALL = [SV('safety.pii.card_account_number', 'Card / account', 'critical'), SV('safety.pii.gov_id', 'Gov ID', 'critical'),
  SV('safety.pii.phone_number', 'Phone', 'high'), SV('safety.pii.email', 'Email', 'high'), SV('safety.pii.street_address', 'Address', 'high'),
  SV('safety.pii.date_of_birth', 'DOB', 'high'), SV('safety.pii.private_name', 'Name', 'low')];
const PROF = [SV('safety.profanity', 'Profanity', 'medium')];
const TOXS = [SV('safety.toxicity.insult_harassment', 'Insult', 'medium'), SV('safety.toxicity.hate_identity', 'Hate', 'high'),
  SV('safety.toxicity.sexual_explicit', 'Sexual', 'high'), SV('safety.toxicity.threat_violence', 'Threat', 'critical')];
const RISK = [SV('safety.scam', 'Scam', 'high', 'utt'), SV('safety.child', 'Child safety', 'critical', 'utt'), SV('safety.crisis', 'Crisis', 'critical', 'utt'),
  SV('safety.threat.escalation', 'Escalation', 'high', 'utt')];
const PROFILES = {
  auto: { label: 'Auto', roles: [], safety: PII_ALL.concat(PROF, TOXS, RISK), events: null, phases: [], intent_set: 'general', redact: 'redact+log',
    ents: ['person', 'organization', 'place', 'team', 'legal_ref', 'work_product', 'event_name'], panels: ['domain', 'timeline', 'speakers'] },
  contact_center: { roles: [{ id: 'speaker.role.agent', label: 'Agent' }, { id: 'speaker.role.customer', label: 'Customer' }], roles_heuristic: false,
    safety: PII_ALL.concat(PROF, [SV('safety.scam', 'Scam', 'high', 'utt'), SV('safety.threat.escalation', 'Escalation', 'high', 'utt')]),
    events: [], phases: [], intent_set: 'general', redact: 'redact+log', ents: ['person', 'organization', 'place'],
    panels: ['intents', 'commitments', 'escalation', 'redlog', 'speakers'] },
  legal_court: { roles: R(['judge', 'counsel', 'witness', 'other']), safety: [SV('safety.pii.private_name', 'Name', 'low'), SV('safety.pii.street_address', 'Address', 'high'),
    SV('safety.pii.gov_id', 'Gov ID', 'critical'), SV('safety.pii.date_of_birth', 'DOB', 'high')].concat(PROF),
    events: ['event.legal.objection.recent', 'event.legal.ruling_sustained.recent', 'event.legal.ruling_overruled.recent'],
    phases: [{ group: 'content.legal_phase', label: 'Phase' }], intent_set: 'general', redact: 'off', ents: ['person', 'legal_ref', 'organization', 'place'],
    panels: ['objections', 'phase:content.legal_phase', 'speakers', 'timeline'] },
  sports_media: { roles: R(['play_by_play', 'analyst', 'host', 'reporter', 'athlete', 'coach_staff', 'game_official', 'other']), safety: PROF.concat([TOXS[0], TOXS[1]]),
    events: null, eventPrefix: 'event.sports.', phases: [{ group: 'content.game_state', label: 'Game state' }, { group: 'content.mode', label: 'Mode' }],
    intent_set: 'general', redact: 'off', ents: ['person', 'team', 'place', 'event_name'], panels: ['scores', 'highlights', 'phase:content.game_state', 'speakers'] },
  meetings: { roles: R(['host', 'participant', 'guest', 'other']), safety: [SV('safety.pii.private_name', 'Name', 'low'), SV('safety.pii.email', 'Email', 'high'),
    SV('safety.pii.phone_number', 'Phone', 'high')].concat(PROF), events: ['event.meeting.decision.recent', 'event.meeting.action_item.recent'],
    phases: [{ group: 'content.meeting_phase', label: 'Meeting phase' }], intent_set: 'general', redact: 'off', ents: ['person', 'organization', 'work_product'],
    panels: ['decisions', 'actions', 'phase:content.meeting_phase', 'speakers'] },
  voice_assistant: { roles: R(['device_user', 'other']), safety: [SV('safety.pii.phone_number', 'Phone', 'high'), SV('safety.pii.email', 'Email', 'high'),
    SV('safety.pii.street_address', 'Address', 'high'), SV('safety.pii.private_name', 'Name', 'low')], events: ['discourse.assistant.command_complete'],
    phases: [], intent_set: 'assistant', redact: 'off', ents: ['person', 'place', 'organization', 'work_product'], panels: ['commands', 'latency'] },
  trust_safety_gaming: { roles: [], safety: PROF.concat(TOXS, RISK, [SV('safety.pii.url_handle', 'Handle', 'low'), SV('safety.pii.phone_number', 'Phone', 'high')]),
    events: [], phases: [], intent_set: 'general', redact: 'redact+log', ents: ['person', 'organization'], panels: ['safetyfeed', 'redlog', 'speakers'] },
  news_broadcast: { roles: R(['host', 'co_host', 'reporter', 'guest', 'caller_audience', 'announcer_voiceover', 'other']), safety: PROF.concat([TOXS[1], TOXS[3]]),
    events: ['content.story_start'], phases: [{ group: 'content.subject', label: 'Subject' }, { group: 'content.mode', label: 'Mode' }], intent_set: 'general', redact: 'off',
    ents: ['person', 'place', 'organization', 'event_name'], panels: ['stories', 'phase:content.subject', 'speakers', 'timeline'] },
};
for (const v of VERTS) { const p = PROFILES[v[0]]; p.id = v[0]; p.label = p.label === 'Auto' ? 'Auto' : v[1]; p.engine = v[2]; }
const SEV_DEFAULT = { card_account_number: 'critical', gov_id: 'critical', phone_number: 'high', email: 'high', street_address: 'high', date_of_birth: 'high',
  private_name: 'low', url_handle: 'low', public_contact: 'low', threat_violence: 'critical', self_harm: 'critical', hate_identity: 'high', sexual_explicit: 'high' };

function mergeServerProfiles(list) {
  if (!Array.isArray(list)) return;
  for (const sp of list) {
    if (!PROFILES[sp.id]) {                        // a config without a page layout: the generic (auto) layout, its own redaction
      PROFILES[sp.id] = Object.assign({}, PROFILES.auto, { id: sp.id, label: sp.label || sp.id, engine: sp.domain || 'auto' });
      if (!VERTS.some((v) => v[0] === sp.id)) VERTS.push([sp.id, sp.label || sp.id, sp.domain || 'auto']);
    }
    const p = PROFILES[sp.id];
    if (Array.isArray(sp.roles)) p.roles = sp.roles;
    if (sp.roles_heuristic != null) p.roles_heuristic = sp.roles_heuristic;
    if (Array.isArray(sp.phases)) p.phases = sp.phases;
    if (Array.isArray(sp.events)) { p.events = sp.events.map((e) => e.id); p.eventLabels = {}; for (const e of sp.events) p.eventLabels[e.id] = e.label; }
    if (Array.isArray(sp.acts)) p.acts = sp.acts;
    if (Array.isArray(sp.intents)) p.intents = sp.intents;
    if (sp.intent_set) p.intent_set = sp.intent_set;
    if (sp.show_intent != null) p.show_intent = !!sp.show_intent;
    if (Array.isArray(sp.safety) && sp.safety.length) {
      p.safety = sp.safety.map((s) => ({ id: s.id, label: s.label || pretty(s.id), kind: s.kind || 'span', severity: s.severity || SEV_DEFAULT[last(s.id)] || sevGuess(s.id) }));
    }
    if (sp.redaction && sp.redaction.mode) p.redact = sp.redaction.mode;
    if (Array.isArray(sp.entities) && sp.entities.length) p.ents = sp.entities.map((e) => last(e.id));
    if (sp.redaction && sp.redaction.policy) p.policy = sp.redaction.policy;
    if (Array.isArray(sp.summaries)) p.summaries = sp.summaries;
    p.server = true;
  }
}
function sevGuess(id) {
  if (/^safety\.(child|crisis)/.test(id)) return 'critical';
  if (/^safety\.(scam|threat)/.test(id)) return 'high';
  if (/^safety\.profanity\.mild/.test(id)) return 'low';
  return 'medium';
}

// ------------------------------------------------------------------ state
const S = {
  ws: null, conn: 'connecting', retry: 0, retryAt: 0, lastRx: 0, status: null, schema: 1, resume: null,
  vertical: 'auto', running: false, needsAudio: true, startedAt: 0, lastUpd: 0, done: false, label: '',
  words: [], rev: {}, spans: { T0: [], T1: [], T2: [] }, uttB: new Set(), v1turns: [], events: [], alerts: [], utts: new Map(), ro: null,
  stats: {}, red: { counters: {}, log: [] }, policy: {}, asrTurns: [],
  tmap: new Map(), speakers: [], live: null, sections: [], summary: null, domain: null, enrichRev: 0,
  cc: {}, ccLead: [], banSeen: new Set(), spkIdx: {}, exp: new Set(), follow: true, json: [], jsonDirty: false, lastUpdate: null,
  redSpans: {}, lateSeen: {}, redView: 'overlay', debug: false, clips: [], mode: 'recorded', audioEl: null, tOffset: 0, nowI: -1,
  provider: 'deepgram', ctx: null, node: null, media: null, src: null, view: 'type',
  lanes: { pts: [], tags: {}, ver: 0 }, laneOn: { intent: 1, act: 1, state: 1, safety: 0, events: 1 },
};
const tierSel = () => $('tier').value;
const evTier = () => (tierSel() === 'T2' ? 'T1' : tierSel());

function msg(t, err) { const m = $('msg'); m.textContent = t || ''; m.className = 'msg' + (err ? ' err' : ''); }
function showErr(where, e) {
  const s = (where ? where + ': ' : '') + (e && e.message ? e.message : String(e));
  msg(s.slice(0, 300), true);
  try { console.error('[arc]', s, e && e.stack); } catch (x) { /* ignore */ }
}
window.addEventListener('error', (e) => showErr('page', e.error || e.message));
window.addEventListener('unhandledrejection', (e) => showErr('async', e.reason));
function safe(name, f) { try { return f(); } catch (e) { showErr(name, e); return null; } }

// active profile: the chosen vertical, or in auto the server's locked/active profile
function activeId() {
  if (S.vertical !== 'auto') return S.vertical;
  const d = S.domain;
  if (d && d.active && d.active !== 'auto' && PROFILES[d.active]) return d.active;
  return 'auto';
}
const prof = () => PROFILES[activeId()];

// ------------------------------------------------------------------ connection: reconnect, heartbeat, resume
const BACKOFF = [500, 1000, 2000, 3000];
let retryTimer = null, hbTimer = null;
function setConn(state, text) {
  S.conn = state;
  const el = $('conn'); el.dataset.s = state; el.setAttribute('data-s', state);
  $('connTxt').textContent = text || '';
}
function connect() {
  clearTimeout(retryTimer);
  if (S.ws) { try { S.ws.onclose = null; S.ws.close(); } catch (e) { /* ignore */ } S.ws = null; }
  setConn(S.retry ? 'reconnecting' : 'connecting', S.retry ? 'Reconnecting to the server (attempt ' + (S.retry + 1) + ')…' : 'Connecting to the server…');
  let ws;
  try { ws = new WebSocket(WS_URL); } catch (e) { scheduleReconnect('cannot open the websocket: ' + e.message); return; }
  ws.binaryType = 'arraybuffer';
  ws.onopen = () => { S.retry = 0; S.lastRx = Date.now(); setConn('open', ''); };
  ws.onmessage = (e) => {
    S.lastRx = Date.now();
    let m; try { m = JSON.parse(e.data); } catch (x) { showErr('bad message', x); return; }
    onMsg(m);
  };
  ws.onerror = () => { /* onclose follows */ };
  ws.onclose = () => { if (S.ws === ws) { S.ws = null; onDisconnect(); } };
  S.ws = ws;
}
function onDisconnect() {
  if (S.running && S.resume) msg('connection lost: reconnecting and resuming (the server keeps the session for ' + (S.orphanS || 45) + ' s)', true);
  else if (S.running && S.needsAudio) { stopCapture(); msg('connection lost during a live stream: capture stopped; the transcript so far is kept', true); }
  scheduleReconnect();
}
function saveToken(t) { S.resume = t || null; try { if (t) sessionStorage.setItem('arctag_resume', t); else sessionStorage.removeItem('arctag_resume'); } catch (e) { /* ignore */ } }
function scheduleReconnect(why) {
  const d = BACKOFF[Math.min(S.retry, BACKOFF.length - 1)]; S.retry++;
  S.retryAt = Date.now() + d;
  setConn(S.retry > 6 ? 'offline' : 'reconnecting', (why ? why + ' · ' : '') + (S.retry > 6 ? 'Server unreachable. ' : 'Connection lost. ') + 'Retrying in ' + Math.round(d / 1000) + ' s…');
  retryTimer = setTimeout(connect, d);
}
function send(o) { if (S.ws && S.ws.readyState === 1) { S.ws.send(JSON.stringify(o)); return true; } return false; }
function heartbeat() {
  const now = Date.now();
  if (S.ws && S.ws.readyState === 1) {
    if (now - S.lastRx > 40000) { setConn('stalled', 'No data from the server for 40 s: reconnecting…'); connect(); return; }
    if (now - S.lastRx > 15000) send({ type: 'status' });
    if (S.running && !S.needsAudio && S.lastUpd && now - S.lastUpd > 25000 && !S.done) {
      setConn('stalled', 'The replay has sent no updates for ' + Math.round((now - S.lastUpd) / 1000) + ' s. It may have stopped on the server; press Stop, then Replay.');
    } else if (S.conn === 'stalled') setConn('open', '');
  }
}
$('bRetry').onclick = () => { S.retry = 0; connect(); };

function onMsg(m) {
  logJson(m);
  const t = m.type;
  if (t === 'hello') {
    S.status = m; if (m.profiles) { mergeServerProfiles(m.profiles); fillVerticals(); }
    liveAvail(m.live, m.precisions, m.precision);
    if (S.resume && (S.running || S.reloadResume) && !S.resumeSent) { S.resumeSent = true; S.reloadResume = false; send({ type: 'resume', token: S.resume }); }
    else if (S.running && !S.needsAudio && !S.resume && !S.done && S.lastUpd) {
      setRunning(false); msg('the connection dropped during the replay; the transcript so far is kept. Press Replay to run it again.', true);
    }
    renderAll(true);
  } else if (t === 'started' || t === 'resumed') {
    S.needsAudio = !!m.needs_audio; S.schema = m.schema || S.schema; saveToken(m.resume_token || m.resume || S.resume); S.resumeSent = false; S.orphanS = m.orphan_s || S.orphanS;
    S.awaitStart = false; onStartedAudio();
    if (t === 'resumed') { if (!S.running) { S.label = S.label || 'Resumed session'; setRunning(true); } if (S.needsAudio && !S.media) msg('resumed the session, but this page has no audio capture: press Stop to finish it'); }
    if (S.status && m.step) S.status.step = m.step;
    msg((t === 'resumed' ? 'resumed · ' : '') + (S.label || 'streaming') + (S.needsAudio && m.provider ? ' via ' + m.provider : '') + ' · ' + nz(m.precision, '') + (m.config ? ' · configs/' + m.config : '') + (m.profile && PROFILES[m.profile] ? ' · ' + PROFILES[m.profile].label + ' profile' : ''));
    renderAll(true);
  } else if (t === 'update') safe('update', () => onUpdate(m));
  else if (t === 'ping' || t === 'noop') { /* heartbeat: lastRx already updated */ }
  else if (t === 'resume_failed') { S.resumeSent = false; saveToken(null); if (S.running) { stopCapture(); } msg((m.msg || 'could not resume the session') + ' · the transcript so far is kept', true); renderAll(true); }
  else if (t === 'error') {
    if (S.resumeSent && /resume/i.test(m.msg || '')) { S.resume = null; setRunning(false); msg('could not resume the stream after reconnecting; the transcript so far is kept', true); return; }
    S.awaitStart = false; S.pendingPlay = false; msg(m.msg || 'server error', true); if (m.code === 'auth' || m.code === 'limit' || S.running) stopCapture();
  } else if (t === 'status') {
    if (m.reconnecting) setConn('reconnecting', 'Deepgram connection dropped: the server is re-opening it (capture continues)…');
    else if (m.reconnected) { setConn('open', ''); msg('Deepgram reconnected · ' + nz(m.msg, '')); }
    else if (m.level === 'error') msg(m.msg, true);
    else if (m.level === 'warn') msg('warning · ' + nz(m.msg, ''));
    else if (m.closed) { msg((m.msg || 'stream closed') + ' · final results shown'); }
    else if (m.msg) msg(m.msg);
  } else if (t === 'done') {
    if (S.awaitStart) return;                       // the previous session's end, after a restart (seek)
    S.done = true; saveToken(null);
    if (m.redact && m.redact.counters) S.red.counters = m.redact.counters;
    const RS = { finished: 'finished', replay_done: 'replay finished', user_stop: 'stopped', session_cap: 'stopped: per-session limit reached', daily_cap: 'stopped: daily Deepgram cap reached',
      orphan_timeout: 'ended: the page was disconnected too long', reconnect_failed: 'ended: Deepgram could not be reconnected', tagger_failed: 'ended: tagger failed repeatedly', server_shutdown: 'ended: server restarting' };
    stopCapture(); msg((S.label || 'stream') + ' ' + (RS[m.reason] || 'finished') + ' · ' + S.words.length + ' words · ' + turnList().length + ' turns'); renderAll(true);
  } else if (t === 'stopped') { if (S.awaitStart) return; saveToken(null); stopCapture(); renderAll(true); }
  else if (t === 'export') download(m.data);
}

// ------------------------------------------------------------------ updates
function onUpdate(m) {
  S.lastUpdate = m; S.lastUpd = Date.now();
  if (m.resync) { if (!m.from) S.words = []; S.events = []; S.alerts = []; S.asrTurns = []; S.rev = {}; S.spans = { T0: [], T1: [], T2: [] }; S.v1turns = []; S.cc = {}; evCache.key = ''; tlCache.key = ''; }
  if (S.conn === 'stalled') setConn('open', '');
  const now = performance.now(), from = nz(m.from, 0), words = m.words || [];
  for (let j = 0; j < words.length; j++) {
    const i = from + j, nw = words[j], old = S.words[i];
    if (old && old.w !== nw.w && !old.r && !nw.r) S.rev[i] = { old: old.w, t: now }; else if (old && (old.r || nw.r)) delete S.rev[i];
  }
  S.words.length = Math.min(S.words.length, from);
  for (const w of words) S.words.push(w);
  if (m.n != null) S.words.length = m.n;
  const n = S.words.length;
  for (const t of ['T0', 'T1', 'T2']) {
    const x = m.tiers && m.tiers[t]; if (!x) continue;
    S.spans[t] = S.spans[t].filter((s) => s.b <= x.a0 && s.b <= n).concat(x.spans || []);
  }
  for (const b of m.utt_bounds || []) S.uttB.add(b);
  for (const b of Array.from(S.uttB)) if (b > n) S.uttB.delete(b);
  if (m.turns && m.turns.length) { const a0 = m.turns[0].a; S.v1turns = S.v1turns.filter((t) => t.a < a0).concat(m.turns); }
  if (!m.enrich && m.sections) S.sections = m.sections;
  for (const x of m.asr_turns || []) S.asrTurns.push(x);
  const wall = Date.now();
  for (const e of m.events || []) {
    S.events.push(e);
    if (e.tag === 'discourse.assistant.command_complete' && e.tier === 'T0' && !(e.i in S.cc)) S.cc[e.i] = wall;
  }
  if (S.events.length > 4000) S.events.splice(0, S.events.length - 4000);
  for (const a of m.alerts || []) { S.alerts.push(a); maybeHighBanner(a); }
  if (S.alerts.length > 2000) S.alerts.splice(0, S.alerts.length - 2000);
  if (m.readouts && Object.keys(m.readouts).length) S.ro = m.readouts;
  for (const u of m.utt_done || []) {
    S.utts.set(u.w1, Object.assign({}, u, { wall })); S.uttB.add(u.w1);
    const cc = Object.keys(S.cc).map(Number).filter((i) => i >= u.w0 && i < u.w1);
    if (cc.length) { const lead = wall - S.cc[Math.min.apply(null, cc)]; if (lead >= 0) S.ccLead.push(lead); }
  }
  S.stats = m.stats || S.stats; S.stats.server_ms = m.server_ms;
  if (m.redact) {
    S.red.counters = m.redact.counters || S.red.counters; if (m.redact.log) S.red.log = m.redact.log;
    if (Array.isArray(m.redact.spans)) { S.redSpans = {}; for (const x of m.redact.spans) S.redSpans[x.id] = x; }
    if (m.redact.display) S.redDisplay = m.redact.display;
  }
  if (BM.node) safe('bleep', () => bmWords(m));
  if (m.enrich) onEnrich(m.enrich);
  if (m.lanes) lanesApply(m.lanes);
  else if (m.resync) { S.lanes.pts = []; S.lanes.ver++; }
  schedule();
}
function onEnrich(e) {
  S.schema = 2; S.enrichRev = nz(e.rev, S.enrichRev);
  if (e.domain) S.domain = e.domain;
  if (e.turns) {
    for (const t of e.turns.upsert || []) {
      if (!t.intents) t.intents = t.intent ? [t.intent] : [];
      if (t.role && (t.role.top == null || t.role.below_min)) { const d0 = (t.role.dist || [])[0]; t.role = Object.assign({}, t.role, { top: d0 ? d0.tag : null, label: d0 ? d0.label : 'role ?', p: d0 ? d0.p : 0, on: false }); }
      S.tmap.set(t.seq, t);
    }
    if (e.turns.n != null) for (const k of Array.from(S.tmap.keys())) if (k >= e.turns.n) S.tmap.delete(k);
  }
  if (e.speakers) S.speakers = e.speakers;
  if (e.live) S.live = e.live;
  if (e.sections) S.sections = e.sections;
  if (e.state) S.state = e.state;
  if (e.summary) S.summary = e.summary;
}

let pending = false, lastSlow = 0, slowTimer = null;
function schedule() { if (!pending) { pending = true; requestAnimationFrame(() => { pending = false; renderAll(); }); } }
function renderAll(force) {
  safe('transcript', renderTranscript); safe('safety strip', renderSafety); safe('status', renderStatus); safe('domain', renderDetected);
  const t = performance.now();
  if (force || t - lastSlow > 300) { lastSlow = t; safe('panels', renderPanels); safe('developer', renderDev); }
  else if (!slowTimer) slowTimer = setTimeout(() => { slowTimer = null; renderAll(true); }, 320);
}

// ------------------------------------------------------------------ v1 -> v2 turn conversion
function sortedB() { return Array.from(S.uttB).filter((b) => b > 0 && b <= S.words.length).sort((a, b) => a - b); }
function labelOf(tag) {
  const p = prof();
  for (const L0 of [p.roles, p.acts, p.intents]) if (L0) for (const x of L0) if (x.id === tag) return x.label;
  if (p.eventLabels && p.eventLabels[tag]) return p.eventLabels[tag];
  return pretty(tag);
}
const ASSIST_INTENT = /^intent\.(alarm|audio|calendar|cooking|datetime|email|general_(joke|quirky)|iot|lists|music|news_query|play|qa|recommendation|social|takeaway|transport|weather)/;
function intentAllowed(tag) {
  const p = prof();
  if (p.intents && p.intents.length) return p.intents.some((x) => x.id === tag);
  return p.intent_set === 'assistant' ? ASSIST_INTENT.test(tag) : !ASSIST_INTENT.test(tag);
}
function roleAllowed(tag) { const p = prof(); return !p.roles || !p.roles.length || p.roles.some((x) => x.id === tag); }
function aggReadouts(list) {             // word-weighted mean of the utterance read-outs inside a turn (v1 only)
  const acc = {};
  let W = 0;
  for (const u of list) {
    const w = Math.max(1, u.w1 - u.w0); W += w;
    for (const key of ['intent', 'act', 'role']) for (const r of (u.readouts && u.readouts[key]) || []) {
      const k = key + '|' + r.tag, a = acc[k] || (acc[k] = { key, tag: r.tag, s: 0, thr: r.thr }); a.s += w * r.p;
    }
  }
  const out = { intent: [], act: [], role: [] };
  for (const a of Object.values(acc)) { const p = Math.round(1000 * a.s / W) / 1000; out[a.key].push({ tag: a.tag, p, thr: a.thr, on: p >= a.thr }); }
  for (const k in out) out[k].sort((x, y) => y.p - x.p);
  return out;
}
function v1Turns() {
  const B = sortedB(), utts = Array.from(S.utts.values()).sort((a, b) => a.w0 - b.w0), n = S.words.length;
  const base = S.v1turns.length ? S.v1turns : (n ? [{ a: 0, b: n, spk: null, src: 'start' }] : []);
  const out = [];
  for (let k = 0; k < base.length; k++) {
    const t = base[k], a = t.a, b = k + 1 < base.length ? base[k + 1].a : n, isLast = k === base.length - 1;
    const inside = utts.filter((u) => u.w0 >= a && u.w0 < b);
    let ro = inside.length ? aggReadouts(inside) : null, interim = false;
    if (isLast && S.ro && !S.ro.final && (!inside.length || inside[inside.length - 1].w1 < n)) { ro = { intent: S.ro.intent || [], act: S.ro.act || [], role: S.ro.role || [] }; interim = true; }
    const acts = ro ? ro.act.filter((r) => /^discourse\.act\./.test(r.tag) && !/ritual_procedural$/.test(r.tag)) : [];
    const dd = ro ? ro.act.find((r) => r.tag === 'discourse.assistant.device_directed') : null;
    let role = null;
    if (t.role) {
      const dist = [{ tag: t.role.tag, p: t.role.p }].concat(t.role.alt || []).filter((r) => roleAllowed(r.tag)).map((r) => ({ tag: r.tag, label: labelOf(r.tag), p: r.p }));
      const top = dist[0] || { tag: t.role.tag, p: t.role.p };
      role = { top: top.tag, label: labelOf(top.tag), p: top.p, thr: t.role.thr, on: top.tag === t.role.tag ? t.role.on : false, dist, raw: true };
    }
    const allWordsFinal = S.words.slice(a, b).every((w) => w && w.f);
    out.push({ id: 'v' + a, seq: k, a, b, open: isLast && !S.done, final: !isLast && allWordsFinal, src: srcV1(t.src), spk: t.spk, readout: interim ? 'interim' : 'turn_end',
      role, act: acts[0] ? Object.assign({ top: acts[0].tag, label: labelOf(acts[0].tag) }, acts[0]) : null,
      intents: ro ? ro.intent.filter((r) => intentAllowed(r.tag)).slice(0, 3).map((r) => Object.assign({ label: labelOf(r.tag) }, r)) : [],
      device_directed: dd ? { p: dd.p, thr: dd.thr, on: dd.on } : null,
      yield: t.yield ? { on: true } : null, interrupt: { on: !!t.interrupt }, /* turn_yield: no badge unless the server sends one */ arc_change: t.arc_change, events: null, v1: true });
  }
  void B;
  return out;
}
function srcV1(s) { return { 'Flux EndOfTurn': 'flux_eot', diarization: 'asr_speaker', 'ASR utterance end': 'asr_utt_end', 'arc speaker.change': 'arc_speaker_change' }[s] || s || 'start'; }
let tlCache = { key: '', list: [] };
function turnList() {
  const key = S.schema + ':' + S.enrichRev + ':' + S.words.length + ':' + S.v1turns.length + ':' + S.utts.size + ':' + activeId() + ':' + (S.ro ? S.ro.j : '') + ':' + S.done;
  if (tlCache.key === key) return tlCache.list;
  let list;
  if (S.tmap.size) list = Array.from(S.tmap.values()).sort((x, y) => x.seq - y.seq);
  else list = v1Turns();
  tlCache = { key, list };
  return list;
}
function turnAt(i) { let r = null; for (const t of turnList()) { if (t.a <= i) r = t; else break; } return r; }
function spkKey(spk) { return spk == null ? null : String(spk); }
function spkName(spk) {
  if (spk == null || spk === '') return 'Speaker';
  if (typeof spk === 'number') return 'Speaker ' + (spk + 1);
  const m = /^dg(\d+)$/.exec(spk); return m ? 'Speaker ' + (+m[1] + 1) : String(spk);
}
const SPK_COL = ['var(--s1)', 'var(--s2)', 'var(--s3)', 'var(--s4)', 'var(--s5)', 'var(--s6)'];
function spkColor(spk) {
  const k = spkKey(spk); if (k == null) return 'var(--faint)';
  if (!(k in S.spkIdx)) S.spkIdx[k] = Object.keys(S.spkIdx).length;
  return SPK_COL[S.spkIdx[k] % SPK_COL.length];
}
function roleOfSpk(spk) {
  const k = spkKey(spk); if (k == null) return null;
  for (const s of S.speakers || []) if (spkKey(s.spk) === k) return s.role;
  return null;
}
function owner(i) { const t = turnAt(i); if (!t) return ''; const r = t.role && t.role.on ? ' · ' + (t.role.label || labelOf(t.role.top)) : ''; return spkName(t.spk) + r; }
function text(a, b) { return S.words.slice(a, b).map((w) => (w ? w.w : '')).filter(Boolean).join(' '); }

// ------------------------------------------------------------------ events (per profile)
function eventAllowed(tag) {
  const p = prof();
  if (/^(speaker\.change|discourse\.turn_yield|discourse\.interruption)$/.test(tag)) return false;
  if (p.events === null || p.events === undefined) return p.eventPrefix ? tag.indexOf(p.eventPrefix) === 0 : /^(event\.|content\.story_start|discourse\.assistant\.command_complete)/.test(tag);
  return p.events.indexOf(tag) >= 0 || (!!p.eventPrefix && tag.indexOf(p.eventPrefix) === 0);
}
function cluster(list) {
  const by = {}, out = [];
  for (const e of list) (by[e.tag] || (by[e.tag] = [])).push(e);
  for (const tag of Object.keys(by)) {
    const L0 = by[tag].sort((a, b) => a.i - b.i), gap = /\.(recent|imminent)$/.test(tag) ? 12 : 3;
    let best = null, lastI = -1e9;
    for (const e of L0) { if (best && e.i - lastI > gap) { out.push(best); best = null; } if (!best || e.p > best.p) best = e; lastI = e.i; }
    if (best) out.push(best);
  }
  return out.sort((a, b) => a.i - b.i);
}
let evCache = { key: '', list: [] };
function profileEvents() {         // on-threshold events of the active profile, deduplicated, in word order
  const key = evTier() + ':' + S.events.length + ':' + S.words.length + ':' + activeId() + ':' + S.enrichRev;
  if (evCache.key === key) return evCache.list;
  const t = evTier(), n = S.words.length;
  let list;
  if (S.tmap.size) {                 // v2: the server's per-turn events (peak-picked, profile display floor applied)
    list = [];
    for (const tn of S.tmap.values()) for (const e of tn.events || []) if (eventAllowed(e.tag) && e.i < n) list.push(e);
    list.sort((x, y) => x.i - y.i);
  } else list = cluster(S.events.filter((e) => e.tier === t && e.i < n && eventAllowed(e.tag)));
  evCache = { key, list };
  return list;
}
function eventsOfTurn(t) {
  if (t.events) return t.events.filter((e) => eventAllowed(e.tag));
  return profileEvents().filter((e) => e.i >= t.a && e.i < t.b);
}
function evClass(tag) {
  if (tag.indexOf('event.sports.') === 0) return 'sports';
  if (tag.indexOf('event.legal.') === 0) return 'legal';
  if (tag.indexOf('event.meeting.') === 0) return 'meeting';
  if (tag.indexOf('discourse.assistant') === 0) return 'assist';
  if (tag.indexOf('safety.') === 0) return 'safety';
  return '';
}
function evName(e) { return e.label || labelOf(e.tag) + (/\.imminent$/.test(e.tag) ? ' (imminent)' : ''); }

// ------------------------------------------------------------------ transcript (turn cards, keyed + diffed in place)
// Information hierarchy: per turn only speaker (+ role when confident), one act chip, at most one intent chip (assistant / contact
// center). Session-state changes (phase / mode / subject / game state) are thin dividers between turns, after hysteresis.
// Domain events are icons in the left margin (and cards in the side timeline). Everything else is behind "show all tags (debug)".
const cards = new Map();           // turn id -> { el, sig }
const INTENT_VERTS = { voice_assistant: 1, contact_center: 1 };
function covering(t, a, b) {
  const per = {};
  const p = prof(), ents = p.ents || [];
  for (const s of S.spans[t]) {
    if (s.b <= a || s.a >= b) continue;
    if (s.fam === 'entity' && ents.indexOf(last(s.tag)) < 0) continue;
    if ((s.fam === 'pii' || s.fam === 'profanity') && !safetyCat(s.tag)) continue;
    for (let i = Math.max(a, s.a); i < Math.min(b, s.b); i++) (per[i] || (per[i] = [])).push(s);
  }
  return per;
}
function roleBadge(t) {
  const r = t.role, fd = t.readout === 'interim' ? ' fd' : '';
  if (!r || (!r.on && !S.debug)) return '';
  const lab = r.label || labelOf(r.top), heur = r.heuristic || prof().roles_heuristic ? ' heur' : '';
  const dist = (r.dist || []).slice(0, 8);
  const bars = dist.map((d) => '<div class="bar' + (d.tag === r.top && r.on ? ' on' : '') + '"><span class="lb">' + esc(d.label || labelOf(d.tag)) + '</span><span class="tr"><span class="fl" style="width:' +
    (100 * d.p).toFixed(1) + '%"></span></span><span class="v">' + d.p.toFixed(2) + '</span></div>').join('');
  const note = r.raw ? 'raw p, model top-3 within this domain' : 'over this domain’s roles' + (r.mass != null ? ' · mass ' + r.mass.toFixed(2) : '');
  return '<span class="badge role' + (r.on ? '' : ' off') + heur + fd + '" data-role="1" title="click to pin the distribution">' + esc(lab) + (r.on ? '' : '?') +
    (S.debug ? ' <span class="pv">' + pct(r.p) + '</span>' : '') + '<span class="pop"><h4>Speaker role · ' + pct(r.p) + '</h4>' + (bars || '<div class="empty">no distribution</div>') +
    '<div class="note">' + esc(note) + (r.thr != null ? ' · thr ' + r.thr : '') + (heur ? ' · derived (heuristic)' : '') + '</div></span></span>';
}
function intentsDebug(t) {
  const L0 = (t.intents || []).slice(0, 3);
  if (!L0.length) return '';
  return '<div class="ints' + (t.readout === 'interim' ? ' fd' : '') + '">' + L0.map((r) => '<span class="ib' + (r.on ? ' on' : '') + '" title="' + esc(r.tag) + ' p=' + r.p + (r.thr != null ? ' (thr ' + r.thr + ')' : '') + '">' +
    esc(r.label || labelOf(r.tag)) + '<span class="tr"><span class="fl" style="width:' + (100 * Math.min(1, r.p)).toFixed(0) + '%"></span></span><span class="v">' + r.p.toFixed(2) + '</span></span>').join('') + '</div>';
}
const SRC_LAB = { start: 'start', flux_eot: 'Flux EoT', asr_speaker: 'diarized', asr_utt_end: 'ASR endpoint', arc_speaker_change: 'Arc change' };
const SRC_LONG = { start: 'first turn', flux_eot: 'opened by a Deepgram Flux EndOfTurn', asr_speaker: 'opened by a diarized speaker-id change', asr_utt_end: 'opened by an ASR utterance end', arc_speaker_change: 'opened by an Arc speaker.change event (no diarization)' };
const EV_ICON = { sports: '●', legal: '⚖', meeting: '✓', assist: '▸', safety: '!', '': '◆' };

// session-state dividers: per group keep a change only when the new state lasts >= MIN_DWELL words (the newest change once it has
// lasted that long, or the stream ended) and differs from the state shown before it. Server sections are already thresholded.
const MIN_DWELL = { 'content.subject': 40, 'content.mode': 25 };
let divCache = { key: '', list: [] };
function stateChanges() {
  const n = S.words.length, key = S.sections.length + ':' + (S.sections.length ? S.sections[S.sections.length - 1].i : '') + ':' + Math.floor(n / 5) + ':' + S.done + ':' + activeId();
  if (divCache.key === key) return divCache.list;
  const groups = allowedSectionGroups(), by = {};
  for (const s of S.sections) if (groups.indexOf(s.group) >= 0 || (S.schema === 2 && !groups.length)) (by[s.group] || (by[s.group] = [])).push(s);
  const out = [];
  for (const g of Object.keys(by)) {
    const L0 = by[g].slice().sort((a, b) => a.i - b.i), dwell = MIN_DWELL[g] || 15;
    let cur = null;
    for (let k = 0; k < L0.length; k++) {
      const s = L0[k], end = k + 1 < L0.length ? L0[k + 1].i : (S.done ? n + dwell : n);
      if (s.type !== 'state_change' && end - s.i < dwell) continue;   // flap (server state_change events already passed hysteresis)
      if (cur && cur.tag === s.tag) continue;               // same state resumed after a flap
      out.push({ i: s.i, group: g, tag: s.tag, label: s.label, prev: cur ? cur.tag : null, p: s.p }); cur = s;
    }
  }
  out.sort((a, b) => a.i - b.i);
  divCache = { key, list: out };
  return out;
}
function dividerHtml(d) {
  return '<div class="sdiv" data-j="' + d.i + '" title="' + esc(d.tag) + ' p=' + d.p + ' · held ≥ ' + (MIN_DWELL[d.group] || 15) + ' words"><span>' + esc(groupLabel(d.group)) + ': <b>' + esc(d.label || pretty(d.tag)) + '</b>' +
    (d.prev ? ' <span class="from">from ' + esc(pretty(d.prev)) + '</span>' : '') + '</span></div>';
}
// redaction overlay: category / probability / source of the span a redacted word belongs to (redact.spans, else the redaction log)
const RCAT = { card: 'PII·card', gov_id: 'PII·gov ID', phone: 'PII·phone', email: 'PII·email', address: 'PII·address', dob: 'PII·DOB', name: 'PII·name',
  handle: 'PII·handle', contact: 'PII·contact', profanity: 'profanity', toxicity: 'toxicity' };
const TAG2CAT = [[/card_account/, 'card'], [/gov_id/, 'gov_id'], [/phone/, 'phone'], [/email/, 'email'], [/street_address/, 'address'], [/date_of_birth/, 'dob'],
  [/private_name/, 'name'], [/url_handle/, 'handle'], [/public_contact/, 'contact'], [/profanity/, 'profanity'], [/toxicity/, 'toxicity']];
function redInfo(w, i) {
  const sp = S.redSpans[w.rid], lg = sp ? null : S.red.log.find((x) => x.id === w.rid);
  let tag = (sp && sp.tag) || (lg && lg.tag) || '';
  if (!tag) { const c = (covering(tierSel(), i, i + 1)[i] || []).find((x) => x.fam === 'pii' || x.fam === 'profanity'); if (c) tag = c.tag; }
  let cat = (sp && sp.cat) || null;
  if (!cat) for (const m of TAG2CAT) if (m[0].test(tag)) { cat = m[1]; break; }
  if (!cat && lg && lg.type) cat = String(lg.type).toLowerCase();
  let label = RCAT[cat] || (cat ? String(cat) : 'redacted');
  if (cat === 'toxicity' && tag) label = 'toxicity·' + last(tag).split('_')[0];
  const b = tag ? safetyCat(tag) : null, sev = b ? b.severity : (cat === 'card' || cat === 'gov_id' ? 'critical' : cat === 'name' || cat === 'handle' || cat === 'contact' ? 'low' : 'high');
  const src = sp ? sp.src : lg ? (lg.tier_masked && lg.tier_masked !== 'T0' ? 'late_mask' : 'T0') : '';
  const fam = /profanity|toxicity/.test(cat || tag) ? 'lang' : 'pii';
  return { cat, tag, label, cls: 'rc-' + fam + ' sev-' + sev, p: sp ? sp.p : lg ? nz(lg.conf_decided, lg.conf) : null, src, delay: sp ? sp.delay_words : null,
    tier: sp ? sp.tier : lg ? lg.tier_masked : null, status: sp ? sp.status : lg ? lg.status : null, late: src === 'late_mask' || (sp && sp.tier && sp.tier !== 'T0') };
}
function lateFresh(id) { const t = S.lateSeen[id]; const now = Date.now(); if (t == null) { S.lateSeen[id] = now; return true; } return now - t < 1600; }
// ------------------------------------------------------------------ read-out lanes (update.lanes): per-word intent / act / state /
// utterance-safety change points + event markers, drawn as thin rows under each word of the newest LANE_TURNS turns.
// A row cell shows every label at or above its threshold as stacked stripes (opacity grows with p above thr); when none is on, the
// strongest near-threshold label (>= 0.5 thr) as a faint stripe. The top label is written where it changes. Hover: top-3 with p / thr.
const LANE_TURNS = 60;
const LANE_ROWS = [['intent', 'Intent'], ['act', 'Act'], ['state', 'State'], ['safety', 'Utt. safety'], ['events', 'Events']];
const LANE_PAL = ['var(--c1)', 'var(--c2)', 'var(--c3)', 'var(--c4)', 'var(--c5)', 'var(--c6)', 'var(--c7)', 'var(--c8)', 'var(--c9)', 'var(--c10)'];   // brand-derived categorical tokens (app.css)
function laneOn(id) { return !!S.laneOn[id]; }
function laneCol(tag) { let h = 0; for (let i = 0; i < tag.length; i++) h = (h * 31 + tag.charCodeAt(i)) | 0; return LANE_PAL[Math.abs(h) % LANE_PAL.length]; }
function lanesApply(L) {
  const P = S.lanes.pts;
  if (L.full) P.length = 0;
  else { let k = P.length; while (k > 0 && P[k - 1][0] >= L.from) k--; P.length = k; }
  for (const x of L.pts || []) P.push(x);
  Object.assign(S.lanes.tags, L.tags || {});
  if (L.n != null) { let k = P.length; while (k > 0 && P[k - 1][0] >= L.n) k--; P.length = k; }
  if (P.length > 30000) P.splice(0, P.length - 25000);
  S.lanes.ver++;
}
function laneRows() {                              // [{id, lane (server lane name), label}] of the rows switched on
  const rows = [];
  for (const [id, label] of LANE_ROWS) {
    if (!laneOn(id)) continue;
    if (id === 'state') { for (const g of allowedSectionGroups()) rows.push({ id: g, lane: g, label: groupLabel(g) }); continue; }
    rows.push({ id, lane: id === 'events' ? null : id, label });
  }
  return rows;
}
function laneAllowed(lane, tag) {
  if (lane === 'intent') return intentAllowed(tag);
  if (lane === 'safety') return !!safetyCat(tag) || S.debug;
  return true;
}
function laneDerive(pt, row) {                     // cached per point + profile: on labels, faint top, title
  const key = activeId() + (S.debug ? 'd' : '');
  if (pt._k === key) return pt._d;
  const all = (pt[2] || []).filter((x) => laneAllowed(row.lane, x[0])).map((x) => ({ tag: x[0], p: x[1], thr: nz(S.lanes.tags[x[0]], 1) }));
  const on = all.filter((x) => x.p >= x.thr);
  const top = on.length ? on[0] : all[0] || null;
  const name = (x) => (row.lane === 'safety' && safetyCat(x.tag) ? safetyCat(x.tag).label + ' · ' : '') + labelOf(x.tag);
  const title = row.label + (all.length ? '' : ': nothing near threshold') + all.slice(0, 3).map((x) => '\n' + name(x) + '   p ' + x.p.toFixed(2) + ' · thr ' + x.thr.toFixed(2) + (x.p >= x.thr ? '  ✓' : '')).join('') +
    '\n(read-out pooled up to this word; from word ' + pt[0] + ')';
  pt._k = key; pt._d = { on, top, title, lab: top ? name(top) + (on.length > 1 ? ' +' + (on.length - 1) : '') : '' };
  return pt._d;
}
let laneCache = { key: '' };
function laneCtx(turns) {
  const rows = laneRows(), n = S.words.length;
  if (!rows.length || !turns.length) { if (laneCache.key) { laneCache = { key: '' }; laneLegend(); } return null; }
  const ls = turns[Math.max(0, turns.length - LANE_TURNS)].a;
  const key = S.lanes.ver + ':' + ls + ':' + n + ':' + rows.map((r) => r.id).join(',') + ':' + activeId() + ':' + S.events.length + ':' + evTier() + ':' + S.debug;
  if (laneCache.key === key) return laneCache.ctx;
  const P = S.lanes.pts, vals = {};
  let k0 = P.length; while (k0 > 0 && P[k0 - 1][0] >= ls) k0--;
  for (const r of rows) {
    if (!r.lane) continue;
    const arr = new Array(Math.max(0, n - ls)).fill(null);
    let cur = null;
    for (let k = k0 - 1; k >= 0; k--) if (P[k][1] === r.lane) { cur = P[k]; break; }
    let k = k0;
    for (let i = ls; i < n; i++) { while (k < P.length && P[k][0] <= i) { if (P[k][1] === r.lane) cur = P[k]; k++; } arr[i - ls] = cur; }
    vals[r.id] = arr;
  }
  const ev = {};
  if (rows.some((r) => r.id === 'events')) {
    const t = evTier();
    for (const e of S.events) if (e.i >= ls && e.i < n && e.tier === t && (e.q === 'ev_at' || e.q === 'ev_imm') && (S.debug || eventAllowed(e.tag))) (ev[e.i] || (ev[e.i] = [])).push(e);
  }
  const ctx = { ls, rows, vals, ev };
  laneCache = { key, ctx }; laneLegend();
  return ctx;
}
function laneCells(lc, i, t) {
  let h = '<span class="lns">';
  for (const r of lc.rows) {
    if (r.id === 'events') {
      const es = lc.ev[i];
      h += es ? '<span class="lc lev">' + es.slice(0, 3).map((e) => '<b class="em ev-' + (evClass(e.tag) || 'gen') + '" title="' + esc(evName(e) + ' · ' + e.tag + '\np ' + e.p + ' · thr ' + nz(e.thr, '?') + ' · ' + e.q + ' ' + e.tier) + '"></b>').join('') + '</span>' : '<span class="lc lev"></span>';
      continue;
    }
    const pt = lc.vals[r.id][i - lc.ls];
    if (!pt) { h += '<span class="lc"></span>'; continue; }
    const d = laneDerive(pt, r);
    if (r.lane === 'safety' && !d.on.length) { h += '<span class="lc" title="' + esc(d.title) + '"></span>'; continue; }   // off safety rows: hover only, never a mark
    const prev = i > t.a ? lc.vals[r.id][i - 1 - lc.ls] : null, pd = prev ? laneDerive(prev, r) : null;
    const showLab = d.top && (i === t.a || !pd || !pd.top || pd.lab !== d.lab);
    let s = '';
    if (d.on.length) for (const x of d.on.slice(0, 3)) s += '<b style="background:' + laneCol(x.tag) + ';opacity:' + (0.55 + 0.45 * Math.min(1, (x.p - x.thr) / Math.max(0.05, 1 - x.thr))).toFixed(2) + '"></b>';
    else if (d.top) s = '<b class="nr" style="background:' + laneCol(d.top.tag) + '"></b>';
    h += '<span class="lc' + (d.on.length ? '' : ' faint') + '" title="' + esc(d.title) + '">' + s + (showLab ? '<i class="ll' + (d.on.length ? '' : ' off') + '" style="--lc:' + laneCol(d.top.tag) + '">' + esc(d.lab) + '</i>' : '') + '</span>';
  }
  return h + '</span>';
}
(function laneToggles() {
  try { const v = JSON.parse(localStorage.getItem('arctag_lanes') || 'null'); if (v && typeof v === 'object') Object.assign(S.laneOn, v); } catch (e) { /* ignore */ }
  for (const cb of document.querySelectorAll('#laneTg input[data-lane]')) {
    cb.checked = laneOn(cb.dataset.lane);
    cb.onchange = () => { S.laneOn[cb.dataset.lane] = cb.checked ? 1 : 0; try { localStorage.setItem('arctag_lanes', JSON.stringify(S.laneOn)); } catch (e) { /* ignore */ } renderAll(true); };
  }
})();
function laneLegend() {
  const rows = laneRows();
  $('laneLeg').textContent = ''; $('laneLeg').title = rows.length ? 'rows under each word: ' + rows.map((r) => r.label).join(' · ') + ' (newest ' + LANE_TURNS + ' turns)' : '';
}
function cardHtml(t, ctx) {
  const interim = t.readout === 'interim', fd = interim ? ' fd' : '', dbg = S.debug;
  const w0 = S.words[t.a] || {};
  const evs = eventsOfTurn(t), evAt = {};
  for (const e of evs) (evAt[e.i] || (evAt[e.i] = [])).push(e);
  let h = '';
  for (const d of ctx.divs) if (d.i >= t.a && d.i < t.b) h += dividerHtml(d);
  // left margin: domain event icons
  h += '<div class="trow"><div class="gut">' + evs.slice(0, 4).map((e) => { const c = evClass(e.tag); return '<span class="gi ' + c + '" data-j="' + e.i + '" title="' + esc(evName(e)) + ' · p=' + e.p + ' at “' + esc((S.words[e.i] || {}).w || '') + '”">' + EV_ICON[c] + '</span>'; }).join('') +
    (evs.length > 4 ? '<span class="gi more" title="' + evs.length + ' events">+' + (evs.length - 4) + '</span>' : '') + '</div>';
  h += '<div class="turn' + (t.open ? ' open' : '') + (interim ? ' interim' : '') + (S.exp.has(t.id) ? ' exp' : '') + '" style="border-left-color:' + ctx.col + ';--spk:' + ctx.col + '">';
  h += '<div class="th"><span class="spk"><span>' + esc(spkName(t.spk)) + '</span></span>';
  h += roleBadge(t);
  if (t.act && (t.act.on || dbg)) h += '<span class="badge act' + fd + (t.act.on === false ? ' off' : '') + '" title="dialog act · ' + esc(t.act.top) + ' p=' + t.act.p + (t.act.thr != null ? ' (thr ' + t.act.thr + ')' : '') + '">' + esc(t.act.label || labelOf(t.act.top)) + (dbg ? ' <span class="pv">' + pct(t.act.p) + '</span>' : '') + '</span>';
  if (!dbg && (prof().show_intent != null ? prof().show_intent : INTENT_VERTS[activeId()])) {
    const it = (t.intents || [])[0];
    if (it && it.on) h += '<span class="badge int' + fd + '" title="intent · ' + esc(it.tag) + ' p=' + it.p + '">' + esc(it.label || labelOf(it.tag)) + '</span>';
  }
  if (t.device_directed && t.device_directed.on) h += '<span class="badge dev' + fd + '" title="discourse.assistant.device_directed p=' + t.device_directed.p + '">to device</span>';
  h += '<span class="right"><span class="tm">' + fmtT(w0.t0) + '</span>';
  if (dbg) {
    if (t.interrupt && t.interrupt.on) h += '<span class="badge intr">interruption</span>';
    h += '<span class="badge src" title="turn boundary: ' + esc(SRC_LONG[t.src] || t.src) + '">' + esc(SRC_LAB[t.src] || t.src) + (t.arc_change ? ' · Arc ✓' : '') + '</span>';
    if (t.yield && !t.open) h += '<span class="badge yield' + (t.yield.on ? ' on' : '') + '" title="Arc discourse.turn_yield' + (t.yield.p != null ? ' p=' + t.yield.p + ' (thr ' + t.yield.thr + ')' : '') + '">' + (t.yield.on ? '⏎ yield' : 'no yield') + '</span>';
  }
  h += '</span></div>';
  if (dbg) h += intentsDebug(t);
  h += '<div class="txt' + (ctx.lanes && t.b > ctx.lanes.ls ? ' lanes' : '') + '">';
  const per = ctx.per, end = Math.min(t.b, S.words.length);
  for (let i = t.a; i < end; i++) {
    const w = S.words[i];
    if (!w) continue;
    const cov = per[i] || [], red = w.r === 'mask' || w.r === 'tag' || w.r === 'pend';
    let cls = 'w' + (w.f ? '' : ' i'), txt = w.w, lab = '';
    if (red && S.redView === 'redacted') {                    // exactly what a downstream consumer receives
      txt = w.rx != null ? w.rx : w.w;
      if (txt === '' || (w.r === 'mask' && !txt)) continue;
      cls += w.r === 'tag' ? ' rd ' + redInfo(w, i).cls : ' red';
    } else if (red) {                                         // overlay: original words, category overlay on top
      if (!txt) continue;
      const ri = redInfo(w, i);
      cls += ' rd ' + ri.cls + (w.r === 'pend' ? ' held' : '') + (ri.late && lateFresh(w.rid) ? ' late' : '');
      const prevW = i > t.a ? S.words[i - 1] : null;
      const prevRed = prevW && (prevW.r === 'mask' || prevW.r === 'tag' || prevW.r === 'pend');
      if (!prevRed || (prevW.rid !== w.rid && redInfo(prevW, i - 1).label !== ri.label)) lab = '<span class="rl">' + esc(ri.label) + '</span>';
    } else {
      if (!txt) continue;
      const ent = cov.find((s) => s.fam === 'entity');
      if (ent) cls += ' e e-' + last(ent.tag); else if (dbg && cov.some((s) => s.fam === 'number')) cls += ' n';
      if (cov.some((s) => s.fam === 'pii')) cls += ' pii'; else if (cov.some((s) => s.fam === 'profanity')) cls += ' prof';
    }
    if (evAt[i]) cls += ' evw ev-' + evClass(evAt[i][0].tag);
    if (i === ctx.nowI) cls += ' now';
    const rv = S.rev[i]; let pre = '';
    if (dbg && rv && ctx.now - rv.t < 2000) { cls += ' rev'; pre = '<s class="old">' + esc(rv.old) + '</s>'; }
    const ws = '<span class="' + cls + '" data-i="' + i + '">' + esc(txt) + '</span>';
    h += ctx.lanes && i >= ctx.lanes.ls ? pre + lab + '<span class="wl">' + ws + laneCells(ctx.lanes, i, t) + '</span><wbr>' : pre + lab + ws + ' ';
  }
  h += '</div>';
  const foot = [];
  for (const s of t.safety || []) if (s.kind === 'utt' && s.on && safetyCat(s.tag)) foot.push('<span class="flag" title="' + esc(s.tag) + ' (utterance read-out)\np ' + s.p + ' · thr ' + nz(s.thr, '?') + '">' + esc(safetyCat(s.tag).label) + ' <span class="pv">' + (+s.p).toFixed(2) + (s.thr != null ? ' / ' + (+s.thr).toFixed(2) : '') + '</span></span>');
  if (foot.length) h += '<div class="tf">' + foot.join('') + '</div>';
  return h + '</div></div>';
}
function groupLabel(g) {
  for (const p of prof().phases || []) if (p.group === g) return p.label;
  return { 'content.subject': 'Topic', 'content.mode': 'Mode', 'content.legal_phase': 'Phase', 'content.meeting_phase': 'Meeting phase', 'content.game_state': 'Game state' }[g] || pretty(g + '.x').replace(/ ?X$/, '') || g;
}
function allowedSectionGroups() { return (prof().phases || []).map((p) => p.group); }

function nowWord() {                       // the word being spoken: audio time vs word times (offset = where the ASR stream started)
  const a = S.audioEl;
  if (!a || a.paused || !S.words.length) return -1;
  const t = a.currentTime - (S.tOffset || 0);
  let lo = 0, hi = S.words.length - 1, best = -1;
  while (lo <= hi) { const m = (lo + hi) >> 1, w = S.words[m]; if (w && w.t0 != null && w.t0 <= t) { best = m; lo = m + 1; } else hi = m - 1; }
  if (best >= 0) { const w = S.words[best]; if (w.t1 != null && t > w.t1 + 0.6) return -1; }
  return best;
}
function renderTranscript() {
  const el = $('transcript');
  const turns = turnList();
  if (!S.words.length || !turns.length) {
    if (cards.size) { for (const c of cards.values()) c.el.remove(); cards.clear(); }
    $('trEmpty').hidden = false; $('trinfo').textContent = ''; return;
  }
  $('trEmpty').hidden = true;
  const t = tierSel(), now = performance.now();
  const vis = turns.slice(-150);
  const a0 = vis[0].a;
  S.nowI = nowWord();
  const ctx = { now, divs: stateChanges(), per: covering(t, a0, S.words.length), nowI: S.nowI, lanes: laneCtx(vis) };
  const keep = new Set();
  for (const tn of vis) {
    keep.add(tn.id);
    ctx.col = spkColor(tn.spk);
    const h = cardHtml(tn, ctx);
    let c = cards.get(tn.id);
    if (!c) {
      const div = document.createElement('div'); div.className = 'tw'; div.dataset.id = tn.id;
      el.appendChild(div); c = { el: div, sig: '' }; cards.set(tn.id, c);
    }
    c.el.dataset.a = String(tn.a);
    if (c.sig !== h) { c.el.innerHTML = h; c.sig = h; }
  }
  for (const [id, c] of Array.from(cards.entries())) if (!keep.has(id)) { c.el.remove(); cards.delete(id); }
  if (S.follow) el.scrollTop = el.scrollHeight;
  $('bFollow').hidden = S.follow;
  const nf = S.words.filter((w) => w && w.f).length;
  $('trinfo').textContent = S.words.length + ' words · ' + nf + ' final · ' + turns.length + ' turns' + (S.debug ? ' · tier ' + t : '') + (S.schema === 2 ? '' : ' · schema v1');
  if (S.debug && Object.keys(S.rev).length) { for (const k of Object.keys(S.rev)) if (now - S.rev[k].t > 2000) delete S.rev[k]; setTimeout(schedule, 700); }
}
$('transcript').addEventListener('scroll', () => {
  const el = $('transcript');
  S.follow = el.scrollHeight - el.scrollTop - el.clientHeight < 60;
  $('bFollow').hidden = S.follow;
});
$('bFollow').onclick = () => { S.follow = true; const el = $('transcript'); el.scrollTop = el.scrollHeight; $('bFollow').hidden = true; };
$('transcript').addEventListener('click', (e) => {
  const b = e.target.closest && e.target.closest('[data-role]');
  if (b) { const card = b.closest('.tw'); if (card) { const id = card.dataset.id; if (S.exp.has(id)) S.exp.delete(id); else S.exp.add(id); renderAll(); } return; }
  const j = e.target.closest && e.target.closest('[data-j]'); if (j) jump(+j.dataset.j);
  const w = e.target.closest && e.target.closest('.w');          // click a word: seek the audio there
  if (w && S.audioEl && !S.audioLive) { const wd = S.words[+w.dataset.i]; if (wd && wd.t0 != null) seekAudio(wd.t0 + (S.tOffset || 0)); }
});
function jump(i) {
  const t = turnAt(i); if (!t) return;
  const c = cards.get(t.id); if (!c) { msg('that turn has scrolled out of the rendered window'); return; }
  S.follow = false; $('bFollow').hidden = false;
  c.el.scrollIntoView({ block: 'center', behavior: 'smooth' });
  c.el.classList.add('flash'); setTimeout(() => c.el.classList.remove('flash'), 1400);
  const w = c.el.querySelector('[data-i="' + i + '"]'); if (w) { w.classList.add('hit'); setTimeout(() => w.classList.remove('hit'), 1400); }
}
document.addEventListener('mousemove', (e) => {
  const tip = $('tip'), w = e.target && e.target.closest && e.target.closest('#transcript .w');
  if (!w) { tip.style.display = 'none'; return; }
  const i = +w.dataset.i, wd = S.words[i]; if (!wd) return;
  const cov = covering(tierSel(), i, i + 1)[i] || [];
  let s = '"' + wd.w + '"';
  if (wd.r === 'mask' || wd.r === 'tag' || wd.r === 'pend') {
    const ri = redInfo(wd, i);
    s = ri.label + (ri.p != null ? '  p=' + (+ri.p).toFixed(2) : '') + (ri.src ? '  source ' + ri.src : '') + (ri.delay ? '  (+' + ri.delay + ' words late)' : '') + (ri.tier ? '  ' + ri.tier : '') + (ri.status ? '  ' + ri.status : '') +
      '\n' + (wd.r === 'tag' ? 'tagged only (shown in clear downstream)' : wd.r === 'pend' ? 'held: shown once its read-out decides' : 'masked downstream as ' + (wd.rx != null ? '"' + wd.rx + '"' : 'a placeholder'));
    if (wd.rx == null && wd.r === 'mask') s += '\nthis server sends masked text only; the original stays on the server';
    tip.textContent = s; tip.style.display = 'block'; tip.style.left = Math.min(e.clientX + 14, innerWidth - 450) + 'px'; tip.style.top = (e.clientY + 16) + 'px'; return;
  }
  if (S.debug) s += '  ' + (wd.f ? 'final' : 'interim') + '  t=' + nz(wd.t0, '?') + '–' + nz(wd.t1, '?') + 's';
  if (wd.r !== 'mask' && wd.r !== 'pend') {
    for (const x of cov) s += '\n' + (x.fam === 'number' ? 'number · ' : '') + labelOf(x.tag) + '  ' + pct(x.p) + (x.alt ? '  (or ' + labelOf(x.alt.tag) + ' ' + pct(x.alt.p) + ')' : '') + (S.debug ? '  (' + x.tag + ', thr ' + x.thr + ')' : '');
    for (const ev of eventsOfTurn(turnAt(i) || { a: i, b: i + 1 })) if (ev.i === i) s += '\n◆ ' + evName(ev) + '  ' + pct(ev.p);
  }
  if (!cov.length && !S.debug && !/\bevw\b/.test(w.className)) return void (tip.style.display = 'none');
  tip.textContent = s; tip.style.display = 'block';
  tip.style.left = Math.min(e.clientX + 14, innerWidth - 450) + 'px'; tip.style.top = (e.clientY + 16) + 'px';
});

// ------------------------------------------------------------------ safety strip (profile categories only)
// display buckets for the strip: a profile's (fine-grained) safety tags fold into these; only buckets the profile contains are shown
const BUCKETS = [
  ['card', 'Card / account', 'critical', /^safety\.pii\.card_account_number/], ['gov_id', 'Gov ID', 'critical', /^safety\.pii\.gov_id/],
  ['phone', 'Phone', 'high', /^safety\.pii\.phone_number/], ['email', 'Email', 'high', /^safety\.pii\.email/], ['address', 'Address', 'high', /^safety\.pii\.street_address/],
  ['dob', 'DOB', 'high', /^safety\.pii\.date_of_birth/], ['name', 'Name', 'low', /^safety\.pii\.private_name/], ['handle', 'Handle', 'low', /^safety\.pii\.url_handle/],
  ['contact', 'Public contact', 'low', /^safety\.pii\.public_contact/], ['pii_req', 'PII request / leak', 'medium', /^safety\.pii\.(request|leak)/],
  ['profanity', 'Profanity', 'medium', /^safety\.profanity\./], ['insult', 'Insult', 'medium', /^safety\.toxicity\.insult_harassment/],
  ['hate', 'Hate', 'high', /^safety\.toxicity\.hate_identity/], ['sexual', 'Sexual', 'high', /^safety\.toxicity\.sexual_explicit/],
  ['threat', 'Threat', 'critical', /^safety\.toxicity\.threat_violence/], ['illegal', 'Illegal / regulated', 'medium', /^safety\.toxicity\.illegal_regulated/],
  ['crisis', 'Self-harm / crisis', 'critical', /^safety\.(crisis\.|toxicity\.self_harm)/], ['escalation', 'Escalation', 'high', /^safety\.threat\.escalation/],
  ['scam', 'Scam', 'high', /^safety\.scam\./], ['child', 'Child safety', 'critical', /^safety\.child\./],
  ['solicit', 'Solicitation / promo', 'medium', /^safety\.(solicitation\.|content\.promo_redirect)/],
];
let bucketCache = { id: '', list: [] };
function profileBuckets() {
  const id = activeId() + ':' + (prof().server ? 's' : 'c');
  if (bucketCache.id === id) return bucketCache.list;
  const ent = (prof().safety || []).map((x) => x.id);
  const list = BUCKETS.filter((b) => ent.some((e) => b[3].test(e) || b[3].test(e + '.x'))).map((b) => ({ id: b[0], label: b[1], severity: b[2], re: b[3] }));
  bucketCache = { id, list };
  return list;
}
function safetyCat(tag) { for (const b of profileBuckets()) if (b.re.test(tag)) return b; return null; }
function safetyCounts() {
  const c = {}, mx = {};
  const add = (tag, n, p) => { const k = safetyCat(tag); if (!k) return; c[k.id] = (c[k.id] || 0) + n; if (p != null) mx[k.id] = Math.max(mx[k.id] || 0, p); };
  const sm = S.summary;
  if (sm && (sm.pii || sm.safety)) {
    for (const src of [sm.pii && sm.pii.by_tag, sm.safety && sm.safety.by_tag]) if (src) for (const tag of Object.keys(src)) add(tag, src[tag]);
  } else {                                     // v1: one per distinct span (tier shown, T1 preferred) + on utterance read-outs
    const t = evTier(), seen = {};
    for (const s of S.spans[t]) if (s.fam === 'pii' || s.fam === 'profanity') { const k = s.tag + ':' + s.a; if (!seen[k]) { seen[k] = 1; add(s.tag, 1, s.p); } }
    for (const a of S.alerts) if (a.kind === 'utterance') add(a.tag, 1, a.p);
  }
  return [c, mx];
}
function renderSafety() {
  const res = safetyCounts(), c = res[0], mx = res[1];
  let h = '';
  for (const k of profileBuckets()) {
    const n = c[k.id] || 0;
    h += '<span class="sc ' + k.severity + (n ? '' : ' zero') + '" title="' + esc(k.id) + ' · severity ' + k.severity + (mx[k.id] ? ' · max p ' + mx[k.id].toFixed(2) : '') + '"><i></i>' + esc(k.label) + ' <b>' + n + '</b>' + (n && (k.severity === 'critical' || k.severity === 'high') ? '<span class="sev">' + k.severity + '</span>' : '') + '</span>';
  }
  if (!h) h = '<span class="note">no safety categories in this profile</span>';
  if ($('scats')._h !== h) { $('scats').innerHTML = h; $('scats')._h = h; }
}
function hiBanner(html) { $('hibanTxt').innerHTML = html; $('hiban').hidden = false; clearTimeout(S.hbT); S.hbT = setTimeout(() => { $('hiban').hidden = true; }, 12000); }
$('bHiban').onclick = () => { $('hiban').hidden = true; };
function maybeHighBanner(a) {
  const k = safetyCat(a.tag); if (!k || k.severity !== 'critical') return;
  if (a.p < Math.max((a.thr || 0) + 0.15, 0.7)) return;
  const key = k.id + ':' + a.w0; if (S.banSeen.has(key)) return; S.banSeen.add(key);
  hiBanner('<b>' + esc(k.label) + '</b> · ' + esc(a.tag) + ' p=' + a.p + ' at word ' + a.w0 + ($('rmode').value === 'off' ? ' · redaction is off' : ''));
}

// ------------------------------------------------------------------ header status / detected domain
function renderStatus() {
  const st = S.status || {}, x = S.stats || {}, lim = st.limits || {};
  const f = (v, u) => (v == null ? '–' : v + (u || '')), ms = (v) => (v == null ? '–' : Math.round(v * 1000) + ' ms');
  const h = '<span>arctag <b>' + f(st.version) + '</b> · ' + f(nz(x.precision, st.precision)) + (x.profile ? ' · ' + esc(x.profile) : '') + '</span>' +
    '<span>ASR audio→word <b>' + ms(x.asr_lat_p50) + '</b> p50</span>' +
    '<span>tagger <b>' + f(x.word_p50, ' ms') + '</b>/update p50 · ' + f(x.word_p95, ' ms') + ' p95</span>' +
    '<span>words/s <b>' + f(x.words_per_s) + '</b></span>' +
    '<span>live ASR <b>' + (st.live && st.live.enabled ? esc(st.live.provider || 'on') : 'off') + '</b></span>' +
    '<span>' + (S.running ? '<b>' + esc(S.label || 'streaming') + '</b>' : S.done ? 'finished' : 'idle') + '</span>';
  if ($('status')._h !== h) { $('status').innerHTML = h; $('status')._h = h; }
  const sum = (S.running ? '\u25CF ' + (S.label || 'streaming') : S.done ? 'finished' : 'idle') + ' · arctag ' + f(st.version) + ' · tagger ' + f(x.word_p50, ' ms') + '/update · live ASR ' + (st.live && st.live.enabled ? 'on' : 'off');
  void lim;
  if ($('footSum').textContent !== sum) $('footSum').textContent = sum;
  $('honest').textContent = 'Streaming tagger';
  $('honest').title = 'arctag C library ' + f(st.version) + ' · ' + f(st.precision) + ' · CPU';
}
function renderDetected() {
  const el = $('detected');
  if (S.vertical !== 'auto') { el.hidden = true; $('trTitle').textContent = 'Transcript · ' + PROFILES[S.vertical].label; return; }
  el.hidden = false;
  const d = S.domain;
  let h, cls = 'det';
  if (!d || (!d.detected && d.requested === 'auto')) { h = 'Domain: auto (generic thresholds; pick a vertical for domain tags)'; cls += ' wait'; }
  else if (d.detected && PROFILES[d.detected]) h = 'Detected: ' + PROFILES[d.detected].label + ' · ' + pct(d.conf) + (d.locked ? '' : ' (tentative)');
  else { h = 'Detecting domain… ' + nz(d.evidence_words, 0) + ' words'; cls += ' wait'; }
  el.className = cls; el.textContent = h;
  $('trTitle').textContent = 'Transcript · ' + (activeId() === 'auto' ? 'auto' : PROFILES[activeId()].label + ' (auto)');
}

// ------------------------------------------------------------------ side panels
function item(i, body, p, cls) {
  const w = S.words[i] || {};
  return '<div class="it' + (cls ? ' ' + cls : '') + '" data-j="' + i + '"><span class="t">' + fmtT(w.t0) + '</span><span>' + body + '</span><span class="p">' + (p != null ? (+p).toFixed(2) : '') + '</span></div>';
}
function quote(a, b) { return '<span class="q">' + esc(text(a, Math.min(b, a + 40))) + '</span>'; }
const L = (rows, empty) => '<div class="list">' + (rows.length ? rows.join('') : '<div class="empty">' + empty + '</div>') + '</div>';
const closed = () => turnList().filter((t) => !t.open || S.done);
const onIntent = (t, re) => (t.intents || []).find((r) => r.on && re.test(r.tag));
function turnItem(t, label, p) { return item(t.a, '<b>' + esc(label) + '</b> <span class="who">' + esc(owner(t.a)) + '</span><br>' + quote(t.a, t.b), p); }
function sumList(key) { return S.summary && Array.isArray(S.summary[key]) ? S.summary[key] : null; }
function timelineEvents() {
  const s = sumList('timeline');
  if (s) return s.filter((e) => eventAllowed(e.tag));
  return profileEvents();
}
function evItem(e, extra) {
  const t = turnAt(e.i);
  return item(e.i, '<b>' + esc(evName(e)) + '</b> <span class="who">' + esc(owner(e.i)) + '</span>' + (extra || '') + (t ? '<br>' + quote(t.a, t.b) : ''), e.p);
}
const PANELS = {
  domain: {
    title: 'Domain detection', render() {
      const d = S.domain;
      if (!d) return '<div class="empty">' + (S.schema === 2 ? 'waiting for evidence…' : 'domain detection needs the v2 server (update.enrich)') + '</div>';
      const sc = Object.keys(d.scores || {}).map((k) => [k, d.scores[k]]).sort((a, b) => b[1] - a[1]).slice(0, 7);
      return '<div class="note" style="margin-bottom:6px">' + (d.detected ? '<b>' + esc((PROFILES[d.detected] || {}).label || d.detected) + '</b> ' + pct(d.conf) + (d.locked ? ' · locked' : ' · tentative') : 'not locked yet') +
        ' · ' + nz(d.evidence_words, 0) + ' words of evidence' + (d.candidate ? ' · challenger ' + esc((PROFILES[d.candidate.id] || {}).label || d.candidate.id) + ' ' + pct(d.candidate.conf) : '') + '</div>' +
        sc.map((x) => '<div class="bar' + (x[0] === d.detected ? ' on' : '') + '"><span class="lb">' + esc((PROFILES[x[0]] || {}).label || x[0]) + '</span><span class="tr"><span class="fl" style="width:' + (100 * x[1]).toFixed(1) + '%"></span></span><span class="v">' + x[1].toFixed(2) + '</span></div>').join('');
    },
  },
  timeline: { title: 'Events timeline', render: () => L(timelineEvents().slice(-150).reverse().map((e) => evItem(e)), 'on-threshold events of this domain appear here') },
  objections: {
    title: 'Objections → rulings', render() {
      let rows = [];
      const s = sumList('objections');
      if (s) rows = s.map((o) => item(nz(o.i, o.a), '<b>' + (o.kind === 'objection_intent' ? 'Objecting turn' : 'Objection') + '</b> <span class="who">' + esc(owner(nz(o.i, o.a))) + '</span>' +
        (o.ruling ? ' → <b>' + esc(o.ruling.label || pretty(o.ruling.tag).replace(/^Ruling /, '')) + '</b> <span class="who">' + esc(owner(o.ruling.i)) + '</span>' : ' → <span class="note">no ruling yet</span>'), o.p));
      else {
        const ev = profileEvents();
        for (const e of ev) {
          if (e.tag.indexOf('event.legal.objection') !== 0) continue;
          const rl = ev.find((x) => x.i > e.i && x.i - e.i < 40 && /ruling_/.test(x.tag));
          rows.push(evItem(e, rl ? ' → <b>' + esc(pretty(rl.tag).replace(/^Ruling /, '')) + '</b>' : ' → <span class="note">no ruling yet</span>'));
        }
      }
      return L(rows.reverse(), 'objections and rulings appear here');
    },
  },
  scores: {
    title: 'Scores & fouls', render() {
      const s = sumList('scores');
      const fouls = profileEvents().filter((e) => /penalty_foul|turnover|review_challenge|injury|timeout_called|substitution/.test(e.tag));
      const sc = s ? s.map((x) => Object.assign({}, x, { nums: x.numbers })) : profileEvents().filter((e) => /\.(score|touchdown|field_goal|wicket|boundary)\.recent$/.test(e.tag));
      const rows = sc.concat(fouls).sort((a, b) => a.i - b.i).slice(-100).reverse().map((e) => {
        const nums = (e.nums || S.spans[evTier()].filter((x) => x.tag === 'entity.number.score' && Math.abs(x.a - e.i) <= 12)).map((x) => text(x.a, x.b)).filter(Boolean);
        return evItem(e, nums.length ? ' <span class="note">' + esc(nums.slice(0, 2).join(' · ')) + '</span>' : '');
      });
      return L(rows, 'score, foul and stoppage events appear here');
    },
  },
  highlights: {
    title: 'Highlight markers (clip t−8 s … t+4 s)', render() {
      const ev = timelineEvents().filter((e) => /\.recent$/.test(e.tag));
      return L(ev.slice(-60).reverse().map((e) => { const w = S.words[e.i] || {}; return item(e.i, '<b>' + esc(evName(e)) + '</b> <span class="note">clip ' + fmtT(Math.max(0, (w.t1 || 0) - 8)) + '–' + fmtT((w.t1 || 0) + 4) + '</span>', e.p); }), 'highlight clips appear when sports events fire');
    },
  },
  decisions: { title: 'Decisions', render: () => decisionRows('decision', 'event.meeting.decision', /^intent\.decide_agree$/, 'decisions: meeting decision events and decide/agree turns') },
  actions: { title: 'Action items · owner = speaker', render: () => decisionRows('action_item', 'event.meeting.action_item', /^intent\.assign_task$/, 'action items: meeting action-item events and assign-task turns') },
  intents: {
    title: 'Customer & agent intents', render: () => L(closed().filter((t) => t.intents && t.intents[0] && t.intents[0].on).slice(-80).reverse().map((t) => turnItem(t, t.intents[0].label || labelOf(t.intents[0].tag), t.intents[0].p)), 'on-threshold turn intents appear here'),
  },
  commitments: {
    title: 'Commitments / offers', render: () => L(closed().filter((t) => (t.act && t.act.on && /commitment_offer$/.test(t.act.top)) || onIntent(t, /^intent\.offer_help$/)).slice(-60).reverse()
      .map((t) => turnItem(t, t.act && /commitment_offer$/.test(t.act.top) ? 'Commitment / offer' : 'Offer help', t.act ? t.act.p : null)), 'commitments (act commitment_offer, intent offer_help) appear here'),
  },
  escalation: {
    title: 'Complaints / escalation', render() {
      const rows = closed().map((t) => {
        const it = onIntent(t, /^intent\.(complain_criticize|warn_threaten|disagree_object)$/);
        const sf = (t.safety || []).find((s) => s.kind === 'utt' && s.on && /threat\.escalation/.test(s.tag));
        return it ? turnItem(t, it.label || labelOf(it.tag), it.p) : sf ? turnItem(t, 'Escalation', sf.p) : '';
      }).filter(Boolean);
      for (const a of S.alerts) if (a.kind === 'utterance' && /threat\.escalation/.test(a.tag) && !S.tmap.size) rows.push(item(a.w0, '<b>Escalation</b> <span class="who">' + esc(owner(a.w0)) + '</span>', a.p));
      return L(rows.slice(-60).reverse(), 'complaints, warnings and escalation cues appear here');
    },
  },
  redlog: {
    title: 'PII redaction log', render() {
      const c = S.red.counters || {}, lg = S.red.log || [], mode = $('rmode').value;
      const k = '<div class="kpis"><div class="kpi"><b>' + nz(c.spans, 0) + '</b><span>spans redacted</span></div><div class="kpi"><b>' + (c.release_rate == null ? '–' : Math.round(100 * c.release_rate) + '%') +
        '</b><span>released (false alarm)</span></div><div class="kpi"><b>' + (c.ttm_p50 == null ? '–' : Math.round(c.ttm_p50) + ' ms') + '</b><span>time-to-mask p50</span></div></div>';
      if (mode === 'off') return k + '<div class="empty">turn Redaction on to see each masked span</div>';
      if (!lg.length) return k + '<div class="empty">nothing redacted yet</div>';
      return k + '<div class="list"><table class="t"><tr><th>#</th><th>time</th><th>type</th><th class="num">conf</th><th>status</th></tr>' + lg.slice(-120).reverse().map((r) =>
        '<tr class="' + (r.released ? 'released' : '') + '" data-j="' + r.w0 + '" title="' + esc(r.tag) + ' · words ' + r.w0 + '–' + r.w1 + '"><td>' + r.id + '</td><td>' + fmtT(r.t0) + '</td><td>' + esc(r.type) +
        '</td><td class="num">' + (+nz(r.conf_decided, r.conf)).toFixed(2) + '</td><td>' + (r.released ? 'released' : esc(r.status)) + '</td></tr>').join('') + '</table></div>' +
        '<div style="display:flex;justify-content:flex-end;margin-top:6px"><button class="sm" data-act="export">Export redacted JSON</button></div>';
    },
  },
  safetyfeed: {
    title: 'Safety feed', render() {
      const rows = [];
      const seen = {};
      for (const a of S.alerts.slice(-300)) {
        const k = safetyCat(a.tag); if (!k) continue;
        const key = k.id + ':' + a.w0; if (seen[key]) continue; seen[key] = 1;
        rows.push(item(a.w0, '<b>' + esc(k.label) + '</b> <span class="note">' + esc(k.severity) + ' · ' + esc(a.kind) + (a.thr != null ? ' · thr ' + (+a.thr).toFixed(2) : '') + '</span> <span class="who">' + esc(owner(a.w0)) + '</span>', a.p));
      }
      return L(rows.slice(-120).reverse(), 'profanity, toxicity, scam and PII flags appear here');
    },
  },
  commands: {
    title: 'Commands: intent + slots', render() {
      const rows = closed().filter((t) => (t.device_directed && t.device_directed.on) || (t.act && t.act.on && /command_request$/.test(t.act.top))).slice(-60).reverse().map((t) => {
        const it = (t.intents || [])[0];
        const slots = (t.numbers || S.spans[evTier()].filter((s) => (s.fam === 'number' || s.fam === 'entity') && s.a >= t.a && s.b <= t.b)).map((s) => (s.label || pretty(s.tag)) + ': ' + text(s.a, s.b)).slice(0, 4);
        const cc = eventsOfTurn(t).some((e) => e.tag === 'discourse.assistant.command_complete');
        return item(t.a, '<b>' + esc(it ? it.label || labelOf(it.tag) : 'Command') + '</b>' + (t.device_directed && t.device_directed.on ? ' <span class="note">device-directed ' + pct(t.device_directed.p) + '</span>' : '') +
          (cc ? ' <span class="note">· complete</span>' : '') + '<br>' + quote(t.a, t.b) + (slots.length ? '<br><span class="note">slots: ' + esc(slots.join(' · ')) + '</span>' : ''), it ? it.p : null);
      });
      return L(rows, 'device-directed commands appear here');
    },
  },
  latency: {
    title: 'Command-complete latency', render() {
      const x = S.stats || {}, l = S.ccLead.slice().sort((a, b) => a - b), md = l.length ? Math.round(l[Math.floor(l.length / 2)]) : null;
      return '<div class="kpis"><div class="kpi"><b>' + (md == null ? '–' : md + ' ms') + '</b><span>command complete ahead of ASR endpoint (median, n=' + l.length + ')</span></div>' +
        '<div class="kpi"><b>' + nz(x.word_p50, '–') + ' ms</b><span>tagger per word p50</span></div><div class="kpi"><b>' + (x.asr_lat_p50 == null ? '–' : Math.round(1000 * x.asr_lat_p50) + ' ms') + '</b><span>ASR audio→word p50</span></div></div>' +
        '';
    },
  },
  stories: {
    title: 'Stories & subjects', render() {
      const rows = timelineEvents().filter((e) => e.tag === 'content.story_start').map((e) => evItem(e))
        .concat(S.sections.filter((s) => s.group === 'content.subject' || s.tag === 'content.breaking_news').map((s) => item(s.i, '<b>' + esc(s.tag === 'content.breaking_news' ? 'Breaking news' : 'Subject: ' + (s.label || pretty(s.tag))) + '</b>', s.p)));
      return L(rows.slice(-80).reverse(), 'story starts and subject changes appear here');
    },
  },
  speakers: {
    title: 'Speakers & roles', render() {
      let L0 = S.speakers && S.speakers.length ? S.speakers.map((s) => ({ spk: s.spk, turns: s.n_turns, words: s.n_words, role: s.role, a: firstTurnOf(s.spk) })) : null;
      if (!L0) {
        const by = {};
        for (const t of turnList()) { if (t.spk == null) continue; const k = spkKey(t.spk); const e = by[k] || (by[k] = { spk: t.spk, turns: 0, words: 0, role: null, a: t.a }); e.turns++; e.words += t.b - t.a; if (t.role) e.role = t.role; }
        L0 = Object.values(by);
      }
      if (!L0.length) return '<div class="empty">' + (S.provider === 'flux' && S.needsAudio ? 'Flux has no diarization: switch to nova-3 for speaker ids and roles' : 'speakers appear with diarized speaker ids') + '</div>';
      return L0.sort((a, b) => b.words - a.words).slice(0, 8).map((e) => {
        const r = e.role, dist = r && r.dist ? r.dist.slice(0, 4) : [];
        return '<div class="spkrow" data-j="' + nz(e.a, 0) + '"><span><span style="color:' + spkColor(e.spk) + '">●</span> <b>' + esc(spkName(e.spk)) + '</b> ' + (r ? '· ' + esc(r.label || labelOf(r.top)) + ' ' + pct(r.p) + (r.heuristic ? ' <span class="note">heuristic</span>' : '') : '') +
          '</span><span class="note">' + e.turns + ' turns · ' + e.words + ' words</span><div class="bars">' + dist.map((d) => '<div class="bar' + (d.tag === r.top ? ' on' : '') + '"><span class="lb">' + esc(d.label || labelOf(d.tag)) + '</span><span class="tr"><span class="fl" style="width:' + (100 * d.p).toFixed(1) + '%"></span></span><span class="v">' + d.p.toFixed(2) + '</span></div>').join('') + '</div></div>';
      }).join('') + (prof().roles_heuristic ? '<div class="note" title="agent / customer roles are derived (heuristic), not a model output">roles: heuristic</div>' : '');
    },
  },
};
function firstTurnOf(spk) { const k = spkKey(spk); const t = turnList().find((x) => spkKey(x.spk) === k); return t ? t.a : 0; }
function decisionRows(kind, evPrefix, intentRe, empty) {
  const s = sumList('decisions');
  let rows;
  if (s) rows = s.filter((d) => d.kind === kind).map((d) => { const i = nz(d.i, d.a); const t = turnAt(i); return item(i, '<b>' + esc(d.label || pretty(d.tag)) + '</b> <span class="who">' + esc(owner(i)) + '</span>' + (t ? '<br>' + quote(t.a, t.b) : ''), d.p); });
  else rows = profileEvents().filter((e) => e.tag.indexOf(evPrefix) === 0).map((e) => evItem(e))
    .concat(closed().map((t) => { const it = onIntent(t, intentRe); return it ? turnItem(t, it.label || labelOf(it.tag), it.p) : ''; }).filter(Boolean));
  return L(rows.slice(-80).reverse(), empty);
}
const TRACK_COL = ['var(--c1)', 'var(--c2)', 'var(--c3)', 'var(--c4)', 'var(--c5)', 'var(--c6)', 'var(--c7)', 'var(--c8)', 'var(--c9)', 'var(--c10)'];
function phasePanel(group) {
  const secs = stateChanges().filter((s) => s.group === group), n = Math.max(1, S.words.length);
  const st = S.state && S.state[group], live = st ? { label: st.label, top: st.tag, p: st.conf, on: true } : S.live && S.live.phases && S.live.phases[group];
  const tags = []; for (const s of secs) if (tags.indexOf(s.tag) < 0) tags.push(s.tag);
  const col = (tag) => TRACK_COL[tags.indexOf(tag) % TRACK_COL.length];
  let track = '';
  for (let k = 0; k < secs.length; k++) {
    const a = secs[k].i, b = k + 1 < secs.length ? secs[k + 1].i : n;
    track += '<span data-j="' + a + '" title="' + esc(secs[k].label || pretty(secs[k].tag)) + ' · words ' + a + '–' + b + '" style="flex:' + Math.max(1, b - a) + ';background:' + col(secs[k].tag) + '"></span>';
  }
  const cur = live ? '<div class="note" style="margin-bottom:4px">now: <b>' + esc(live.label || pretty(live.top)) + '</b> ' + pct(live.p) + (live.on ? '' : ' (below threshold)') + '</div>' : '';
  if (!secs.length) return cur + '<div class="empty">' + esc(groupLabel(group)) + ' changes (held 3 words above threshold) appear here</div>';
  return cur + '<div class="track">' + (secs[0].i > 0 ? '<span style="flex:' + secs[0].i + ';background:transparent"></span>' : '') + track + '</div>' +
    '<div class="tlab">' + tags.map((t) => '<span><i style="background:' + col(t) + '"></i>' + esc(pretty(t)) + '</span>').join('') + '</div>' +
    L(secs.slice(-40).reverse().map((s) => item(s.i, '<b>' + esc(s.label || pretty(s.tag)) + '</b>' + (s.prev ? ' <span class="note">from ' + esc(pretty(s.prev)) + '</span>' : ''), s.p)), '');
}
function panelIds() {
  const id = activeId(), p = PROFILES[id];
  const ids = p.panels.slice();
  if (S.vertical === 'auto' && id !== 'auto') ids.unshift('domain');
  return ids;
}
let panelKey = '';
const panelEls = {};
function renderPanels() {
  const host = $('panels'), ids = panelIds(), key = ids.join(',');
  if (key !== panelKey) {
    host.innerHTML = '';
    for (const k of Object.keys(panelEls)) delete panelEls[k];
    for (const id of ids) {
      const sec = document.createElement('section'); sec.className = 'card';
      const ttl = id.indexOf('phase:') === 0 ? groupLabel(id.slice(6)) + ' track' : PANELS[id].title;
      sec.innerHTML = '<h3><span>' + esc(ttl) + '</span></h3>';
      const body = document.createElement('div'); sec.appendChild(body); host.appendChild(sec);
      panelEls[id] = body;
    }
    panelKey = key;
  }
  for (const id of ids) {
    const el = panelEls[id]; if (!el) continue;
    let h;
    try { h = id.indexOf('phase:') === 0 ? phasePanel(id.slice(6)) : PANELS[id].render(); }
    catch (e) { h = '<div class="perr">this panel failed to render: ' + esc(e.message) + '</div>'; try { console.error(e); } catch (x) { /* ignore */ } }
    if (el._h !== h) { el.innerHTML = h; el._h = h; }
  }
}
$('panels').addEventListener('click', (e) => {
  const ex = e.target.closest && e.target.closest('[data-act="export"]'); if (ex) { send({ type: 'export' }); return; }
  const it = e.target.closest && e.target.closest('[data-j]'); if (it) jump(+it.dataset.j);
});

// ------------------------------------------------------------------ legend
function renderLegend() {
  const p = prof(), ents = p.ents || [];
  const cols = { person: 'var(--person)', team: 'var(--team)', organization: 'var(--org)', place: 'var(--place)', legal_ref: 'var(--legal)', work_product: 'var(--work)', event_name: 'var(--evname)', term: '#b9bdc4', game_entity: 'var(--work)', game_map: 'var(--place)' };
  const hasEv = p.events === null || p.events === undefined || (p.events && p.events.length);
  $('legend').innerHTML = '<span class="lg-h">Legend</span>' + ents.filter((e) => e !== 'term').map((e) => '<span><span class="w e e-' + e + '">' + esc(e.replace(/_/g, ' ')) + '</span></span>').join('') +
    '<span><span class="w rd rc-pii sev-high">PII</span> <span class="w rd rc-lang sev-medium">profanity</span> safety overlay (hover: category, p, source)</span>' +
    (hasEv ? '<span><span class="gi">\u25C6</span> domain event (margin; listed on the right)</span>' : '') +
    (p.phases && p.phases.length ? '<span><span class="sdiv mini"><span>phase / topic change</span></span></span>' : '') +
    '<span><span class="badge role">Role</span> shown only when confident (hover for the distribution)</span><span class="lg-i">grey = still being spoken</span>';
  void cols;
}
// ------------------------------------------------------------------ developer drawer
function logJson(m) {
  if ($('jP').checked) return;
  let o = null;
  if (m.type === 'update') {
    o = { type: 'update', from: m.from, n: m.n };
    if ($('jW').checked && m.words && m.words.length) o.words = m.words;
    if ($('jS').checked && m.tiers) { const t = {}; for (const k of Object.keys(m.tiers)) if (m.tiers[k].spans && m.tiers[k].spans.length) t[k] = m.tiers[k]; if (Object.keys(t).length) o.tiers = t; }
    if ($('jE').checked) { if (m.events && m.events.length) o.events = m.events; if (m.alerts && m.alerts.length) o.alerts = m.alerts; }
    if ($('jT').checked && m.enrich) { const e = m.enrich, x = { rev: e.rev, domain: e.domain }; if (e.turns && e.turns.upsert && e.turns.upsert.length) x.turns = e.turns; if (e.summary) x.summary = e.summary; o.enrich = x; }
    if ($('jR').checked && m.redact) o.redact = m.redact;
    if ($('jO').checked) { o.readouts = m.readouts; o.stats = m.stats; }
    if (Object.keys(o).length <= 3) return;
  } else if (m.type !== 'hello' || $('jO').checked) o = m;
  if (!o) return;
  S.json.unshift(JSON.stringify(o, null, 1).slice(0, 5000));
  if (S.json.length > 40) S.json.length = 40;
  S.jsonDirty = true;
}
function renderDev() { if ($('dev').hidden || !S.jsonDirty) return; $('json').textContent = S.json.join('\n\n') || 'waiting for a stream…'; S.jsonDirty = false; }
$('bDev').onclick = () => { $('dev').hidden = !$('dev').hidden; S.jsonDirty = true; renderDev(); };
$('bDevX').onclick = () => { $('dev').hidden = true; };
function samplePayload() {
  if (S.lastUpdate) return JSON.stringify(S.lastUpdate, null, 1);
  return JSON.stringify({ type: 'update', from: 12, n: 14, words: [{ w: 'call', f: false, t0: 4.1, t1: 4.3, spk: 0 }, { w: '[PHONE]', f: false, t0: 4.4, t1: 5.9, spk: 0, r: 'mask', rid: 3 }],
    tiers: { T0: { a0: 0, spans: [{ tag: 'safety.pii.phone_number', fam: 'pii', a: 13, b: 14, p: 0.91, thr: 0.3 }] } }, events: [], alerts: [],
    enrich: { schema: 2, rev: 1, vertical: 'contact_center', domain: { active: 'contact_center', conf: 1 }, turns: { n: 1, upsert: [{ id: 't0', seq: 0, a: 0, b: 14, open: true, readout: 'interim' }] } } }, null, 1);
}
$('bCopy').onclick = () => {
  const s = samplePayload();
  const ok = () => { $('copyNote').textContent = 'copied (' + s.length + ' chars)'; };
  try { navigator.clipboard.writeText(s).then(ok, () => fallbackCopy(s, ok)); } catch (e) { fallbackCopy(s, ok); }
};
function fallbackCopy(s, ok) { const t = document.createElement('textarea'); t.value = s; document.body.appendChild(t); t.select(); try { document.execCommand('copy'); ok(); } catch (e) { $('copyNote').textContent = 'copy failed'; } t.remove(); }

// ------------------------------------------------------------------ tabs
for (const b of document.querySelectorAll('#tabs button')) {
  b.onclick = () => {
    S.view = b.dataset.v;
    for (const x of document.querySelectorAll('#tabs button')) x.classList.toggle('on', x === b);
    $('live').classList.toggle('on', S.view === 'live');
    if ($('type')) $('type').classList.toggle('on', S.view === 'type');
    document.body.classList.toggle('vtype', S.view === 'type');
    document.dispatchEvent(new CustomEvent('arc:view', { detail: S.view }));             // type.js (Type mode) listens
  };
}

// ------------------------------------------------------------------ vertical / provider / scenarios
function applyVertical(v, fromUser) {
  if (!PROFILES[v]) v = 'auto';
  S.vertical = v; $('vertical').value = v;
  const p = PROFILES[v];
  if (!S.running) { $('rmode').value = p.redact === 'redact' || p.redact === 'redact+log' ? 'redact' : 'off'; S.policy = Object.assign({}, p.policy || {}); }
  panelKey = ''; tlCache.key = ''; evCache.key = '';
  if (S.running && fromUser) { send({ type: 'vertical', vertical: v }); }
  try { localStorage.setItem('arctag_vertical', v); } catch (e) { /* ignore */ }
  renderLegend(); renderAll(true);
}
for (const b of document.querySelectorAll('#viewSeg button')) {
  b.onclick = () => { S.redView = b.dataset.rv; for (const x of document.querySelectorAll('#viewSeg button')) x.classList.toggle('on', x === b); if (S.running) send({ type: 'redact', display: S.redView }); renderAll(true); };
}
$('privacy').onchange = () => { document.body.classList.toggle('privacy', $('privacy').checked); };
$('dbg').onchange = () => { S.debug = $('dbg').checked; document.body.classList.toggle('debug', S.debug); renderAll(true); };
function fillVerticals() {
  const cur = S.vertical;
  $('vertical').innerHTML = VERTS.map((v) => '<option value="' + v[0] + '">' + esc(v[1]) + '</option>').join('');
  $('vertical').value = PROFILES[cur] ? cur : 'auto';
}
fillVerticals();
$('vertical').onchange = () => applyVertical($('vertical').value, true);
function liveAvail(live, precs, prec) {            // hello.live: the Live tab needs DEEPGRAM_API_KEY on the server
  const on = !!(live && live.enabled);
  S.liveOn = on;
  for (const b of ['bTab', 'bMic']) { $(b).disabled = !on || S.running; $(b).title = on ? '' : 'live ASR is off on this server (set DEEPGRAM_API_KEY)'; }
  $('liveNote').hidden = on; $('liveNote').textContent = on ? '' : (live && live.note) || 'Live mic needs DEEPGRAM_API_KEY on the server.';
  if (Array.isArray(precs)) for (const o of $('prec').options) o.disabled = precs.indexOf(o.value) < 0;
  if (prec && !S.running && !S.precTouched) $('prec').value = prec;
}
$('prec').onchange = () => { S.precTouched = true; };

function baseConfig() {
  const p = PROFILES[S.vertical];
  return { provider: S.provider, model: 'nova-3', language: 'en', endpointing: +$('endp').value, diarize: true, precision: $('prec').value,
    vertical: S.vertical, domain: p.engine, tier: tierSel(), redact_mode: $('rmode').value, redact_display: S.redView, redact_policy: Object.keys(S.policy).length ? S.policy : undefined };
}
function resetView() {
  Object.assign(S, { words: [], rev: {}, spans: { T0: [], T1: [], T2: [] }, uttB: new Set(), v1turns: [], events: [], alerts: [], utts: new Map(), ro: null, stats: {},
    red: { counters: {}, log: [] }, asrTurns: [], tmap: new Map(), speakers: [], live: null, sections: [], state: null, summary: null, domain: null, enrichRev: 0,
    cc: {}, ccLead: [], banSeen: new Set(), spkIdx: {}, exp: new Set(), follow: true, lastUpdate: null, done: false, lastUpd: 0, resumeSent: false, redSpans: {}, lateSeen: {},
    lanes: { pts: [], tags: {}, ver: (S.lanes ? S.lanes.ver : 0) + 1 } });
  saveToken(null);
  tlCache.key = ''; evCache.key = ''; panelKey = '';
  $('hiban').hidden = true; renderAll(true);
}
function startSession(cfg) {
  if (!S.ws || S.ws.readyState !== 1) { msg('not connected to the server yet; wait for the connection banner to clear', true); return false; }
  resetView(); S.startedAt = Date.now(); S.awaitStart = true; send({ type: 'start', config: cfg }); setRunning(true); return true;
}
function setRunning(on) {
  S.running = on;
  for (const b of ['bTab', 'bMic']) $(b).disabled = on || !S.liveOn;
  $('bStop').disabled = !on;
}
function onStartedAudio() { S.pendingPlay = false; }
function seekAudio() { /* no audio player in this demo (no replay clips) */ }
async function audioGraph(stream, tab) {
  const ctx = new AudioContext({ latencyHint: 'interactive' });
  await ctx.audioWorklet.addModule('static/worklet.js');
  const src = ctx.createMediaStreamSource(stream);
  if (tab) {                                     // tab share: 16 kHz chunks + the bleep monitor (delayed tab audio) in one worklet
    const node = new AudioWorkletNode(ctx, 'tabmon', { numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [2], channelCount: 2, channelCountMode: 'clamped-max',
      processorOptions: { delay: bmDelay(), maxDelay: BM_MAX + 0.2, monitor: false } });
    node.port.onmessage = (e) => { const d = e.data; if (d && d.pcm) bmChunk(d); else if (d && d.t === 'acct') bmAcct(d); };
    node.connect(ctx.destination); src.connect(node);
    S.ctx = ctx; S.node = node; S.src = src; BM.node = node; BM.ratio = ctx.sampleRate / 16000;
    return;
  }
  const node = new AudioWorkletNode(ctx, 'down16k', { numberOfInputs: 1, numberOfOutputs: 1, channelCount: 2, channelCountMode: 'clamped-max' });
  node.port.onmessage = (e) => { if (S.ws && S.ws.readyState === 1 && S.running) S.ws.send(e.data); };
  const mute = ctx.createGain(); mute.gain.value = 0; node.connect(mute).connect(ctx.destination);
  src.connect(node);
  S.ctx = ctx; S.node = node; S.src = src;
}
// ------------------------------------------------------------------ bleep monitor (tab share only; audio only, the text redaction is unchanged)
// The shared tab is silenced (getDisplayMedia suppressLocalAudioPlayback) and played here through a delay line. Every word the server
// sends with r = "mask" (current redaction mode + policy) becomes a bleep region [t0 - pad, t1 + pad]: Deepgram word times are seconds
// of the audio this page sent, mapped to worklet frames via the sent chunks (a segment restarts after any unsent chunk). Regions only
// grow (masks never unmask). A word whose mask arrives after its start has played is counted as "too late" (its start was audible).
const BM_PAD = 0.04, BM_MIN = 0.5, BM_MAX = 3.0, BM_DEF = 1.0;
const BM = { node: null, ratio: 3, segs: [], sent: 0, lastSeq: -1, words: [], nkey: 0, ok: 0, late: 0, part: 0, byCat: {}, suppressed: false, track: null,
  supported: !!(navigator.mediaDevices && navigator.mediaDevices.getSupportedConstraints && navigator.mediaDevices.getSupportedConstraints().suppressLocalAudioPlayback) };
function bmPref(k, d) { try { const v = localStorage.getItem('arctag_bleep_' + k); return v == null ? d : v; } catch (e) { return d; } }
function bmSave(k, v) { try { localStorage.setItem('arctag_bleep_' + k, String(v)); } catch (e) { /* ignore */ } }
function bmDelay() { const v = +$('bleepDelay').value; return Math.max(BM_MIN, Math.min(BM_MAX, isFinite(v) ? v : BM_DEF)); }
function bmReset() { Object.assign(BM, { segs: [], sent: 0, lastSeq: -1, words: [], nkey: 0, ok: 0, late: 0, part: 0, byCat: {} }); bmRender(); }
function bmChunk(d) {
  if (!(S.ws && S.ws.readyState === 1 && S.running)) return;      // unsent chunk: the next sent one starts a new segment
  if (d.seq !== BM.lastSeq + 1 || !BM.segs.length) { BM.segs.push({ s0: BM.sent, f0: d.f0 }); if (BM.segs.length > 200) BM.segs.splice(0, BM.segs.length - 200); }
  S.ws.send(d.pcm); BM.sent += d.pcm.byteLength / 2; BM.lastSeq = d.seq;
}
function bmFrame(t) {                          // Deepgram stream seconds -> worklet input frame
  const s = Math.max(0, t) * 16000, G = BM.segs;
  if (!G.length) return null;
  let g = G[0];
  for (let i = G.length - 1; i >= 0; i--) if (G[i].s0 <= s) { g = G[i]; break; }
  return g.f0 + (s - g.s0) * BM.ratio;
}
function bmWords(m) {
  if (!$('bleepOn').checked || !BM.suppressed) return;
  const from = nz(m.from, 0), ws = m.words || [];
  for (let j = 0; j < ws.length; j++) {
    const w = ws[j];
    if (w.r !== 'mask' || w.t0 == null || w.t1 == null) continue;
    const a = w.t0 - BM_PAD, b = w.t1 + BM_PAD;
    let hit = null;                            // the same word seen before (interim revisions keep roughly the same times)
    for (let k = BM.words.length - 1; k >= 0 && k >= BM.words.length - 300; k--) {
      const x = BM.words[k], ov = Math.min(b, x.b) - Math.max(a, x.a);
      if (ov > 0.5 * Math.min(b - a, x.b - x.a)) { hit = x; break; }
    }
    if (hit && a >= hit.a - 1e-3 && b <= hit.b + 1e-3) continue;
    const fa = bmFrame(a), fb = bmFrame(b);
    if (fa == null) continue;
    const sp = S.redSpans[w.rid], cat = sp ? sp.cat : 'other';
    if (hit) { hit.a = Math.min(hit.a, a); hit.b = Math.max(hit.b, b); BM.node.port.postMessage({ t: 'bleep', a: fa, b: fb }); continue; }
    const key = ++BM.nkey; BM.words.push({ a, b, key, i: from + j });
    if (BM.words.length > 2000) BM.words.splice(0, 1000);
    BM.node.port.postMessage({ t: 'bleep', a: fa, b: fb, key, cat });
  }
}
function bmAcct(d) {
  BM.ok = d.ok; BM.late = d.late; BM.part = d.part;
  const c = BM.byCat[d.cat || 'other'] || (BM.byCat[d.cat || 'other'] = { ok: 0, late: 0 });
  if (d.st === 'ok') c.ok++; else c.late++;
  bmRender();
}
function bmMode() {                            // what the monitor output plays now
  if (!BM.node) return;
  if (!BM.suppressed) { BM.node.port.postMessage({ t: 'monitor', on: false }); return; }   // the tab plays itself: avoid double audio
  BM.node.port.postMessage({ t: 'delay', s: $('bleepOn').checked ? bmDelay() : 0 });          // monitor off: plain pass-through, no delay
  BM.node.port.postMessage({ t: 'monitor', on: true });
}
function bmRender() {
  const on = $('bleepOn').checked, live = !!BM.node;
  $('bleepDelayV').textContent = bmDelay().toFixed(1) + ' s';
  const late = BM.late + BM.part;
  $('bleepCt').innerHTML = !live ? '<span class="dim">tab share only</span>' : !on ? '<span class="dim">monitor off</span>' :
    '<b>' + BM.ok + '</b> bleeped · <b class="' + (late ? 'bad' : '') + '">' + late + '</b> too late';
  $('bleepCt').title = Object.keys(BM.byCat).length ? Object.entries(BM.byCat).map(([k, v]) => k + ': ' + v.ok + ' bleeped, ' + v.late + ' too late').join('\n') +
    (BM.part ? '\n(' + BM.part + ' of the late ones were bleeped after their first part had played)' : '') : 'words masked by the current redaction mode and policy are bleeped in the monitored tab audio';
  let warn = '';
  if (!BM.supported) warn = 'this browser cannot silence the shared tab (Chrome 109+ needed): tab audio also plays unredacted';
  else if (live && !BM.suppressed && on) warn = 'the shared source was not silenced (share a Chrome tab with “Share tab audio”): tab audio also plays unredacted, monitor muted';
  else if (live && on && $('rmode').value === 'off') warn = 'redaction is off: nothing is bleeped';
  $('bleepWarn').textContent = warn; $('bleepWarn').hidden = !warn;
}
async function bmToggle() {
  bmSave('on', $('bleepOn').checked ? 1 : 0);
  if (BM.node && $('bleepOn').checked && !BM.suppressed && BM.track && BM.track.applyConstraints) {    // shared without suppression: try now
    try { await BM.track.applyConstraints({ suppressLocalAudioPlayback: true }); BM.suppressed = (BM.track.getSettings() || {}).suppressLocalAudioPlayback === true; } catch (e) { /* unsupported */ }
  }
  bmMode(); bmRender();
}
$('bleepDelay').value = String(Math.max(BM_MIN, Math.min(BM_MAX, +bmPref('delay', BM_DEF) || BM_DEF)));
$('bleepOn').checked = bmPref('on', '1') !== '0';
$('bleepOn').onchange = bmToggle;
$('bleepDelay').oninput = bmRender;
$('bleepDelay').onchange = () => { bmSave('delay', bmDelay()); bmMode(); bmRender(); };
bmRender();
function liveCfg() { S.label = 'Live · nova-3'; S.needsAudio = true; return baseConfig(); }
async function startTab() {
  let stream;
  const bleep = $('bleepOn').checked;
  try {
    stream = await navigator.mediaDevices.getDisplayMedia({ video: true, audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, suppressLocalAudioPlayback: bleep },
      selfBrowserSurface: 'exclude', preferCurrentTab: false, systemAudio: 'include' });
  } catch (e) { msg('tab share cancelled: ' + e.message, true); return; }
  if (!stream.getAudioTracks().length) { stream.getTracks().forEach((t) => t.stop()); msg('no audio track: pick a Chrome tab and tick "Share tab audio"', true); return; }
  const tr = stream.getAudioTracks()[0];
  tr.onended = () => stopAll();
  S.media = stream; BM.track = tr;
  try { BM.suppressed = bleep && (tr.getSettings() || {}).suppressLocalAudioPlayback === true; } catch (e) { BM.suppressed = false; }
  try { await audioGraph(new MediaStream(stream.getAudioTracks()), true); } catch (e) { showErr('audio', e); stopCapture(); return; }
  bmReset(); bmMode();
  if (!startSession(liveCfg())) stopCapture();
  bmRender();
}
async function startMic() {
  let stream;
  try { stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 } }); } catch (e) { msg('microphone denied: ' + e.message, true); return; }
  S.media = stream;
  try { await audioGraph(stream); } catch (e) { showErr('audio', e); stopCapture(); return; }
  if (!startSession(liveCfg())) stopCapture();
}
function stopCapture() {
  if (S.audioEl && !S.audioEl.paused && !S.pendingPlay) { try { S.audioEl.pause(); } catch (e) { /* ignore */ } }
  S.audioLive = false;
  if (S.media) { S.media.getTracks().forEach((t) => t.stop()); S.media = null; }
  if (S.ctx) { try { S.ctx.close(); } catch (e) { /* ignore */ } S.ctx = null; }
  BM.node = null; BM.track = null; BM.suppressed = false; bmRender();
  setRunning(false);
}
function stopAll() { if (S.node) S.node.port.postMessage('flush'); send({ type: 'finish' }); setTimeout(stopCapture, 200); }
$('bTab').onclick = startTab; $('bMic').onclick = startMic;
$('bStop').onclick = () => { if (S.needsAudio) stopAll(); else { send({ type: 'stop' }); stopCapture(); } };
$('tier').onchange = () => { send({ type: 'set', tier: tierSel() }); evCache.key = ''; renderAll(true); };
function sendRedact() { if (S.running) send({ type: 'redact', mode: $('rmode').value, policy: S.policy }); renderAll(true); }
$('rmode').onchange = () => { sendRedact(); bmRender(); };
function download(data) {
  const blob = new Blob([JSON.stringify(data, null, 1)], { type: 'application/json' });
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob);
  a.download = 'arctag-redacted-' + new Date().toISOString().replace(/[:.]/g, '-') + '.json'; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}

(() => {
  const q = new URLSearchParams(location.search);
  let v = 'auto'; try { v = localStorage.getItem('arctag_vertical') || 'auto'; } catch (e) { /* ignore */ }
  try { const tk = sessionStorage.getItem('arctag_resume'); if (tk) { S.resume = tk; S.reloadResume = true; } } catch (e) { /* ignore */ }
  const qv = q.get('vertical') || q.get('preset');
  if (qv) v = PROFILES[qv] ? qv : LEGACY_VERT[qv] || v;
  applyVertical(v);
  { const b0 = document.querySelector('#tabs button[data-v="type"]'); if (b0) b0.click(); }      // Type is the default view
  if (q.get('view') === 'live' || q.get('view') === 'type') { const b = document.querySelector('#tabs button[data-v="' + q.get('view') + '"]'); if (b) setTimeout(() => b.click(), 0); }
  if (q.get('dev')) $('dev').hidden = false;
  connect();
  hbTimer = setInterval(heartbeat, 5000);
})();
