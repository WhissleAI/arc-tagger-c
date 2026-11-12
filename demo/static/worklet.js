// AudioWorklet: mono mix -> anti-aliased downsample to 16 kHz -> Int16 chunks of 100 ms (1600 samples) posted to the page.
class Down16k extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / 16000;          // e.g. 3 at 48 kHz
    this.acc = 0; this.sum = 0; this.cnt = 0; // box filter over each output sample's input span (= ratio input samples)
    this.buf = new Int16Array(1600); this.n = 0;
    this.port.onmessage = (e) => { if (e.data === 'flush') this.flush(); };
  }
  flush() {
    if (this.n > 0) { this.port.postMessage(this.buf.slice(0, this.n).buffer, []); this.n = 0; }
  }
  process(inputs) {
    const inp = inputs[0];
    if (!inp || inp.length === 0 || !inp[0]) return true;
    const L = inp[0].length, C = inp.length;
    for (let i = 0; i < L; i++) {
      let x = 0;
      for (let c = 0; c < C; c++) x += inp[c][i];
      x /= C;
      this.sum += x; this.cnt++; this.acc += 1;
      if (this.acc >= this.ratio) {
        this.acc -= this.ratio;
        let y = this.sum / this.cnt; this.sum = 0; this.cnt = 0;
        y = Math.max(-1, Math.min(1, y));
        this.buf[this.n++] = y < 0 ? y * 32768 : y * 32767;
        if (this.n === this.buf.length) {
          const out = this.buf; this.buf = new Int16Array(1600); this.n = 0;
          this.port.postMessage(out.buffer, [out.buffer]);
        }
      }
    }
    return true;
  }
}
registerProcessor('down16k', Down16k);

// AudioWorklet for a shared tab: the same 16 kHz PCM16 chunks as down16k (posted as {pcm, f0, seq}: f0 = input frame of the chunk's
// first sample, seq = chunk number, so the page can map Deepgram word times to frames), PLUS a monitor output: the tab audio at
// the native rate through a delay line, with bleep regions (input-frame coordinates [a, b)) replaced by a 1 kHz tone with 5 ms
// crossfade ramps. A region that arrives after its start has played is bleeped from the playhead on and reported as late.
class TabMon extends AudioWorkletProcessor {
  constructor(opts) {
    super();
    const o = (opts && opts.processorOptions) || {};
    this.ratio = sampleRate / 16000;
    this.acc = 0; this.sum = 0; this.cnt = 0; this.buf = new Int16Array(1600); this.n = 0; this.seq = 0; this.f0 = 0;
    this.N = Math.ceil(sampleRate * (o.maxDelay || 3.2)) + 512;
    this.ring = [new Float32Array(this.N), new Float32Array(this.N)];
    this.fr = 0;                                   // input frames processed so far (the frame clock of f0 / regions)
    this.delay = Math.round(sampleRate * (o.delay || 0.9)); this.nextDelay = null;
    this.on = o.monitor !== false; this.g = this.on ? 1 : 0; this.fade = 1 / Math.round(0.01 * sampleRate);
    this.regs = [];                                // merged, sorted [{a, b}]
    this.ramp = Math.round(0.005 * sampleRate); this.bridge = Math.round(0.08 * sampleRate);
    this.ph = 0; this.dph = 2 * Math.PI * 1000 / sampleRate; this.amp = o.amp || 0.2;
    this.keys = new Set(); this.nOk = 0; this.nLate = 0; this.nPart = 0;
    this.port.onmessage = (e) => this.onmsg(e.data);
  }
  onmsg(d) {
    if (d === 'flush') { this.flush(); return; }
    if (!d || typeof d !== 'object') return;
    if (d.t === 'delay') this.nextDelay = Math.max(0, Math.min(this.N - 256, Math.round(sampleRate * d.s)));
    else if (d.t === 'monitor') this.on = !!d.on;
    else if (d.t === 'bleep') this.addRegion(d);
  }
  addRegion(d) {
    const head = this.fr - this.delay;               // next source frame to be played
    let a = Math.floor(d.a), b = Math.ceil(d.b);
    if (!(b > a)) return;
    let st = 'ok';
    if (b <= head) st = 'late'; else if (a < head) { st = 'partial'; a = head; }
    if (st !== 'late') this.merge(a, b);
    if (d.key != null && !this.keys.has(d.key)) {       // count each word once (later extensions of the same word are not counted)
      this.keys.add(d.key);
      if (st === 'ok') this.nOk++; else if (st === 'late') this.nLate++; else this.nPart++;
      this.port.postMessage({ t: 'acct', key: d.key, cat: d.cat, st, lead: (head - Math.floor(d.a)) / sampleRate, ok: this.nOk, late: this.nLate, part: this.nPart });
    }
  }
  merge(a, b) {
    const R = this.regs; const out = []; let placed = false;
    for (const r of R) {
      if (r.b + this.bridge < a || b + this.bridge < r.a) { if (!placed && b + this.bridge < r.a) { out.push({ a, b }); placed = true; } out.push(r); }
      else { a = Math.min(a, r.a); b = Math.max(b, r.b); }
    }
    if (!placed) out.push({ a, b });
    out.sort((x, y) => x.a - y.a); this.regs = out;
  }
  flush() {
    if (this.n > 0) { const pcm = this.buf.slice(0, this.n).buffer; this.port.postMessage({ pcm, f0: this.f0, seq: this.seq++ }, [pcm]); this.n = 0; }
  }
  process(inputs, outputs) {
    const inp = inputs[0], out = outputs[0];
    const C = inp && inp.length ? inp.length : 0, L = (out && out[0] && out[0].length) || (C ? inp[0].length : 128);
    const N = this.N, r0 = this.ring[0], r1 = this.ring[1];
    if (this.nextDelay != null && this.g <= 0) { this.delay = this.nextDelay; this.nextDelay = null; }
    const want = this.on && this.nextDelay == null ? 1 : 0;
    while (this.regs.length && this.regs[0].b < this.fr - this.delay - N) this.regs.shift();
    let ri = 0;
    for (let i = 0; i < L; i++) {
      const f = this.fr + i;
      const xl = C ? inp[0][i] : 0, xr = C > 1 ? inp[1][i] : xl;
      // ---- 16 kHz path (identical to down16k)
      if (this.n === 0 && this.cnt === 0) this.f0 = f;
      const x = (xl + xr) * 0.5;
      this.sum += C > 1 ? x : xl; this.cnt++; this.acc += 1;
      if (this.acc >= this.ratio) {
        this.acc -= this.ratio;
        let y = this.sum / this.cnt; this.sum = 0; this.cnt = 0;
        y = Math.max(-1, Math.min(1, y));
        this.buf[this.n++] = y < 0 ? y * 32768 : y * 32767;
        if (this.n === this.buf.length) {
          const pcm = this.buf.buffer; this.port.postMessage({ pcm, f0: this.f0, seq: this.seq++ }, [pcm]);
          this.buf = new Int16Array(1600); this.n = 0;
        }
      }
      // ---- monitor: delay line + bleep
      const k = f % N; r0[k] = xl; r1[k] = xr;
      if (!out || !out.length) continue;
      const g = f - this.delay; let yl = 0, yr = 0;
      if (g >= 0) { const j = g % N; yl = r0[j]; yr = r1[j]; }
      while (ri < this.regs.length && this.regs[ri].b <= g) ri++;
      let w = 0;
      if (ri < this.regs.length) {
        const rg = this.regs[ri];
        if (g >= rg.a && g < rg.b) w = Math.min(1, (g - rg.a + 1) / this.ramp, (rg.b - g) / this.ramp);
      }
      if (w > 0) {
        const tone = this.amp * Math.sin(this.ph);
        yl = yl * (1 - w) + tone * w; yr = yr * (1 - w) + tone * w;
      }
      this.ph += this.dph; if (this.ph > 6.283185307179586) this.ph -= 6.283185307179586;
      if (this.g !== want) this.g = want > this.g ? Math.min(1, this.g + this.fade) : Math.max(0, this.g - this.fade);
      out[0][i] = yl * this.g; if (out.length > 1) out[1][i] = yr * this.g;
    }
    this.fr += L;
    return true;
  }
}
registerProcessor('tabmon', TabMon);
