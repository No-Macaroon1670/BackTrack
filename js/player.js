// Multi-track playback engine: named tracks (vocal, backing, harmonics, …)
// through independent gain nodes. DOM-free — the UI passes gain values in
// and reads position out, so this stays testable and app.js remains the
// only module that touches element ids.

export class Player {
  constructor() {
    this.ctx = null;
    this.gains = new Map(); // name -> GainNode (persist across plays)
    this.sources = new Map(); // name -> BufferSource (live play only)
    this.startedAt = 0;
    this.duration = 0;
    this.playing = false;
  }

  ensure() {
    if (!this.ctx) this.ctx = new AudioContext();
  }

  gainNode(name) {
    this.ensure();
    if (!this.gains.has(name)) {
      const g = this.ctx.createGain();
      g.connect(this.ctx.destination);
      this.gains.set(name, g);
    }
    return this.gains.get(name);
  }

  /**
   * @param {Object<string, {buffer: AudioBuffer, gain: number}>} tracks
   */
  async play(tracks, { onEnded = () => {} } = {}) {
    this.ensure();
    await this.ctx.resume();
    this.stop();

    const t = this.ctx.currentTime + 0.05;
    let longestSrc = null;
    this.duration = 0;
    for (const [name, { buffer, gain }] of Object.entries(tracks)) {
      const g = this.gainNode(name);
      g.gain.value = gain;
      const src = this.ctx.createBufferSource();
      src.buffer = buffer;
      src.connect(g);
      src.start(t);
      this.sources.set(name, src);
      if (buffer.duration > this.duration) {
        this.duration = buffer.duration;
        longestSrc = src;
      }
    }
    if (longestSrc) {
      longestSrc.onended = () => {
        if (this.playing) {
          this.playing = false;
          onEnded();
        }
      };
    }
    this.startedAt = t;
    this.playing = true;
  }

  /** Joins a new track mid-playback at the current position. */
  addTrack(name, buffer, gain) {
    if (!this.playing) return;
    const offset = this.position();
    if (offset >= buffer.duration) return;
    const g = this.gainNode(name);
    g.gain.value = gain;
    const src = this.ctx.createBufferSource();
    src.buffer = buffer;
    src.connect(g);
    src.start(this.ctx.currentTime, offset);
    this.sources.set(name, src);
  }

  hasTrack(name) {
    return this.sources.has(name);
  }

  setGain(name, v) {
    if (this.gains.has(name)) this.gains.get(name).gain.value = v;
  }

  stop() {
    this.playing = false;
    for (const s of this.sources.values()) {
      try { s.onended = null; s.stop(); } catch { /* already stopped */ }
    }
    this.sources.clear();
  }

  /** Seconds into the tracks' shared timeline (0 when stopped). */
  position() {
    if (!this.playing || !this.ctx) return 0;
    return Math.max(0, this.ctx.currentTime - this.startedAt);
  }
}
