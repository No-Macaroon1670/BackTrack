// Microphone capture. Records via MediaRecorder, then decodes the compressed
// take into an AudioBuffer for analysis and mixing.

export class Recorder {
  constructor() {
    this.stream = null;
    this.ctx = null;
    this.analyser = null;
    this.mediaRecorder = null;
    this.chunks = [];
    this.startedAt = 0;
  }

  async start() {
    // Defence in depth: never leak a previous capture's live tracks, whatever
    // the caller does. A stream left running keeps the mic indicator lit.
    if (this.stream) {
      try { this.stream.getTracks().forEach((t) => t.stop()); } catch { /* already gone */ }
      try { this.ctx?.close(); } catch { /* already closed */ }
      this.stream = null;
      this.ctx = null;
    }
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        // Voice-call processing mangles singing; ask the browser to leave it raw.
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
      },
    });
    this.ctx = new AudioContext();
    const source = this.ctx.createMediaStreamSource(this.stream);
    this.analyser = this.ctx.createAnalyser();
    this.analyser.fftSize = 2048;
    source.connect(this.analyser);

    this.chunks = [];
    this.mediaRecorder = new MediaRecorder(this.stream);
    this.mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) this.chunks.push(e.data);
    };
    this.mediaRecorder.start();
    this.startedAt = performance.now();
  }

  elapsed() {
    return (performance.now() - this.startedAt) / 1000;
  }

  /** Current input level, 0..1, for the live meter. */
  level() {
    if (!this.analyser) return 0;
    const data = new Float32Array(this.analyser.fftSize);
    this.analyser.getFloatTimeDomainData(data);
    let sum = 0;
    for (let i = 0; i < data.length; i++) sum += data[i] * data[i];
    return Math.min(1, Math.sqrt(sum / data.length) * 6);
  }

  /** Stops recording and resolves with the decoded AudioBuffer. */
  async stop() {
    const buffer = await new Promise((resolve, reject) => {
      this.mediaRecorder.onstop = async () => {
        try {
          const blob = new Blob(this.chunks, { type: this.mediaRecorder.mimeType });
          const arrayBuffer = await blob.arrayBuffer();
          resolve(await this.ctx.decodeAudioData(arrayBuffer));
        } catch (err) {
          reject(err);
        }
      };
      this.mediaRecorder.stop();
    });
    this.stream.getTracks().forEach((t) => t.stop());
    await this.ctx.close();
    this.stream = null;
    this.ctx = null;
    this.analyser = null;
    return buffer;
  }
}
