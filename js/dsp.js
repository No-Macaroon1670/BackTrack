// Shared DSP primitives used across the analysis pipeline: mono downmix,
// linear resampling, running statistics, and the YIN pitch estimator (shared
// by the fallback tracker in analyze.js and the neural-note cross-validation
// in notes.js).
//
// Analysis runs on a 16 kHz mono downmix; pitch resolution at vocal
// frequencies is unaffected and it keeps the O(frames * lags * window) YIN
// loop fast enough for the main thread.

export const ANALYSIS_SR = 16000;
export const FRAME_SIZE = 1024;
export const HOP = 256; // 16 ms
export const F_MIN = 70;
export const F_MAX = 800;
export const YIN_THRESHOLD = 0.15;
export const CLARITY_MIN = 0.5; // 1 - cmndf; below this a frame counts as unvoiced
export const RMS_GATE = 0.005;

/** Pitch class 0-11 of a MIDI number, handling negatives. */
export const pitchClass = (midi) => ((midi % 12) + 12) % 12;

export const midiToHz = (m) => 440 * 2 ** ((m - 69) / 12);

/** Hann window of length n. */
export function hann(n) {
  const w = new Float32Array(n);
  for (let i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (n - 1));
  return w;
}

/** In-place iterative radix-2 Cooley-Tukey FFT (n must be a power of two). */
export function fft(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wr = Math.cos(ang), wi = Math.sin(ang);
    const halfLen = len >> 1;
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < halfLen; k++) {
        const ar = re[i + k], ai = im[i + k];
        const br = re[i + k + halfLen], bi = im[i + k + halfLen];
        const tr = br * cr - bi * ci, ti = br * ci + bi * cr;
        re[i + k] = ar + tr; im[i + k] = ai + ti;
        re[i + k + halfLen] = ar - tr; im[i + k + halfLen] = ai - ti;
        const ncr = cr * wr - ci * wi;
        ci = cr * wi + ci * wr;
        cr = ncr;
      }
    }
  }
}

/** Median of an array (upper-middle element for even lengths). */
export function median(arr) {
  const s = [...arr].sort((a, b) => a - b);
  return s[s.length >> 1];
}

export function toMono(audioBuffer) {
  const out = new Float32Array(audioBuffer.length);
  for (let c = 0; c < audioBuffer.numberOfChannels; c++) {
    const ch = audioBuffer.getChannelData(c);
    for (let i = 0; i < ch.length; i++) out[i] += ch[i] / audioBuffer.numberOfChannels;
  }
  return out;
}

export function resample(input, srIn, srOut) {
  if (srIn === srOut) return input;
  const ratio = srIn / srOut;
  const outLen = Math.floor(input.length / ratio);
  const out = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const pos = i * ratio;
    const i0 = Math.floor(pos);
    const frac = pos - i0;
    out[i] = input[i0] * (1 - frac) + (input[Math.min(i0 + 1, input.length - 1)]) * frac;
  }
  return out;
}

/** RMS of the analysis signal over the time window [t0, t1] (seconds). */
export function winRms(x, t0, t1) {
  const a = Math.max(0, Math.round(t0 * ANALYSIS_SR));
  const b = Math.min(x.length, Math.round(t1 * ANALYSIS_SR));
  if (b - a < 8) return 0;
  let sum = 0;
  for (let i = a; i < b; i++) sum += x[i] * x[i];
  return Math.sqrt(sum / (b - a));
}

/** Reusable YIN working buffers + lag bounds for the configured pitch range. */
export function createYin() {
  const tauMax = Math.ceil(ANALYSIS_SR / F_MIN);
  const tauMin = Math.floor(ANALYSIS_SR / F_MAX);
  const W = FRAME_SIZE - tauMax; // integration window
  return { tauMin, tauMax, W, d: new Float32Array(tauMax + 1), cmndf: new Float32Array(tauMax + 1) };
}

/**
 * One YIN frame at sample offset `off`. `yin` is a context from createYin()
 * (its d/cmndf buffers are reused across calls to avoid per-frame allocation).
 * Returns { t, voiced, midi, clarity }.
 */
export function yinFrame(x, off, yin) {
  const { tauMin, tauMax, W, d, cmndf } = yin;
  const t = off / ANALYSIS_SR;

  let rms = 0;
  for (let j = 0; j < W; j++) rms += x[off + j] * x[off + j];
  rms = Math.sqrt(rms / W);
  if (rms < RMS_GATE) return { t, voiced: false, midi: 0, clarity: 0 };

  // Difference function (from tau=1 so the cumulative-mean normalization is correct).
  for (let tau = 1; tau <= tauMax; tau++) {
    let sum = 0;
    for (let j = 0; j < W; j++) {
      const diff = x[off + j] - x[off + j + tau];
      sum += diff * diff;
    }
    d[tau] = sum;
  }

  // Cumulative mean normalized difference.
  cmndf[0] = 1;
  let running = 0;
  for (let tau = 1; tau <= tauMax; tau++) {
    running += d[tau];
    cmndf[tau] = running > 0 ? (d[tau] * tau) / running : 1;
  }

  // First dip below threshold, refined to its local minimum; else global min.
  let bestTau = -1;
  for (let tau = tauMin; tau <= tauMax; tau++) {
    if (cmndf[tau] < YIN_THRESHOLD) {
      while (tau + 1 <= tauMax && cmndf[tau + 1] < cmndf[tau]) tau++;
      bestTau = tau;
      break;
    }
  }
  if (bestTau === -1) {
    let min = Infinity;
    for (let tau = tauMin; tau <= tauMax; tau++) {
      if (cmndf[tau] < min) { min = cmndf[tau]; bestTau = tau; }
    }
  }

  // Parabolic interpolation around the minimum.
  let tauEst = bestTau;
  if (bestTau > tauMin && bestTau < tauMax) {
    const a = cmndf[bestTau - 1], b = cmndf[bestTau], c = cmndf[bestTau + 1];
    const denom = a - 2 * b + c;
    if (Math.abs(denom) > 1e-9) tauEst = bestTau + (a - c) / (2 * denom);
  }

  const freq = ANALYSIS_SR / tauEst;
  const clarity = 1 - cmndf[bestTau];
  const voiced = clarity > CLARITY_MIN && freq >= F_MIN && freq <= F_MAX;
  const midiVal = voiced ? 69 + 12 * Math.log2(freq / 440) : 0;
  return { t, voiced, midi: midiVal, clarity };
}
