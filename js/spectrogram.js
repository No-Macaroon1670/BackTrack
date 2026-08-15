// Log-frequency (constant-Q-style) spectrogram for the editor underlay.
// STFT via a self-contained radix-2 FFT, then each output pixel row is
// mapped to a musical pitch through the *same* noteLayout the note overlay
// uses — so the harmonic stacks line up vertically with the detected-note
// bars, and an octave error is visible at a glance.

import { toMono, resample, fft, hann } from "./dsp.js";

const SPEC_SR = 16000;
const FFT_SIZE = 2048; // 128 ms window: ~2 bins/semitone at low vocal pitches
const HOP = 512; // 32 ms

/**
 * Renders the take's spectrogram into an offscreen canvas sized to match the
 * player canvas, using `layout` (from views.noteLayout with the editor's
 * fixed pitch range) so x=time and y=pitch align with the note overlay.
 * @returns {HTMLCanvasElement}
 */
export function computeSpectrogram(buffer, width, height, layout) {
  const x = resample(toMono(buffer), buffer.sampleRate, SPEC_SR);
  const frames = stft(x, FFT_SIZE, HOP);
  const nFrames = frames.length;
  const half = FFT_SIZE / 2;
  const binHz = SPEC_SR / FFT_SIZE;
  const secPerFrame = HOP / SPEC_SR;
  const duration = buffer.duration;

  // Precompute, per output row, the FFT bin band covering that row's
  // semitone (fixed because the pitch range is fixed).
  const rowLo = new Int32Array(height);
  const rowHi = new Int32Array(height);
  for (let y = 0; y < height; y++) {
    const midi = layout.fromY(y + 0.5);
    const fLo = 440 * 2 ** ((midi - 0.5 - 69) / 12);
    const fHi = 440 * 2 ** ((midi + 0.5 - 69) / 12);
    rowLo[y] = Math.max(1, Math.floor(fLo / binHz));
    rowHi[y] = Math.min(half - 1, Math.ceil(fHi / binHz));
  }

  let maxMag = 1e-9;
  for (const mag of frames) {
    for (let i = 1; i < half; i++) if (mag[i] > maxMag) maxMag = mag[i];
  }
  const norm = 1 / Math.log1p(maxMag * 8);

  const off = document.createElement("canvas");
  off.width = width;
  off.height = height;
  const ctx = off.getContext("2d");
  const img = ctx.createImageData(width, height);
  const d = img.data;

  for (let px = 0; px < width; px++) {
    const t = (px / width) * duration;
    const mag = frames[Math.max(0, Math.min(nFrames - 1, Math.round(t / secPerFrame)))];
    for (let y = 0; y < height; y++) {
      let m = 0;
      for (let b = rowLo[y]; b <= rowHi[y]; b++) if (mag[b] > m) m = mag[b];
      const v = Math.min(1, Math.log1p(m * 8) * norm) ** 0.85;
      const idx = (y * width + px) * 4;
      // Dark-navy → magenta ramp: low green so the green/red note bars pop.
      d[idx] = 255 * v ** 1.3;
      d[idx + 1] = 255 * v ** 2.2 * 0.7;
      d[idx + 2] = 255 * (0.15 + 0.85 * v ** 1.1);
      d[idx + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  return off;
}

function stft(x, fftSize, hop) {
  const win = hann(fftSize);
  const half = fftSize / 2;
  const nFrames = Math.max(1, Math.floor((x.length - fftSize) / hop) + 1);
  const re = new Float64Array(fftSize);
  const im = new Float64Array(fftSize);
  const frames = [];
  for (let f = 0; f < nFrames; f++) {
    const off = f * hop;
    for (let i = 0; i < fftSize; i++) {
      re[i] = (off + i < x.length ? x[off + i] : 0) * win[i];
      im[i] = 0;
    }
    fft(re, im);
    const mag = new Float32Array(half);
    for (let i = 0; i < half; i++) mag[i] = Math.hypot(re[i], im[i]);
    frames.push(mag);
  }
  return frames;
}

