// Pitch-domain analysis: tuning-offset estimation and Krumhansl-Schmuckler
// key detection. Both operate on the detected note list (pitch class weighted
// by duration), independent of the raw audio.

import { pitchClass } from "./dsp.js";

// Krumhansl-Kessler key profiles.
const MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88];
const MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17];

/**
 * Duration-weighted circular mean of each note's deviation from the nearest
 * equal-tempered pitch, in cents. Circular so a chorus of +45¢ and -48¢
 * deviations (straddling the wrap) doesn't cancel to a bogus 0.
 */
export function estimateTuning(notes) {
  let sinSum = 0, cosSum = 0;
  for (const n of notes) {
    const frac = n.midiFloat - Math.round(n.midiFloat); // -0.5..0.5 semitones
    const angle = 2 * Math.PI * frac;
    sinSum += n.dur * Math.sin(angle);
    cosSum += n.dur * Math.cos(angle);
  }
  const offsetSemis = Math.atan2(sinSum, cosSum) / (2 * Math.PI);
  return Math.round(offsetSemis * 100);
}

export function detectKey(notes) {
  const hist = new Array(12).fill(0);
  for (const n of notes) hist[pitchClass(n.midi)] += n.dur;

  let best = { score: -Infinity, tonic: 0, mode: "major" };
  for (const [mode, profile] of [["major", MAJOR_PROFILE], ["minor", MINOR_PROFILE]]) {
    for (let tonic = 0; tonic < 12; tonic++) {
      const score = correlation(hist, profile, tonic);
      if (score > best.score) best = { score, tonic, mode };
    }
  }
  return { tonic: best.tonic, mode: best.mode };
}

function correlation(hist, profile, rotation) {
  const n = 12;
  let mh = 0, mp = 0;
  for (let i = 0; i < n; i++) { mh += hist[i]; mp += profile[i]; }
  mh /= n; mp /= n;
  let num = 0, dh = 0, dp = 0;
  for (let i = 0; i < n; i++) {
    const h = hist[(i + rotation) % 12] - mh;
    const p = profile[i] - mp;
    num += h * p;
    dh += h * h;
    dp += p * p;
  }
  return dh > 0 && dp > 0 ? num / Math.sqrt(dh * dp) : -Infinity;
}
