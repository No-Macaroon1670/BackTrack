// Backing-track synthesis. Schedules the arrangement's events into an
// OfflineAudioContext: detuned-saw pads through a lowpass, triangle bass,
// synthesized kick/snare/hat, a generated-IR send reverb, and a master
// compressor. Also handles final mixdown and WAV encoding.

import { nearestSample } from "./sampler.js";

const midiToFreq = (m) => 440 * Math.pow(2, (m - 69) / 12);

const STYLE_SOUND = {
  pop: { padCutoff: 1400, padAttack: 0.04, padLevel: 0.085, guitarLevel: 0.2, stringLevel: 0.055, stringAttack: 0.25, hatDecay: 0.04, reverb: 0.16 },
  ballad: { padCutoff: 2200, padAttack: 0.01, padLevel: 0.1, guitarLevel: 0.26, stringLevel: 0.09, stringAttack: 0.4, hatDecay: 0.05, reverb: 0.24 },
  lofi: { padCutoff: 900, padAttack: 0.08, padLevel: 0.09, guitarLevel: 0.18, stringLevel: 0.07, stringAttack: 0.3, hatDecay: 0.03, reverb: 0.14 },
  folk: { padCutoff: 2400, padAttack: 0.005, padLevel: 0.075, guitarLevel: 0.24, stringLevel: 0.075, stringAttack: 0.2, hatDecay: 0.09, reverb: 0.12 },
};

/**
 * Renders the backing track to an AudioBuffer at the given sample rate
 * (use the vocal's rate so mixing is sample-aligned). Pass `samples`
 * ({ chord, bass } maps from sampler.js) to play recorded instruments
 * through the same event/velocity pipeline; null renders the synth voices.
 */
export async function renderBacking(arr, sampleRate, samples = null, timbre = null) {
  const sound = STYLE_SOUND[arr.style];
  const length = Math.ceil(arr.totalDur * sampleRate);
  const ctx = new OfflineAudioContext(2, length, sampleRate);

  // Master chain: sum -> compressor -> output.
  const master = ctx.createGain();
  master.gain.value = 0.9;
  const comp = ctx.createDynamicsCompressor();
  comp.threshold.value = -14;
  comp.ratio.value = 3;
  comp.attack.value = 0.004;
  comp.release.value = 0.25;
  master.connect(comp);
  comp.connect(ctx.destination);

  // Send reverb with a synthetic exponential-decay noise impulse.
  const verb = ctx.createConvolver();
  verb.buffer = makeImpulse(ctx, 1.6, 2.5);
  const verbGain = ctx.createGain();
  verbGain.gain.value = sound.reverb;
  verb.connect(verbGain);
  verbGain.connect(master);

  // Detune every pitched instrument to the singer's tuning offset so the
  // backing is in tune with the voice, not with A=440.
  const detune = arr.tuningCents || 0;

  // Per-role routing. `samples` is the per-role map from loadVoiceSamples
  // ({ chords, harmony, bass, melody }); the legacy { chord, bass } shape
  // still works. Roles without a font render through synth/wave paths.
  const v = arr.voices ?? { chords: arr.instrument ?? "keys", harmony: "follow", bass: "acoustic", melody: "off" };
  const resolve = (r) => (r === "follow" ? v.chords : r);
  const S = samples
    ? (samples.chords !== undefined || samples.harmony !== undefined
      ? samples
      : { chords: samples.chord ?? null, harmony: samples.chord ?? null, bass: samples.bass ?? null, melody: null })
    : { chords: null, harmony: null, bass: null, melody: null };

  // Voice-timbre synth: an oscillator built from the singer's own measured
  // overtone recipe (see notes.voiceTimbre), for any role set to "voice".
  const needsWave = [v.chords, resolve(v.harmony), resolve(v.melody)].includes("voice");
  const voiceWave = needsWave && timbre
    ? ctx.createPeriodicWave(new Float32Array(timbre.length), timbre)
    : null;

  // Chords
  if (v.chords === "voice" && voiceWave) {
    for (const pad of arr.pads) scheduleVoicePad(ctx, master, verb, pad, voiceWave, detune);
  } else if (S.chords) {
    for (const pad of arr.pads) scheduleSampledChord(ctx, master, verb, pad, S.chords, detune, v.chords);
  } else if (v.chords === "guitar") {
    const pluckCache = new Map();
    for (const pad of arr.pads) scheduleGuitar(ctx, master, verb, pad, sound, pluckCache, detune);
  } else if (v.chords === "violin") {
    for (const pad of arr.pads) scheduleStrings(ctx, master, verb, pad, sound, detune);
  } else {
    for (const pad of arr.pads) schedulePad(ctx, master, verb, pad, sound, detune);
  }

  // Bass
  if (v.bass !== "synth" && S.bass) {
    for (const note of arr.bass) scheduleSampledBass(ctx, master, note, S.bass, detune);
  } else {
    for (const note of arr.bass) scheduleBass(ctx, master, note, detune);
  }

  // Harmony voice: shadows the sung melody, so it plays with the singer's
  // own timing. Voice timbre when chosen; else its sampled instrument,
  // softly; else a mellow triangle so it reads as a voice, not a lead.
  const hRole = resolve(v.harmony);
  for (const h of arr.harmony ?? []) {
    if (hRole === "voice" && voiceWave) scheduleVoiceHarmony(ctx, master, verb, h, voiceWave, detune);
    else if (S.harmony) scheduleSampledHarmony(ctx, master, verb, h, S.harmony, detune);
    else scheduleSynthHarmony(ctx, master, verb, h, detune);
  }

  // Melody double: an instrument playing the sung line, louder than harmony.
  const mRole = resolve(v.melody);
  for (const m of arr.melody ?? []) {
    if (mRole === "voice" && voiceWave) scheduleVoiceHarmony(ctx, master, verb, m, voiceWave, detune, 0.2);
    else if (S.melody) scheduleSampledHarmony(ctx, master, verb, m, S.melody, detune, 0.45);
    else scheduleSynthHarmony(ctx, master, verb, m, detune, 0.17);
  }
  for (const hit of arr.drums) {
    const v = hit.vel ?? 1;
    if (hit.type === "kick") scheduleKick(ctx, master, hit.t, v);
    else if (hit.type === "snare") scheduleSnare(ctx, master, verb, hit.t, v);
    else if (hit.type === "hat") scheduleHat(ctx, master, hit.t, sound.hatDecay, v);
    else if (hit.type === "crash") scheduleCrash(ctx, master, verb, hit.t, v);
  }

  return ctx.startRendering();
}

/**
 * Recorded-sample chord voice: one buffer source per chord tone, detuned to
 * the exact pitch and the singer's tuning, gated at the event's duration.
 * Strings get a longer release and a wetter reverb send; piano and guitar
 * ring naturally and stay drier.
 */
function scheduleSampledChord(ctx, master, verb, pad, font, detune, instrument) {
  const sustained = instrument === "violin";
  const level = (sustained ? 0.38 : 0.5) * (pad.vel ?? 1) * (font.gain ?? 1);
  const release = sustained ? 0.25 : 0.12;
  const send = ctx.createGain();
  send.gain.value = sustained ? 0.9 : 0.35;
  send.connect(verb);
  pad.midis.forEach((midi, vi) => {
    const s = nearestSample(font, midi);
    if (!s) return;
    const t = pad.t + vi * (pad.strum || 0);
    const src = ctx.createBufferSource();
    src.buffer = s.buffer;
    src.detune.value = (midi - s.midi) * 100 + detune;
    const g = ctx.createGain();
    g.gain.setValueAtTime(level, t);
    g.gain.setValueAtTime(level, t + pad.dur);
    g.gain.setTargetAtTime(0, t + pad.dur, release);
    src.connect(g);
    g.connect(master);
    g.connect(send);
    src.start(t);
    src.stop(t + pad.dur + release * 6);
  });
}

/**
 * Chord pad rendered with the singer's measured overtone recipe: two
 * slightly detuned wave oscillators per chord tone with a slow choir-like
 * attack and delayed-onset vibrato — a choir of the singer.
 */
function scheduleVoicePad(ctx, master, verb, pad, wave, detune) {
  const level = 0.075 * (pad.vel ?? 1);
  const send = ctx.createGain();
  send.gain.value = 1.1; // voices sit in a hall
  send.connect(verb);
  pad.midis.forEach((midi, vi) => {
    const t = pad.t + vi * (pad.strum || 0);
    const gain = ctx.createGain();

    const lfo = ctx.createOscillator();
    lfo.frequency.value = 4.8 + vi * 0.3;
    const vibDepth = ctx.createGain();
    vibDepth.gain.setValueAtTime(0, t);
    vibDepth.gain.setValueAtTime(0, t + 0.3);
    vibDepth.gain.linearRampToValueAtTime(9, t + 0.8); // cents
    lfo.connect(vibDepth);
    lfo.start(t);
    lfo.stop(t + pad.dur + 0.8);

    for (const cents of [-5, 5]) {
      const osc = ctx.createOscillator();
      osc.setPeriodicWave(wave);
      osc.frequency.value = midiToFreq(midi);
      osc.detune.value = cents + detune;
      vibDepth.connect(osc.detune);
      osc.connect(gain);
      osc.start(t);
      osc.stop(t + pad.dur + 0.8);
    }
    gain.connect(master);
    gain.connect(send);
    gain.gain.setValueAtTime(0, t);
    gain.gain.linearRampToValueAtTime(level, t + 0.15);
    gain.gain.setValueAtTime(level, Math.max(t + 0.15, t + pad.dur - 0.1));
    gain.gain.setTargetAtTime(0, t + pad.dur, 0.2);
  });
}

function scheduleVoiceHarmony(ctx, master, verb, h, wave, detune, base = 0.13) {
  const osc = ctx.createOscillator();
  osc.setPeriodicWave(wave);
  osc.frequency.value = midiToFreq(h.midi);
  osc.detune.value = detune;
  const lfo = ctx.createOscillator();
  lfo.frequency.value = 5.3;
  const vib = ctx.createGain();
  vib.gain.setValueAtTime(0, h.t);
  vib.gain.setValueAtTime(0, h.t + 0.25);
  vib.gain.linearRampToValueAtTime(10, h.t + 0.7);
  lfo.connect(vib);
  vib.connect(osc.detune);
  const g = ctx.createGain();
  const level = base * (h.vel ?? 1);
  g.gain.setValueAtTime(0, h.t);
  g.gain.linearRampToValueAtTime(level, h.t + 0.05);
  g.gain.setValueAtTime(level, Math.max(h.t + 0.05, h.t + h.dur - 0.06));
  g.gain.setTargetAtTime(0, h.t + h.dur - 0.03, 0.08);
  osc.connect(g);
  g.connect(master);
  g.connect(verb);
  osc.start(h.t);
  osc.stop(h.t + h.dur + 0.5);
  lfo.start(h.t);
  lfo.stop(h.t + h.dur + 0.5);
}

function scheduleSampledHarmony(ctx, master, verb, h, font, detune, base = 0.3) {
  const s = nearestSample(font, h.midi);
  if (!s) return;
  const src = ctx.createBufferSource();
  src.buffer = s.buffer;
  src.detune.value = (h.midi - s.midi) * 100 + detune;
  const level = base * (h.vel ?? 1) * (font.gain ?? 1);
  const g = ctx.createGain();
  g.gain.setValueAtTime(0, h.t);
  g.gain.linearRampToValueAtTime(level, h.t + 0.02);
  g.gain.setValueAtTime(level, h.t + h.dur);
  g.gain.setTargetAtTime(0, h.t + h.dur, 0.15);
  const send = ctx.createGain();
  send.gain.value = 0.6;
  src.connect(g);
  g.connect(master);
  g.connect(send);
  send.connect(verb);
  src.start(h.t);
  src.stop(h.t + h.dur + 1);
}

function scheduleSynthHarmony(ctx, master, verb, h, detune, base = 0.11) {
  const osc = ctx.createOscillator();
  osc.type = "triangle";
  osc.frequency.value = midiToFreq(h.midi);
  osc.detune.value = detune;
  const lp = ctx.createBiquadFilter();
  lp.type = "lowpass";
  lp.frequency.value = 1800;
  const g = ctx.createGain();
  const level = base * (h.vel ?? 1);
  g.gain.setValueAtTime(0, h.t);
  g.gain.linearRampToValueAtTime(level, h.t + 0.03);
  g.gain.setValueAtTime(level, Math.max(h.t + 0.03, h.t + h.dur - 0.05));
  g.gain.setTargetAtTime(0, h.t + h.dur - 0.02, 0.06);
  osc.connect(lp);
  lp.connect(g);
  g.connect(master);
  g.connect(verb);
  osc.start(h.t);
  osc.stop(h.t + h.dur + 0.4);
}

function scheduleSampledBass(ctx, master, note, font, detune) {
  const s = nearestSample(font, note.midi);
  if (!s) return;
  const src = ctx.createBufferSource();
  src.buffer = s.buffer;
  src.detune.value = (note.midi - s.midi) * 100 + detune;
  const level = 0.68 * (note.vel ?? 1) * (font.gain ?? 1);
  const g = ctx.createGain();
  g.gain.setValueAtTime(level, note.t);
  g.gain.setValueAtTime(level, note.t + note.dur);
  g.gain.setTargetAtTime(0, note.t + note.dur, 0.08);
  src.connect(g);
  g.connect(master);
  src.start(note.t);
  src.stop(note.t + note.dur + 0.5);
}

function makeImpulse(ctx, seconds, decay) {
  const len = Math.ceil(seconds * ctx.sampleRate);
  const buf = ctx.createBuffer(2, len, ctx.sampleRate);
  for (let c = 0; c < 2; c++) {
    const data = buf.getChannelData(c);
    for (let i = 0; i < len; i++) {
      data[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / len, decay);
    }
  }
  return buf;
}

function schedulePad(ctx, master, verb, pad, sound, detune = 0) {
  pad.midis.forEach((midi, vi) => {
    const t = pad.t + vi * (pad.strum || 0);
    const dur = pad.dur;
    const gain = ctx.createGain();
    const filter = ctx.createBiquadFilter();
    filter.type = "lowpass";
    filter.frequency.value = sound.padCutoff;
    filter.Q.value = 0.5;

    // Two slightly detuned saws per voice for width.
    for (const cents of [-5, 5]) {
      const osc = ctx.createOscillator();
      osc.type = "sawtooth";
      osc.frequency.value = midiToFreq(midi);
      osc.detune.value = cents + detune;
      osc.connect(filter);
      osc.start(t);
      osc.stop(t + dur + 0.6);
    }
    filter.connect(gain);
    gain.connect(master);
    gain.connect(verb);

    const level = sound.padLevel * (pad.vel ?? 1);
    gain.gain.setValueAtTime(0, t);
    gain.gain.linearRampToValueAtTime(level, t + sound.padAttack);
    gain.gain.setValueAtTime(level, Math.max(t + sound.padAttack, t + dur - 0.05));
    gain.gain.setTargetAtTime(0, t + dur, 0.12);
  });
}

/**
 * Bowed strings: detuned saws through a lowpass plus a body-resonance peak,
 * with a slow bow attack and delayed-onset vibrato — the two cues that read
 * as "bowed" rather than "synth pad".
 */
function scheduleStrings(ctx, master, verb, pad, sound, detune = 0) {
  pad.midis.forEach((midi, vi) => {
    const t = pad.t;
    const dur = pad.dur;
    const gain = ctx.createGain();
    const lp = ctx.createBiquadFilter();
    lp.type = "lowpass";
    lp.frequency.value = 2800;
    lp.Q.value = 0.7;
    const body = ctx.createBiquadFilter();
    body.type = "peaking";
    body.frequency.value = 1100;
    body.Q.value = 1.5;
    body.gain.value = 6;

    // Delayed-onset vibrato, slightly different rate per voice for ensemble.
    const lfo = ctx.createOscillator();
    lfo.frequency.value = 5.2 + vi * 0.35;
    const vibDepth = ctx.createGain();
    vibDepth.gain.setValueAtTime(0, t);
    vibDepth.gain.setValueAtTime(0, t + 0.35);
    vibDepth.gain.linearRampToValueAtTime(12, t + 0.9); // cents
    lfo.connect(vibDepth);
    lfo.start(t);
    lfo.stop(t + dur + 0.8);

    for (const cents of [-6, 6]) {
      const osc = ctx.createOscillator();
      osc.type = "sawtooth";
      osc.frequency.value = midiToFreq(midi);
      osc.detune.value = cents + detune;
      vibDepth.connect(osc.detune);
      osc.connect(lp);
      osc.start(t);
      osc.stop(t + dur + 0.8);
    }
    lp.connect(body);
    body.connect(gain);
    gain.connect(master);
    // Strings sit in a hall: stronger reverb send than the other instruments.
    const send = ctx.createGain();
    send.gain.value = 1.6;
    gain.connect(send);
    send.connect(verb);

    const level = sound.stringLevel * (pad.vel ?? 1);
    gain.gain.setValueAtTime(0, t);
    gain.gain.linearRampToValueAtTime(level, t + sound.stringAttack);
    gain.gain.setValueAtTime(level, Math.max(t + sound.stringAttack, t + dur - 0.1));
    gain.gain.setTargetAtTime(0, t + dur, 0.18);
  });
}

/**
 * Karplus-Strong plucked string, rendered into a cached AudioBuffer.
 * The damping exponent is normalized by frequency so low and high strings
 * decay at a similar musical rate.
 */
function karplusPluck(ctx, midi, cache, detune = 0) {
  if (cache.has(midi)) return cache.get(midi);
  const sr = ctx.sampleRate;
  const freq = midiToFreq(midi) * Math.pow(2, detune / 1200);
  const seconds = 2.0;
  const N = Math.round(sr / freq);
  const len = Math.ceil(seconds * sr);
  const out = new Float32Array(len);

  // Excitation: noise burst, lightly lowpassed for a warmer (less wiry) attack.
  const ring = new Float32Array(N);
  let prev = 0;
  for (let i = 0; i < N; i++) {
    const n = Math.random() * 2 - 1;
    ring[i] = 0.5 * n + 0.5 * prev;
    prev = ring[i];
  }

  const damp = Math.pow(0.4, 1 / freq); // ~0.4x amplitude per second for any pitch
  let idx = 0;
  for (let i = 0; i < len; i++) {
    const cur = ring[idx];
    const next = ring[(idx + 1) % N];
    out[i] = cur;
    ring[idx] = damp * 0.5 * (cur + next);
    idx = (idx + 1) % N;
  }

  const buf = ctx.createBuffer(1, len, sr);
  buf.copyToChannel(out, 0);
  cache.set(midi, buf);
  return buf;
}

function scheduleGuitar(ctx, master, verb, pad, sound, cache, detune = 0) {
  const delay = pad.strum || 0.012;
  pad.midis.forEach((midi, i) => {
    const t = pad.t + i * delay; // midis are pre-ordered for down/up strums
    const src = ctx.createBufferSource();
    src.buffer = karplusPluck(ctx, midi, cache, detune);
    const gain = ctx.createGain();
    const level = sound.guitarLevel * (pad.vel ?? 1);
    gain.gain.setValueAtTime(level, t);
    gain.gain.setValueAtTime(level, t + pad.dur);
    gain.gain.setTargetAtTime(0, t + pad.dur, 0.07);
    src.connect(gain);
    gain.connect(master);
    gain.connect(verb);
    src.start(t);
    src.stop(t + pad.dur + 0.45);
  });
}

function scheduleBass(ctx, master, note, detune = 0) {
  const t = note.t;
  const osc = ctx.createOscillator();
  osc.type = "triangle";
  osc.frequency.value = midiToFreq(note.midi);
  osc.detune.value = detune;
  const filter = ctx.createBiquadFilter();
  filter.type = "lowpass";
  filter.frequency.value = 550;
  const gain = ctx.createGain();
  osc.connect(filter);
  filter.connect(gain);
  gain.connect(master);

  const level = 0.26 * (note.vel ?? 1);
  gain.gain.setValueAtTime(0, t);
  gain.gain.linearRampToValueAtTime(level, t + 0.012);
  gain.gain.setValueAtTime(level, Math.max(t + 0.012, t + note.dur - 0.04));
  gain.gain.setTargetAtTime(0, t + note.dur, 0.05);
  osc.start(t);
  osc.stop(t + note.dur + 0.4);
}

function scheduleKick(ctx, master, t, vel = 1) {
  const osc = ctx.createOscillator();
  osc.type = "sine";
  osc.frequency.setValueAtTime(120, t);
  osc.frequency.exponentialRampToValueAtTime(45, t + 0.12);
  const gain = ctx.createGain();
  gain.gain.setValueAtTime(0.55 * vel, t);
  gain.gain.exponentialRampToValueAtTime(0.001, t + 0.25);
  osc.connect(gain);
  gain.connect(master);
  osc.start(t);
  osc.stop(t + 0.3);
}

function scheduleSnare(ctx, master, verb, t, vel = 1) {
  // Noise body.
  const noise = noiseSource(ctx, 0.25);
  const bp = ctx.createBiquadFilter();
  bp.type = "bandpass";
  bp.frequency.value = 1800;
  bp.Q.value = 0.8;
  const nGain = ctx.createGain();
  nGain.gain.setValueAtTime(0.32 * vel, t);
  nGain.gain.exponentialRampToValueAtTime(0.001, t + 0.18);
  noise.connect(bp);
  bp.connect(nGain);
  nGain.connect(master);
  nGain.connect(verb);
  noise.start(t);

  // Tonal thump.
  const osc = ctx.createOscillator();
  osc.type = "sine";
  osc.frequency.value = 185;
  const oGain = ctx.createGain();
  oGain.gain.setValueAtTime(0.18 * vel, t);
  oGain.gain.exponentialRampToValueAtTime(0.001, t + 0.08);
  osc.connect(oGain);
  oGain.connect(master);
  osc.start(t);
  osc.stop(t + 0.1);
}

function scheduleHat(ctx, master, t, decay, vel = 1) {
  const noise = noiseSource(ctx, decay + 0.05);
  const hp = ctx.createBiquadFilter();
  hp.type = "highpass";
  hp.frequency.value = 7000;
  const gain = ctx.createGain();
  gain.gain.setValueAtTime(0.12 * vel, t);
  gain.gain.exponentialRampToValueAtTime(0.001, t + decay);
  noise.connect(hp);
  hp.connect(gain);
  gain.connect(master);
  noise.start(t);
}

/** Crash-ish cymbal: bright noise with a long decay, into the send reverb. */
function scheduleCrash(ctx, master, verb, t, vel = 1) {
  const noise = noiseSource(ctx, 1.1);
  const hp = ctx.createBiquadFilter();
  hp.type = "highpass";
  hp.frequency.value = 5000;
  const gain = ctx.createGain();
  gain.gain.setValueAtTime(0.16 * vel, t);
  gain.gain.exponentialRampToValueAtTime(0.001, t + 1.0);
  noise.connect(hp);
  hp.connect(gain);
  gain.connect(master);
  gain.connect(verb);
  noise.start(t);
}

function noiseSource(ctx, seconds) {
  const len = Math.ceil(seconds * ctx.sampleRate);
  const buf = ctx.createBuffer(1, len, ctx.sampleRate);
  const data = buf.getChannelData(0);
  for (let i = 0; i < len; i++) data[i] = Math.random() * 2 - 1;
  const src = ctx.createBufferSource();
  src.buffer = buf;
  return src;
}

/**
 * Synthesizes an audible melody line from parsed-MIDI notes, standing in
 * for the vocal track: a soft triangle lead with a touch of vibrato.
 */
export async function renderMelodyPreview(notes, sampleRate) {
  const lengthSec = notes[notes.length - 1].end + 1;
  const ctx = new OfflineAudioContext(1, Math.ceil(lengthSec * sampleRate), sampleRate);
  const master = ctx.createGain();
  master.gain.value = 0.9;
  master.connect(ctx.destination);

  for (const n of notes) {
    const osc = ctx.createOscillator();
    osc.type = "triangle";
    osc.frequency.value = midiToFreq(n.midi);
    const lfo = ctx.createOscillator();
    lfo.frequency.value = 5.5;
    const vib = ctx.createGain();
    vib.gain.value = 8; // cents
    lfo.connect(vib);
    vib.connect(osc.detune);
    const lp = ctx.createBiquadFilter();
    lp.type = "lowpass";
    lp.frequency.value = 2400;
    const gain = ctx.createGain();
    const level = 0.28 * (0.6 + 0.4 * (n.amp ?? 1));
    gain.gain.setValueAtTime(0, n.start);
    gain.gain.linearRampToValueAtTime(level, n.start + 0.02);
    gain.gain.setValueAtTime(level, Math.max(n.start + 0.02, n.end - 0.04));
    gain.gain.setTargetAtTime(0, n.end - 0.02, 0.05);
    osc.connect(lp);
    lp.connect(gain);
    gain.connect(master);
    osc.start(n.start);
    osc.stop(n.end + 0.3);
    lfo.start(n.start);
    lfo.stop(n.end + 0.3);
  }
  return ctx.startRendering();
}

/**
 * Renders a layer-note list (harmonics / probability layers) to audio for
 * in-app audition: one soft sine per note, gain scaled by `prob` with the
 * same gamma the MIDI export uses for velocity — so what you hear tracks
 * what the exported file will do.
 */
export async function renderLayers(notes, sampleRate) {
  let end = 0;
  for (const n of notes) if (n.end > end) end = n.end;
  const ctx = new OfflineAudioContext(1, Math.ceil((end + 0.4) * sampleRate), sampleRate);
  const master = ctx.createGain();
  master.gain.value = 0.9;
  master.connect(ctx.destination);

  for (const n of notes) {
    const osc = ctx.createOscillator();
    osc.type = "sine";
    osc.frequency.value = midiToFreq(n.midi);
    const gain = ctx.createGain();
    const level = 0.16 * Math.pow(n.prob ?? 1, 0.75);
    gain.gain.setValueAtTime(0, n.start);
    gain.gain.linearRampToValueAtTime(level, n.start + 0.02);
    gain.gain.setValueAtTime(level, Math.max(n.start + 0.02, n.end - 0.05));
    gain.gain.setTargetAtTime(0, n.end - 0.03, 0.04);
    osc.connect(gain);
    gain.connect(master);
    osc.start(n.start);
    osc.stop(n.end + 0.3);
  }
  return ctx.startRendering();
}

/**
 * Mixes vocal + backing at the given gains into a 16-bit stereo WAV blob.
 * Pass a null vocal to export the backing alone.
 */
export function mixToWavBlob(vocalBuf, backingBuf, vocalGain, backingGain) {
  const sr = (vocalBuf ?? backingBuf).sampleRate;
  const length = Math.max(vocalBuf ? vocalBuf.length : 0, backingBuf.length);
  const out = [new Float32Array(length), new Float32Array(length)];

  for (let c = 0; c < 2; c++) {
    if (vocalBuf) {
      const v = vocalBuf.getChannelData(Math.min(c, vocalBuf.numberOfChannels - 1));
      for (let i = 0; i < v.length; i++) out[c][i] += v[i] * vocalGain;
    }
    const b = backingBuf.getChannelData(Math.min(c, backingBuf.numberOfChannels - 1));
    for (let i = 0; i < b.length; i++) out[c][i] += b[i] * backingGain;
  }

  // Normalize down if the sum clips.
  let peak = 0;
  for (let c = 0; c < 2; c++) {
    for (let i = 0; i < length; i++) peak = Math.max(peak, Math.abs(out[c][i]));
  }
  if (peak > 0.98) {
    const scale = 0.98 / peak;
    for (let c = 0; c < 2; c++) {
      for (let i = 0; i < length; i++) out[c][i] *= scale;
    }
  }

  return encodeWav(out, sr);
}

function encodeWav(channels, sampleRate) {
  const nCh = channels.length;
  const length = channels[0].length;
  const bytesPerSample = 2;
  const blockAlign = nCh * bytesPerSample;
  const dataSize = length * blockAlign;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  const writeStr = (off, s) => {
    for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i));
  };
  writeStr(0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, nCh, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * blockAlign, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true);
  writeStr(36, "data");
  view.setUint32(40, dataSize, true);

  let off = 44;
  for (let i = 0; i < length; i++) {
    for (let c = 0; c < nCh; c++) {
      const s = Math.max(-1, Math.min(1, channels[c][i]));
      view.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7fff, true);
      off += 2;
    }
  }
  return new Blob([buffer], { type: "audio/wav" });
}
