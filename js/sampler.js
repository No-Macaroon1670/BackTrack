// Sampled instruments: real recorded notes from the FluidR3_GM General-MIDI
// soundfont (Creative Commons Attribution 3.0 — credited in the page footer
// and README), pre-rendered per note as mp3 by the midi-js-soundfonts
// project, fetched from the same CDN pattern as the transcription model and
// decoded once per session. render.js plays these through the identical
// event/velocity pipeline as the synth voices, so the "Sampled sound" toggle
// swaps timbre without touching the arrangement — and if the CDN is
// unreachable, generation falls back to synthesis, exactly like the
// Basic Pitch -> YIN fallback on the analysis side.

const SF_BASE = "https://cdn.jsdelivr.net/gh/gleitz/midi-js-soundfonts@gh-pages/FluidR3_GM/";

// instrument choice -> soundfont ("voice" / "off" / "synth" have none)
const FONTS = {
  keys: "acoustic_grand_piano",
  guitar: "acoustic_guitar_steel",
  violin: "string_ensemble_1",
  acoustic: "acoustic_bass",
  electric: "electric_bass_finger",
};
const CHORD_FONTS = { keys: FONTS.keys, guitar: FONTS.guitar, violin: FONTS.violin };
const BASS_FONT = FONTS.acoustic;

const NOTE_OFFSETS = { C: 0, Db: 1, D: 2, Eb: 3, E: 4, F: 5, Gb: 6, G: 7, Ab: 8, A: 9, Bb: 10, B: 11 };

function nameToMidi(name) {
  const m = /^([A-G]b?)(\d)$/.exec(name);
  return m ? 12 * (+m[2] + 1) + NOTE_OFFSETS[m[1]] : null;
}

const fontCache = new Map(); // font name -> Promise<Map<midi, AudioBuffer>>
let decodeCtx = null; // shared: browsers cap live AudioContexts

async function loadFont(font) {
  if (fontCache.has(font)) return fontCache.get(font);
  const promise = (async () => {
    decodeCtx ??= new AudioContext();
    const res = await fetch(`${SF_BASE}${font}-mp3.js`);
    if (!res.ok) throw new Error(`soundfont ${font}: HTTP ${res.status}`);
    const text = await res.text();
    const start = text.indexOf("{", text.indexOf(`MIDI.Soundfont.${font}`));
    const end = text.lastIndexOf("}");
    if (start < 0 || end <= start) throw new Error(`soundfont ${font}: unexpected format`);
    // The files are JS object literals with a trailing comma — valid JS,
    // invalid JSON — so strip it before parsing.
    const table = JSON.parse(text.slice(start, end + 1).replace(/,\s*}$/, "}"));

    const map = new Map();
    await Promise.all(Object.entries(table).map(async ([name, dataUri]) => {
      const midi = nameToMidi(name);
      if (midi === null) return;
      const ab = await (await fetch(dataUri)).arrayBuffer();
      // decodeAudioData detaches its input; each fetch owns its buffer.
      map.set(midi, await decodeCtx.decodeAudioData(ab));
    }));
    if (!map.size) throw new Error(`soundfont ${font}: no notes decoded`);

    // The pre-rendered mp3s sit ~20 dB below full scale, varying per font.
    // Measure the font's median attack peak once and store the gain that
    // brings it to a common reference, so render levels stay font-agnostic.
    const peaks = [];
    for (const buf of map.values()) {
      const d = buf.getChannelData(0);
      let peak = 0;
      const n = Math.min(d.length, buf.sampleRate);
      for (let i = 0; i < n; i++) { const a = Math.abs(d[i]); if (a > peak) peak = a; }
      peaks.push(peak);
    }
    peaks.sort((a, b) => a - b);
    map.gain = Math.min(24, 0.35 / Math.max(0.01, peaks[Math.floor(peaks.length / 2)]));
    return map;
  })();
  fontCache.set(font, promise);
  promise.catch(() => fontCache.delete(font)); // failed loads may be retried
  return promise;
}

/**
 * Loads the sample maps needed to render an arrangement with the given
 * chord instrument. Cached per session; the browser HTTP cache makes later
 * sessions near-instant. Throws when offline — callers fall back to synth.
 *
 * @returns {Promise<{chord: Map<midi, AudioBuffer>, bass: Map<midi, AudioBuffer>}>}
 */
export async function loadSamples(instrument) {
  const [chord, bass] = await Promise.all([
    loadFont(CHORD_FONTS[instrument] ?? CHORD_FONTS.keys),
    loadFont(BASS_FONT),
  ]);
  return { chord, bass };
}

/**
 * Per-role sample maps for a voices config ({ chords, harmony, bass,
 * melody }). Roles whose choice has no soundfont — "voice", "off",
 * "synth" — come back null and render through their synth/wave paths.
 * "follow" resolves to the chords choice. Fonts load in parallel and are
 * shared across roles via the cache.
 */
export async function loadVoiceSamples(voices) {
  const resolve = (r) => (r === "follow" ? voices.chords : r);
  const need = {
    chords: voices.chords,
    harmony: resolve(voices.harmony),
    bass: voices.bass,
    melody: resolve(voices.melody),
  };
  const out = {};
  await Promise.all(Object.entries(need).map(async ([role, choice]) => {
    const font = FONTS[choice];
    out[role] = font ? await loadFont(font) : null;
  }));
  return out;
}

/** Nearest sampled note to `midi` (FluidR3 sets carry every semitone). */
export function nearestSample(map, midi) {
  if (map.has(midi)) return { buffer: map.get(midi), midi };
  for (let d = 1; d <= 6; d++) {
    if (map.has(midi + d)) return { buffer: map.get(midi + d), midi: midi + d };
    if (map.has(midi - d)) return { buffer: map.get(midi - d), midi: midi - d };
  }
  return null;
}
