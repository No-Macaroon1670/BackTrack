// Standard MIDI File (format 1) export for notation software. The tracked
// beat grid becomes a per-beat tempo map, so a take with drifting tempo
// still lands on proper bars and beats when imported into MuseScore, Logic,
// Ableton, etc. Tracks: conductor (tempo/time-sig/key-sig/chord markers),
// vocal melody, chords, bass, drums. Notes are quantized to a 16th grid —
// raw vocal timing makes unreadable notation.

const TPQ = 480; // ticks per quarter note
const STEP = TPQ / 4; // sixteenth-note grid

const DRUM_NOTES = { kick: 36, snare: 38, hat: 42, crash: 49 };
const DRUM_VELS = { kick: 100, snare: 95, hat: 60, crash: 105 };

// Legato gap-fill: extend a melody note to meet the next when the rest
// between them is at most this long. Sung notes leave gaps (breaths,
// consonants, energy decay) that sound choppy on dry MIDI patches; longer
// rests are real phrase boundaries and are preserved.
const LEGATO_MAX_GAP = 2 * TPQ;

/**
 * @param {object} analysis result of analyzeVocal (notes, key)
 * @param {object} arr result of buildArrangement (pads, bass, drums, bars, chordSymbols)
 * @param {"full"|"backing"|"melody"|"layers"} parts which tracks to include;
 *   the conductor track (tempo map, key, chord markers) is always present.
 *   "layers" = picked melody + every pitch the model considered plausible,
 *   with velocity encoding its probability (pass notes via options.layerNotes).
 * @param {{legato?: boolean, layerNotes?: Array}} options
 * @returns {Blob} .mid file contents
 */
export function arrangementToMidiBlob(analysis, arr, parts = "full", { legato = true, layerNotes = null, layersName = "Probability layers" } = {}) {
  const grid = buildGrid(arr.bars, analysis.notes);
  const tick = (t) => timeToTick(t, grid);
  const q = (tk) => Math.max(0, Math.round(tk / STEP) * STEP);

  // --- Conductor track ---
  const conductor = [
    { tick: 0, data: metaText(0x03, "BackTrack") },
    { tick: 0, data: [0xff, 0x58, 0x04, 0x04, 0x02, 0x24, 0x08] }, // 4/4
    { tick: 0, data: keySigEvent(analysis.key) },
  ];
  for (let k = 0; k < grid.beats.length - 1; k++) {
    const us = Math.min(0xffffff, Math.round((grid.beats[k + 1] - grid.beats[k]) * 1e6));
    conductor.push({ tick: k * TPQ, data: [0xff, 0x51, 0x03, (us >> 16) & 0xff, (us >> 8) & 0xff, us & 0xff] });
  }
  arr.chordSymbols.forEach((c, i) => {
    conductor.push({ tick: q(tick(arr.bars[i].bt[0])), data: metaText(0x06, c.name) });
  });

  // --- Melody (quantized, overlaps trimmed) ---
  const melody = analysis.notes
    .map((n) => {
      const start = q(tick(n.start));
      const end = Math.max(start + STEP, q(tick(n.end)));
      return { tick: start, dur: end - start, midi: n.midi, vel: 96 };
    })
    .sort((a, b) => a.tick - b.tick);
  for (let i = 0; i + 1 < melody.length; i++) {
    const gap = melody[i + 1].tick - melody[i].tick;
    if (gap > 0 && melody[i].dur > gap) melody[i].dur = gap;
  }
  if (legato) {
    for (let i = 0; i + 1 < melody.length; i++) {
      const rest = melody[i + 1].tick - (melody[i].tick + melody[i].dur);
      if (rest > 0 && rest <= LEGATO_MAX_GAP) melody[i].dur += rest;
    }
  }

  // --- Chords / bass / drums from the arrangement ---
  const chords = [];
  for (const p of arr.pads) {
    const start = q(tick(p.t));
    const dur = Math.max(STEP, q(tick(p.t + p.dur)) - start);
    const vel = Math.min(110, Math.round(60 + 45 * (p.vel ?? 1)));
    for (const m of p.midis) chords.push({ tick: start, dur, midi: m, vel });
  }
  const bass = arr.bass.map((n) => {
    const start = q(tick(n.t));
    return {
      tick: start,
      dur: Math.max(STEP, q(tick(n.t + n.dur)) - start),
      midi: n.midi,
      vel: Math.min(110, Math.round(90 * (n.vel ?? 1))),
    };
  });
  const drums = arr.drums.map((d) => ({
    tick: q(tick(d.t)),
    dur: STEP / 2,
    midi: DRUM_NOTES[d.type] ?? 42,
    vel: Math.max(20, Math.round((DRUM_VELS[d.type] ?? 70) * (d.vel ?? 1))),
  }));

  // Steel guitar / string ensemble / grand piano per the chosen instrument.
  const chordProgram = arr.instrument === "guitar" ? 25 : arr.instrument === "violin" ? 48 : arr.instrument === "voice" ? 52 : 0;
  const chordName = arr.instrument === "guitar" ? "Guitar" : arr.instrument === "violin" ? "Strings" : arr.instrument === "voice" ? "Voice" : "Keys";
  const tracks = [trackBytes(conductor)];
  if (parts !== "backing") {
    tracks.push(trackBytes(noteEvents("Vocal melody", 0, 52, melody))); // choir aahs
  }
  if (parts === "layers") {
    if (!layerNotes?.length) throw new Error("no probability data for this take");
    // Velocity encodes the model's probability (gamma-lifted so mid values
    // stay audible); quantized to the same grid as the melody.
    const layers = layerNotes.map((n) => {
      const start = q(tick(n.start));
      return {
        tick: start,
        dur: Math.max(STEP, q(tick(n.end)) - start),
        midi: Math.round(n.midi),
        vel: Math.max(8, Math.min(127, Math.round(127 * Math.pow(n.prob, 0.75)))),
      };
    });
    tracks.push(trackBytes(noteEvents(layersName, 3, 52, layers)));
  } else if (parts !== "melody") {
    tracks.push(trackBytes(noteEvents(chordName, 1, chordProgram, chords)));
    if (bass.length) tracks.push(trackBytes(noteEvents("Bass", 2, 32, bass))); // acoustic bass
    tracks.push(trackBytes(noteEvents("Drums", 9, null, drums))); // channel 10
    const roleProgram = (choice) =>
      choice === "guitar" ? 25 : choice === "violin" ? 48 : choice === "keys" ? 0 : 52;
    const resolveRole = (r) => (r === "follow" ? (arr.voices?.chords ?? arr.instrument) : r);
    const quantizeLine = (evts) => evts.map((h) => {
      const start = q(tick(h.t));
      return {
        tick: start,
        dur: Math.max(STEP, q(tick(h.t + h.dur)) - start),
        midi: h.midi,
        vel: Math.min(105, Math.round(60 + 45 * (h.vel ?? 1))),
      };
    });
    if (arr.harmony?.length) {
      tracks.push(trackBytes(noteEvents("Harmony voice", 4, roleProgram(resolveRole(arr.voices?.harmony ?? "follow")), quantizeLine(arr.harmony))));
    }
    if (arr.melody?.length) {
      tracks.push(trackBytes(noteEvents("Melody double", 5, roleProgram(resolveRole(arr.voices?.melody ?? "off")), quantizeLine(arr.melody))));
    }
  }

  const header = [
    0x4d, 0x54, 0x68, 0x64, 0, 0, 0, 6, // MThd, length 6
    0, 1, // format 1
    0, tracks.length,
    (TPQ >> 8) & 0xff, TPQ & 0xff,
  ];
  const bytes = [...header];
  for (const t of tracks) bytes.push(...t);
  return new Blob([new Uint8Array(bytes)], { type: "audio/midi" });
}

/**
 * Beat times serving as the tick grid: tick 0 is the first downbeat, or one
 * extrapolated bar earlier when the melody starts with a pickup.
 */
function buildGrid(bars, notes) {
  const beats = [];
  const first = bars[0].bt;
  const interval = first[1] - first[0];
  if (notes.some((n) => n.start < first[0] - 0.01)) {
    for (let i = 4; i >= 1; i--) beats.push(first[0] - i * interval);
  }
  for (const bar of bars) for (let i = 0; i < 4; i++) beats.push(bar.bt[i]);
  beats.push(bars[bars.length - 1].bt[4]);
  return { beats };
}

function timeToTick(t, grid) {
  const b = grid.beats;
  const last = b.length - 1;
  if (t <= b[0]) return 0;
  if (t >= b[last]) {
    const intv = b[last] - b[last - 1];
    return Math.round((last + (t - b[last]) / intv) * TPQ);
  }
  let lo = 0, hi = last;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (b[mid] <= t) lo = mid;
    else hi = mid;
  }
  return Math.round((lo + (t - b[lo]) / (b[hi] - b[lo])) * TPQ);
}

function noteEvents(name, channel, program, notes) {
  const ev = [{ tick: 0, data: metaText(0x03, name) }];
  if (program !== null) ev.push({ tick: 0, data: [0xc0 | channel, program] });
  for (const n of notes) {
    ev.push({ tick: n.tick, data: [0x90 | channel, n.midi & 0x7f, n.vel & 0x7f] });
    ev.push({ tick: n.tick + n.dur, data: [0x80 | channel, n.midi & 0x7f, 0] });
  }
  return ev;
}

/** Serializes events (sorted; note-offs before note-ons at equal ticks). */
function trackBytes(events) {
  const order = (e) => {
    const status = e.data[0] & 0xf0;
    if (e.data[0] === 0xff) return 0;
    if (status === 0x80) return 1;
    return 2;
  };
  events.sort((a, b) => a.tick - b.tick || order(a) - order(b));
  const body = [];
  let prev = 0;
  for (const e of events) {
    body.push(...vlq(e.tick - prev), ...e.data);
    prev = e.tick;
  }
  body.push(0x00, 0xff, 0x2f, 0x00); // end of track
  const len = body.length;
  return [
    0x4d, 0x54, 0x72, 0x6b, // MTrk
    (len >> 24) & 0xff, (len >> 16) & 0xff, (len >> 8) & 0xff, len & 0xff,
    ...body,
  ];
}

function vlq(n) {
  const bytes = [n & 0x7f];
  n >>= 7;
  while (n > 0) {
    bytes.unshift((n & 0x7f) | 0x80);
    n >>= 7;
  }
  return bytes;
}

function metaText(type, text) {
  // Keep meta text plain ASCII for maximum importer compatibility.
  const ascii = text.replace(" · ", " - ");
  const chars = [...ascii].map((c) => {
    const code = c.charCodeAt(0);
    return code >= 0x20 && code < 0x7f ? code : 0x20;
  });
  return [0xff, type, ...vlq(chars.length), ...chars];
}

// ---------------------------------------------------------------------------
// MIDI file input
// ---------------------------------------------------------------------------

const MAX_INPUT_SECONDS = 300;

/**
 * Parses a Standard MIDI File into the analysis-note format. All non-drum
 * tracks are pooled and reduced to a single top melodic line; the tempo map
 * becomes a beat grid (quarter notes), so generation follows the file's
 * tempo changes exactly.
 *
 * @param {ArrayBuffer} arrayBuffer
 * @returns {{notes, beats, duration, tempo}}
 */
export function parseMidiFile(arrayBuffer) {
  const b = new Uint8Array(arrayBuffer);
  const str = (o, n) => String.fromCharCode(...b.slice(o, o + n));
  if (str(0, 4) !== "MThd") throw new Error("not a Standard MIDI File");
  const nTracks = (b[10] << 8) | b[11];
  const division = (b[12] << 8) | b[13];
  if (division & 0x8000) throw new Error("SMPTE-timed MIDI files aren't supported");
  const tpq = division || 480;

  const tempos = []; // { tick, us }
  const rawNotes = [];
  let pos = 14;
  for (let t = 0; t < nTracks; t++) {
    if (str(pos, 4) !== "MTrk") throw new Error("malformed MIDI (missing MTrk chunk)");
    const len = (b[pos + 4] << 24) | (b[pos + 5] << 16) | (b[pos + 6] << 8) | b[pos + 7];
    let p = pos + 8;
    const end = p + len;
    let tick = 0;
    let running = 0;
    const open = new Map(); // (channel<<8)|note -> { tick, vel }
    while (p < end) {
      let delta = 0;
      while (b[p] & 0x80) delta = (delta << 7) | (b[p++] & 0x7f);
      delta = (delta << 7) | (b[p++] & 0x7f);
      tick += delta;

      let status = b[p];
      if (status & 0x80) {
        p++;
        if (status < 0xf0) running = status;
      } else {
        status = running;
      }

      if (status === 0xff) {
        const type = b[p++];
        let l = 0;
        while (b[p] & 0x80) l = (l << 7) | (b[p++] & 0x7f);
        l = (l << 7) | (b[p++] & 0x7f);
        if (type === 0x51) tempos.push({ tick, us: (b[p] << 16) | (b[p + 1] << 8) | b[p + 2] });
        p += l;
      } else if (status === 0xf0 || status === 0xf7) {
        let l = 0;
        while (b[p] & 0x80) l = (l << 7) | (b[p++] & 0x7f);
        l = (l << 7) | (b[p++] & 0x7f);
        p += l;
      } else {
        const kind = status & 0xf0;
        const ch = status & 0x0f;
        if (kind === 0x90 || kind === 0x80) {
          const note = b[p++];
          const vel = b[p++];
          if (ch !== 9) { // skip drum channel
            const key = (ch << 8) | note;
            if (kind === 0x90 && vel > 0) {
              if (!open.has(key)) open.set(key, { tick, vel });
            } else {
              const o = open.get(key);
              if (o && tick > o.tick) {
                rawNotes.push({ startTick: o.tick, endTick: tick, midi: note, vel: o.vel });
              }
              open.delete(key);
            }
          }
        } else if (kind === 0xc0 || kind === 0xd0) {
          p += 1;
        } else {
          p += 2;
        }
      }
    }
    pos = end;
  }

  // Tempo map -> tick-to-seconds conversion (default 120 BPM).
  tempos.sort((a, c) => a.tick - c.tick);
  if (tempos.length === 0 || tempos[0].tick > 0) tempos.unshift({ tick: 0, us: 500000 });
  let acc = 0;
  const segs = tempos.map((tp, i) => {
    if (i > 0) acc += ((tp.tick - tempos[i - 1].tick) / tpq) * (tempos[i - 1].us / 1e6);
    return { tick: tp.tick, sec: acc, spq: tp.us / 1e6 };
  });
  const tickToSec = (tick) => {
    let s = segs[0];
    for (const seg of segs) {
      if (seg.tick <= tick) s = seg;
      else break;
    }
    return s.sec + ((tick - s.tick) / tpq) * s.spq;
  };

  let notes = rawNotes
    .map((n) => {
      const start = tickToSec(n.startTick);
      const end = tickToSec(n.endTick);
      return { start, end, dur: end - start, midi: n.midi, midiFloat: n.midi, amp: n.vel / 127 };
    })
    .filter((n) => n.dur > 0.02 && n.midi >= 24 && n.midi <= 100 && n.start < MAX_INPUT_SECONDS)
    .sort((a, c) => a.start - c.start || c.midi - a.midi);
  notes = topLine(notes);
  if (notes.length === 0) throw new Error("no melody notes found (drum-only file?)");

  const lastTick = Math.max(...rawNotes.map((n) => n.endTick));
  const duration = Math.min(tickToSec(lastTick), MAX_INPUT_SECONDS) + 0.5;
  const beats = [];
  const nBeats = Math.ceil(lastTick / tpq) + 8;
  for (let i = 0; i < nBeats; i++) {
    const t = tickToSec(i * tpq);
    if (t > duration + 5) break;
    beats.push(t);
  }
  const intervals = [];
  for (let i = 1; i < beats.length; i++) intervals.push(beats[i] - beats[i - 1]);
  intervals.sort((a, c) => a - c);
  const median = intervals[Math.floor(intervals.length / 2)] || 0.5;
  const tempo = Math.round(60 / median);

  return { notes, beats, duration, tempo };
}

/**
 * Reduces possibly-polyphonic MIDI (chords, multiple instruments) to a
 * single melodic line, preferring the highest concurrent note.
 */
function topLine(notes) {
  const out = [];
  for (const n of notes) {
    const last = out[out.length - 1];
    if (!last || n.start >= last.end - 0.01) {
      out.push({ ...n });
      continue;
    }
    if (n.midi > last.midi) {
      if (n.start - last.start > 0.05) {
        last.end = n.start; // trim the lower note under the new higher one
        last.dur = last.end - last.start;
        out.push({ ...n });
      } else {
        out[out.length - 1] = { ...n }; // same attack: keep the top of the chord
      }
    } // lower concurrent note: drop
  }
  return out.filter((n) => n.dur > 0.02);
}

// Circle-of-fifths position (number of sharps, negative = flats) per tonic.
const MAJOR_SF = [0, -5, 2, -3, 4, -1, 6, 1, -4, 3, -2, 5];
const MINOR_SF = [-3, 4, -1, 6, 1, -4, 3, -2, 5, 0, -5, 2];

function keySigEvent(key) {
  const sf = (key.mode === "major" ? MAJOR_SF : MINOR_SF)[key.tonic];
  return [0xff, 0x59, 0x02, sf & 0xff, key.mode === "minor" ? 1 : 0];
}
