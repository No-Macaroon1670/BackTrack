// Transcription accuracy metrics, faithful to mir_eval's definitions so the
// numbers are comparable to published systems.
//
//   Note level (mir_eval.transcription): a max bipartite matching of
//   estimated to reference notes under onset (±50 ms), pitch (±50 cents),
//   and optional offset (±max(50 ms, 20% of note length)) tolerances, then
//   precision / recall / F-measure. Reported three ways: onset-only,
//   onset+pitch (COnP), onset+pitch+offset (COnPOff).
//
//   Frame level (mir_eval.melody): raw pitch accuracy, raw chroma accuracy,
//   and voicing, comparing an F0 sequence on a fixed time grid.

export const midiToHz = (m) => 440 * 2 ** ((m - 69) / 12);
export const hzToMidi = (hz) => 69 + 12 * Math.log2(hz / 440);

// ---------------------------------------------------------------------------
// Note-level
// ---------------------------------------------------------------------------

/**
 * @param {Array<{onset,offset,midi}>} ref reference notes
 * @param {Array<{onset,offset,midi}>} est estimated notes
 * @returns {{onset, onp, onpoff, nRef, nEst}} each of onset/onp/onpoff is
 *   { precision, recall, f }
 */
export function noteTranscriptionMetrics(ref, est, opts = {}) {
  const { onsetTol = 0.05, pitchTolCents = 50, offsetRatio = 0.2, offsetMinTol = 0.05 } = opts;
  const centsDiff = (a, b) => Math.abs(100 * (a - b)); // MIDI difference in cents

  const score = (needPitch, needOffset) => {
    const adj = ref.map((r) => {
      const list = [];
      for (let j = 0; j < est.length; j++) {
        const e = est[j];
        if (Math.abs(e.onset - r.onset) > onsetTol) continue;
        if (needPitch && centsDiff(e.midi, r.midi) > pitchTolCents) continue;
        if (needOffset) {
          const tol = Math.max(offsetMinTol, offsetRatio * (r.offset - r.onset));
          if (Math.abs(e.offset - r.offset) > tol) continue;
        }
        list.push(j);
      }
      return list;
    });
    const matched = maxBipartiteMatching(adj, est.length);
    const precision = est.length ? matched / est.length : 0;
    const recall = ref.length ? matched / ref.length : 0;
    const f = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
    return { precision, recall, f };
  };

  return {
    onset: score(false, false),
    onp: score(true, false),
    onpoff: score(true, true),
    nRef: ref.length,
    nEst: est.length,
  };
}

/** Maximum bipartite matching (Kuhn's augmenting paths). adj[i] = est indices. */
function maxBipartiteMatching(adj, nEst) {
  const matchEst = new Array(nEst).fill(-1); // est j -> ref i
  const tryKuhn = (u, seen) => {
    for (const v of adj[u]) {
      if (seen[v]) continue;
      seen[v] = true;
      if (matchEst[v] === -1 || tryKuhn(matchEst[v], seen)) {
        matchEst[v] = u;
        return true;
      }
    }
    return false;
  };
  let count = 0;
  for (let u = 0; u < adj.length; u++) {
    if (tryKuhn(u, new Array(nEst).fill(false))) count++;
  }
  return count;
}

// ---------------------------------------------------------------------------
// Frame-level (melody)
// ---------------------------------------------------------------------------

/** Piecewise-constant F0 (Hz) from a note list, sampled at `times`. 0 = unvoiced. */
export function notesToF0(notes, times) {
  const sorted = [...notes].sort((a, b) => a.start - b.start);
  const out = new Float64Array(times.length);
  let k = 0;
  for (let i = 0; i < times.length; i++) {
    const t = times[i];
    while (k < sorted.length && sorted[k].end <= t) k++;
    let hz = 0;
    for (let j = k; j < sorted.length && sorted[j].start <= t; j++) {
      if (t >= sorted[j].start && t < sorted[j].end) { hz = midiToHz(sorted[j].midi); break; }
    }
    out[i] = hz;
  }
  return out;
}

/**
 * @param {Float64Array|number[]} refHz reference F0 per frame (0 = unvoiced)
 * @param {Float64Array|number[]} estHz estimated F0 on the same grid
 * @returns {{rpa, rca, voicingRecall, voicingFalseAlarm, overall}}
 */
export function melodyMetrics(refHz, estHz, { tolCents = 50 } = {}) {
  let voicedRef = 0, unvoicedRef = 0;
  let estVoicedWhereRefVoiced = 0, falseAlarm = 0;
  let correct = 0, correctChroma = 0, correctFrames = 0;

  for (let i = 0; i < refHz.length; i++) {
    const rf = refHz[i], ef = estHz[i];
    const rV = rf > 0, eV = ef > 0;
    if (rV) {
      voicedRef++;
      if (eV) {
        estVoicedWhereRefVoiced++;
        const cents = 1200 * Math.log2(ef / rf);
        if (Math.abs(cents) <= tolCents) { correct++; correctFrames++; }
        let cc = ((cents % 1200) + 1200) % 1200;
        cc = Math.min(cc, 1200 - cc);
        if (cc <= tolCents) correctChroma++;
      }
    } else {
      unvoicedRef++;
      if (eV) falseAlarm++;
      else correctFrames++; // correctly unvoiced
    }
  }

  const total = voicedRef + unvoicedRef;
  return {
    rpa: voicedRef ? correct / voicedRef : 0,
    rca: voicedRef ? correctChroma / voicedRef : 0,
    voicingRecall: voicedRef ? estVoicedWhereRefVoiced / voicedRef : 0,
    voicingFalseAlarm: unvoicedRef ? falseAlarm / unvoicedRef : 0,
    overall: total ? correctFrames / total : 0,
  };
}

/** Mean of a metric field across per-clip result objects. */
export function mean(values) {
  if (!values.length) return 0;
  return values.reduce((a, b) => a + b, 0) / values.length;
}
