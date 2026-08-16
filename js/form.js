// Song form: finding the repeated material in a sung melody.
//
// Everything downstream of the arranger has so far treated every bar as an
// independent problem, so the same phrase sung twice could be harmonised two
// different ways — which sounds less like a choice than like the band
// forgetting the song. Real songs are built from repetition and contrast, so
// the arranger needs to know which stretches ARE the same stretch.
//
// Phrases are cut at the singer's own breath rests (the boundaries the drum
// fills already use), compared by their sequence of pitch INTERVALS so a
// phrase repeated a step higher still matches, and clustered greedily into
// labels: A B A B, verse/chorus, and so on.

// Segmenting on breath rests was tried first and measured useless on real
// humming: the median rest in a sung take is ~0.04 beats — people hum
// legato — and relaxing the threshold from 1.3 beats to 0.45 moved a 33 s
// take from 5 phrases to 7. There is no rest structure at the scale of a
// musical line, so repeats have to be found from the melody itself.

/**
 * Melodic shape, transposition-invariant: the semitone steps between
 * consecutive notes. "Do re mi" and "sol la ti" both read [2, 2].
 */
function intervals(notes, p) {
  const iv = [];
  for (let i = p.i0 + 1; i <= p.i1; i++) {
    iv.push(Math.round((notes[i].midiFloat ?? notes[i].midi) - (notes[i - 1].midiFloat ?? notes[i - 1].midi)));
  }
  return iv;
}

/** Longest common subsequence length — tolerant of an inserted or dropped note. */
function lcs(a, b) {
  const prev = new Array(b.length + 1).fill(0);
  const cur = new Array(b.length + 1).fill(0);
  for (let i = 1; i <= a.length; i++) {
    cur[0] = 0;
    for (let j = 1; j <= b.length; j++) {
      cur[j] = a[i - 1] === b[j - 1] ? prev[j - 1] + 1 : Math.max(prev[j], cur[j - 1]);
    }
    for (let j = 0; j <= b.length; j++) prev[j] = cur[j];
  }
  return prev[b.length];
}

function similarity(a, b) {
  if (!a.length || !b.length) return 0;
  // Wildly different lengths are different material, whatever they share.
  const ratio = Math.min(a.length, b.length) / Math.max(a.length, b.length);
  if (ratio < 0.6) return 0;
  return (2 * lcs(a, b)) / (a.length + b.length);
}

/**
 * Finds repeated melodic material anywhere in the take, without needing the
 * singer to leave a gap at the seams.
 *
 * Works on the interval sequence, so a phrase sung a tone higher still
 * matches, and scans it by LAG: for every offset d, the positions where
 * step i equals step i+d form runs, and a long run means "this stretch is
 * that stretch, d notes later". Runs are allowed one wrong step so a single
 * mis-transcribed note doesn't sever a real repeat. Longest repeats win, and
 * anything left over becomes its own section.
 *
 * @returns {{phrases: Array<{i0,i1,startBar,endBar,label,occurrence}>, nLabels: number}}
 */
export function detectForm(notes, bars, { minLen = 8, threshold = 0.72 } = {}) {
  if (!notes?.length || !bars?.length) return { phrases: [], nLabels: 0 };
  const iv = intervals(notes, { i0: 0, i1: notes.length - 1 });
  const N = iv.length;

  // Candidate repeats, compared as WINDOWS rather than as runs of identical
  // steps. Requiring an exact run was tried and found nothing on real takes:
  // two performances of the same line rarely transcribe to the same note
  // count, and a single inserted note shifts every interval after it, so an
  // exact match never survives. Longest-common-subsequence tolerates those
  // insertions and deletions, which is precisely the error mode here.
  const cands = [];
  for (const W of [16, 12, 8]) {
    if (W > N / 2) continue;
    for (let i = 0; i + W <= N; i++) {
      const a = iv.slice(i, i + W);
      for (let j = i + W; j + W <= N; j++) {
        const s = similarity(a, iv.slice(j, j + W));
        if (s >= threshold) cands.push({ i, d: j - i, len: W, score: s });
      }
    }
  }
  // Prefer long, then strong: a big approximate repeat beats a short exact one.
  cands.sort((a, b) => b.len - a.len || b.score - a.score);

  // Claim the longest repeats first; a note belongs to at most one section.
  const label = new Array(notes.length).fill(-1);
  let next = 0;
  const free = (a, b) => { for (let k = a; k <= b; k++) if (label[k] >= 0) return false; return true; };
  for (const c of cands) {
    const a0 = c.i, a1 = c.i + c.len;          // interval run -> note span
    const b0 = c.i + c.d, b1 = c.i + c.d + c.len;
    if (b1 >= notes.length) continue;
    if (!free(a0, a1) || !free(b0, b1)) continue;
    const L = next++;
    for (let k = a0; k <= a1; k++) label[k] = L;
    for (let k = b0; k <= b1; k++) label[k] = L;
  }

  // Contiguous stretches of one label become phrases; unlabelled runs get
  // their own label so every note belongs to some section.
  const phrases = [];
  let k = 0;
  while (k < notes.length) {
    const lab = label[k];
    let j = k;
    while (j + 1 < notes.length && label[j + 1] === lab) j++;
    phrases.push({ i0: k, i1: j, label: lab >= 0 ? lab : next++ });
    k = j + 1;
  }

  const seen = new Map();
  for (const p of phrases) {
    const c = seen.get(p.label) ?? 0;
    p.occurrence = c;
    seen.set(p.label, c + 1);
    const t0 = notes[p.i0].start, t1 = notes[p.i1].end;
    p.startBar = bars.findIndex((b) => t0 < b.bt[4]);
    p.endBar = bars.reduce((acc, b, i) => (t1 > b.bt[0] ? i : acc), 0);
    if (p.startBar < 0) p.startBar = 0;
    if (p.endBar < p.startBar) p.endBar = p.startBar;
  }
  return { phrases, nLabels: next };
}

/**
 * Forces repeats of the same phrase to share a harmonisation, in place.
 *
 * Only occurrences spanning the same number of bars are aligned — a repeat
 * that got stretched or truncated is left to its own chords rather than
 * forced into a shape that no longer fits. `segChords` holds two segments
 * per bar.
 *
 * @returns {number} how many segments were rewritten
 */
export function unifyRepeats(form, segChords) {
  const byLabel = new Map();
  for (const p of form.phrases) {
    if (!byLabel.has(p.label)) byLabel.set(p.label, []);
    byLabel.get(p.label).push(p);
  }
  let changed = 0;
  for (const group of byLabel.values()) {
    if (group.length < 2) continue;
    const first = group[0];
    const span = first.endBar - first.startBar;
    for (const p of group.slice(1)) {
      if (p.endBar - p.startBar !== span) continue; // different shape: leave it
      for (let b = 0; b <= span; b++) {
        for (const half of [0, 1]) {
          const src = 2 * (first.startBar + b) + half;
          const dst = 2 * (p.startBar + b) + half;
          if (!segChords[src] || !segChords[dst]) continue;
          if (segChords[dst] !== segChords[src]) changed++;
          segChords[dst] = segChords[src];
        }
      }
    }
  }
  return changed;
}
