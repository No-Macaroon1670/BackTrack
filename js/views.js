// Canvas / DOM rendering for the player panel: the waveform with its
// detected-note overlay, and the chord strip with playback highlighting.
// Elements are passed in — app.js is the only module that knows element ids.

/**
 * Maps between note coordinates and canvas pixels. `range` fixes the pitch
 * window (so the editor's notes don't rescale while one is dragged); without
 * it the window is derived from the notes with a little padding.
 */
export function noteLayout(canvas, duration, notes, range = null) {
  const W = canvas.width, H = canvas.height, pad = 6;
  let lo, hi;
  if (range) {
    ({ lo, hi } = range);
  } else {
    lo = Infinity; hi = -Infinity;
    for (const n of notes) { lo = Math.min(lo, n.midi); hi = Math.max(hi, n.midi); }
    if (!Number.isFinite(lo)) { lo = 57; hi = 69; }
    lo -= 2; hi += 2;
  }
  const span = Math.max(1, hi - lo);
  const inner = H - 2 * pad;
  return {
    lo, hi, span,
    toX: (t) => (t / duration) * W,
    fromX: (x) => (x / W) * duration,
    toY: (m) => H - pad - ((m - lo) / span) * inner,
    fromY: (y) => lo + ((H - pad - y) / inner) * span,
  };
}

/**
 * Waveform (or spectrogram) with the detected notes as pitch-positioned
 * bars. When `background` (an offscreen spectrogram canvas) is given it is
 * blitted as the base instead of the waveform.
 */
export function drawWaveform(canvas, vocalBuffer, notes, { selectedIndex = -1, selectedIndices = null, range = null, background = null, marquee = null } = {}) {
  const c = canvas.getContext("2d");
  c.clearRect(0, 0, canvas.width, canvas.height);
  if (background) {
    c.drawImage(background, 0, 0, canvas.width, canvas.height);
  } else {
    const data = vocalBuffer.getChannelData(0);
    const step = Math.ceil(data.length / canvas.width);
    const mid = canvas.height / 2;
    c.fillStyle = "rgba(93, 214, 255, 0.4)"; // dimmed so the note overlay reads
    for (let xPx = 0; xPx < canvas.width; xPx++) {
      let min = 0, max = 0;
      const base = xPx * step;
      for (let j = 0; j < step && base + j < data.length; j++) {
        const v = data[base + j];
        if (v < min) min = v;
        if (v > max) max = v;
      }
      c.fillRect(xPx, mid + min * mid, 1, Math.max(1, (max - min) * mid));
    }
  }

  if (!notes?.length) return;
  const L = noteLayout(canvas, vocalBuffer.duration, notes, range);
  notes.forEach((n, i) => {
    const x = L.toX(n.start);
    const w = Math.max(2, L.toX(n.end) - x);
    const y = L.toY(n.midi);
    const sel = selectedIndices ? selectedIndices.has(i) : i === selectedIndex;
    const h = sel ? 5 : 3;
    // Confidence shading: bars fade with the model's support for the note as
    // drawn (selection stays solid so faint notes are still easy to grab).
    c.globalAlpha = sel || n.conf === undefined ? 1 : 0.25 + 0.75 * n.conf;
    c.fillStyle = sel ? "#ff5d73" : "#6fe3a5";
    c.fillRect(x, y - h / 2, w, h);
  });
  c.globalAlpha = 1;

  if (marquee) {
    c.strokeStyle = "rgba(255, 255, 255, 0.8)";
    c.setLineDash([4, 3]);
    c.lineWidth = 1;
    c.strokeRect(
      Math.min(marquee.x0, marquee.x1) + 0.5,
      Math.min(marquee.y0, marquee.y1) + 0.5,
      Math.abs(marquee.x1 - marquee.x0),
      Math.abs(marquee.y1 - marquee.y0),
    );
    c.setLineDash([]);
  }
}

export function drawChordStrip(strip, chordSymbols) {
  strip.innerHTML = "";
  for (const c of chordSymbols) {
    const cell = document.createElement("div");
    cell.className = "chord-cell";
    cell.innerHTML = `${c.name}<span class="m">bar ${c.measure + 1}</span>`;
    strip.appendChild(cell);
  }
}

/** Highlights the bar currently playing; pass -1 to clear. */
export function highlightChord(strip, index) {
  const cells = strip.children;
  for (let i = 0; i < cells.length; i++) {
    cells[i].classList.toggle("active", i === index);
  }
}

/** Index of the bar containing time `t`, or -1. Bars carry beat times in bt[0..4]. */
export function barIndexAt(bars, t) {
  for (let i = 0; i < bars.length; i++) {
    if (t >= bars[i].bt[0] && t < bars[i].bt[4]) return i;
  }
  return -1;
}
