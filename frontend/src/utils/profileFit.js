/**
 * Shared Profile Fit calculation utilities.
 *
 * calculateProfileFit  — maps a raw model probability to a user-friendly score
 *                        based on historical calibration data (Apr 2026).
 *
 * normalizeTop3Fits    — normalizes the top 3 Profile Fit scores so they
 *                        always sum to exactly 100 (using the
 *                        largest-remainder method for clean integers).
 */

/**
 * Maps a raw model probability (0–100) to a "Profile Fit" percentage
 * based on calibration bins from Apr 2026 (26 classes, 1 497 test samples,
 * 84.97 % overall accuracy).
 *
 *  Raw band       → Mapped range
 *    0 – 5 %      → 0 – 35 %    (linear ramp – statistically unreliable)
 *    5 – 10 %     → 35 – 49 %
 *   10 – 15 %     → 49 – 62 %
 *   15 – 20 %     → 62 – 80 %
 *   20 – 30 %     → 80 – 83 %
 *   30 – 50 %     → 83 – 92 %
 *   50 – 100 %    → 92 – 97 %
 */
export const calculateProfileFit = (rawProbability) => {
  const p = rawProbability;

  if (p < 5)  return Math.round((p / 5) * 35);
  if (p < 10) return Math.round(35 + ((p - 5) / 5) * 14);
  if (p < 15) return Math.round(49 + ((p - 10) / 5) * 13);
  if (p < 20) return Math.round(62 + ((p - 15) / 5) * 18);
  if (p < 30) return Math.round(80 + ((p - 20) / 10) * 3);
  if (p < 50) return Math.round(83 + ((p - 30) / 20) * 9);
  return Math.round(92 + ((p - 50) / 50) * 5);
};

/**
 * Given an array of top-3 prediction objects (each with `raw_confidence`),
 * returns an array of 3 integer percentages that sum to exactly 100.
 *
 * Uses the largest-remainder method so the numbers are always whole and
 * always add up.
 *
 * @param {Array<{raw_confidence?: number}>} topPredictions
 * @returns {number[]}  e.g. [48, 30, 22]
 */
export const normalizeTop3Fits = (topPredictions = []) => {
  const top3 = topPredictions.slice(0, 3);
  if (top3.length === 0) return [];

  const rawFits = top3.map((p) => calculateProfileFit(p.raw_confidence ?? 0));
  const totalFit = rawFits.reduce((a, b) => a + b, 0) || 1;

  // Exact proportional shares
  const exactShares = rawFits.map((f) => (f / totalFit) * 100);

  // Floor each share
  const floored = exactShares.map((s) => Math.floor(s));
  let remainder = 100 - floored.reduce((a, b) => a + b, 0);

  // Distribute leftover 1-point increments to the entries with the
  // largest fractional remainders first.
  const remainders = exactShares.map((s, i) => ({ i, r: s - floored[i] }));
  remainders.sort((a, b) => b.r - a.r);
  for (const item of remainders) {
    if (remainder <= 0) break;
    floored[item.i] += 1;
    remainder -= 1;
  }

  return floored;
};
