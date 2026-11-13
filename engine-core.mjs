/**
 * Red & Black Prediction Engine — pure math/core.
 * Used by validate-engine.mjs and inlined into engine.html.
 * Predictions are always formed from history strictly before the outcome
 * they are scored against (walk-forward, no leakage).
 */

export const WINDOWS = [3, 4, 5, 6, 10, 20, 25, 50, 100];
export const ALPHA = 0.05;
export const MIN_N_SIGNIFICANCE = 30;
export const LAPLACE_A = 1;
export const LAPLACE_B = 1;

// ---------------------------------------------------------------------------
// Sequence helpers
// ---------------------------------------------------------------------------

export function parseSequence(text) {
  if (text == null) return [];
  const s = String(text).toUpperCase().replace(/RED/g, "R").replace(/BLACK/g, "B");
  const out = [];
  for (const ch of s) {
    if (ch === "R" || ch === "B") out.push(ch);
  }
  return out;
}

export function currentStreak(seq) {
  if (!seq.length) return { color: null, length: 0 };
  const color = seq[seq.length - 1];
  let length = 1;
  for (let i = seq.length - 2; i >= 0; i--) {
    if (seq[i] === color) length++;
    else break;
  }
  return { color, length };
}

// ---------------------------------------------------------------------------
// Special functions
// ---------------------------------------------------------------------------

const LANCZOS = [
  0.99999999999980993, 676.5203681218851, -1259.1392167224028,
  771.32342877765313, -176.61502916214059, 12.507343278686905,
  -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
];

export function logGamma(z) {
  if (z < 0.5) {
    return Math.log(Math.PI) - Math.log(Math.abs(Math.sin(Math.PI * z))) - logGamma(1 - z);
  }
  const x0 = z - 1;
  let x = LANCZOS[0];
  for (let i = 1; i < LANCZOS.length; i++) x += LANCZOS[i] / (x0 + i);
  const t = x0 + 7.5;
  return 0.5 * Math.log(2 * Math.PI) + (x0 + 0.5) * Math.log(t) - t + Math.log(x);
}

function betacf(a, b, x) {
  const MAXIT = 200;
  const EPS = 3e-14;
  const FPMIN = 1e-300;
  const qab = a + b;
  const qap = a + 1;
  const qam = a - 1;
  let c = 1;
  let d = 1 - (qab * x) / qap;
  if (Math.abs(d) < FPMIN) d = FPMIN;
  d = 1 / d;
  let h = d;
  for (let m = 1; m <= MAXIT; m++) {
    const m2 = 2 * m;
    let aa = (m * (b - m) * x) / ((qam + m2) * (a + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c;
    if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d;
    h *= d * c;
    aa = (-(a + m) * (qab + m) * x) / ((a + m2) * (qap + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c;
    if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) < EPS) return h;
  }
  return h;
}

export function regularizedIncompleteBeta(a, b, x) {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  if (!(a > 0) || !(b > 0) || !Number.isFinite(x)) return NaN;
  const lbeta = logGamma(a) + logGamma(b) - logGamma(a + b);
  const logFront = a * Math.log(x) + b * Math.log(1 - x) - lbeta;
  const front = Math.exp(logFront);
  if (!Number.isFinite(front)) {
    // Extreme tails: fall back via symmetry / clamp
    if (x < (a - 1) / (a + b - 2)) return 0;
    return 1;
  }
  const switchPoint = (a + 1) / (a + b + 2);
  if (x < switchPoint) {
    return (front / a) * betacf(a, b, x);
  }
  return 1 - (front / b) * betacf(b, a, 1 - x);
}

export function erf(x) {
  // Abramowitz and Stegun 7.1.26
  const sign = x < 0 ? -1 : 1;
  const a = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * a);
  const y =
    1 -
    (((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t +
      0.254829592) *
      t *
      Math.exp(-a * a));
  return sign * y;
}

export function erfc(x) {
  return 1 - erf(x);
}

export function normalCdf(z) {
  return 0.5 * erfc(-z / Math.SQRT2);
}

export function twoSidedNormalP(z) {
  if (!Number.isFinite(z)) return 1;
  const p = 2 * (1 - normalCdf(Math.abs(z)));
  return Math.min(1, Math.max(0, p));
}

/**
 * P(X >= k | n, p) for a binomial. Uses the regularized incomplete beta,
 * with a normal continuity-corrected fallback for huge n or unstable beta.
 */
function logBinom(n, k) {
  if (k < 0 || k > n) return -Infinity;
  return logGamma(n + 1) - logGamma(k + 1) - logGamma(n - k + 1);
}

function binomialTailLogSum(from, to, n, p) {
  // sum_{i=from..to} C(n,i) p^i (1-p)^{n-i} via log-sum-exp
  const logp = Math.log(p);
  const logq = Math.log(1 - p);
  // Peak term in the requested range
  const mode = Math.round(n * p);
  const peak = Math.min(to, Math.max(from, mode));
  const logPeak = logBinom(n, peak) + peak * logp + (n - peak) * logq;
  let sum = 0;
  for (let i = from; i <= to; i++) {
    const li = logBinom(n, i) + i * logp + (n - i) * logq;
    sum += Math.exp(li - logPeak);
  }
  const val = Math.exp(logPeak) * sum;
  return Number.isFinite(val) ? val : NaN;
}

export function binomialPValueGte(k, n, p = 0.5) {
  if (n <= 0) return 1;
  if (k <= 0) return 1;
  if (k > n) return 0;
  if (p <= 0) return 0;
  if (p >= 1) return 1;
  // Exact-ish log-sum-exp for moderate n; for huge n or a huge tail, use a
  // continuity-corrected normal or the incomplete-beta identity.
  const tailLen = n - k + 1;
  const lowerLen = k; // terms in P(X <= k-1)
  if (n <= 20000 && Math.min(tailLen, lowerLen) <= 8000) {
    let val;
    if (tailLen <= lowerLen) {
      val = binomialTailLogSum(k, n, n, p);
    } else {
      const lower = k <= 0 ? 0 : binomialTailLogSum(0, k - 1, n, p);
      val = 1 - lower;
    }
    if (Number.isFinite(val)) return Math.min(1, Math.max(0, val));
  }
  const mu = n * p;
  const sd = Math.sqrt(n * p * (1 - p));
  if (n >= 2000) {
    const z = (k - 0.5 - mu) / sd;
    return Math.min(1, Math.max(0, 1 - normalCdf(z)));
  }
  const val = regularizedIncompleteBeta(k, n - k + 1, p);
  if (!Number.isFinite(val)) {
    const z = (k - 0.5 - mu) / sd;
    return Math.min(1, Math.max(0, 1 - normalCdf(z)));
  }
  return Math.min(1, Math.max(0, val));
}

export function wilsonCI(k, n, z = 1.95996398454) {
  if (n <= 0) return { lo: 0, hi: 1, center: 0.5 };
  const phat = k / n;
  const z2 = z * z;
  const denom = 1 + z2 / n;
  const center = (phat + z2 / (2 * n)) / denom;
  const margin = (z * Math.sqrt((phat * (1 - phat) + z2 / (4 * n)) / n)) / denom;
  return {
    lo: Math.max(0, center - margin),
    hi: Math.min(1, center + margin),
    center,
  };
}

/** Regularized upper incomplete gamma Q(s, x) = Γ(s,x)/Γ(s). */
function gammaincc(s, x) {
  if (x <= 0) return 1;
  if (!(s > 0)) return NaN;
  if (x < s + 1) {
    // Series for P, then Q = 1-P
    let ap = s;
    let sum = 1 / s;
    let del = sum;
    for (let i = 0; i < 200; i++) {
      ap += 1;
      del *= x / ap;
      sum += del;
      if (Math.abs(del) < Math.abs(sum) * 1e-14) break;
    }
    const p = sum * Math.exp(-x + s * Math.log(x) - logGamma(s));
    return Math.min(1, Math.max(0, 1 - p));
  }
  // Continued fraction for Q
  const FPMIN = 1e-300;
  let b = x + 1 - s;
  let c = 1 / FPMIN;
  let d = 1 / b;
  let h = d;
  for (let i = 1; i <= 200; i++) {
    const an = -i * (i - s);
    b += 2;
    d = an * d + b;
    if (Math.abs(d) < FPMIN) d = FPMIN;
    c = b + an / c;
    if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) < 1e-14) break;
  }
  const q = h * Math.exp(-x + s * Math.log(x) - logGamma(s));
  return Math.min(1, Math.max(0, q));
}

export function chiSquarePValue(x, df) {
  if (x <= 0) return 1;
  if (!(df > 0) || !Number.isFinite(x)) return 1;
  if (df === 1) return Math.min(1, Math.max(0, erfc(Math.sqrt(x / 2))));
  return gammaincc(df / 2, x / 2);
}

// ---------------------------------------------------------------------------
// Randomness tests
// ---------------------------------------------------------------------------

export function runsTest(seq) {
  const n = seq.length;
  if (n < 12) {
    return { name: "Runs (Wald–Wolfowitz)", n, nR: 0, nB: 0, runs: 0, z: null, p: null, skip: "Need at least 12 outcomes." };
  }
  let nR = 0;
  let nB = 0;
  for (const x of seq) {
    if (x === "R") nR++;
    else nB++;
  }
  if (nR === 0 || nB === 0) {
    return { name: "Runs (Wald–Wolfowitz)", n, nR, nB, runs: 1, z: null, p: null, skip: "Only one colour observed." };
  }
  let runs = 1;
  for (let i = 1; i < n; i++) if (seq[i] !== seq[i - 1]) runs++;
  const mu = (2 * nR * nB) / n + 1;
  const varR =
    (2 * nR * nB * (2 * nR * nB - nR - nB)) / (n * n * (n - 1));
  const sd = Math.sqrt(varR);
  const z = sd > 0 ? (runs - mu) / sd : 0;
  const p = twoSidedNormalP(z);
  return { name: "Runs (Wald–Wolfowitz)", n, nR, nB, runs, mu, sd, z, p, skip: null };
}

export function chiSquareFrequency(seq) {
  const n = seq.length;
  if (n < 8) {
    return { name: "χ² frequency (P=50/50)", n, nR: 0, nB: 0, chi2: null, df: 1, p: null, skip: "Need at least 8 outcomes." };
  }
  let nR = 0;
  for (const x of seq) if (x === "R") nR++;
  const nB = n - nR;
  const exp = n / 2;
  const chi2 = ((nR - exp) ** 2 + (nB - exp) ** 2) / exp;
  const p = chiSquarePValue(chi2, 1);
  return { name: "χ² frequency (P=50/50)", n, nR, nB, chi2, df: 1, p, skip: null };
}

export function autocorrelations(seq, maxLag = 5) {
  const n = seq.length;
  const results = [];
  const x = seq.map((c) => (c === "R" ? 1 : 0));
  const mean = x.reduce((a, b) => a + b, 0) / (n || 1);
  let ss = 0;
  for (const v of x) ss += (v - mean) ** 2;
  for (let lag = 1; lag <= maxLag; lag++) {
    if (n < lag + 12 || ss <= 0) {
      results.push({
        name: `Autocorrelation lag ${lag}`,
        lag,
        n: n - lag,
        r: null,
        z: null,
        p: null,
        skip: n < lag + 12 ? "Need more outcomes." : "Zero variance (one colour only).",
      });
      continue;
    }
    let num = 0;
    for (let i = 0; i < n - lag; i++) num += (x[i] - mean) * (x[i + lag] - mean);
    // Standard sample autocorrelation (divide by full SS)
    const r = num / ss;
    const se = 1 / Math.sqrt(n - lag);
    const z = r / se;
    const p = twoSidedNormalP(z);
    results.push({ name: `Autocorrelation lag ${lag}`, lag, n: n - lag, r, z, p, skip: null });
  }
  return results;
}

function serialChi(seq, order) {
  const n = seq.length;
  const nGrams = n - order + 1;
  const nCats = 2 ** order;
  const label = order === 2 ? "Serial pairs" : order === 3 ? "Serial triples" : `Serial ${order}-grams`;
  if (nGrams < nCats * 5) {
    return {
      name: label,
      order,
      n: nGrams,
      chi2: null,
      df: nCats - 1,
      counts: null,
      p: null,
      skip: `Need at least ${nCats * 5} overlapping ${order}-grams.`,
    };
  }
  const counts = new Array(nCats).fill(0);
  const keys = [];
  for (let i = 0; i < nGrams; i++) {
    let code = 0;
    let key = "";
    for (let k = 0; k < order; k++) {
      const bit = seq[i + k] === "R" ? 1 : 0;
      code = (code << 1) | bit;
      key += seq[i + k];
    }
    counts[code]++;
    if (keys.length < nCats && !keys.includes(key)) keys.push(key);
  }
  const exp = nGrams / nCats;
  let chi2 = 0;
  for (let i = 0; i < nCats; i++) chi2 += (counts[i] - exp) ** 2 / exp;
  const df = nCats - 1;
  const p = chiSquarePValue(chi2, df);
  const countMap = {};
  for (let i = 0; i < nCats; i++) {
    let key = "";
    for (let k = order - 1; k >= 0; k--) key += (i & (1 << k)) ? "R" : "B";
    countMap[key] = counts[i];
  }
  return { name: label, order, n: nGrams, chi2, df, counts: countMap, expected: exp, p, skip: null };
}

export function serialTests(seq) {
  return [serialChi(seq, 2), serialChi(seq, 3)];
}

export function allRandomnessTests(seq) {
  const tests = [runsTest(seq), chiSquareFrequency(seq), ...autocorrelations(seq, 5), ...serialTests(seq)];
  const evaluated = tests.filter((t) => t.p != null);
  const nEval = evaluated.length;
  const significantRaw = evaluated.filter((t) => t.p < ALPHA);
  const significantBonferroni = evaluated.filter((t) => t.p * nEval < ALPHA);
  return { tests, nEval, significantRaw, significantBonferroni };
}

// ---------------------------------------------------------------------------
// Strategy walk-forward
// ---------------------------------------------------------------------------

function keyOf(arr) {
  return arr.join("");
}

function pickFromProb(pR) {
  if (pR > 0.5) return "R";
  if (pR < 0.5) return "B";
  return null;
}

function makeMajority(window, bayesian) {
  const W = window == null ? Infinity : window;
  const name = bayesian
    ? window == null
      ? "Bayesian Beta-Binomial (all)"
      : `Bayesian Beta-Binomial W${window}`
    : window == null
      ? "Majority frequency (all)"
      : `Majority frequency W${window}`;
  const id = bayesian
    ? window == null
      ? "bayes-all"
      : `bayes-w${window}`
    : window == null
      ? "maj-all"
      : `maj-w${window}`;
  return {
    id,
    name,
    family: bayesian ? "bayesian" : "majority",
    window: window,
    init() {
      return { buf: [], r: 0, b: 0 };
    },
    predict(st) {
      const r = st.r;
      const b = st.b;
      const n = r + b;
      if (n === 0) return { pick: null, pR: 0.5, pB: 0.5 };
      let pR;
      if (bayesian) pR = (LAPLACE_A + r) / (LAPLACE_A + LAPLACE_B + n);
      else pR = r / n;
      return { pick: pickFromProb(pR), pR, pB: 1 - pR };
    },
    update(st, x) {
      st.buf.push(x);
      if (x === "R") st.r++;
      else st.b++;
      if (st.buf.length > W) {
        const y = st.buf.shift();
        if (y === "R") st.r--;
        else st.b--;
      }
    },
  };
}

function makeMarkov(order, { raw, window } = {}) {
  const W = window == null ? Infinity : window;
  const id = raw
    ? `pattern-${order}`
    : window == null
      ? `markov-${order}`
      : `markov-${order}-w${window}`;
  const name = raw
    ? `Pattern-frequency last-${order}`
    : window == null
      ? `Markov order ${order} (Laplace)`
      : `Markov order ${order} W${window} (Laplace)`;
  return {
    id,
    name,
    family: raw ? "pattern" : "markov",
    window,
    order,
    init() {
      return { buf: [], map: new Map() };
    },
    _get(st, state) {
      let rec = st.map.get(state);
      if (!rec) {
        rec = { r: 0, b: 0 };
        st.map.set(state, rec);
      }
      return rec;
    },
    _probFor(st, k, rawMode) {
      if (st.buf.length < k) return null;
      const state = keyOf(st.buf.slice(st.buf.length - k));
      const rec = st.map.get(state) || { r: 0, b: 0 };
      const n = rec.r + rec.b;
      if (rawMode) {
        if (n === 0) return null;
        return rec.r / n;
      }
      return (rec.r + LAPLACE_A) / (n + LAPLACE_A + LAPLACE_B);
    },
    predict(st) {
      if (raw) {
        for (let k = order; k >= 1; k--) {
          const pR = this._probFor(st, k, true);
          if (pR == null) continue;
          const pick = pickFromProb(pR);
          if (pick != null || k === 1) return { pick, pR, pB: 1 - pR, usedOrder: k };
        }
        return { pick: null, pR: 0.5, pB: 0.5, usedOrder: 0 };
      }
      const pR = this._probFor(st, order, false);
      if (pR == null) return { pick: null, pR: 0.5, pB: 0.5 };
      return { pick: pickFromProb(pR), pR, pB: 1 - pR };
    },
    update(st, x) {
      if (st.buf.length >= order) {
        const state = keyOf(st.buf.slice(st.buf.length - order));
        const rec = this._get(st, state);
        if (x === "R") rec.r++;
        else rec.b++;
      }
      st.buf.push(x);
      if (st.buf.length > W) {
        // Drop the oldest symbol and the transition it started, if any
        // Window applies to the number of symbols; remove the transition
        // from buf[0]..buf[order-1] -> buf[order] before shifting.
        // After a long run buf length is W+? We cap symbols at W.
        // When we exceeded W we need to undo the transition that used the
        // dropped symbol as the first element of the state.
        // Sequence after push has length W+1. Oldest is buf[0], the
        // transition starting at 0 is state buf[0..order-1] -> buf[order].
        if (st.buf.length - 1 >= order) {
          const state = keyOf(st.buf.slice(0, order));
          const y = st.buf[order];
          const rec = st.map.get(state);
          if (rec) {
            if (y === "R") rec.r = Math.max(0, rec.r - 1);
            else rec.b = Math.max(0, rec.b - 1);
            if (rec.r === 0 && rec.b === 0) st.map.delete(state);
          }
        }
        st.buf.shift();
      }
    },
  };
}

function makeStreakFollow() {
  return {
    id: "streak-follow",
    name: "Streak-follow",
    family: "streak",
    init() {
      return { last: null, streak: 0, n: 0, repeats: 0 };
    },
    predict(st) {
      if (st.last == null) return { pick: null, pR: 0.5, pB: 0.5 };
      const pContinue = st.n > 0 ? (st.repeats + LAPLACE_A) / (st.n + LAPLACE_A + LAPLACE_B) : 0.5;
      const pR = st.last === "R" ? pContinue : 1 - pContinue;
      return { pick: st.last, pR, pB: 1 - pR };
    },
    update(st, x) {
      if (st.last != null) {
        st.n++;
        if (x === st.last) st.repeats++;
      }
      st.streak = st.last === x ? st.streak + 1 : 1;
      st.last = x;
    },
  };
}

function makeStreakReverse() {
  return {
    id: "streak-reverse",
    name: "Streak-reverse (anti-gambler)",
    family: "streak",
    init() {
      return { last: null, streak: 0, n2: 0, breaks2: 0 };
    },
    predict(st) {
      // Only fade a run once it is at least length 2; otherwise abstain.
      // This is distinct from strict alternation (always flip the last colour).
      if (st.last == null || st.streak < 2) return { pick: null, pR: 0.5, pB: 0.5 };
      const pBreak = st.n2 > 0 ? (st.breaks2 + LAPLACE_A) / (st.n2 + LAPLACE_A + LAPLACE_B) : 0.5;
      const pick = st.last === "R" ? "B" : "R";
      const pR = pick === "R" ? pBreak : 1 - pBreak;
      return { pick, pR, pB: 1 - pR };
    },
    update(st, x) {
      if (st.last != null && st.streak >= 2) {
        st.n2++;
        if (x !== st.last) st.breaks2++;
      }
      st.streak = st.last === x ? st.streak + 1 : 1;
      st.last = x;
    },
  };
}

function makeAlternation() {
  return {
    id: "alternation",
    name: "Alternation",
    family: "alternation",
    init() {
      return { last: null, n: 0, switches: 0 };
    },
    predict(st) {
      if (st.last == null) return { pick: null, pR: 0.5, pB: 0.5 };
      const pSwitch = st.n > 0 ? (st.switches + LAPLACE_A) / (st.n + LAPLACE_A + LAPLACE_B) : 0.5;
      const pick = st.last === "R" ? "B" : "R";
      const pR = pick === "R" ? pSwitch : 1 - pSwitch;
      return { pick, pR, pB: 1 - pR };
    },
    update(st, x) {
      if (st.last != null) {
        st.n++;
        if (x !== st.last) st.switches++;
      }
      st.last = x;
    },
  };
}


function sigmoid(z) {
  const zz = Math.max(-20, Math.min(20, z));
  return 1 / (1 + Math.exp(-zz));
}

function makeLogistic() {
  const LAGS = 10;
  const LR = 0.08;
  const L2 = 0.001;
  const nFeat = 1 + LAGS + 2;
  return {
    id: "logistic-online",
    name: "Online logistic (lags 1–10 + run length)",
    family: "ml",
    init() {
      return { w: new Float64Array(nFeat), buf: [], streak: 0, last: null };
    },
    features(st) {
      const f = new Float64Array(nFeat);
      f[0] = 1;
      for (let i = 0; i < LAGS; i++) {
        const idx = st.buf.length - 1 - i;
        f[1 + i] = idx >= 0 ? (st.buf[idx] === "R" ? 1 : 0) : 0.5;
      }
      f[1 + LAGS] = st.streak / 10;
      f[2 + LAGS] = st.last === "R" ? 1 : st.last === "B" ? -1 : 0;
      return f;
    },
    predict(st) {
      if (st.buf.length === 0) return { pick: null, pR: 0.5, pB: 0.5 };
      const f = this.features(st);
      let z = 0;
      for (let i = 0; i < nFeat; i++) z += st.w[i] * f[i];
      const pR = sigmoid(z);
      return { pick: pickFromProb(pR), pR, pB: 1 - pR };
    },
    update(st, x) {
      const y = x === "R" ? 1 : 0;
      if (st.buf.length > 0) {
        const f = this.features(st);
        let z = 0;
        for (let i = 0; i < nFeat; i++) z += st.w[i] * f[i];
        const p = sigmoid(z);
        for (let i = 0; i < nFeat; i++) {
          st.w[i] -= LR * ((p - y) * f[i] + L2 * st.w[i]);
        }
      }
      st.buf.push(x);
      if (st.buf.length > 16) st.buf.shift();
      st.streak = st.last === x ? st.streak + 1 : 1;
      st.last = x;
    },
  };
}

export function buildStrategies() {
  const list = [];
  list.push(makeMajority(null, false));
  for (const w of WINDOWS) list.push(makeMajority(w, false));
  list.push(makeMajority(null, true));
  for (const w of [10, 20, 50, 100]) list.push(makeMajority(w, true));
  for (const k of [1, 2, 3, 4]) list.push(makeMarkov(k, { raw: false }));
  for (const k of [3, 4, 5, 6]) list.push(makeMarkov(k, { raw: true }));
  list.push(makeStreakFollow());
  list.push(makeStreakReverse());
  list.push(makeAlternation());
  list.push(makeLogistic());
  return list;
}

/**
 * Walk-forward evaluation: at each t, predict from seq[0..t-1] only, then update with seq[t].
 * Returns per-strategy hits/n, accuracy, Wilson CI, one-sided binomial p vs 50%, and the
 * prediction that would be made for the NEXT outcome (after the full sequence).
 */

export function reliabilityBins(pairs, nBins = 10) {
  const bins = [];
  for (let i = 0; i < nBins; i++) bins.push({ lo: i / nBins, hi: (i + 1) / nBins, n: 0, sumP: 0, sumY: 0 });
  for (const { p, y } of pairs) {
    const i = Math.min(nBins - 1, Math.max(0, Math.floor(p * nBins)));
    bins[i].n++;
    bins[i].sumP += p;
    bins[i].sumY += y;
  }
  return bins.map((b) => ({
    lo: b.lo,
    hi: b.hi,
    n: b.n,
    meanP: b.n ? b.sumP / b.n : null,
    freq: b.n ? b.sumY / b.n : null,
  }));
}

export function walkForward(seq, strategies) {
  const strats = strategies || buildStrategies();
  const results = [];
  for (const s of strats) {
    const st = s.init();
    let hits = 0;
    let n = 0;
    let brierSum = 0;
    let brierN = 0;
    let loglossSum = 0;
    const pairs = [];
    // Trailing window hits for reporting
    const trail = [];
    let seenR = 0;
    let seenN = 0;
    let vsBaseHits = 0;
    let vsBaseN = 0;
    let brierBaseSum = 0;
    for (let t = 0; t < seq.length; t++) {
      const pred = s.predict(st);
      const y = seq[t] === "R" ? 1 : 0;
      const pR = pred.pR;
      if (pR != null && Number.isFinite(pR)) {
        const p = Math.min(1 - 1e-12, Math.max(1e-12, pR));
        brierSum += (p - y) ** 2;
        loglossSum += -(y * Math.log(p) + (1 - y) * Math.log(1 - p));
        brierN++;
        pairs.push({ p, y });
        if (seenN > 0) {
          const pBase = seenR / seenN;
          brierBaseSum += (pBase - y) ** 2;
        }
      }
      if (pred.pick != null) {
        const ok = pred.pick === seq[t];
        if (ok) hits++;
        n++;
        trail.push(ok);
        if (seenN > 0) {
          const basePick = seenR > seenN - seenR ? "R" : seenR < seenN - seenR ? "B" : null;
          if (basePick != null) {
            vsBaseN++;
            if ((pred.pick === seq[t]) === (basePick === seq[t])) {
              /* tie with base — counted separately via accuracies */
            }
            if (pred.pick === seq[t]) vsBaseHits += 0; // placeholder kept for structure
          }
        }
      }
      s.update(st, seq[t]);
      seenN++;
      if (seq[t] === "R") seenR++;
    }
    const next = s.predict(st);
    const acc = n > 0 ? hits / n : null;
    const pValue = n > 0 ? binomialPValueGte(hits, n, 0.5) : null;
    const ci = n > 0 ? wilsonCI(hits, n) : null;
    const trailing = {};
    for (const w of WINDOWS) {
      if (trail.length === 0) {
        trailing[w] = { n: 0, hits: 0, acc: null };
        continue;
      }
      const slice = trail.slice(-w);
      let h = 0;
      for (const ok of slice) if (ok) h++;
      trailing[w] = { n: slice.length, hits: h, acc: h / slice.length };
    }
    const brier = brierN > 0 ? brierSum / brierN : null;
    const brier50 = 0.25;
    const brierBase = brierN > 0 && seenN > 1 ? brierBaseSum / Math.max(1, brierN - 1) : null;
    const logloss = brierN > 0 ? loglossSum / brierN : null;
    const calib = reliabilityBins(pairs, 10);
    results.push({
      id: s.id,
      name: s.name,
      family: s.family,
      hits,
      n,
      acc,
      accMinus50: acc != null ? acc - 0.5 : null,
      pValue,
      ci,
      brier,
      brier50,
      brierMinus50: brier != null ? brier - brier50 : null,
      brierBase,
      brierMinusBase: brier != null && brierBase != null ? brier - brierBase : null,
      logloss,
      calib,
      next,
      trailing,
    });
  }
  return results;
}

export function evaluateStrategySignificance(row, nTests) {
  const eligible = row.n >= MIN_N_SIGNIFICANCE && row.pValue != null && row.acc != null && row.acc > 0.5;
  const rawSig = eligible && row.pValue < ALPHA;
  const correctedP = row.pValue != null && nTests > 0 ? Math.min(1, row.pValue * nTests) : null;
  const bonferroniSig = eligible && correctedP != null && correctedP < ALPHA;
  return { eligible, rawSig, correctedP, bonferroniSig };
}

/**
 * Ensemble: the strategy with the best walk-forward accuracy, used as a call
 * only if it is significant after Bonferroni correction among eligible strategies.
 * Ties in accuracy are broken by smaller p-value, then larger n.
 */
export function ensembleFromResults(results) {
  const eligiblePool = results.filter((r) => r.n >= MIN_N_SIGNIFICANCE && r.acc != null);
  const nTests = Math.max(1, eligiblePool.length);
  const annotated = results.map((r) => ({ ...r, sig: evaluateStrategySignificance(r, nTests) }));
  const ranked = annotated
    .filter((r) => r.n >= MIN_N_SIGNIFICANCE && r.acc != null)
    .slice()
    .sort((a, b) => {
      const d = (b.acc ?? 0) - (a.acc ?? 0);
      if (d !== 0) return d;
      const p = (a.pValue ?? 1) - (b.pValue ?? 1);
      if (p !== 0) return p;
      return (b.n ?? 0) - (a.n ?? 0);
    });
  const best = ranked[0] || null;
  const use = best && best.sig.bonferroniSig && best.next && best.next.pick != null;
  return { annotated, nTests, best, use };
}

export function confidenceLabel({ use, n, ci, bonferroniSig }) {
  if (!use || !bonferroniSig || n < MIN_N_SIGNIFICANCE) return "Low";
  if (n >= 100 && ci && ci.lo > 0.52) return "High";
  if (n >= MIN_N_SIGNIFICANCE && ci && ci.lo > 0.5) return "Medium";
  return "Low";
}

export function chanceOdds(streakLen) {
  if (streakLen <= 0) return { n: 0, p: 1, label: "—" };
  const p = Math.pow(0.5, streakLen);
  const denom = Math.round(1 / p);
  return { n: streakLen, p, label: `1 in ${denom.toLocaleString("en-US")} (0.5^${streakLen})` };
}

function fmtPct(x, digits = 1) {
  if (x == null || !Number.isFinite(x)) return "—";
  return (x * 100).toFixed(digits) + "%";
}

function fmtP(p) {
  if (p == null || !Number.isFinite(p)) return "—";
  if (p < 1e-6) return p.toExponential(2);
  if (p < 0.001) return p.toExponential(2);
  return p.toFixed(3);
}

export function buildReason({ ensemble, tests, seq }) {
  const n = seq.length;
  const nR = seq.filter((c) => c === "R").length;
  const nB = n - nR;
  const { annotated, nTests, best, use } = ensemble;
  const randSig = tests.significantBonferroni;
  const randRaw = tests.significantRaw;

  if (!use) {
    let text =
      "No predictive edge detected - results consistent with a fair random game";
    if (n === 0) {
      return "No results yet. Until evidence appears, the next colour is treated as a fair 50/50. Enter outcomes one at a time after the prediction is locked.";
    }
    if (best) {
      const ci = best.ci;
      text +=
        `. Best walk-forward was ${best.name} at ${fmtPct(best.acc)} (${best.hits}/${best.n}), ` +
        `one-sided binomial p=${fmtP(best.pValue)} vs 50%, Bonferroni-adjusted p=${fmtP(best.sig.correctedP)} ` +
        `over ${nTests} eligible strateg${nTests === 1 ? "y" : "ies"}` +
        (ci ? `, Wilson 95% CI ${fmtPct(ci.lo)}–${fmtPct(ci.hi)}` : "") +
        ". That is not significant after multiple-testing correction, so the engine will not override a 50/50 call.";
    } else {
      text += `. Only ${n} result${n === 1 ? "" : "s"} recorded (${nR} red, ${nB} black). A strategy needs at least ${MIN_N_SIGNIFICANCE} scored walk-forward predictions before it can be tested against chance.`;
    }
    if (randSig.length) {
      text +=
        " Note: after Bonferroni correction, randomness test(s) rejected a fair i.i.d. coin: " +
        randSig.map((t) => `${t.name} p=${fmtP(t.p)}`).join("; ") +
        ". That is a hint of structure, not by itself a usable next-colour call — wait until a strategy's walk-forward edge is also significant.";
    } else if (randRaw.length && !randSig.length) {
      text +=
        ` Uncorrected randomness tests with p<0.05: ${randRaw.map((t) => `${t.name} p=${fmtP(t.p)}`).join("; ")}.` +
        ` After correcting for ${tests.nEval} tests, none remain significant.`;
    }
    return text;
  }

  const ci = best.ci;
  const pick = best.next.pick === "R" ? "RED" : "BLACK";
  const pR = best.next.pR;
  let text =
    `Ensemble selects ${best.name}: walk-forward accuracy ${fmtPct(best.acc)} (${best.hits}/${best.n}), ` +
    `one-sided binomial p=${fmtP(best.pValue)} vs 50%, Bonferroni-adjusted p=${fmtP(best.sig.correctedP)} ` +
    `over ${nTests} eligible strategies` +
    (ci ? `, Wilson 95% CI ${fmtPct(ci.lo)}–${fmtPct(ci.hi)}` : "") +
    `. Next call is ${pick} with model P(RED)=${fmtPct(pR)} and P(BLACK)=${fmtPct(1 - pR)}.`;

  const streak = currentStreak(seq);
  if (streak.length) {
    text += ` Current run: ${streak.length} ${streak.color === "R" ? "RED" : "BLACK"}.`;
  }
  text += ` Sample so far: ${n} outcomes (${nR} red, ${nB} black, ${fmtPct(nR / n)} red).`;
  if (randSig.length) {
    text +=
      " Supporting randomness tests (Bonferroni-significant): " +
      randSig.map((t) => `${t.name} p=${fmtP(t.p)}`).join("; ") +
      ".";
  }
  return text;
}

// ---------------------------------------------------------------------------
// Context / data support, change-points, timing, model confidence
// ---------------------------------------------------------------------------

/**
 * Historical occurrences of the current k-length suffix (the "context")
 * and how often the next colour was R vs B. Only uses positions whose
 * following outcome is already known — no leakage from the future.
 */
export function dataSupport(seq, ks = [3, 4]) {
  const out = [];
  for (const k of ks) {
    if (seq.length < k) {
      out.push({
        k,
        context: seq.join(""),
        count: 0,
        nextR: 0,
        nextB: 0,
        note: `Need at least ${k} outcomes to form a length-${k} context.`,
      });
      continue;
    }
    const context = seq.slice(-k).join("");
    let nextR = 0;
    let nextB = 0;
    for (let i = 0; i + k < seq.length; i++) {
      let ok = true;
      for (let j = 0; j < k; j++) {
        if (seq[i + j] !== seq[seq.length - k + j]) {
          ok = false;
          break;
        }
      }
      if (!ok) continue;
      if (seq[i + k] === "R") nextR++;
      else nextB++;
    }
    const count = nextR + nextB;
    const laplacePR = (nextR + 1) / (count + 2);
    out.push({
      k,
      context,
      count,
      nextR,
      nextB,
      pR: count > 0 ? nextR / count : 0.5,
      pRLaplace: laplacePR,
      note:
        count === 0
          ? `Current ${k}-colour context ${context} has 0 earlier occurrences (Laplace split 1–1).`
          : `Context ${context} appeared ${count} earlier time${count === 1 ? "" : "s"}; next was RED ${nextR} (${((nextR / count) * 100).toFixed(1)}%), BLACK ${nextB} (${((nextB / count) * 100).toFixed(1)}%).`,
    });
  }
  return out;
}

export function formatDataSupport(rows) {
  if (!rows || !rows.length) return "No context yet.";
  return rows.map((r) => r.note).join(" ");
}

/**
 * Two-sided CUSUM on a Bernoulli red-rate with target p0 (default 50%).
 * k is the allowance, h the decision threshold.
 */
export function cusumBernoulli(seq, { p0 = 0.5, k = 0.05, h = 8 } = {}) {
  let sp = 0;
  let sm = 0;
  let lastZeroP = 0;
  let lastZeroM = 0;
  let firstAlarm = null;
  for (let t = 0; t < seq.length; t++) {
    const x = seq[t] === "R" ? 1 : 0;
    sp = Math.max(0, sp + (x - p0) - k);
    sm = Math.max(0, sm + (p0 - x) - k);
    if (sp === 0) lastZeroP = t + 1;
    if (sm === 0) lastZeroM = t + 1;
    if (!firstAlarm && sp > h) firstAlarm = { side: "red-rate up", t: t + 1, estimateStart: lastZeroP, value: sp };
    if (!firstAlarm && sm > h) firstAlarm = { side: "red-rate down", t: t + 1, estimateStart: lastZeroM, value: sm };
  }
  const active = sp > h || sm > h;
  const alarm = active
    ? { side: sp > h ? "red-rate up" : "red-rate down", t: seq.length, estimateStart: sp > h ? lastZeroP : lastZeroM, value: sp > h ? sp : sm }
    : null;
  let summary;
  if (active) {
    summary = `CUSUM currently in alarm: ${alarm.side} (S+=${sp.toFixed(2)}, S−=${sm.toFixed(2)}, h=${h}); excursion started near outcome ${alarm.estimateStart}. Pre-change data may be stale.`;
  } else if (firstAlarm) {
    summary = `No active CUSUM alarm (S+=${sp.toFixed(2)}, S−=${sm.toFixed(2)}, h=${h}). A threshold crossing occurred earlier at outcome ${firstAlarm.t} and has since returned to baseline.`;
  } else {
    summary = `No CUSUM alarm (S+=${sp.toFixed(2)}, S−=${sm.toFixed(2)}, threshold h=${h}).`;
  }
  return {
    name: "CUSUM (Bernoulli red rate)",
    sp,
    sm,
    h,
    k,
    p0,
    alarm,
    firstAlarm,
    flagged: active,
    n: seq.length,
    summary,
  };
}

/**
 * Truncated Bayesian online change-point detection (Adams & MacKay 2007 style)
 * for a Bernoulli rate with a Beta(a0,b0) prior and constant hazard.
 */
export function bocpdBernoulli(seq, { hazard = 1 / 100, maxRun = 200, a0 = 1, b0 = 1 } = {}) {
  const n = seq.length;
  if (n === 0) {
    return { name: "Bayesian online change-point", n: 0, mapRunLength: 0, mostRecentChange: null, flagged: false, summary: "No data." };
  }
  const prefix = new Int32Array(n + 1);
  for (let i = 0; i < n; i++) prefix[i + 1] = prefix[i] + (seq[i] === "R" ? 1 : 0);

  let growth = new Float64Array(maxRun + 1);
  growth[0] = 1;
  let lastMap = 0;
  let mostRecentChange = null;

  for (let t = 0; t < n; t++) {
    const xR = seq[t] === "R";
    const lim = Math.min(t, maxRun);
    const pred = new Float64Array(lim + 1);
    for (let r = 0; r <= lim; r++) {
      const reds = r === 0 ? 0 : prefix[t] - prefix[t - r];
      const p = (a0 + reds) / (a0 + b0 + r);
      pred[r] = xR ? p : 1 - p;
    }
    const neu = new Float64Array(maxRun + 1);
    let acc0 = 0;
    for (let r = 0; r <= lim; r++) acc0 += growth[r] * pred[r] * hazard;
    neu[0] = acc0;
    for (let r = 0; r <= lim; r++) {
      const g = growth[r] * pred[r] * (1 - hazard);
      if (r + 1 <= maxRun) neu[r + 1] += g;
      else neu[maxRun] += g;
    }
    let s = 0;
    for (let i = 0; i <= maxRun; i++) s += neu[i];
    if (s <= 0) s = 1;
    for (let i = 0; i <= maxRun; i++) neu[i] /= s;
    growth = neu;

    let map = 0;
    for (let r = 1; r <= maxRun; r++) if (growth[r] > growth[map]) map = r;
    if (map <= 1 && lastMap >= 20 && t > 0) {
      mostRecentChange = { t: t + 1, previousRun: lastMap };
    }
    lastMap = map;
  }

  return {
    name: "Bayesian online change-point",
    n,
    hazard,
    mapRunLength: lastMap,
    mostRecentChange,
    flagged: !!mostRecentChange,
    summary: mostRecentChange
      ? `BOCPD: MAP run length is now ${lastMap}. A possible change was flagged at outcome ${mostRecentChange.t} after a run of ${mostRecentChange.previousRun}.`
      : `BOCPD: no change flagged. MAP run length of the current regime is ${lastMap} (hazard 1/${Math.round(1 / hazard)}).`,
  };
}

function mean(arr) {
  if (!arr.length) return 0;
  let s = 0;
  for (const x of arr) s += x;
  return s / arr.length;
}
function variance(arr) {
  if (arr.length < 2) return 0;
  const m = mean(arr);
  let s = 0;
  for (const x of arr) s += (x - m) ** 2;
  return s / (arr.length - 1);
}

function studentTCdfAbsP(t, df) {
  // two-sided p-value of |T| under Student-t(df)
  if (!Number.isFinite(t) || !(df > 0)) return 1;
  const x = df / (df + t * t);
  // P(|T|>t) = I_x(df/2, 1/2)
  const p = regularizedIncompleteBeta(df / 2, 0.5, x);
  return Math.min(1, Math.max(0, p));
}

/**
 * Optional timing-effect tests. `entries` is [{ color: 'R'|'B', ts: number|null }].
 * ts is epoch milliseconds. Returns skip if too few timestamps.
 */
export function timingTests(entries) {
  const name = "Timing effects";
  if (!entries || entries.length < 20) {
    return { name, skip: "Need at least 20 timestamped results to test timing.", pInterval: null, pHour: null };
  }
  const withTs = entries.filter((e) => e && (e.color === "R" || e.color === "B") && e.ts != null && Number.isFinite(e.ts));
  if (withTs.length < 20) {
    return { name, skip: "Need at least 20 timestamped results to test timing.", n: withTs.length, pInterval: null, pHour: null };
  }
  const logR = [];
  const logB = [];
  for (let i = 1; i < withTs.length; i++) {
    const dt = withTs[i].ts - withTs[i - 1].ts;
    if (!(dt > 0)) continue;
    const v = Math.log(dt);
    if (withTs[i].color === "R") logR.push(v);
    else logB.push(v);
  }
  let pInterval = null;
  let tStat = null;
  let df = null;
  let skipT = null;
  if (logR.length >= 8 && logB.length >= 8) {
    const mR = mean(logR);
    const mB = mean(logB);
    const vR = variance(logR);
    const vB = variance(logB);
    const nR = logR.length;
    const nB = logB.length;
    const se = Math.sqrt(vR / nR + vB / nB);
    tStat = se > 0 ? (mR - mB) / se : 0;
    df = (vR / nR + vB / nB) ** 2 / ((vR / nR) ** 2 / Math.max(1, nR - 1) + (vB / nB) ** 2 / Math.max(1, nB - 1));
    pInterval = studentTCdfAbsP(tStat, df);
  } else {
    skipT = "Not enough inter-arrival samples in both colours.";
  }

  // Hour-of-day χ² if the span covers at least 4 distinct hours and 30 points
  let pHour = null;
  let hourSkip = null;
  const hours = new Array(24).fill(0);
  let nH = 0;
  const redH = new Array(24).fill(0);
  for (const e of withTs) {
    const d = new Date(e.ts);
    if (Number.isNaN(d.getTime())) continue;
    const h = d.getHours();
    hours[h]++;
    if (e.color === "R") redH[h]++;
    nH++;
  }
  const used = [];
  for (let h = 0; h < 24; h++) if (hours[h] >= 5) used.push(h);
  if (nH >= 30 && used.length >= 4) {
    let chi2 = 0;
    const pGlobal = withTs.filter((e) => e.color === "R").length / withTs.length;
    for (const h of used) {
      const expR = hours[h] * pGlobal;
      const expB = hours[h] * (1 - pGlobal);
      if (expR <= 0 || expB <= 0) continue;
      chi2 += (redH[h] - expR) ** 2 / expR + (hours[h] - redH[h] - expB) ** 2 / expB;
    }
    const dfH = used.length - 1;
    pHour = chiSquarePValue(chi2, dfH);
  } else {
    hourSkip = "Need 30+ timestamps across at least 4 hours (with 5+ each) for an hour-of-day test.";
  }

  const flagged = (pInterval != null && pInterval < ALPHA) || (pHour != null && pHour < ALPHA);
  let summary;
  if (pInterval != null) {
    summary = `Welch t-test on log inter-arrival times before RED vs BLACK: t=${tStat.toFixed(2)} (df≈${df.toFixed(1)}), two-sided p=${pInterval < 0.001 ? pInterval.toExponential(2) : pInterval.toFixed(3)}.`;
  } else {
    summary = skipT || "Interval test not run.";
  }
  if (pHour != null) summary += ` Hour-of-day χ² p=${pHour < 0.001 ? pHour.toExponential(2) : pHour.toFixed(3)}.`;
  else if (hourSkip) summary += " " + hourSkip;
  if (!flagged) summary += " No significant timing effect at 5%.";

  return {
    name,
    skip: null,
    n: withTs.length,
    pInterval,
    tStat,
    df,
    pHour,
    flagged,
    summary,
  };
}

/**
 * MODEL CONFIDENCE % = 100 * (1 − Bonferroni-adjusted one-sided p-value of
 * the best walk-forward model versus a 50/50 baseline). This is evidence
 * strength, NOT the probability that the next colour is correct.
 */
export function modelConfidencePct(ensemble) {
  const best = ensemble && ensemble.best;
  if (!best || best.pValue == null || best.n < MIN_N_SIGNIFICANCE) {
    return { pct: 0, pAdjusted: null, pRaw: null, bestName: null, note: "Insufficient walk-forward samples (need ≥ 30) to compute model confidence." };
  }
  const pAdj = best.sig ? best.sig.correctedP : best.pValue;
  const pct = 100 * (1 - pAdj);
  return {
    pct,
    pAdjusted: pAdj,
    pRaw: best.pValue,
    bestName: best.name,
    significant: !!(best.sig && best.sig.bonferroniSig),
    note:
      `Model confidence is 1 − p (Bonferroni-adjusted p=${pAdj < 0.001 ? pAdj.toExponential(2) : pAdj.toFixed(3)}) ` +
      `that ${best.name} is no better than a fair 50/50 coin on walk-forward data. ` +
      `It is not the probability that the next colour is correct.`,
  };
}

export function patternDetected(analysis) {
  if (!analysis || !analysis.use || !analysis.ensemble || !analysis.ensemble.best) {
    return { text: "none significant", detail: "No strategy beats chance at a Bonferroni-corrected 5% level." };
  }
  const b = analysis.ensemble.best;
  const pick = b.next && b.next.pick === "R" ? "RED" : b.next && b.next.pick === "B" ? "BLACK" : "—";
  const acc = b.acc != null ? (b.acc * 100).toFixed(1) + "%" : "—";
  const detail =
    `${b.name} selected: walk-forward ${acc} (${b.hits}/${b.n}), next call ${pick}` +
    (b.next && b.next.pR != null ? `, model P(RED)=${(b.next.pR * 100).toFixed(1)}%` : "") +
    ".";
  return { text: b.name, detail };
}

export function baselineScores(seq) {
  // Walking base-rate (majority of history so far) and constant 50/50 Brier.
  let r = 0;
  let hits = 0;
  let n = 0;
  let brierBase = 0;
  let brierN = 0;
  for (let t = 0; t < seq.length; t++) {
    const y = seq[t] === "R" ? 1 : 0;
    if (t > 0) {
      const pBase = r / t;
      brierBase += (pBase - y) ** 2;
      brierN++;
      const pick = r > t - r ? "R" : r < t - r ? "B" : null;
      if (pick != null) {
        n++;
        if (pick === seq[t]) hits++;
      }
    }
    if (seq[t] === "R") r++;
  }
  return {
    majority: { hits, n, acc: n ? hits / n : null, pValue: n ? binomialPValueGte(hits, n, 0.5) : null },
    brier50: 0.25,
    brierBase: brierN ? brierBase / brierN : null,
  };
}

export function analyze(seq, { timestamps } = {}) {
  const results = walkForward(seq);
  const ensemble = ensembleFromResults(results);
  const tests = allRandomnessTests(seq);
  const use = ensemble.use;
  const best = ensemble.best;
  let prediction, pRed, pBlack;
  if (use) {
    prediction = best.next.pick;
    pRed = best.next.pR;
    pBlack = 1 - pRed;
  } else {
    prediction = null;
    pRed = 0.5;
    pBlack = 0.5;
  }
  const nForConf = use ? best.n : seq.length;
  const conf = confidenceLabel({
    use,
    n: nForConf,
    ci: use ? best.ci : null,
    bonferroniSig: use ? best.sig.bonferroniSig : false,
  });
  const reason = buildReason({ ensemble, tests, seq });
  const noEdgeBanner =
    !use && tests.significantBonferroni.length === 0
      ? "No predictive edge detected - results consistent with a fair random game"
      : null;
  const support = dataSupport(seq, [3, 4]);
  const confPct = modelConfidencePct(ensemble);
  const pattern = patternDetected({ use, ensemble });
  const cusum = cusumBernoulli(seq);
  const bocpd = bocpdBernoulli(seq);
  const baselines = baselineScores(seq);
  let timing = null;
  if (timestamps && timestamps.length) {
    const entries = seq.map((color, i) => ({ color, ts: timestamps[i] != null ? timestamps[i] : null }));
    timing = timingTests(entries);
  }
  if (cusum.flagged || bocpd.flagged) {
    // annotate reason with change-point caution (does not rewrite locked history)
  }
  return {
    prediction,
    pRed,
    pBlack,
    confidence: conf,
    modelConfidence: confPct,
    patternDetected: pattern,
    dataSupport: support,
    dataSupportText: formatDataSupport(support),
    reason,
    noEdgeBanner,
    use,
    ensemble,
    tests,
    results: ensemble.annotated,
    baselines,
    cusum,
    bocpd,
    timing,
    n: seq.length,
  };
}

export function lockedPredictionRecord(analysis, now = Date.now()) {
  const pred = analysis.prediction;
  return {
    ts: now,
    prediction: pred,
    pRed: analysis.pRed,
    pBlack: analysis.pBlack,
    confidence: analysis.confidence,
    modelConfidencePct: analysis.modelConfidence ? analysis.modelConfidence.pct : 0,
    strategy: analysis.use && analysis.ensemble.best ? analysis.ensemble.best.id : "no-edge",
    strategyName: analysis.use && analysis.ensemble.best ? analysis.ensemble.best.name : "No edge (50/50)",
    pattern: analysis.patternDetected ? analysis.patternDetected.text : "none significant",
    dataSupport: analysis.dataSupportText,
    reason: analysis.reason,
  };
}

// ===========================================================================
// OMNIPREDICT — five engines + performance-weighted, shrinkage-aware master
// ===========================================================================

export const OMNI_ETA = 14;
export const OMNI_RHO = 0.45;
export const OMNI_PRIOR_N = 20;
export const OMNI_LAMBDA = 0.97;
export const OMNI_MIN_N_CALL = 30;
const P_CLIP = 1e-6;
const LOGLOSS_50 = Math.log(2);

function clipP(p) {
  return Math.min(1 - P_CLIP, Math.max(P_CLIP, p));
}
export function logLoss(p, y) {
  const q = clipP(p);
  return -(y * Math.log(q) + (1 - y) * Math.log(1 - q));
}
export function brierScore(p, y) {
  return (p - y) ** 2;
}

export function formatLastColours(seq, max = 5) {
  const s = seq.slice(-max);
  if (!s.length) return "—";
  return s.join("-");
}

export function colourFromP(p) {
  if (p == null || !Number.isFinite(p)) return "EVEN";
  if (p > 0.5) return "RED";
  if (p < 0.5) return "BLACK";
  return "EVEN";
}

function recGet(map, key) {
  let rec = map.get(key);
  if (!rec) {
    rec = { r: 0, b: 0 };
    map.set(key, rec);
  }
  return rec;
}

function laplacePR(rec) {
  const n = rec.r + rec.b;
  return (rec.r + 1) / (n + 2);
}

function makeSequenceEngine() {
  const maps = [null, new Map(), new Map(), new Map(), new Map(), new Map()];
  return {
    id: "sequence",
    name: "SEQUENCE",
    buf: [],
    maps,
    lastDetail: "",
    usedK: 0,
    predict() {
      const buf = this.buf;
      for (let k = 5; k >= 1; k--) {
        if (buf.length < k) continue;
        const key = buf.slice(-k).join("");
        const rec = maps[k].get(key) || { r: 0, b: 0 };
        const n = rec.r + rec.b;
        const need = k === 5 ? 5 : k === 4 ? 4 : k === 3 ? 3 : k === 2 ? 2 : 0;
        if (n >= need) {
          const pR = laplacePR(rec);
          this.usedK = k;
          this.lastDetail = `Length-${k} context ${key} seen ${n} time${n === 1 ? "" : "s"} → next ${rec.r} red, ${rec.b} black (Laplace P(RED)=${(pR * 100).toFixed(1)}%).`;
          return pR;
        }
      }
      this.usedK = 0;
      this.lastDetail = "No context with enough earlier occurrences; backing off to 50/50.";
      return 0.5;
    },
    update(x) {
      const buf = this.buf;
      for (let k = 1; k <= 5; k++) {
        if (buf.length < k) continue;
        const key = buf.slice(-k).join("");
        const rec = recGet(maps[k], key);
        if (x === "R") rec.r++;
        else rec.b++;
      }
      buf.push(x);
    },
  };
}

function makeMarkovOrderEngine() {
  const Kmax = 4;
  const maps = [null, new Map(), new Map(), new Map(), new Map()];
  const ll = [0, 0, 0, 0, 0];
  const nn = [0, 0, 0, 0, 0];
  return {
    id: "markov",
    name: "MARKOV",
    buf: [],
    maps,
    ll,
    nn,
    lastDetail: "",
    chosenK: 0,
    lastPs: {},
    predict() {
      const buf = this.buf;
      const ps = {};
      for (let k = 1; k <= Kmax; k++) {
        if (buf.length < k) continue;
        const key = buf.slice(-k).join("");
        const rec = maps[k].get(key) || { r: 0, b: 0 };
        ps[k] = laplacePR(rec);
      }
      this.lastPs = ps;
      const candidates = Object.keys(ps).map(Number);
      if (!candidates.length) {
        this.chosenK = 0;
        this.lastDetail = "No transitions yet.";
        return 0.5;
      }
      let bestK = candidates[0];
      let bestAvg = nn[bestK] > 0 ? ll[bestK] / nn[bestK] : Infinity;
      for (const k of candidates) {
        const avg = nn[k] > 0 ? ll[k] / nn[k] : Infinity;
        if (avg < bestAvg - 1e-15 || (avg === Infinity && bestAvg === Infinity && k < bestK)) {
          // Prefer the order with the lowest walk-forward log-loss; if none have
          // been scored yet, start at the lowest order that exists (less overfit).
          if (avg !== Infinity || bestAvg !== Infinity) {
            bestAvg = avg;
            bestK = k;
          }
        } else if (avg !== Infinity && Math.abs(avg - bestAvg) < 1e-15 && k < bestK) {
          bestK = k;
        }
      }
      if (bestAvg === Infinity) bestK = Math.min(...candidates);
      this.chosenK = bestK;
      const parts = candidates.map((k) => `k=${k} logloss ${nn[k] ? (ll[k] / nn[k]).toFixed(3) : "—"}`).join(", ");
      this.lastDetail = `Order ${bestK} selected by walk-forward log-loss. ${parts}. Laplace P(RED)=${(ps[bestK] * 100).toFixed(1)}%.`;
      return ps[bestK];
    },
    update(x) {
      const buf = this.buf;
      const y = x === "R" ? 1 : 0;
      for (let k = 1; k <= Kmax; k++) {
        if (buf.length < k) continue;
        const key = buf.slice(-k).join("");
        const rec = maps[k].get(key) || { r: 0, b: 0 };
        const p = laplacePR(rec);
        ll[k] += logLoss(p, y);
        nn[k]++;
        const recU = recGet(maps[k], key);
        if (x === "R") recU.r++;
        else recU.b++;
      }
      buf.push(x);
    },
  };
}

function makeBayesianForgetEngine(lambda = OMNI_LAMBDA) {
  return {
    id: "bayesian",
    name: "BAYESIAN",
    lambda,
    alpha: 1,
    beta: 1,
    lastDetail: "",
    predict() {
      const pR = this.alpha / (this.alpha + this.beta);
      this.lastDetail = `Beta-Binomial with forget λ=${this.lambda}: posterior Beta(${this.alpha.toFixed(2)}, ${this.beta.toFixed(2)}), P(RED)=${(pR * 100).toFixed(1)}%.`;
      return pR;
    },
    update(x) {
      this.alpha = this.lambda * this.alpha + (x === "R" ? 1 : 0);
      this.beta = this.lambda * this.beta + (x === "B" ? 1 : 0);
    },
  };
}

function makeMLEngine() {
  const LAGS = 10;
  const LR = 0.08;
  const L2 = 0.001;
  const nFeat = 1 + LAGS + 2;
  return {
    id: "ml",
    name: "MACHINE LEARNING",
    w: new Float64Array(nFeat),
    buf: [],
    streak: 0,
    last: null,
    lastDetail: "",
    features() {
      const f = new Float64Array(nFeat);
      f[0] = 1;
      for (let i = 0; i < LAGS; i++) {
        const idx = this.buf.length - 1 - i;
        f[1 + i] = idx >= 0 ? (this.buf[idx] === "R" ? 1 : 0) : 0.5;
      }
      f[1 + LAGS] = this.streak / 10;
      f[2 + LAGS] = this.last === "R" ? 1 : this.last === "B" ? -1 : 0;
      return f;
    },
    predict() {
      if (this.buf.length === 0) {
        this.lastDetail = "No observations yet; prior P(RED)=50%.";
        return 0.5;
      }
      const f = this.features();
      let z = 0;
      for (let i = 0; i < nFeat; i++) z += this.w[i] * f[i];
      z = Math.max(-20, Math.min(20, z));
      const pR = 1 / (1 + Math.exp(-z));
      this.lastDetail = `Online logistic (lags 1–10 + run length), walk-forward updates only. P(RED)=${(pR * 100).toFixed(1)}%.`;
      return pR;
    },
    update(x) {
      const y = x === "R" ? 1 : 0;
      if (this.buf.length > 0) {
        const f = this.features();
        let z = 0;
        for (let i = 0; i < nFeat; i++) z += this.w[i] * f[i];
        z = Math.max(-20, Math.min(20, z));
        const p = 1 / (1 + Math.exp(-z));
        for (let i = 0; i < nFeat; i++) {
          this.w[i] -= LR * ((p - y) * f[i] + L2 * this.w[i]);
        }
      }
      this.buf.push(x);
      if (this.buf.length > 16) this.buf.shift();
      this.streak = this.last === x ? this.streak + 1 : 1;
      this.last = x;
    },
  };
}

function makeAnomalyEngine() {
  return {
    id: "anomaly",
    name: "ANOMALY",
    n: 0,
    nR: 0,
    last: null,
    runs: 0,
    trans: { RR: 0, RB: 0, BR: 0, BB: 0 },
    sp: 0,
    sm: 0,
    lastZeroP: 0,
    lastZeroM: 0,
    window: [],
    lastDetail: "",
    lastMessage: "Insufficient evidence",
    predict() {
      const verdict = this.inspect();
      this.lastMessage = verdict.message;
      this.lastDetail = verdict.detail;
      return verdict.pR; // null if no contribution
    },
    inspect() {
      const n = this.n;
      if (n < OMNI_MIN_N_CALL) {
        return {
          pR: null,
          message: "Insufficient evidence",
          detail: `Need at least ${OMNI_MIN_N_CALL} outcomes before anomaly tests are scored (have ${n}).`,
          tests: [],
        };
      }
      const nR = this.nR;
      const nB = n - nR;
      const tests = [];

      const exp = n / 2;
      const chi2 = ((nR - exp) ** 2 + (nB - exp) ** 2) / exp;
      const pChi = chiSquarePValue(chi2, 1);
      tests.push({ name: "χ² frequency", p: pChi, z: null, stat: chi2, dir: nR === nB ? "even" : nR > nB ? "red-bias" : "black-bias" });

      if (nR > 0 && nB > 0 && n >= 12) {
        const mu = (2 * nR * nB) / n + 1;
        const varR = (2 * nR * nB * (2 * nR * nB - nR - nB)) / (n * n * (n - 1));
        const sd = Math.sqrt(Math.max(0, varR));
        const z = sd > 0 ? (this.runs - mu) / sd : 0;
        const pRuns = twoSidedNormalP(z);
        tests.push({ name: "Runs", p: pRuns, z, stat: this.runs, dir: z < 0 ? "sticky" : "alternating" });
      }

      const nG = this.trans.RR + this.trans.RB + this.trans.BR + this.trans.BB;
      if (nG >= 20) {
        const e = nG / 4;
        let chiS = 0;
        for (const k of ["RR", "RB", "BR", "BB"]) chiS += (this.trans[k] - e) ** 2 / e;
        const pSer = chiSquarePValue(chiS, 3);
        const same = this.trans.RR + this.trans.BB;
        tests.push({ name: "Serial pairs", p: pSer, z: null, stat: chiS, dir: same > nG / 2 ? "sticky" : "alternating" });
      }

      const h = 8;
      const cusumActive = this.sp > h || this.sm > h;
      if (cusumActive) {
        tests.push({
          name: "CUSUM",
          p: 0.01,
          z: null,
          stat: Math.max(this.sp, this.sm),
          dir: this.sp > this.sm ? "red-rate up" : "red-rate down",
        });
      }

      const evaluated = tests.filter((t) => t.p != null);
      const nT = Math.max(1, evaluated.length);
      const sig = evaluated.filter((t) => t.p * nT < ALPHA);
      if (!sig.length) {
        const raw = evaluated.filter((t) => t.p < ALPHA);
        return {
          pR: null,
          message: "Insufficient evidence",
          detail: raw.length
            ? `Raw p<0.05 on ${raw.map((t) => `${t.name} p=${t.p < 0.001 ? t.p.toExponential(2) : t.p.toFixed(3)}`).join("; ")}, but none survive Bonferroni correction over ${nT} tests.`
            : `No randomness or change-point test is significant after correcting for ${nT} tests.`,
          tests,
        };
      }

      sig.sort((a, b) => a.p - b.p);
      const top = sig[0];
      let pR = 0.5;
      let signal = "";
      if (top.dir === "red-bias" || top.dir === "black-bias") {
        pR = (nR + 1) / (n + 2);
        signal = `${top.dir === "red-bias" ? "red" : "black"} frequency bias (${((nR / n) * 100).toFixed(1)}% red, ${top.name} p=${top.p < 0.001 ? top.p.toExponential(2) : top.p.toFixed(3)})`;
      } else if (top.dir === "sticky" || top.dir === "alternating") {
        if (this.last == null) pR = 0.5;
        else {
          const same = this.trans.RR + this.trans.BB;
          const pStay = (same + 1) / (nG + 2);
          const pRepeat = top.dir === "sticky" ? pStay : 1 - pStay;
          pR = this.last === "R" ? pRepeat : 1 - pRepeat;
          signal = `${top.dir} dependence (${top.name} p=${top.p < 0.001 ? top.p.toExponential(2) : top.p.toFixed(3)}; P(repeat)≈${(pStay * 100).toFixed(1)}%)`;
        }
      } else if (top.dir === "red-rate up" || top.dir === "red-rate down") {
        const w = this.window;
        const wR = w.filter((c) => c === "R").length;
        pR = w.length ? (wR + 1) / (w.length + 2) : 0.5;
        signal = `possible ${top.dir} (CUSUM S+=${this.sp.toFixed(2)}, S−=${this.sm.toFixed(2)}; recent-window P(RED)=${(pR * 100).toFixed(1)}%)`;
      }
      if (pR == null || Math.abs(pR - 0.5) < 0.01) {
        return {
          pR: null,
          message: `Detected signal: ${signal || top.name}, but no directional probability away from 50%`,
          detail: "A test rejected fairness, yet the implied next-colour probability is essentially 50/50, so ANOMALY does not vote.",
          tests,
        };
      }
      return {
        pR,
        message: `Detected signal: ${signal}`,
        detail: `ANOMALY contributes P(RED)=${(pR * 100).toFixed(1)}% from the leading significant test (${top.name}).`,
        tests,
      };
    },
    update(x) {
      if (this.last != null) {
        const key = this.last + x;
        this.trans[key]++;
        if (x !== this.last) this.runs++;
      } else {
        this.runs = 1;
      }
      this.last = x;
      this.n++;
      if (x === "R") this.nR++;
      const p0 = 0.5;
      const k = 0.05;
      const xv = x === "R" ? 1 : 0;
      this.sp = Math.max(0, this.sp + (xv - p0) - k);
      this.sm = Math.max(0, this.sm + (p0 - xv) - k);
      if (this.sp === 0) this.lastZeroP = this.n;
      if (this.sm === 0) this.lastZeroM = this.n;
      this.window.push(x);
      if (this.window.length > 30) this.window.shift();
    },
  };
}

function emptyMetrics() {
  return {
    n: 0,
    hits: 0,
    acc: null,
    brierSum: 0,
    loglossSum: 0,
    brier: null,
    logloss: null,
    pValue: null,
    ci: null,
    roll: { 20: [], 50: [], 100: [] },
  };
}

function hitPush(m, pR, y) {
  if (pR == null || !Number.isFinite(pR)) return;
  const pick = pR > 0.5 ? 1 : pR < 0.5 ? 0 : null;
  if (pick != null) {
    const ok = pick === y;
    if (ok) m.hits++;
    m.n++;
    for (const w of [20, 50, 100]) {
      m.roll[w].push(ok);
      if (m.roll[w].length > w) m.roll[w].shift();
    }
  }
  m.brierSum += brierScore(pR, y);
  m.loglossSum += logLoss(pR, y);
}

function finaliseMetrics(m) {
  const scored = m.n;
  // Brier/logloss are accumulated on every probabilistic forecast, including 0.5
  // We stored brier only in hitPush when pR is not null — for 0.5, pick is null so n (hard calls)
  // may be smaller. Track forecast count separately? hitPush always adds brier when pR != null.
  // Use brierN via hits+misses for hard n, and infer brier n from... I need a forecast count.
  return m;
}

function rollingAcc(arr) {
  if (!arr.length) return null;
  let h = 0;
  for (const ok of arr) if (ok) h++;
  return { n: arr.length, hits: h, acc: h / arr.length };
}

function completeMetrics(m) {
  const forecastN = m.brierN != null ? m.brierN : null;
  m.acc = m.n > 0 ? m.hits / m.n : null;
  m.pValue = m.n > 0 ? binomialPValueGte(m.hits, m.n, 0.5) : null;
  m.ci = m.n > 0 ? wilsonCI(m.hits, m.n) : null;
  m.brier = m.brierN > 0 ? m.brierSum / m.brierN : null;
  m.logloss = m.loglossN > 0 ? m.loglossSum / m.loglossN : null;
  m.rolling = {};
  for (const w of [20, 50, 100]) m.rolling[w] = rollingAcc(m.roll[w]);
  return m;
}

function hitPush2(m, pR, y) {
  if (pR == null || !Number.isFinite(pR)) return;
  if (m.brierN == null) m.brierN = 0;
  if (m.loglossN == null) m.loglossN = 0;
  m.brierSum += brierScore(pR, y);
  m.loglossSum += logLoss(pR, y);
  m.brierN++;
  m.loglossN++;
  const pick = pR > 0.5 ? 1 : pR < 0.5 ? 0 : null;
  if (pick != null) {
    const ok = pick === y;
    if (ok) m.hits++;
    m.n++;
    for (const w of [20, 50, 100]) {
      m.roll[w].push(ok);
      if (m.roll[w].length > w) m.roll[w].shift();
    }
  }
}

function engineWeights(metricsMap, ids, eta = OMNI_ETA) {
  const raw = {};
  let sum = 0;
  for (const id of ids) {
    const m = metricsMap[id];
    const nLL = m.loglossN || 0;
    const meanLL = (m.loglossSum + LOGLOSS_50 * OMNI_PRIOR_N) / (nLL + OMNI_PRIOR_N);
    const meanB = ((m.brierSum || 0) + 0.25 * OMNI_PRIOR_N) / ((m.brierN || 0) + OMNI_PRIOR_N);
    // Blend log-loss and Brier advantages vs a 50/50 baseline.
    const adv = 0.6 * (meanLL - LOGLOSS_50) + 0.4 * (meanB - 0.25);
    let w = Math.exp(-eta * adv);
    // Down-weight engines that have not beaten a 50/50 log-loss after enough forecasts.
    if (nLL >= 40 && meanLL >= LOGLOSS_50 - 1e-9) w *= 0.2;
    raw[id] = w;
    sum += w;
  }
  const out = {};
  for (const id of ids) out[id] = sum > 0 ? raw[id] / sum : 1 / ids.length;
  return out;
}

export function shrinkEnsemble(pEns, kEff, nScores, rho = OMNI_RHO) {
  const k = Math.max(1, kEff);
  const corrShrink = 1 / (1 + (k - 1) * rho);
  const sizeShrink = (nScores || 0) / ((nScores || 0) + OMNI_PRIOR_N);
  const s = corrShrink * sizeShrink;
  const p = 0.5 + s * (pEns - 0.5);
  return { p, s, corrShrink, sizeShrink, kEff: k, rho };
}

function publishedFrom(pShrunk, rawMetrics) {
  const n = rawMetrics.n;
  const ci = n > 0 ? wilsonCI(rawMetrics.hits, rawMetrics.n) : null;
  const significant = n >= OMNI_MIN_N_CALL && ci && ci.lo > 0.5 && rawMetrics.hits / n > 0.5;
  if (!significant) return { p: 0.5, call: null, significant: false, ci, n };
  const call = pShrunk > 0.5 ? "R" : pShrunk < 0.5 ? "B" : null;
  return { p: call == null ? 0.5 : pShrunk, call, significant: call != null, ci, n };
}

/**
 * Single O(n) walk-forward pass of the engines and the master.
 * Weights at time t use only losses from 0..t-1. Predictions never see y_t.
 *
 * Optional opts.suits is a parallel array of suit ids (e.g. "R♣") the same
 * length as seq. When omitted, the suit specialist does not vote.
 */
function liveEligible(m) {
  if (!m || m.n < OMNI_MIN_N_CALL) return false;
  const ci = wilsonCI(m.hits, m.n);
  return !!(ci && ci.lo > 0.5 && m.hits / m.n > 0.5);
}

function makeFrequencyEngine() {
  let r = 0;
  let n = 0;
  return {
    id: "frequency",
    name: "FREQUENCY",
    lastDetail: "",
    predict() {
      const pR = (r + 1) / (n + 2);
      this.lastDetail = `Laplace colour rate (${r} red, ${n - r} black in ${n}) → P(RED)=${(pR * 100).toFixed(1)}%.`;
      return pR;
    },
    update(x) {
      n++;
      if (x === "R") r++;
    },
  };
}

function makeAntiMajorityEngine() {
  let r = 0;
  let n = 0;
  let fadeH = 0;
  let fadeN = 0;
  return {
    id: "antiMajority",
    name: "ANTI-MAJORITY",
    lastDetail: "",
    fading: false,
    predict() {
      const pMaj = (r + 1) / (n + 2);
      const lo = fadeN >= OMNI_MIN_N_CALL ? wilsonCI(fadeH, fadeN).lo : null;
      this.fading = fadeN >= OMNI_MIN_N_CALL && lo != null && lo > 0.5 && fadeH / fadeN > 0.5;
      if (this.fading) {
        const pR = 1 - pMaj;
        this.lastDetail = `Fade is significant (Wilson lo ${(lo * 100).toFixed(1)}%, ${fadeH}/${fadeN} against the running majority). Inverting the majority → P(RED)=${(pR * 100).toFixed(1)}%.`;
        return pR;
      }
      this.lastDetail =
        fadeN < OMNI_MIN_N_CALL
          ? `Watching whether results fade the running majority (${fadeN}/${OMNI_MIN_N_CALL} trials). Not voting yet.`
          : `No significant fade (against-majority ${fadeH}/${fadeN}, Wilson lo ${lo == null ? "—" : (lo * 100).toFixed(1)}%). Not voting.`;
      return pMaj;
    },
    update(x) {
      const pMaj = (r + 1) / (n + 2);
      const bet = pMaj > 0.5 ? "R" : pMaj < 0.5 ? "B" : null;
      if (bet != null) {
        fadeN++;
        if (x !== bet) fadeH++;
      }
      n++;
      if (x === "R") r++;
    },
  };
}

function makeShoeEngine(red = 26, black = 26) {
  let R = red;
  let B = black;
  const capR = red;
  const capB = black;
  return {
    id: "shoe",
    name: "SHOE COUNT",
    lastDetail: "",
    predict() {
      const t = R + B;
      const pR = t <= 0 ? 0.5 : R / t;
      this.lastDetail = `Remaining ${R} red / ${B} black of a ${capR + capB}-card shoe → P(RED)=${(pR * 100).toFixed(1)}%. Votes only after its own walk-forward edge is verified.`;
      return pR;
    },
    update(x) {
      if (R + B <= 0) {
        R = capR;
        B = capB;
      }
      if (x === "R") R--;
      else B--;
    },
  };
}

function makeBocpdEngine(hazard = 1 / 80, maxRun = 120) {
  let growth = new Float64Array(maxRun + 1);
  growth[0] = 1;
  const prefix = [0];
  return {
    id: "bocpd",
    name: "BOCPD",
    lastDetail: "",
    mapRun: 0,
    predict() {
      const t = prefix.length - 1;
      let p = 0;
      let s = 0;
      const lim = Math.min(t, maxRun);
      for (let r = 0; r <= lim; r++) {
        const reds = r === 0 ? 0 : prefix[t] - prefix[t - r];
        const pr = (1 + reds) / (2 + r);
        p += growth[r] * pr;
        s += growth[r];
      }
      const pR = s > 0 ? p / s : 0.5;
      this.lastDetail = `Bayesian online change-point predictive (hazard 1/${Math.round(1 / hazard)}, MAP run ${this.mapRun}). P(RED)=${(pR * 100).toFixed(1)}%.`;
      return pR;
    },
    update(x) {
      const t = prefix.length - 1;
      const xR = x === "R";
      const lim = Math.min(t, maxRun);
      const pred = new Float64Array(lim + 1);
      for (let r = 0; r <= lim; r++) {
        const reds = r === 0 ? 0 : prefix[t] - prefix[t - r];
        const pr = (1 + reds) / (2 + r);
        pred[r] = xR ? pr : 1 - pr;
      }
      const neu = new Float64Array(maxRun + 1);
      let acc0 = 0;
      for (let r = 0; r <= lim; r++) acc0 += growth[r] * pred[r] * hazard;
      neu[0] = acc0;
      for (let r = 0; r <= lim; r++) {
        const g = growth[r] * pred[r] * (1 - hazard);
        if (r + 1 <= maxRun) neu[r + 1] += g;
        else neu[maxRun] += g;
      }
      let s = 0;
      for (let i = 0; i <= maxRun; i++) s += neu[i];
      if (s <= 0) s = 1;
      for (let i = 0; i <= maxRun; i++) neu[i] /= s;
      growth = neu;
      prefix.push(prefix[t] + (xR ? 1 : 0));
      let map = 0;
      for (let r = 1; r <= maxRun; r++) if (growth[r] > growth[map]) map = r;
      this.mapRun = map;
    },
  };
}

function makeSuitEngine() {
  const map = new Map();
  let last = null;
  return {
    id: "suit",
    name: "SUIT",
    lastDetail: "",
    predict() {
      if (!last) {
        this.lastDetail = "No previous suit yet.";
        return 0.5;
      }
      const rec = map.get(last) || { r: 0, b: 0 };
      const pR = (rec.r + 1) / (rec.r + rec.b + 2);
      this.lastDetail = `After ${last}: ${rec.r} red, ${rec.b} black (Laplace P(RED)=${(pR * 100).toFixed(1)}%). Votes only after a verified walk-forward edge.`;
      return pR;
    },
    update(x, suit) {
      if (last) {
        let rec = map.get(last);
        if (!rec) {
          rec = { r: 0, b: 0 };
          map.set(last, rec);
        }
        if (x === "R") rec.r++;
        else rec.b++;
      }
      last = suit || null;
    },
  };
}

export function omniWalkForward(seq, { snapshotWeightsEvery = 0, contextOverride, suits } = {}) {
  const seqEng = makeSequenceEngine();
  const markovEng = makeMarkovOrderEngine();
  const bayesEng = makeBayesianForgetEngine();
  const mlEng = makeMLEngine();
  const anomEng = makeAnomalyEngine();
  const freqEng = makeFrequencyEngine();
  const fadeEng = makeAntiMajorityEngine();
  const shoeEng = makeShoeEngine();
  const bocpdEng = makeBocpdEngine();
  const suitEng = makeSuitEngine();
  const hasSuits = Array.isArray(suits) && suits.length === seq.length;

  const metrics = {
    sequence: emptyMetrics(),
    markov: emptyMetrics(),
    bayesian: emptyMetrics(),
    ml: emptyMetrics(),
    anomaly: emptyMetrics(),
    frequency: emptyMetrics(),
    antiMajority: emptyMetrics(),
    shoe: emptyMetrics(),
    bocpd: emptyMetrics(),
    suit: emptyMetrics(),
    masterRaw: emptyMetrics(),
    masterPublished: emptyMetrics(),
    baseline50: emptyMetrics(),
    baselineRate: emptyMetrics(),
  };
  let seenR = 0;
  const weightHistory = [];
  const idsCore = ["sequence", "markov", "bayesian", "ml", "frequency"];

  function collectVotes() {
    const pSeq = seqEng.predict();
    const pMk = markovEng.predict();
    const pBy = bayesEng.predict();
    const pMl = mlEng.predict();
    const pAn = anomEng.predict();
    const pFr = freqEng.predict();
    const pFade = fadeEng.predict();
    const pShoe = shoeEng.predict();
    const pBo = bocpdEng.predict();
    const pSu = hasSuits ? suitEng.predict() : null;
    const votes = [
      { id: "sequence", p: pSeq },
      { id: "markov", p: pMk },
      { id: "bayesian", p: pBy },
      { id: "ml", p: pMl },
      { id: "frequency", p: pFr },
    ];
    if (pAn != null) votes.push({ id: "anomaly", p: pAn });
    // Anti-majority votes only while a verified fade is active (otherwise it
    // would double-count the frequency engine).
    if (fadeEng.fading) votes.push({ id: "antiMajority", p: pFade });
    if (liveEligible(metrics.shoe)) votes.push({ id: "shoe", p: pShoe });
    if (liveEligible(metrics.bocpd)) votes.push({ id: "bocpd", p: pBo });
    if (hasSuits && liveEligible(metrics.suit)) votes.push({ id: "suit", p: pSu });
    const weightIds = idsCore.concat(votes.map((v) => v.id).filter((id) => !idsCore.includes(id)));
    const w = engineWeights(metrics, weightIds, OMNI_ETA);
    let num = 0;
    let den = 0;
    for (const v of votes) {
      const wi = w[v.id] != null ? w[v.id] : 0;
      num += wi * v.p;
      den += wi;
    }
    const pEns = den > 0 ? num / den : 0.5;
    const kEff = Math.max(1, votes.length);
    const sh = shrinkEnsemble(pEns, kEff, metrics.masterRaw.brierN || 0);
    return {
      pSeq, pMk, pBy, pMl, pAn, pFr, pFade, pShoe, pBo, pSu, votes, w, pEns, sh,
    };
  }

  for (let t = 0; t < seq.length; t++) {
    const y = seq[t] === "R" ? 1 : 0;
    const step = collectVotes();
    const pub = publishedFrom(step.sh.p, metrics.masterRaw);
    const pBase = t > 0 ? seenR / t : 0.5;

    hitPush2(metrics.sequence, step.pSeq, y);
    hitPush2(metrics.markov, step.pMk, y);
    hitPush2(metrics.bayesian, step.pBy, y);
    hitPush2(metrics.ml, step.pMl, y);
    if (step.pAn != null) hitPush2(metrics.anomaly, step.pAn, y);
    hitPush2(metrics.frequency, step.pFr, y);
    hitPush2(metrics.antiMajority, step.pFade, y);
    hitPush2(metrics.shoe, step.pShoe, y);
    hitPush2(metrics.bocpd, step.pBo, y);
    if (hasSuits) hitPush2(metrics.suit, step.pSu, y);
    hitPush2(metrics.masterRaw, step.sh.p, y);
    hitPush2(metrics.masterPublished, pub.p, y);
    hitPush2(metrics.baseline50, 0.5, y);
    hitPush2(metrics.baselineRate, pBase, y);

    if (snapshotWeightsEvery && (t % snapshotWeightsEvery === 0 || t === seq.length - 1)) {
      weightHistory.push({ t, weights: { ...step.w }, pEns: step.pEns, pShrunk: step.sh.p, published: pub.p });
    }

    seqEng.update(seq[t]);
    markovEng.update(seq[t]);
    bayesEng.update(seq[t]);
    mlEng.update(seq[t]);
    anomEng.update(seq[t]);
    freqEng.update(seq[t]);
    fadeEng.update(seq[t]);
    shoeEng.update(seq[t]);
    bocpdEng.update(seq[t]);
    if (hasSuits) suitEng.update(seq[t], suits[t]);
    if (seq[t] === "R") seenR++;
  }

  for (const k of Object.keys(metrics)) completeMetrics(metrics[k]);

  function withOverride(eng, fn) {
    if (!contextOverride || !contextOverride.length) return fn();
    const saved = eng.buf ? eng.buf.slice() : null;
    if (eng.buf) {
      const head = eng.buf.slice(0, Math.max(0, eng.buf.length - contextOverride.length));
      eng.buf.length = 0;
      if (saved.length >= contextOverride.length) {
        for (const c of head.concat(contextOverride)) eng.buf.push(c);
      } else {
        for (const c of contextOverride) eng.buf.push(c);
      }
    }
    const out = fn();
    if (saved && eng.buf) {
      eng.buf.length = 0;
      for (const c of saved) eng.buf.push(c);
    }
    return out;
  }

  const pSeqN = withOverride(seqEng, () => seqEng.predict());
  const pMkN = withOverride(markovEng, () => markovEng.predict());
  const pByN = bayesEng.predict();
  const pMlN = mlEng.predict();
  const pAnN = anomEng.predict();
  const pFrN = freqEng.predict();
  const pFadeN = fadeEng.predict();
  const pShoeN = shoeEng.predict();
  const pBoN = bocpdEng.predict();
  const pSuN = hasSuits ? suitEng.predict() : null;

  const votesN = [
    { id: "sequence", p: pSeqN },
    { id: "markov", p: pMkN },
    { id: "bayesian", p: pByN },
    { id: "ml", p: pMlN },
    { id: "frequency", p: pFrN },
  ];
  if (pAnN != null) votesN.push({ id: "anomaly", p: pAnN });
  if (fadeEng.fading) votesN.push({ id: "antiMajority", p: pFadeN });
  if (liveEligible(metrics.shoe)) votesN.push({ id: "shoe", p: pShoeN });
  if (liveEligible(metrics.bocpd)) votesN.push({ id: "bocpd", p: pBoN });
  if (hasSuits && liveEligible(metrics.suit)) votesN.push({ id: "suit", p: pSuN });
  const weightIdsN = idsCore.concat(votesN.map((v) => v.id).filter((id) => !idsCore.includes(id)));
  const wN = engineWeights(metrics, weightIdsN, OMNI_ETA);
  let num = 0;
  let den = 0;
  for (const v of votesN) {
    const wi = wN[v.id] != null ? wN[v.id] : 0;
    num += wi * v.p;
    den += wi;
  }
  const pEnsN = den > 0 ? num / den : 0.5;
  const shN = shrinkEnsemble(pEnsN, Math.max(1, votesN.length), metrics.masterRaw.brierN || 0);
  // Publish the live call from the raw walk-forward record (not from the
  // previously published 50/50 series, which delayed honest colours).
  const pubN = publishedFrom(shN.p, metrics.masterRaw);

  const ctx = contextOverride && contextOverride.length ? contextOverride : seq;
  function pack(pR, detail, extra = {}) {
    return { pR, colour: pR == null ? null : colourFromP(pR), detail, ...extra };
  }
  return {
    metrics,
    weightHistory,
    weights: wN,
    votingIds: votesN.map((v) => v.id),
    engines: {
      sequence: pack(pSeqN, seqEng.lastDetail, { usedK: seqEng.usedK }),
      markov: pack(pMkN, markovEng.lastDetail, { order: markovEng.chosenK, orders: markovEng.lastPs }),
      bayesian: pack(pByN, bayesEng.lastDetail),
      ml: pack(pMlN, mlEng.lastDetail),
      frequency: pack(pFrN, freqEng.lastDetail, { contributes: true }),
      antiMajority: pack(pFadeN, fadeEng.lastDetail, { contributes: fadeEng.fading, fading: fadeEng.fading }),
      shoe: pack(pShoeN, shoeEng.lastDetail, { contributes: liveEligible(metrics.shoe) }),
      bocpd: pack(pBoN, bocpdEng.lastDetail, { contributes: liveEligible(metrics.bocpd), mapRun: bocpdEng.mapRun }),
      suit: pack(pSuN, hasSuits ? suitEng.lastDetail : "No suit data in this log.", { contributes: !!(hasSuits && liveEligible(metrics.suit)) }),
      anomaly: {
        pR: pAnN,
        colour: pAnN == null ? null : colourFromP(pAnN),
        message: anomEng.lastMessage,
        detail: anomEng.lastDetail,
        contributes: pAnN != null,
      },
    },
    master: {
      pEns: pEnsN,
      pShrunk: shN.p,
      shrinkage: shN,
      publishedP: pubN.p,
      publishedCall: pubN.call,
      significant: pubN.significant,
      weights: wN,
    },
    lastColours: formatLastColours(ctx, 5),
    n: seq.length,
  };
}

export function omniConfidence(metricsPublished, metricsRaw) {
  // Significance follows the raw walk-forward record (the live publish gate).
  // Displayed accuracy prefers the published series once it has any hard calls.
  const raw = metricsRaw || emptyMetrics();
  const pub = metricsPublished || emptyMetrics();
  const gate = raw.n > 0 ? raw : pub;
  const significant = gate.n >= OMNI_MIN_N_CALL && gate.ci && gate.ci.lo > 0.5;
  const m = pub.n >= OMNI_MIN_N_CALL ? pub : raw;
  const n = m.n;
  const acc = m.acc;
  const ci = m.ci;
  let band = "Low";
  if (significant && n >= 100 && ci && ci.lo > 0.52) band = "High";
  else if (significant && n >= OMNI_MIN_N_CALL && ci && ci.lo > 0.5) band = "Medium";
  const accTxt = acc == null ? "—" : `${(acc * 100).toFixed(1)}% (${m.hits}/${n})`;
  const ciTxt = ci ? `Wilson 95% CI ${(ci.lo * 100).toFixed(1)}–${(ci.hi * 100).toFixed(1)}%` : "no CI yet";
  const text =
    (raw.n || 0) < OMNI_MIN_N_CALL
      ? `CONFIDENCE: Low — only ${raw.n || 0} scored master call${raw.n === 1 ? "" : "s"} so far (need ${OMNI_MIN_N_CALL}). This is measured walk-forward accuracy, not the chance the next colour is correct.`
      : `CONFIDENCE: ${band} — master walk-forward accuracy ${accTxt}, ${ciTxt}, n=${n}. This is independently measured out-of-sample performance, not the probability that the next colour is correct.` +
        (significant ? "" : " The interval still includes 50%, so there is no verified edge.");
  return {
    band,
    significant,
    acc,
    n,
    hits: m.hits,
    ci,
    text,
    rawAcc: raw.acc,
    rawN: raw.n,
  };
}

export function omniAnalyze(seq, opts = {}) {
  const wf = omniWalkForward(seq, opts);
  const conf = omniConfidence(wf.metrics.masterPublished, wf.metrics.masterRaw);
  const tests = allRandomnessTests(seq);
  const cusum = cusumBernoulli(seq);
  const bocpd = bocpdBernoulli(seq);
  const support = dataSupport(seq, [3, 4, 5]);
  const patterns = omniPatternTests(seq, opts);
  const use = wf.master.significant && wf.master.publishedCall != null;
  const prediction = use ? wf.master.publishedCall : null;
  const pRed = use ? wf.master.publishedP : 0.5;
  const pBlack = 1 - pRed;
  const noEdgeBanner =
    !use && tests.significantBonferroni.length === 0
      ? "No predictive edge detected - results consistent with a fair random game"
      : null;
  const reasonParts = [];
  reasonParts.push(
    use
      ? `Master publishes ${prediction === "R" ? "RED" : "BLACK"} at ${(pRed * 100).toFixed(1)}% after correlation shrinkage (s=${wf.master.shrinkage.s.toFixed(2)}, ρ=${OMNI_RHO}).`
      : "Master has no statistically verified walk-forward edge, so the published call is 50/50.",
  );
  const engineTitle = {
    sequence: "SEQUENCE",
    markov: "MARKOV",
    bayesian: "BAYESIAN",
    ml: "MACHINE LEARNING",
    frequency: "FREQUENCY",
  };
  for (const id of ["sequence", "markov", "bayesian", "ml", "frequency"]) {
    const e = wf.engines[id];
    const w = wf.weights[id] || 0;
    if (e) reasonParts.push(`${engineTitle[id]}: ${e.colour} ${(e.pR * 100).toFixed(1)}% (weight ${(w * 100).toFixed(1)}%). ${e.detail}`);
  }
  reasonParts.push(`ANOMALY: ${wf.engines.anomaly.message}.`);
  if (wf.engines.anomaly.contributes) {
    reasonParts.push(`ANOMALY weight ${((wf.weights.anomaly || 0) * 100).toFixed(1)}%. ${wf.engines.anomaly.detail}`);
  }
  for (const id of ["antiMajority", "shoe", "bocpd", "suit"]) {
    const e = wf.engines[id];
    if (!e) continue;
    const tag = e.contributes ? `voting, weight ${((wf.weights[id] || 0) * 100).toFixed(1)}%` : "not voting";
    reasonParts.push(`${e.detail ? id.toUpperCase() : id}: ${tag}. ${e.detail || ""}`.trim());
  }
  reasonParts.push(conf.text);
  return {
    ...wf,
    prediction,
    pRed,
    pBlack,
    confidence: conf,
    tests,
    cusum,
    bocpd,
    patterns,
    dataSupport: support,
    dataSupportText: formatDataSupport(support),
    noEdgeBanner,
    reason: reasonParts.join(" "),
    use,
  };
}

export function omniLockedRecord(analysis, now = Date.now()) {
  return {
    ts: now,
    lastColours: analysis.lastColours,
    sequence: analysis.engines.sequence,
    markov: analysis.engines.markov,
    bayesian: analysis.engines.bayesian,
    ml: analysis.engines.ml,
    frequency: analysis.engines.frequency,
    antiMajority: analysis.engines.antiMajority,
    shoe: analysis.engines.shoe,
    bocpd: analysis.engines.bocpd,
    suit: analysis.engines.suit,
    anomaly: analysis.engines.anomaly,
    weights: analysis.weights,
    votingIds: analysis.votingIds,
    prediction: analysis.prediction,
    pRed: analysis.pRed,
    pBlack: analysis.pBlack,
    pEns: analysis.master.pEns,
    pShrunk: analysis.master.pShrunk,
    significant: analysis.master.significant,
    confidenceBand: analysis.confidence.band,
    confidenceText: analysis.confidence.text,
    reason: analysis.reason,
  };
}

/**
 * Fisher's method: combine independent p-values.
 * X = -2 Σ ln(p_i) ~ χ²(2k).
 */
export function fisherCombinedP(pvalues) {
  const ps = (pvalues || []).map((p) => {
    const x = Number(p);
    if (!Number.isFinite(x)) return 1;
    return Math.min(1, Math.max(1e-16, x));
  });
  const k = ps.length;
  if (!k) return { stat: 0, df: 0, p: 1, k: 0 };
  let stat = 0;
  for (const p of ps) stat += -2 * Math.log(p);
  const df = 2 * k;
  const p = chiSquarePValue(stat, df);
  return { stat, df, p, k };
}

function lagPredictPValue(seq, L) {
  let hits = 0;
  let n = 0;
  for (let t = L; t < seq.length; t++) {
    n++;
    if (seq[t] === seq[t - L]) hits++;
  }
  if (n < 20) return { p: null, acc: null, n, lag: L };
  const acc = hits / n;
  const p = acc >= 0.5 ? binomialPValueGte(hits, n, 0.5) : binomialPValueGte(n - hits, n, 0.5);
  return { p, acc, n, hits, lag: L };
}

/**
 * Pattern tests used for side-by-side device comparison.
 * Each item has a stable `id` so Fisher's method can combine like with like.
 */
export function omniPatternTests(seq, { suits } = {}) {
  const n = seq.length;
  const out = [];
  const nR = seq.filter((c) => c === "R").length;
  const freq = chiSquareFrequency(seq);
  out.push({
    id: "bias",
    name: "Colour bias",
    p: freq.p,
    direction: nR === n - nR ? "even" : nR > n - nR ? "red" : "black",
    stat: { nR, nB: n - nR, n },
    significant: freq.p != null && freq.p < ALPHA,
  });

  const runs = runsTest(seq);
  out.push({
    id: "sticky",
    name: "Streak / alternation",
    p: runs.p,
    direction: runs.z == null ? null : runs.z < 0 ? "sticky" : "alternating",
    stat: { z: runs.z, runs: runs.runs, n },
    significant: runs.p != null && runs.p < ALPHA,
  });

  // Fade: does the outcome go against the running majority?
  let r = 0;
  let fadeH = 0;
  let fadeN = 0;
  for (let t = 0; t < n; t++) {
    if (t > 0) {
      const bet = r > t - r ? "R" : r < t - r ? "B" : null;
      if (bet != null) {
        fadeN++;
        if (seq[t] !== bet) fadeH++;
      }
    }
    if (seq[t] === "R") r++;
  }
  const fadeP = fadeN >= 20 ? binomialPValueGte(fadeH, fadeN, 0.5) : null;
  const fadeAcc = fadeN ? fadeH / fadeN : null;
  out.push({
    id: "fade",
    name: "Anti-majority fade",
    p: fadeP,
    direction: fadeAcc != null && fadeAcc > 0.5 ? "fades-majority" : fadeAcc != null && fadeAcc < 0.5 ? "follows-majority" : null,
    stat: { hits: fadeH, n: fadeN, acc: fadeAcc },
    significant: fadeP != null && fadeP < ALPHA && fadeAcc > 0.5,
  });

  // Shoe: remaining-count walk-forward vs 50/50 (52-card, 26/26).
  let R = 26;
  let B = 26;
  let shoeH = 0;
  let shoeN = 0;
  for (let t = 0; t < n; t++) {
    if (R + B <= 0) {
      R = 26;
      B = 26;
    }
    const pR = R / (R + B);
    if (pR !== 0.5) {
      const pick = pR > 0.5 ? "R" : "B";
      shoeN++;
      if (pick === seq[t]) shoeH++;
    }
    if (seq[t] === "R") R--;
    else B--;
  }
  const shoeP = shoeN >= 20 ? binomialPValueGte(shoeH, shoeN, 0.5) : null;
  const shoeAcc = shoeN ? shoeH / shoeN : null;
  out.push({
    id: "shoe",
    name: "Without-replacement / shoe",
    p: shoeP,
    direction: shoeAcc != null && shoeAcc > 0.5 ? "depletion" : null,
    stat: { hits: shoeH, n: shoeN, acc: shoeAcc },
    significant: shoeP != null && shoeP < ALPHA && shoeAcc > 0.5,
  });

  let bestLag = null;
  for (let L = 2; L <= 12; L++) {
    const row = lagPredictPValue(seq, L);
    if (!row.p || row.acc == null) continue;
    if (!bestLag || row.acc > bestLag.acc) bestLag = row;
  }
  out.push({
    id: "period",
    name: "Periodic lag",
    p: bestLag ? bestLag.p : null,
    direction: bestLag && bestLag.acc > 0.5 ? `lag-${bestLag.lag}` : null,
    stat: bestLag,
    significant: !!(bestLag && bestLag.p < ALPHA && bestLag.acc > 0.5),
  });

  const hasSuits = Array.isArray(suits) && suits.length === n;
  if (hasSuits) {
    const map = new Map();
    let hits = 0;
    let tn = 0;
    for (let t = 1; t < n; t++) {
      const key = suits[t - 1];
      if (!key) continue;
      const rec = map.get(key) || { r: 0, b: 0 };
      const tot = rec.r + rec.b;
      if (tot >= 2) {
        const pR = (rec.r + 1) / (tot + 2);
        if (pR !== 0.5) {
          tn++;
          const pick = pR > 0.5 ? "R" : "B";
          if (pick === seq[t]) hits++;
        }
      }
      if (seq[t] === "R") rec.r++;
      else rec.b++;
      map.set(key, rec);
    }
    const p = tn >= 20 ? binomialPValueGte(hits, tn, 0.5) : null;
    const acc = tn ? hits / tn : null;
    out.push({
      id: "suit",
      name: "Suit-to-next-colour",
      p,
      direction: acc != null && acc > 0.5 ? "suit-predicts-colour" : null,
      stat: { hits, n: tn, acc },
      significant: p != null && p < ALPHA && acc > 0.5,
    });
  } else {
    out.push({
      id: "suit",
      name: "Suit-to-next-colour",
      p: null,
      direction: null,
      stat: { skip: "No suit data." },
      significant: false,
    });
  }

  const cp = bocpdBernoulli(seq);
  const cu = cusumBernoulli(seq);
  out.push({
    id: "changepoint",
    name: "Regime change",
    p: cu.flagged || cp.flagged ? 0.01 : 0.5,
    direction: cu.flagged ? cu.alarm && cu.alarm.side : cp.flagged ? "bocpd" : null,
    stat: { cusum: cu.flagged, bocpd: cp.flagged, mapRun: cp.mapRunLength },
    significant: !!(cu.flagged || cp.flagged),
  });

  return out;
}

function leadingEngine(metrics) {
  const ids = ["frequency", "sequence", "markov", "bayesian", "ml", "anomaly", "antiMajority", "shoe", "bocpd", "suit"];
  let best = null;
  for (const id of ids) {
    const m = metrics[id];
    if (!m || m.n < 20 || m.acc == null) continue;
    if (!best || m.acc > best.acc) best = { id, acc: m.acc, n: m.n, hits: m.hits, ci: m.ci };
  }
  return best;
}

/**
 * Side-by-side comparison of independent device logs.
 * Predictions are never pooled here. Optional `poolSameGame` only adds a
 * diagnostic combined analysis; it does not replace per-device results.
 */
export function omniCompareDevices(devices, { poolSameGame = false } = {}) {
  const rows = [];
  for (const d of devices || []) {
    const seq = (d.seq || []).filter((c) => c === "R" || c === "B");
    const suits = Array.isArray(d.suits) && d.suits.length === seq.length ? d.suits : null;
    const analysis = omniAnalyze(seq, { suits });
    const patterns = analysis.patterns || omniPatternTests(seq, { suits });
    rows.push({
      id: d.id || d.label || `device-${rows.length + 1}`,
      label: d.label || d.id || `Device ${rows.length + 1}`,
      n: seq.length,
      nRed: seq.filter((c) => c === "R").length,
      published: analysis.use,
      prediction: analysis.prediction,
      pRed: analysis.pRed,
      raw: {
        acc: analysis.metrics.masterRaw.acc,
        n: analysis.metrics.masterRaw.n,
        hits: analysis.metrics.masterRaw.hits,
        ci: analysis.metrics.masterRaw.ci,
        significant: analysis.master.significant,
      },
      leading: leadingEngine(analysis.metrics),
      patterns,
      banner: analysis.noEdgeBanner,
    });
  }

  const patternIds = ["bias", "sticky", "fade", "shoe", "period", "suit", "changepoint"];
  const consistency = [];
  const nPat = patternIds.length;
  for (const pid of patternIds) {
    const used = [];
    for (const row of rows) {
      const pat = (row.patterns || []).find((p) => p.id === pid);
      if (pat && pat.p != null) used.push({ label: row.label, p: pat.p, significant: !!pat.significant, direction: pat.direction });
    }
    if (used.length < 2) continue;
    const combined = fisherCombinedP(used.map((u) => u.p));
    const nSig = used.filter((u) => u.significant).length;
    const dirs = used.filter((u) => u.significant && u.direction).map((u) => u.direction);
    const sameDir = dirs.length >= 2 && dirs.every((d) => d === dirs[0]);
    const adjP = Math.min(1, combined.p * nPat);
    consistency.push({
      id: pid,
      name: (rows[0].patterns.find((p) => p.id === pid) || { name: pid }).name,
      devices: used,
      combined,
      adjustedP: adjP,
      nSignificant: nSig,
      sameDirection: sameDir,
      flagged: nSig >= 2 && sameDir && adjP < 0.05,
    });
  }

  let pooled = null;
  if (poolSameGame && rows.length >= 2) {
    const seq = [];
    const suits = [];
    let anySuit = false;
    for (const d of devices) {
      const s = (d.seq || []).filter((c) => c === "R" || c === "B");
      for (let i = 0; i < s.length; i++) {
        seq.push(s[i]);
        const su = d.suits && d.suits[i];
        if (su) anySuit = true;
        suits.push(su || null);
      }
    }
    const analysis = omniAnalyze(seq, { suits: anySuit ? suits : undefined });
    pooled = {
      n: seq.length,
      published: analysis.use,
      prediction: analysis.prediction,
      pRed: analysis.pRed,
      raw: {
        acc: analysis.metrics.masterRaw.acc,
        n: analysis.metrics.masterRaw.n,
        hits: analysis.metrics.masterRaw.hits,
        ci: analysis.metrics.masterRaw.ci,
        significant: analysis.master.significant,
      },
      patterns: analysis.patterns,
      note: "Diagnostic only. This pooled view does not change any device's live predictions.",
    };
  }

  return {
    devices: rows,
    consistency,
    sharedFlags: consistency.filter((c) => c.flagged),
    poolSameGame: !!poolSameGame,
    pooled,
  };
}
