/**
 * Seeding for the demo's training run. Weight initialization, dropout and batch shuffling take no
 * seed in this repo's code, and tfjs seeds an unseeded op from `Math.random()`. Replacing
 * `Math.random` with a seeded generator therefore seeds all of them without changing
 * `packages/core`. On the same machine and thread count, the trained weights then match bit for
 * bit (see "Reproducibility" in docs/credential-demo/credential-demo.md).
 */

export const PRNG_NAME = "mulberry32 replacing Math.random";

/** mulberry32: a small 32-bit generator returning floats in [0, 1). */
export function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return function next(): number {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Replaces `Math.random` with a seeded generator and returns a function that restores it. */
export function seedMathRandom(seed: number): () => void {
  const original = Math.random;
  Math.random = mulberry32(seed);
  return function restore(): void {
    Math.random = original;
  };
}
