/**
 * Weight-difference forensics: what changed between a parent model's weights and a derived
 * model's, tensor by tensor. Different derivations leave different marks. Quantization moves
 * every value by at most half a rounding step; fine-tuning moves values further, in most tensors.
 * So the difference shows whether a derivation event's claim is plausible, even when the
 * derivation can't be rerun.
 */

export interface WeightChangeProfile {
  tensors: number;
  /** Tensors whose values are all exactly the parent's. */
  unchanged: number;
  /** Tensors that changed, but no value by more than half a step of int8 rounding. */
  withinRounding: number;
  /** Tensors with at least one value moved further than that. */
  beyondRounding: number;
  /** The largest change in a tensor, relative to the tensor's size (Euclidean norms). */
  largestRelativeChange: number;
}

/**
 * Compares two sets of float32 weights laid out tensor by tensor with these shapes. A tensor's
 * rounding step is the one int8 quantization would use for the parent's values: their largest
 * absolute value / 127.
 */
export function weightChangeProfile(
  parent: Float32Array,
  derived: Float32Array,
  shapes: number[][],
): WeightChangeProfile {
  if (parent.length !== derived.length) {
    throw new Error(`the models differ in size: ${parent.length} and ${derived.length} values`);
  }
  const profile: WeightChangeProfile = {
    tensors: shapes.length,
    unchanged: 0,
    withinRounding: 0,
    beyondRounding: 0,
    largestRelativeChange: 0,
  };
  let offset = 0;
  for (const shape of shapes) {
    const size = shape.reduce((product, dimension) => product * dimension, 1);
    const before = parent.subarray(offset, offset + size);
    const after = derived.subarray(offset, offset + size);
    offset += size;

    let maxAbs = 0;
    for (const value of before) maxAbs = Math.max(maxAbs, Math.abs(value));
    // Half a step, with room for the float32 error in value = q * scale.
    const halfStep = (Math.fround(maxAbs / 127) / 2) * (1 + 1e-5) + 1e-12;

    let changed = false;
    let beyond = false;
    let difference = 0;
    let norm = 0;
    for (let i = 0; i < size; i++) {
      const d = after[i] - before[i];
      if (d !== 0) changed = true;
      if (Math.abs(d) > halfStep) beyond = true;
      difference += d * d;
      norm += before[i] * before[i];
    }
    if (!changed) profile.unchanged++;
    else if (beyond) profile.beyondRounding++;
    else profile.withinRounding++;
    if (norm > 0) {
      profile.largestRelativeChange = Math.max(
        profile.largestRelativeChange,
        Math.sqrt(difference / norm),
      );
    }
  }
  return profile;
}
