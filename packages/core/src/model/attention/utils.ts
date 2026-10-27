import * as tf from "@tensorflow/tfjs-core";

/**
 * Shared tensor utilities used across the attention variants.
 */

/**
 * Project every vector in a batched sequence through a weight matrix (with optional bias):
 * `[batch, seqLen, dIn] @ [dIn, dOut] -> [batch, seqLen, dOut]`.
 *
 * We flatten the batch and sequence axes into one so a single 2D matMul does the work,
 * then restore the shape. This avoids relying on batched-matMul broadcasting rules.
 */
export function project(input: tf.Tensor3D, weight: tf.Tensor2D, bias?: tf.Tensor1D): tf.Tensor3D {
  return tf.tidy(() => {
    const [batch, seqLen, dIn] = input.shape;
    const dOut = weight.shape[1];

    // Stack all batch*seqLen token vectors into one 2D matrix for a single matMul.
    const flat = tf.reshape(input, [batch * seqLen, dIn]);
    let projected = tf.matMul(flat, weight);
    if (bias) projected = tf.add(projected, bias);

    // Restore the sequence structure: [batch, seqLen, dOut].
    return tf.reshape(projected, [batch, seqLen, dOut]) as tf.Tensor3D;
  });
}

/**
 * A fresh `[dIn, dOut]` weight matrix with PyTorch's default `nn.Linear` initialization:
 * uniform in [-1/sqrt(dIn), 1/sqrt(dIn)]. Matching it makes training follow the book's loss
 * curve; a wider start (e.g. standard deviation 1/sqrt(dIn)) learns noticeably slower.
 */
export function linearWeight(dIn: number, dOut: number): tf.Tensor2D {
  const bound = dIn ** -0.5;
  return tf.randomUniform([dIn, dOut], -bound, bound);
}

/**
 * Add a causal mask to attention scores so position `i` cannot attend to any position
 * `j > i` (no peeking at the future). `scores` has shape `[..., seqLen, seqLen]`; the
 * mask broadcasts over any leading batch and head dimensions.
 */
export function addCausalMask<T extends tf.Tensor>(scores: T, seqLen: number): T {
  return tf.tidy(() => {
    // Lower-triangular (including the diagonal) is allowed; everything above is masked.
    const allowed = tf.linalg.bandPart(tf.ones([seqLen, seqLen]), -1, 0);
    const mask = tf.mul(tf.sub(1, allowed), -1e9); // 0 where allowed, -1e9 above the diagonal
    return tf.add(scores, mask) as T;
  });
}
