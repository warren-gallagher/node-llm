import * as tf from "@tensorflow/tfjs-core";
import type { Attention } from "./types";

/**
 * Chapter 3, lesson 2: simplified self-attention, with NO trainable weights.
 *
 * This is attention stripped to its essence, to build intuition before adding learnable
 * projections:
 *   1. score every pair of tokens by their dot product,
 *   2. scale the scores by the square root of the embedding dimension,
 *   3. softmax each row into weights that sum to 1,
 *   4. produce each context vector as the weighted sum of the inputs.
 *
 * Because there are no query/key/value projections, the output dimension equals the input
 * dimension (`dOut === dIn`).
 */
export class SimplifiedAttention implements Attention {
  /** No trainable weights. */
  readonly weights: tf.Variable[] = [];

  forward(x: tf.Tensor3D): tf.Tensor3D {
    return tf.tidy(() => {
      // scores[b, i, j] = x_i · x_j, shape [batch, seqLen, seqLen].
      const scores = tf.matMul(x, x, false, true);
      // Scale by the square root of the embedding dimension to keep the values stable.
      const dK = x.shape[x.shape.length - 1];
      const scaled = tf.div(scores, Math.sqrt(dK));
      // Softmax over the last axis turns each row into attention weights.
      const weights = tf.softmax(scaled, -1);
      // Context vectors: weighted sum of the input vectors, shape [batch, seqLen, dIn].
      return tf.matMul(weights, x) as tf.Tensor3D;
    });
  }

  dispose(): void {
    // No owned variables to free.
  }
}
