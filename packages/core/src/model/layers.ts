import * as tf from "@tensorflow/tfjs-core";
import { linearWeight, project } from "./attention/utils";

/**
 * Chapter 4, lesson 1: layer normalization.
 *
 * Rescales each token vector to mean 0 and variance 1 across its features, then applies a
 * learned per-feature scale and shift so the model can undo the normalization if useful:
 *   y = scale * (x - mean) / sqrt(variance + eps) + shift
 *
 * Book analog: the book's `LayerNorm` module (equivalent to `nn.LayerNorm`). The variance is
 * the biased one (divide by n, not n - 1), matching PyTorch and GPT-2.
 */
export class LayerNorm {
  readonly weights: tf.Variable[];

  private readonly scale: tf.Variable;
  private readonly shift: tf.Variable;

  constructor(
    dim: number,
    private readonly eps = 1e-5,
  ) {
    this.scale = tf.variable(tf.ones([dim]));
    this.shift = tf.variable(tf.zeros([dim]));
    this.weights = [this.scale, this.shift];
  }

  /** Normalizes over the last axis; any leading axes (batch, sequence) are independent. */
  forward<T extends tf.Tensor>(x: T): T {
    return tf.tidy(() => {
      const mean = tf.mean(x, -1, true);
      const variance = tf.mean(tf.square(tf.sub(x, mean)), -1, true);
      const normalized = tf.div(tf.sub(x, mean), tf.sqrt(tf.add(variance, this.eps)));
      return tf.add(tf.mul(normalized, this.scale), this.shift) as T;
    });
  }

  dispose(): void {
    for (const w of this.weights) w.dispose();
  }
}

/**
 * Chapter 4, lesson 1: GELU activation, in the tanh approximation GPT-2 was trained with:
 *   gelu(x) = 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
 *
 * Like ReLU it passes large positive values and suppresses negative ones, but it is smooth
 * and lets small negative values through, which gives better gradients near zero.
 */
export function gelu<T extends tf.Tensor>(x: T): T {
  return tf.tidy(() => {
    const inner = tf.mul(Math.sqrt(2 / Math.PI), tf.add(x, tf.mul(0.044715, tf.pow(x, 3))));
    return tf.mul(tf.mul(0.5, x), tf.add(1, tf.tanh(inner))) as T;
  });
}

/**
 * Chapter 4, lesson 1: the position-wise feed-forward network inside each transformer block.
 *
 * Expands every token vector to `4 * embDim`, applies GELU, and projects back to `embDim`:
 *   [batch, seqLen, embDim] -> [batch, seqLen, 4 * embDim] -> [batch, seqLen, embDim]
 *
 * Attention moves information between tokens; this network then processes each token on
 * its own. Most of a GPT's parameters live here.
 */
export class FeedForward {
  readonly weights: tf.Variable[];

  private readonly wIn: tf.Variable;
  private readonly bIn: tf.Variable;
  private readonly wOut: tf.Variable;
  private readonly bOut: tf.Variable;

  constructor(embDim: number) {
    const hidden = 4 * embDim;
    this.wIn = tf.variable(linearWeight(embDim, hidden));
    this.bIn = tf.variable(tf.zeros([hidden]));
    this.wOut = tf.variable(linearWeight(hidden, embDim));
    this.bOut = tf.variable(tf.zeros([embDim]));
    this.weights = [this.wIn, this.bIn, this.wOut, this.bOut];
  }

  forward(x: tf.Tensor3D): tf.Tensor3D {
    return tf.tidy(() => {
      const expanded = gelu(project(x, this.wIn as tf.Tensor2D, this.bIn as tf.Tensor1D));
      return project(expanded, this.wOut as tf.Tensor2D, this.bOut as tf.Tensor1D);
    });
  }

  dispose(): void {
    for (const w of this.weights) w.dispose();
  }
}
