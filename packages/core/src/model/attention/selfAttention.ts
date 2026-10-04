import * as tf from "@tensorflow/tfjs-core";
import { addCausalMask, project } from "./utils";
import type { Attention } from "./types";

export interface SelfAttentionConfig {
  /** Input dimension (embedding size). */
  dIn: number;
  /** Output dimension of the attention. */
  dOut: number;
  /** Add learnable bias terms to the Q/K/V projections. Defaults to false. */
  qkvBias?: boolean;
  /** Mask out future positions so each token only attends to itself and earlier ones. */
  causal?: boolean;
}

/**
 * Chapter 3, lesson 3: single-head self-attention with trainable weights.
 *
 * Now the model learns three projections, query, key, and value, and computes scaled
 * dot-product attention:
 *   Q = xWq, K = xWk, V = xWv
 *   scores = (Q Kᵀ) / sqrt(dOut)
 *   context = softmax(scores) V
 *
 * With `causal: true` the scores are masked so position i cannot attend to future tokens.
 */
export class SelfAttention implements Attention {
  readonly weights: tf.Variable[] = [];

  private readonly wQuery: tf.Variable;
  private readonly wKey: tf.Variable;
  private readonly wValue: tf.Variable;
  private readonly bQuery?: tf.Variable;
  private readonly bKey?: tf.Variable;
  private readonly bValue?: tf.Variable;
  private readonly dOut: number;
  private readonly causal: boolean;

  constructor(config: SelfAttentionConfig) {
    const { dIn, dOut, qkvBias = false, causal = false } = config;
    this.dOut = dOut;
    this.causal = causal;

    // Small random init scaled by 1/sqrt(dIn) keeps early activations well-conditioned.
    const scale = dIn ** -0.5;
    this.wQuery = tf.variable(tf.mul(tf.randomNormal([dIn, dOut]), scale));
    this.wKey = tf.variable(tf.mul(tf.randomNormal([dIn, dOut]), scale));
    this.wValue = tf.variable(tf.mul(tf.randomNormal([dIn, dOut]), scale));
    this.weights.push(this.wQuery, this.wKey, this.wValue);

    if (qkvBias) {
      this.bQuery = tf.variable(tf.zeros([dOut]));
      this.bKey = tf.variable(tf.zeros([dOut]));
      this.bValue = tf.variable(tf.zeros([dOut]));
      this.weights.push(this.bQuery, this.bKey, this.bValue);
    }
  }

  forward(x: tf.Tensor3D): tf.Tensor3D {
    return tf.tidy(() => {
      const seqLen = x.shape[1];

      const q = project(x, this.wQuery as tf.Tensor2D, this.bQuery as tf.Tensor1D | undefined);
      const k = project(x, this.wKey as tf.Tensor2D, this.bKey as tf.Tensor1D | undefined);
      const v = project(x, this.wValue as tf.Tensor2D, this.bValue as tf.Tensor1D | undefined);

      // Scaled dot-product scores: [batch, seqLen, seqLen].
      let scores = tf.div(tf.matMul(q, k, false, true), Math.sqrt(this.dOut));
      if (this.causal) scores = addCausalMask(scores, seqLen);

      const weights = tf.softmax(scores, -1);
      return tf.matMul(weights, v) as tf.Tensor3D; // [batch, seqLen, dOut]
    });
  }

  dispose(): void {
    for (const w of this.weights) w.dispose();
  }
}
