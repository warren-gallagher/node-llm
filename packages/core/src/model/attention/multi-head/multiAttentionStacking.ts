import * as tf from "@tensorflow/tfjs-core";
import { SelfAttention } from "../selfAttention";
import type { Attention } from "../types";

export interface MultiAttentionStackingConfig {
  /** Input dimension (embedding size). */
  dIn: number;
  /** Output dimension of each individual head. The final width is `dOut * numHeads`. */
  dOut: number;
  /** Number of independent self-attention heads to stack. */
  numHeads: number;
  /** Add learnable bias terms to each head's Q/K/V projections. Defaults to false. */
  qkvBias?: boolean;
  /** Mask out future positions in every head. Defaults to false. */
  causal?: boolean;
}

/**
 * Chapter 3, lesson 4: multi-head attention by stacking.
 *
 * The most direct way to build multi-head attention: create `numHeads` independent
 * `SelfAttention` heads, run the same input through each one, then concatenate their
 * context vectors along the feature axis.
 *
 *   head_i: [batch, seqLen, dIn] -> [batch, seqLen, dOut]
 *   context = concat([head_0, ..., head_{N-1}], axis=-1) -> [batch, seqLen, dOut * numHeads]
 *
 * Each head owns its own Wq, Wk, Wv, so every head learns to attend differently. This is
 * easy to read but runs one matmul per head; the batched `MultiHeadAttention` does the
 * same work in a single matmul.
 */
export class MultiAttentionStacking implements Attention {
  readonly weights: tf.Variable[] = [];

  private readonly heads: SelfAttention[];

  constructor(config: MultiAttentionStackingConfig) {
    const { dIn, dOut, numHeads, qkvBias = false, causal = false } = config;

    this.heads = Array.from(
      { length: numHeads },
      () => new SelfAttention({ dIn, dOut, qkvBias, causal }),
    );

    for (const head of this.heads) this.weights.push(...head.weights);
  }

  forward(x: tf.Tensor3D): tf.Tensor3D {
    return tf.tidy(() => {
      // Each head produces [batch, seqLen, dOut]; concatenate along the feature axis.
      const perHead = this.heads.map((head) => head.forward(x));
      return tf.concat(perHead, -1) as tf.Tensor3D; // [batch, seqLen, dOut * numHeads]
    });
  }

  dispose(): void {
    for (const head of this.heads) head.dispose();
  }
}
