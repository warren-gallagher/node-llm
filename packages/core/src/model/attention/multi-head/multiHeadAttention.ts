import * as tf from "@tensorflow/tfjs-core";
import { addCausalMask, project } from "../utils";
import type { Attention } from "../types";

export interface MultiHeadAttentionConfig {
  /** Input dimension (embedding size). */
  dIn: number;
  /** Output dimension. Must be divisible by `numHeads`. */
  dOut: number;
  /** Number of attention heads run in parallel. */
  numHeads: number;
  /** Maximum sequence length the model supports. */
  contextLength: number;
  /** Dropout rate applied to the attention weights during training. Defaults to 0. */
  dropout?: number;
  /** Add learnable bias terms to the Q/K/V projections. Defaults to false. */
  qkvBias?: boolean;
  /** Mask out future positions. Defaults to true (the GPT setting). */
  causal?: boolean;
}

/**
 * Chapter 3, lesson 4: multi-head self-attention.
 *
 * The Q/K/V projections are split into `numHeads` independent heads that each attend over
 * a `dOut / numHeads`-sized subspace. The heads run in parallel, their outputs are
 * concatenated, and a final output projection (`Wo`, with bias `bo`) mixes them back
 * together. This is the exact form used inside each transformer block. The output bias
 * matches the book's `nn.Linear` out_proj and GPT-2's `c_proj`, so pretrained weights fit.
 */
export class MultiHeadAttention implements Attention {
  readonly weights: tf.Variable[] = [];

  private readonly wQuery: tf.Variable;
  private readonly wKey: tf.Variable;
  private readonly wValue: tf.Variable;
  private readonly wOut: tf.Variable;
  private readonly bOut: tf.Variable;
  private readonly bQuery?: tf.Variable;
  private readonly bKey?: tf.Variable;
  private readonly bValue?: tf.Variable;

  private readonly dOut: number;
  private readonly numHeads: number;
  private readonly headDim: number;
  private readonly contextLength: number;
  private readonly dropout: number;
  private readonly causal: boolean;

  constructor(config: MultiHeadAttentionConfig) {
    const {
      dIn,
      dOut,
      numHeads,
      contextLength,
      dropout = 0,
      qkvBias = false,
      causal = true,
    } = config;

    if (dOut % numHeads !== 0) {
      throw new Error(`dOut (${dOut}) must be divisible by numHeads (${numHeads})`);
    }

    this.dOut = dOut;
    this.numHeads = numHeads;
    this.headDim = dOut / numHeads;
    this.contextLength = contextLength;
    this.dropout = dropout;
    this.causal = causal;

    const scale = dIn ** -0.5;
    this.wQuery = tf.variable(tf.mul(tf.randomNormal([dIn, dOut]), scale));
    this.wKey = tf.variable(tf.mul(tf.randomNormal([dIn, dOut]), scale));
    this.wValue = tf.variable(tf.mul(tf.randomNormal([dIn, dOut]), scale));
    this.wOut = tf.variable(tf.mul(tf.randomNormal([dOut, dOut]), dOut ** -0.5));
    this.bOut = tf.variable(tf.zeros([dOut]));
    this.weights.push(this.wQuery, this.wKey, this.wValue, this.wOut, this.bOut);

    if (qkvBias) {
      this.bQuery = tf.variable(tf.zeros([dOut]));
      this.bKey = tf.variable(tf.zeros([dOut]));
      this.bValue = tf.variable(tf.zeros([dOut]));
      this.weights.push(this.bQuery, this.bKey, this.bValue);
    }
  }

  forward(x: tf.Tensor3D, training = false): tf.Tensor3D {
    return tf.tidy(() => {
      const [batch, seqLen] = x.shape;
      if (seqLen > this.contextLength) {
        throw new Error(`seqLen (${seqLen}) exceeds contextLength (${this.contextLength})`);
      }

      // Project, then split each [batch, seqLen, dOut] into heads:
      // [batch, seqLen, dOut] -> [batch, numHeads, seqLen, headDim].
      const splitHeads = (t: tf.Tensor3D): tf.Tensor4D => {
        const reshaped = tf.reshape(t, [batch, seqLen, this.numHeads, this.headDim]);
        return tf.transpose(reshaped, [0, 2, 1, 3]) as tf.Tensor4D;
      };

      const q = splitHeads(project(x, this.wQuery as tf.Tensor2D, this.bQuery as tf.Tensor1D | undefined));
      const k = splitHeads(project(x, this.wKey as tf.Tensor2D, this.bKey as tf.Tensor1D | undefined));
      const v = splitHeads(project(x, this.wValue as tf.Tensor2D, this.bValue as tf.Tensor1D | undefined));

      // Scaled dot-product per head: [batch, numHeads, seqLen, seqLen].
      let scores = tf.div(tf.matMul(q, k, false, true), Math.sqrt(this.headDim));
      if (this.causal) scores = addCausalMask(scores, seqLen);

      let weights = tf.softmax(scores, -1);
      if (training && this.dropout > 0) weights = tf.dropout(weights, this.dropout);

      // Context per head, then merge heads back: [batch, seqLen, dOut].
      const context = tf.matMul(weights, v) as tf.Tensor4D; // [batch, numHeads, seqLen, headDim]
      const merged = tf.reshape(
        tf.transpose(context, [0, 2, 1, 3]),
        [batch, seqLen, this.dOut],
      ) as tf.Tensor3D;

      // Final output projection mixes information across heads.
      return project(merged, this.wOut as tf.Tensor2D, this.bOut as tf.Tensor1D);
    });
  }

  dispose(): void {
    for (const w of this.weights) w.dispose();
  }
}
