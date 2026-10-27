import * as tf from "@tensorflow/tfjs-core";
import { MultiHeadAttention } from "./attention";
import { FeedForward, LayerNorm } from "./layers";

/**
 * Chapter 4, lesson 2: a single transformer block.
 *
 * Two sub-layers, each wrapped as "normalize, transform, add back" (pre-LayerNorm, as in
 * GPT-2):
 *   x = x + dropout(attention(norm1(x)))
 *   x = x + dropout(feedForward(norm2(x)))
 *
 * The `x + ...` shortcut (residual) connections give gradients a direct path back through
 * a deep stack of blocks. Input and output are both `[batch, seqLen, embDim]`, so blocks
 * can be stacked any number of times.
 */
export interface TransformerBlockConfig {
  embDim: number;
  numHeads: number;
  contextLength: number;
  dropout?: number;
  qkvBias?: boolean;
}

export class TransformerBlock {
  readonly weights: tf.Variable[];

  private readonly norm1: LayerNorm;
  private readonly attention: MultiHeadAttention;
  private readonly norm2: LayerNorm;
  private readonly feedForward: FeedForward;
  private readonly dropout: number;

  constructor(config: TransformerBlockConfig) {
    this.dropout = config.dropout ?? 0;
    this.norm1 = new LayerNorm(config.embDim);
    this.attention = new MultiHeadAttention({
      dIn: config.embDim,
      dOut: config.embDim,
      numHeads: config.numHeads,
      contextLength: config.contextLength,
      dropout: config.dropout,
      qkvBias: config.qkvBias,
    });
    this.norm2 = new LayerNorm(config.embDim);
    this.feedForward = new FeedForward(config.embDim);

    this.weights = [
      ...this.norm1.weights,
      ...this.attention.weights,
      ...this.norm2.weights,
      ...this.feedForward.weights,
    ];
  }

  /** x: [batch, seqLen, embDim] -> [batch, seqLen, embDim] */
  forward(x: tf.Tensor3D, training = false): tf.Tensor3D {
    return tf.tidy(() => {
      // Attention sub-layer: tokens exchange information.
      const attended = this.drop(this.attention.forward(this.norm1.forward(x), training), training);
      const afterAttention = tf.add(x, attended) as tf.Tensor3D;

      // Feed-forward sub-layer: each token is processed on its own.
      const fed = this.drop(this.feedForward.forward(this.norm2.forward(afterAttention)), training);
      return tf.add(afterAttention, fed) as tf.Tensor3D;
    });
  }

  /** Dropout on a sub-layer's output, only while training. */
  private drop(t: tf.Tensor3D, training: boolean): tf.Tensor3D {
    return training && this.dropout > 0 ? (tf.dropout(t, this.dropout) as tf.Tensor3D) : t;
  }

  dispose(): void {
    this.norm1.dispose();
    this.attention.dispose();
    this.norm2.dispose();
    this.feedForward.dispose();
  }
}
