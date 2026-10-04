import type * as tf from "@tensorflow/tfjs-core";
import { MultiHeadAttention } from "./attention";

/**
 * Chapter 4: a single transformer block (LayerNorm -> attention -> residual ->
 * LayerNorm -> feed-forward -> residual).
 *
 * TODO(ch04): implement LayerNorm, GELU feed-forward, and residual connections.
 */
export interface TransformerBlockConfig {
  embDim: number;
  numHeads: number;
  contextLength: number;
  dropout?: number;
  qkvBias?: boolean;
}

export class TransformerBlock {
  private readonly attention: MultiHeadAttention;

  constructor(config: TransformerBlockConfig) {
    this.attention = new MultiHeadAttention({
      dIn: config.embDim,
      dOut: config.embDim,
      numHeads: config.numHeads,
      contextLength: config.contextLength,
      dropout: config.dropout,
      qkvBias: config.qkvBias,
    });
  }

  forward(_x: tf.Tensor3D): tf.Tensor3D {
    // Uses this.attention once implemented (ch04).
    void this.attention;
    throw new Error("TODO(ch04): implement transformer block forward pass");
  }
}
