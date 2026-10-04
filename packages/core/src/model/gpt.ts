import type * as tf from "@tensorflow/tfjs-core";

/**
 * Chapter 4: the full GPT model (token + positional embeddings -> N transformer blocks ->
 * final LayerNorm -> output head).
 *
 * Mirrors the book's `GPT_CONFIG_124M`.
 */
export interface GptConfig {
  vocabSize: number;
  contextLength: number;
  embDim: number;
  numHeads: number;
  numLayers: number;
  dropout: number;
  qkvBias: boolean;
}

export const GPT_CONFIG_124M: GptConfig = {
  vocabSize: 50257,
  contextLength: 1024,
  embDim: 768,
  numHeads: 12,
  numLayers: 12,
  dropout: 0.1,
  qkvBias: false,
};

export class GptModel {
  constructor(readonly config: GptConfig) {}

  /** tokenIds: [batch, seqLen] -> logits [batch, seqLen, vocabSize] */
  forward(_tokenIds: tf.Tensor2D): tf.Tensor3D {
    throw new Error("TODO(ch04): implement GPT forward pass");
  }

  /** Chapter 5: greedy/temperature text generation. */
  generate(_tokenIds: tf.Tensor2D, _maxNewTokens: number): tf.Tensor2D {
    throw new Error("TODO(ch05): implement generation loop");
  }
}
