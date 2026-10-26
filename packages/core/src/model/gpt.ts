import * as tf from "@tensorflow/tfjs-core";
import { project } from "./attention/utils";
import { TransformerBlock } from "./block";
import { LayerNorm } from "./layers";

/**
 * Chapter 4, lesson 3: the full GPT model (token + positional embeddings -> N transformer
 * blocks -> final LayerNorm -> output head).
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
  readonly weights: tf.Variable[];

  private readonly tokenEmbedding: tf.Variable;
  private readonly positionEmbedding: tf.Variable;
  private readonly blocks: TransformerBlock[];
  private readonly finalNorm: LayerNorm;
  private readonly outHead: tf.Variable;

  constructor(readonly config: GptConfig) {
    const { vocabSize, contextLength, embDim, numLayers } = config;

    // Lookup tables, as in chapter 2: one row per token id and one row per position.
    this.tokenEmbedding = tf.variable(tf.randomNormal([vocabSize, embDim]));
    this.positionEmbedding = tf.variable(tf.randomNormal([contextLength, embDim]));

    this.blocks = Array.from(
      { length: numLayers },
      () =>
        new TransformerBlock({
          embDim,
          numHeads: config.numHeads,
          contextLength,
          dropout: config.dropout,
          qkvBias: config.qkvBias,
        }),
    );
    this.finalNorm = new LayerNorm(embDim);

    // Maps each final token vector to one score (logit) per vocabulary entry. No bias, as in
    // the book and GPT-2.
    this.outHead = tf.variable(tf.mul(tf.randomNormal([embDim, vocabSize]), embDim ** -0.5));

    this.weights = [
      this.tokenEmbedding,
      this.positionEmbedding,
      ...this.blocks.flatMap((block) => block.weights),
      ...this.finalNorm.weights,
      this.outHead,
    ];
  }

  /** tokenIds: [batch, seqLen] -> logits [batch, seqLen, vocabSize] */
  forward(tokenIds: tf.Tensor2D, training = false): tf.Tensor3D {
    return tf.tidy(() => {
      const seqLen = tokenIds.shape[1];
      if (seqLen > this.config.contextLength) {
        throw new Error(`seqLen (${seqLen}) exceeds contextLength (${this.config.contextLength})`);
      }

      // [batch, seqLen, embDim] token vectors plus [seqLen, embDim] position vectors
      // (broadcast over the batch).
      const tokens = tf.gather(this.tokenEmbedding, tokenIds) as tf.Tensor3D;
      const positions = tf.slice(this.positionEmbedding, [0, 0], [seqLen, this.config.embDim]);
      let x = tf.add(tokens, positions) as tf.Tensor3D;
      if (training && this.config.dropout > 0) {
        x = tf.dropout(x, this.config.dropout) as tf.Tensor3D;
      }

      for (const block of this.blocks) x = block.forward(x, training);

      return project(this.finalNorm.forward(x), this.outHead as tf.Tensor2D);
    });
  }

  /** Total number of trainable values across every weight tensor. */
  numParameters(): number {
    return this.weights.reduce((sum, w) => sum + w.size, 0);
  }

  /** Chapter 5: greedy/temperature text generation. */
  generate(_tokenIds: tf.Tensor2D, _maxNewTokens: number): tf.Tensor2D {
    throw new Error("TODO(ch05): implement generation loop");
  }

  dispose(): void {
    this.tokenEmbedding.dispose();
    this.positionEmbedding.dispose();
    for (const block of this.blocks) block.dispose();
    this.finalNorm.dispose();
    this.outHead.dispose();
  }
}
