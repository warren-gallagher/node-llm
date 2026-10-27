import * as tf from "@tensorflow/tfjs-core";
import { linearWeight, project } from "./attention/utils";
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

export interface GenerateOptions {
  /** 0 (default) picks the highest-scoring token; above 0 samples, higher = more random. */
  temperature?: number;
  /** Sample only among the k highest-scoring tokens. */
  topK?: number;
  /** Stop early when every sequence produces this token (e.g. 50256, `<|endoftext|>`). */
  eosId?: number;
  /** Seed for reproducible sampling. */
  seed?: number;
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
    this.outHead = tf.variable(linearWeight(embDim, vocabSize));

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

  /**
   * Chapter 5, lesson 3: text generation.
   *
   * Repeats `maxNewTokens` times: run the model on the last `contextLength` tokens, take the
   * logits at the final position, choose one token from them, and append it. With
   * `temperature` 0 (the default) the choice is greedy: always the highest-scoring token.
   *
   * tokenIds: [batch, seqLen] -> [batch, seqLen + generated]
   */
  generate(
    tokenIds: tf.Tensor2D,
    maxNewTokens: number,
    options: GenerateOptions = {},
  ): tf.Tensor2D {
    const { temperature = 0, topK, eosId, seed } = options;
    let ids = tf.clone(tokenIds);

    for (let step = 0; step < maxNewTokens; step++) {
      const stepSeed = seed === undefined ? undefined : seed + step;
      const next = tf.tidy(() => this.nextToken(ids, temperature, topK, stepSeed));

      // Stop before appending once every sequence in the batch produced the end token.
      if (eosId !== undefined && next.dataSync().every((id) => id === eosId)) {
        next.dispose();
        break;
      }

      const longer = tf.concat([ids, next], 1);
      tf.dispose([ids, next]);
      ids = longer;
    }
    return ids;
  }

  /** Picks one next token per sequence: [batch, seqLen] -> [batch, 1] (int32). */
  private nextToken(
    ids: tf.Tensor2D,
    temperature: number,
    topK: number | undefined,
    seed: number | undefined,
  ): tf.Tensor2D {
    // Crop to the context window; older tokens fall out of view.
    const seqLen = ids.shape[1];
    const start = Math.max(0, seqLen - this.config.contextLength);
    const context = tf.slice(ids, [0, start], [-1, seqLen - start]);

    // Only the last position's logits predict the next token: [batch, vocabSize].
    const logits = this.forward(context);
    let last = tf.squeeze(tf.slice(logits, [0, seqLen - start - 1, 0], [-1, 1, -1]), [1]);

    if (topK !== undefined) {
      // Keep the k highest logits; the rest get -Infinity, which softmax turns into 0.
      const kth = tf.slice(tf.topk(last, topK).values, [0, topK - 1], [-1, 1]);
      last = tf.where(tf.less(last, kth), tf.fill(last.shape, Number.NEGATIVE_INFINITY), last);
    }

    if (temperature > 0) {
      // Dividing by the temperature sharpens (< 1) or flattens (> 1) the distribution.
      return tf.multinomial(tf.div(last, temperature) as tf.Tensor2D, 1, seed) as tf.Tensor2D;
    }
    return tf.expandDims(tf.argMax(last, -1), 1) as tf.Tensor2D;
  }

  dispose(): void {
    this.tokenEmbedding.dispose();
    this.positionEmbedding.dispose();
    for (const block of this.blocks) block.dispose();
    this.finalNorm.dispose();
    this.outHead.dispose();
  }
}
