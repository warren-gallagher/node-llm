import type * as tf from "@tensorflow/tfjs-core";

/**
 * Chapter 3: the common contract shared by every attention variant.
 *
 * Every implementation maps a batched sequence of vectors to another sequence of the same
 * length: `[batch, seqLen, dIn] -> [batch, seqLen, dOut]`. This keeps the variants
 * interchangeable, so a `TransformerBlock` can hold any of them behind this interface.
 *
 * Attention is pure linear algebra (matMul + softmax), so it operates on tensors only.
 * The plain-array vs tensor choice lives at the DataLoader boundary; by the time data
 * reaches attention it is always a tensor (embeddings come out of `tf.gather`).
 */
export interface Attention {
  /**
   * Run the attention forward pass.
   * @param x        input sequence, shape `[batch, seqLen, dIn]`
   * @param training when true, stochastic layers such as dropout are active
   * @returns        output sequence, shape `[batch, seqLen, dOut]`
   */
  forward(x: tf.Tensor3D, training?: boolean): tf.Tensor3D;

  /** Trainable variables the optimizer should update. Empty for weight-free variants. */
  readonly weights: tf.Variable[];

  /** Free owned `tf.Variable`s. They are not garbage-collected. */
  dispose(): void;
}
