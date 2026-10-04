import * as tf from "@tensorflow/tfjs-node";
import { SimplifiedAttention } from "@node-llm/core";

/**
 * Chapter 3, Lesson 2: Simplified self-attention (no trainable weights).
 *
 * We walk the whole mechanism by hand on the sentence "It's a shine bright light" using tiny
 * 4-dimensional embeddings, then confirm the SimplifiedAttention class produces the same
 * context matrix on the batched input.
 *
 * Importing `@tensorflow/tfjs-node` registers the backend that tensor ops need.
 */
function main(): void {
  // Our five token embeddings, shape [5, 4].
  const X = tf.tensor2d([
    [1, 0, 1, 0], // It's
    [0, 1, 0, 1], // a
    [1, 1, 0, 0], // shine
    [0, 0, 1, 1], // bright
    [1, 0, 0, 1], // light
  ]);

  // Step 2: score every pair, X · Xᵀ, shape [5, 5].
  const scores = tf.matMul(X, X, false, true);

  // Step 3: scale by the square root of the embedding dimension.
  const dK = X.shape[1];
  const scaled = scores.div(Math.sqrt(dK));

  // Step 4: softmax over the last axis, each row sums to 1.
  const weights = tf.softmax(scaled, -1);

  // Step 5: blend the value rows into context vectors, shape [5, 4].
  const context = tf.matMul(weights, X);

  console.log("scores:");
  scores.print();
  console.log("attention weights:");
  weights.print();
  console.log("context matrix (step by step):");
  context.print();

  // Confirm the SimplifiedAttention class agrees, on a batch of one: shape [1, 5, 4].
  const attn = new SimplifiedAttention();
  const batched = X.reshape([1, 5, 4]) as tf.Tensor3D;
  const out = attn.forward(batched);
  console.log("context matrix (SimplifiedAttention):");
  out.print();

  // Tensors are not garbage-collected, so dispose once done.
  attn.dispose();
  tf.dispose([X, scores, scaled, weights, context, batched, out]);
}

main();
