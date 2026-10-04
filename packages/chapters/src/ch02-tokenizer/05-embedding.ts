import * as tf from "@tensorflow/tfjs-node";
import { encode as gptEncode } from "gpt-tokenizer";

/**
 * Chapter 2, Lesson 6: Embeddings.
 *
 * An embedding layer is just a [vocabSize, embDim] lookup table: one learned vector per token.
 * "Embedding" a batch of ids is a single `tf.gather`, which selects each token's row and turns
 * ids of shape [batchSize, seqLen] into vectors of shape [batchSize, seqLen, embDim].
 *
 * Importing `@tensorflow/tfjs-node` registers the backend that tensor ops need.
 */
function main(): void {
  const ids = gptEncode("Life is like box of chocolate");

  const vocabSize = 50257; // GPT-2 BPE vocabulary
  const embDim = 8; // tiny for a readable demo; GPT-2 small uses 768

  // The embedding table. Starts random, then gets optimized during training.
  const embedding = tf.randomNormal([vocabSize, embDim]);

  // A batch of one sequence: shape [1, seqLen].
  const batch = tf.tensor2d([ids], undefined, "int32");

  // Look up each token's row: shape [1, seqLen, embDim].
  const vectors = tf.gather(embedding, batch);

  console.log("token ids:", ids);
  console.log("batch shape:", batch.shape); // [1, seqLen]
  console.log("embedded shape:", vectors.shape); // [1, seqLen, embDim]

  // Peek at the vector for the very first token.
  const firstVector = vectors.slice([0, 0, 0], [1, 1, embDim]);
  console.log("first token vector:", Array.from(firstVector.dataSync()));

  // Tensors are not garbage-collected, so dispose once done.
  tf.dispose([embedding, batch, vectors, firstVector]);
}

main();
