import * as tf from "@tensorflow/tfjs-node";
import { DataLoader, LlmDataset, SelfAttention } from "@node-llm/core";
import { encode as gptEncode } from "gpt-tokenizer";

/**
 * Chapter 3, Lesson 3: Self-attention with trainable weights.
 *
 * This runs the full path: real tokens flow through the Dataset and DataLoader as batched
 * tensors, get embedded, and are fed straight into SelfAttention. The class learns three
 * projections Wq, Wk, Wv, then computes scaled dot-product attention:
 *
 *   Q = xWq, K = xWk, V = xWv
 *   context = softmax(Q Kᵀ / sqrt(dOut)) V
 *
 * Importing `@tensorflow/tfjs-node` registers the backend that tensor ops need.
 */
function main(): void {
  const ids = gptEncode("Life is like box of chocolate you never knew what you gonna get");

  const vocabSize = 50257; // GPT-2 BPE vocabulary
  const embDim = 8; // tiny for a readable demo; GPT-2 small uses 768
  const contextSize = 4; // tokens per chunk

  // Chunk the tokens and batch them as tensors, exactly as in the DataLoader lesson.
  const dataset = new LlmDataset(ids, contextSize, contextSize);
  const loader = new DataLoader(dataset, { batchSize: 2, asTensors: true });

  // The token embedding table, learned during training.
  const tokenEmbedding = tf.randomNormal([vocabSize, embDim]);

  // One attention layer with trainable Wq, Wk, Wv. dIn === dOut === embDim here.
  const attn = new SelfAttention({ dIn: embDim, dOut: embDim });

  for (const batch of loader) {
    // Embed every token id in the batch. Shape [batchSize, contextSize, embDim].
    const x = tf.gather(tokenEmbedding, batch.inputs) as tf.Tensor3D;

    // Self-attention over the batch. Shape [batchSize, contextSize, embDim].
    const context = attn.forward(x);

    console.log("batch ids shape:", batch.inputs.shape); // [batchSize, contextSize]
    console.log("embedded input shape:", x.shape); // [batchSize, contextSize, embDim]
    console.log("context shape:", context.shape); // [batchSize, contextSize, embDim]
    console.log("first token context vector:", Array.from(context.slice([0, 0, 0], [1, 1, embDim]).dataSync()));

    tf.dispose([batch.inputs, batch.targets, x, context]);
    break; // one batch is enough for the demo
  }

  // Tensors are not garbage-collected, so dispose once done.
  attn.dispose();
  tf.dispose(tokenEmbedding);
}

main();
