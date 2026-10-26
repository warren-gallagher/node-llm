import * as tf from "@tensorflow/tfjs-node";
import { DataLoader, LlmDataset, MultiAttentionStacking } from "@node-llm/core";
import { encode as gptEncode } from "gpt-tokenizer/encoding/r50k_base";

/**
 * Chapter 3, Lesson 4: multi-head attention by stacking.
 *
 * Same full path as the previous lesson: real tokens flow through the Dataset and
 * DataLoader as batched tensors, get embedded, then feed into MultiAttentionStacking.
 *
 * The stacking class holds `numHeads` independent SelfAttention heads. Each head has its
 * own Wq, Wk, Wv and outputs a context of width `dOut`. We concatenate them along the
 * feature axis, so the final context width is `dOut * numHeads`.
 *
 * Importing `@tensorflow/tfjs-node` registers the backend that tensor ops need.
 */
function main(): void {
  const ids = gptEncode("Life is like box of chocolate you never knew what you gonna get");

  const vocabSize = 50257; // GPT-2 BPE vocabulary
  const embDim = 8; // tiny for a readable demo; GPT-2 small uses 768
  const contextSize = 4; // tokens per chunk

  const numHeads = 4; // heads run in parallel
  const headDim = 2; // width M of each head's context vector; dOut = numHeads * headDim = 8

  // Chunk the tokens and batch them as tensors, exactly as in the DataLoader lesson.
  const dataset = new LlmDataset(ids, contextSize, contextSize);
  const loader = new DataLoader(dataset, { batchSize: 2, asTensors: true });

  // The token embedding table, learned during training.
  const tokenEmbedding = tf.randomNormal([vocabSize, embDim]);

  // Multi-head attention by stacking: numHeads independent SelfAttention heads.
  const attn = new MultiAttentionStacking({ dIn: embDim, dOut: headDim, numHeads, causal: true });

  for (const batch of loader) {
    // Embed every token id in the batch. Shape [batchSize, contextSize, embDim].
    const x = tf.gather(tokenEmbedding, batch.inputs) as tf.Tensor3D;

    // Stacked multi-head attention. Shape [batchSize, contextSize, headDim * numHeads].
    const context = attn.forward(x);

    console.log("batch ids shape:", batch.inputs.shape); // [batchSize, contextSize]
    console.log("embedded input shape:", x.shape); // [batchSize, contextSize, embDim]
    console.log("context shape:", context.shape); // [batchSize, contextSize, headDim * numHeads]
    console.log(
      "first token context vector:",
      Array.from(context.slice([0, 0, 0], [1, 1, headDim * numHeads]).dataSync()),
    );

    tf.dispose([batch.inputs, batch.targets, x, context]);
    break; // one batch is enough for the demo
  }

  // Tensors are not garbage-collected, so dispose once done.
  attn.dispose();
  tf.dispose(tokenEmbedding);
}

main();
