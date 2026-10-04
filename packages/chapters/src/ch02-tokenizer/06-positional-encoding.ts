import * as tf from "@tensorflow/tfjs-node";
import { DataLoader, LlmDataset } from "@node-llm/core";
import { encode as gptEncode } from "gpt-tokenizer";

/**
 * Chapter 2, Lesson 6: Positional encoding.
 *
 * A token embedding gives a token the same vector no matter where it appears. But word
 * order carries meaning, so we add a second table, the positional embedding of shape
 * [contextSize, embDim], and sum the two:
 *
 *   input = tokenEmbedding + positionalEmbedding
 *
 * Here the token ids come batched from the DataLoader (as in the previous lesson), so the
 * positional vectors are broadcast across every sequence in the batch.
 */
function main(): void {
  const ids = gptEncode("Life is like box of chocolate you never knew what you gonna get");

  const vocabSize = 50257; // GPT-2 BPE vocabulary
  const embDim = 8; // tiny for a readable demo; GPT-2 small uses 768
  const contextSize = 4; // tokens per chunk, and the largest position in a chunk

  // Chunk the tokens and batch them as tensors, exactly as in the DataLoader lesson.
  const dataset = new LlmDataset(ids, contextSize, contextSize);
  const loader = new DataLoader(dataset, { batchSize: 2, asTensors: true });

  // Two tables, both learned during training.
  const tokenEmbedding = tf.randomNormal([vocabSize, embDim]);
  const positionalEmbedding = tf.randomNormal([contextSize, embDim]);

  // "Where": position ids 0..contextSize-1, shared by every sequence. Shape [contextSize, embDim].
  const positions = tf.range(0, contextSize, 1, "int32");
  const positionVectors = tf.gather(positionalEmbedding, positions);

  for (const batch of loader) {
    // "What": embed every token id in the batch. Shape [batchSize, contextSize, embDim].
    const tokenVectors = tf.gather(tokenEmbedding, batch.inputs);

    // input = token + position. Broadcasting applies the same positions to each sequence.
    const input = tf.add(tokenVectors, positionVectors);

    console.log("batch ids shape:", batch.inputs.shape); // [batchSize, contextSize]
    console.log("token vectors shape:", tokenVectors.shape); // [batchSize, contextSize, embDim]
    console.log("positional vectors shape:", positionVectors.shape); // [contextSize, embDim]
    console.log("input shape:", input.shape); // [batchSize, contextSize, embDim]

    tf.dispose([batch.inputs, batch.targets, tokenVectors, input]);
    break; // one batch is enough for the demo
  }

  tf.dispose([tokenEmbedding, positionalEmbedding, positions, positionVectors]);
}

main();
