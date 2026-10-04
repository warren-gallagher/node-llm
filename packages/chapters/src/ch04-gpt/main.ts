import { GPT_CONFIG_124M, GptModel } from "@node-llm/core";
import * as tf from "@tensorflow/tfjs-node";

/**
 * Chapter 4: assemble the full GPT and run a forward pass.
 *
 * TODO(ch04): once GptModel.forward is implemented, print logits shape
 * (expect [batch, seqLen, vocabSize]).
 */
function main(): void {
  // A small config for quick local runs (override the 124M defaults).
  const model = new GptModel({
    ...GPT_CONFIG_124M,
    contextLength: 32,
    embDim: 64,
    numHeads: 4,
    numLayers: 2,
  });

  const tokenIds = tf.tensor2d([[1, 2, 3, 4]], [1, 4], "int32");
  const logits = model.forward(tokenIds);
  console.log("logits shape:", logits.shape);
}

main();
