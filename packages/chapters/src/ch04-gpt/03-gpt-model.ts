import { GPT_CONFIG_124M, GptModel } from "@node-llm/core";
import * as tf from "@tensorflow/tfjs-node";
import { encode } from "gpt-tokenizer/encoding/r50k_base";

/**
 * Chapter 4, Lesson 3: the full GPT model at GPT-2 small's size.
 *
 * Builds the 124M configuration, runs two short texts through it, and counts parameters.
 * The weights are random, so the logits mean nothing yet: chapter 5 trains them.
 */
function main(): void {
  const texts = ["Every effort moves you", "Every day holds a"];
  const batch = tf.tensor2d(
    texts.map((text) => encode(text)),
    undefined,
    "int32",
  );

  const model = new GptModel(GPT_CONFIG_124M);
  const logits = model.forward(batch);
  console.log("input shape: ", batch.shape); // [2, 4]
  console.log("logits shape:", logits.shape); // [2, 4, 50257]

  const total = model.numParameters();
  const outHead = GPT_CONFIG_124M.embDim * GPT_CONFIG_124M.vocabSize;
  console.log("total parameters:", total.toLocaleString());
  // GPT-2 reuses the token embedding as the output head ("weight tying"), which is where
  // the "124M" in its name comes from.
  console.log("without a separate output head:", (total - outHead).toLocaleString());
  console.log("size as float32:", `${((total * 4) / 1024 ** 2).toFixed(2)} MB`);

  tf.dispose([batch, logits]);
  model.dispose();
}

main();
