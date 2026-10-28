import * as tf from "@tensorflow/tfjs-node";
import { encode } from "gpt-tokenizer/encoding/r50k_base";
import { LABELS, loadGpt2Classifier } from "./shared";

/**
 * Chapter 6, lesson 4: replace the output head and choose what to train.
 *
 * GPT-2's output head turns each 768-value token vector into 50,257 scores, one per
 * vocabulary token. A classifier needs one score per class, so the head is replaced by a
 * new, randomly initialized one with 2 outputs. Then everything is frozen except the last
 * transformer block, the final LayerNorm and the new head.
 */
async function main(): Promise<void> {
  const model = await loadGpt2Classifier();
  const { embDim, vocabSize } = model.base.config;

  const outHead = embDim * vocabSize;
  const body = model.base.numParameters() - outHead; // everything below GPT-2's own head
  const head = model.headWeights.reduce((sum, w) => sum + w.size, 0);
  const trainable = model.trainableWeights.reduce((sum, w) => sum + w.size, 0);
  console.log(`GPT-2 below its head:  ${body.toLocaleString()} parameters`);
  console.log(`GPT-2's head (unused): ${outHead.toLocaleString()} (${embDim} x ${vocabSize})`);
  console.log(`new head:              ${head.toLocaleString()} (${embDim} x 2, plus 2 biases)`);
  console.log(
    `trainable:             ${trainable.toLocaleString()} (last block, final norm, head)`,
  );

  const ids = encode("Do you have time");
  console.log(`\ninputs: [${ids.join(", ")}]  shape [1, ${ids.length}]`);

  tf.tidy(() => {
    const input = tf.tensor2d([ids], undefined, "int32");
    const outputs = model.forward(input);
    console.log(`outputs: shape [${outputs.shape.join(", ")}]  (batch, tokens, classes)`);
    outputs.print();

    // Only the last token has seen the whole text, so its two scores classify it.
    const last = model.lastTokenLogits(input);
    console.log("last token's scores:");
    last.print();
    const label = tf.argMax(last, -1).dataSync()[0] as number;
    console.log(`class: ${label} (${LABELS[label]}), from an untrained head, so a coin toss`);
  });

  model.dispose();
}

main();
