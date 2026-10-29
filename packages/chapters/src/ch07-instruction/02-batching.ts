import { collateInstructions } from "@node-llm/core";
import * as tf from "@tensorflow/tfjs-node";

/**
 * Chapter 7, lesson 2: organizing examples into batches.
 *
 * Examples have different lengths, but a batch tensor needs equal-length rows. Each batch is
 * padded to its own longest example with `<|endoftext|>` (50256). The targets are the inputs
 * shifted by one, as in chapter 5. In the targets, the first end-of-text token is kept, so
 * the model learns to end its response, and the rest of the padding is replaced by -100,
 * which the loss skips.
 */
function main(): void {
  const batch = [
    [0, 1, 2, 3, 4],
    [5, 6],
    [7, 8, 9],
  ];
  console.log("three examples:", batch);

  const { inputs, targets } = collateInstructions(batch);
  console.log("\ninputs (padded to the longest, plus one end-of-text token, minus the last):");
  console.log(inputs);
  console.log("\ntargets (shifted by one; padding after the first 50256 becomes -100):");
  console.log(targets);

  // Why -100 works: positions with that target are given weight 0 in the loss, so they
  // drop out of the average. The book's two-class example shows it.
  console.log("\ncross-entropy on two toy examples, then with a third:");
  const logits = [
    [-1.0, 1.0],
    [-0.5, 1.5],
    [-0.5, 1.5],
  ];
  console.log(`  2 examples, targets [0, 1]:         ${loss(logits.slice(0, 2), [0, 1])}`);
  console.log(`  3 examples, targets [0, 1, 1]:      ${loss(logits, [0, 1, 1])}`);
  console.log(`  3 examples, targets [0, 1, -100]:   ${loss(logits, [0, 1, -100])}  (same as 2)`);
}

/** Mean cross-entropy, skipping negative targets: the rule `crossEntropyLoss` applies. */
function loss(logits: number[][], targets: number[]): string {
  return tf.tidy(() => {
    const t = tf.tensor1d(targets, "int32");
    const keep = tf.cast(tf.greaterEqual(t, 0), "float32");
    const oneHot = tf.oneHot(tf.maximum(t, tf.scalar(0, "int32")), 2);
    const value = tf.losses.softmaxCrossEntropy(oneHot, tf.tensor2d(logits), keep);
    return (value.dataSync()[0] as number).toFixed(4);
  });
}

main();
