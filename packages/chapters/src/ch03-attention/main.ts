import { MultiHeadAttention } from "@node-llm/core";
import * as tf from "@tensorflow/tfjs-node";

/**
 * Chapter 3: run a forward pass through multi-head self-attention.
 *
 * Importing `@tensorflow/tfjs-node` registers the native backend that core's tfjs-core
 * ops dispatch to.
 *
 * TODO(ch03): once MultiHeadAttention.forward is implemented, feed a random batch and
 * print the output shape / a few attention weights.
 */
function main(): void {
  const attn = new MultiHeadAttention({
    dIn: 8,
    dOut: 8,
    numHeads: 2,
    contextLength: 4,
  });

  const x = tf.randomNormal<tf.Rank.R3>([1, 4, 8]);
  const out = attn.forward(x);
  out.print();
}

main();
