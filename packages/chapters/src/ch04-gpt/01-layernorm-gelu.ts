import { FeedForward, gelu, LayerNorm } from "@node-llm/core";
import * as tf from "@tensorflow/tfjs-node";

/**
 * Chapter 4, Lesson 1: LayerNorm and the GELU feed-forward network.
 *
 * Three small experiments:
 *   1. LayerNorm turns each vector into one with mean 0 and variance 1.
 *   2. GELU compared with ReLU at a few inputs.
 *   3. The feed-forward network keeps the [batch, seqLen, embDim] shape.
 */
function main(): void {
  // 1. Two vectors with 5 features each, deliberately far from mean 0 / variance 1.
  const x = tf.add(tf.mul(tf.randomNormal([2, 5], 0, 1, "float32", 123), 3), 2);
  const norm = new LayerNorm(5);
  const y = norm.forward(x);

  const stats = (t: tf.Tensor) => {
    const { mean, variance } = tf.moments(t, -1);
    return { mean: round(mean), variance: round(variance) };
  };
  console.log("before LayerNorm:", stats(x));
  console.log("after LayerNorm: ", stats(y));

  // 2. GELU lets small negative values through; ReLU zeroes every negative value.
  const inputs = [-3, -1, -0.5, 0, 0.5, 1, 3];
  const t = tf.tensor1d(inputs);
  const geluOut = round(gelu(t));
  const reluOut = round(tf.relu(t));
  console.table(inputs.map((v, i) => ({ x: v, gelu: geluOut[i], relu: reluOut[i] })));

  // 3. Feed-forward at GPT-2 small's width: 768 -> 3072 -> 768.
  const ff = new FeedForward(768);
  const tokens = tf.randomNormal<tf.Rank.R3>([2, 3, 768]);
  const out = ff.forward(tokens);
  const ffParams = ff.weights.reduce((sum, w) => sum + w.size, 0);
  console.log("feed-forward:", tokens.shape, "->", out.shape);
  console.log("feed-forward parameters:", ffParams.toLocaleString());

  tf.dispose([x, y, t, tokens, out]);
  norm.dispose();
  ff.dispose();
}

function round(t: tf.Tensor): number[] {
  return Array.from(t.dataSync(), (v) => Math.round(v * 1e4) / 1e4);
}

main();
