import { gelu, TransformerBlock } from "@node-llm/core";
import * as tf from "@tensorflow/tfjs-node";

/**
 * Chapter 4, Lesson 2: shortcut connections and the transformer block.
 *
 * 1. A deep stack of small layers, run through one backward pass with and without shortcut
 *    (residual) connections. Without shortcuts the gradient shrinks layer by layer on its
 *    way back to the first layers; with them it reaches every layer at a useful size.
 * 2. One transformer block at GPT-2 small's size keeps the input shape and holds about
 *    7 million parameters.
 */
function main(): void {
  for (const useShortcut of [false, true]) {
    console.log(`\nmean |gradient| per layer, shortcut=${useShortcut}`);
    for (const [layer, size] of gradientSizes(useShortcut).entries()) {
      console.log(`  layer ${layer}: ${size.toExponential(2)}`);
    }
  }

  const block = new TransformerBlock({ embDim: 768, numHeads: 12, contextLength: 1024 });
  const x = tf.randomNormal<tf.Rank.R3>([2, 4, 768]);
  const out = block.forward(x);
  const params = block.weights.reduce((sum, w) => sum + w.size, 0);
  console.log("\ntransformer block:", x.shape, "->", out.shape);
  console.log("block parameters:", params.toLocaleString());

  tf.dispose([x, out]);
  block.dispose();
}

/**
 * Five layers of width 3 (then a final layer to 1 output), GELU between them. Returns the
 * mean absolute gradient of each layer's weight matrix for one example.
 */
function gradientSizes(useShortcut: boolean): number[] {
  const sizes = [3, 3, 3, 3, 3, 1];
  const layers = sizes
    .slice(0, -1)
    .map((dIn, i) =>
      tf.variable(tf.randomNormal([dIn, sizes[i + 1] as number], 0, 0.5, "float32", 7 + i)),
    );

  const input = tf.tensor2d([[1, 0, -1]]);
  const target = tf.tensor2d([[0]]);
  const { value, grads } = tf.variableGrads(() => {
    let x: tf.Tensor2D = input;
    for (const w of layers) {
      const y = gelu(tf.matMul(x, w) as tf.Tensor2D);
      // A shortcut only works when input and output have the same shape.
      x = useShortcut && y.shape[1] === x.shape[1] ? (tf.add(x, y) as tf.Tensor2D) : y;
    }
    return tf.losses.meanSquaredError(target, x) as tf.Scalar;
  }, layers);

  const result = layers.map(
    (w) => tf.mean(tf.abs(grads[w.name] as tf.Tensor)).dataSync()[0] as number,
  );
  tf.dispose([input, target, value, ...Object.values(grads), ...layers]);
  return result;
}

main();
