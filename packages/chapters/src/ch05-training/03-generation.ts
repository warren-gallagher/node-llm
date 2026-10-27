import * as tf from "@tensorflow/tfjs-node";
import { idsToText, loadCheckpoint, textToIds } from "./shared";

/**
 * Chapter 5, Lesson 3: turning logits into text.
 *
 * 1. On a made-up five-word vocabulary: how temperature reshapes the probabilities, and
 *    how top-k removes unlikely choices before sampling.
 * 2. On the model trained in lesson 2: greedy decoding vs. sampling with temperature and
 *    top-k.
 */
function main(): void {
  // 1. Pretend these are the logits for the next token after "The cat sat on the".
  const words = ["mat", "sofa", "roof", "moon", "banana"];
  const logits = tf.tensor1d([4.0, 3.2, 2.1, 0.5, -1.0]);

  for (const temperature of [0.5, 1, 2]) {
    const probs = tf.softmax(tf.div(logits, temperature));
    const counts = sampleCounts(logits, temperature, 1000);
    console.log(`\ntemperature ${temperature}`);
    console.table(
      words.map((word, i) => ({
        word,
        probability: (probs.dataSync()[i] as number).toFixed(3),
        "picked (of 1000)": counts[i],
      })),
    );
    probs.dispose();
  }

  // Top-k with k = 2: everything below the 2nd-highest logit becomes -Infinity (probability 0).
  const kth = tf.topk(logits, 2).values.dataSync()[1] as number;
  const masked = tf.where(tf.less(logits, kth), tf.fill([5], Number.NEGATIVE_INFINITY), logits);
  const topKProbs = Array.from(tf.softmax(masked).dataSync(), (p) => p.toFixed(3));
  console.log(
    "\ntop-k (k = 2) probabilities:",
    Object.fromEntries(words.map((w, i) => [w, topKProbs[i]])),
  );
  tf.dispose([logits, masked]);

  // 2. The model trained in lesson 2.
  const model = loadCheckpoint("verdict-124m");
  if (!model) {
    console.log(
      "\nNo trained model yet. Run lesson 2 first: pnpm --filter @node-llm/chapters ch05:2",
    );
    return;
  }

  const prompt = textToIds("Every effort moves you");
  const greedy = model.generate(prompt, 25);
  console.log(`\ngreedy:            ${JSON.stringify(idsToText(greedy))}`);

  for (const seed of [1, 2, 3]) {
    const sampled = model.generate(prompt, 25, { temperature: 1.4, topK: 25, seed });
    console.log(`temp 1.4, top-25 (seed ${seed}): ${JSON.stringify(idsToText(sampled))}`);
    sampled.dispose();
  }

  tf.dispose([prompt, greedy]);
  model.dispose();
}

/** Draws `n` samples from softmax(logits / temperature) and counts how often each index wins. */
function sampleCounts(logits: tf.Tensor1D, temperature: number, n: number): number[] {
  const scaled = tf.div(logits, temperature) as tf.Tensor1D;
  const draws = tf.multinomial(scaled, n, 123).dataSync();
  const counts = new Array<number>(logits.size).fill(0);
  for (const i of draws) counts[i] = (counts[i] ?? 0) + 1;
  scaled.dispose();
  return counts;
}

main();
