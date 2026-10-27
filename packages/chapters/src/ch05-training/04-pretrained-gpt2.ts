import { evaluateLoss, type GptModel } from "@node-llm/core";
import * as tf from "@tensorflow/tfjs-node";
import { decode } from "gpt-tokenizer/encoding/r50k_base";
import { ensureGpt2Weights, GPT2_CONFIG, loadGpt2 } from "./gpt2";
import { idsToText, loadCheckpoint, loadVerdict, makeLoaders, textToIds } from "./shared";

/**
 * Chapter 5, extra lesson (not in the book's outline; from Raschka's chapter 5): OpenAI's
 * pretrained GPT-2 small, loaded into this repo's own GptModel.
 *
 * 1. Download (once, about 550 MB, git-ignored) and load the weights.
 * 2. Compare its loss on "The Verdict" with lesson 2's model, which trained on the story.
 * 3. Look at its next-token probabilities for a few prompts.
 * 4. Generate text, greedy and sampled.
 */
async function main(): Promise<void> {
  const model = loadGpt2(await ensureGpt2Weights());
  const total = model.numParameters();
  const outHead = GPT2_CONFIG.embDim * GPT2_CONFIG.vocabSize;
  console.log(`GPT-2 small loaded: ${(total - outHead).toLocaleString()} parameters`);
  console.log("(the output head is a transposed copy of the token embedding, counted once)\n");

  // 2. Loss on the story, using the same 256-token windows as lessons 1 and 2.
  const { train, val } = makeLoaders(loadVerdict(), 256);
  console.log("loss on The Verdict          training text   validation text");
  console.log(
    `  GPT-2 (pretrained on web)    ${fmt(evaluateLoss(model, train))}          ${fmt(evaluateLoss(model, val))}`,
  );
  const ours = loadCheckpoint("verdict-124m");
  if (ours) {
    console.log(
      `  lesson 2 (trained on story)  ${fmt(evaluateLoss(ours, train))}          ${fmt(evaluateLoss(ours, val))}`,
    );
    ours.dispose();
  }
  console.log("  untrained model              ~10.98          ~10.98");

  // 3. What GPT-2 expects next.
  for (const prompt of ["The capital of France is", "Every effort moves you", "1, 2, 3, 4,"]) {
    console.log(`\nmost likely next tokens after ${JSON.stringify(prompt)}:`);
    for (const { token, probability } of topNextTokens(model, prompt, 5)) {
      console.log(`  ${JSON.stringify(token).padEnd(14)} ${(probability * 100).toFixed(1)}%`);
    }
  }

  // 4. Generation.
  const prompts = ["Every effort moves you", "In a shocking finding, scientists discovered"];
  for (const text of prompts) {
    const prompt = textToIds(text);
    const greedy = model.generate(prompt, 40);
    console.log(`\ngreedy:\n  ${JSON.stringify(idsToText(greedy))}`);
    for (const seed of [1, 2]) {
      const sampled = model.generate(prompt, 40, { temperature: 1, topK: 40, seed });
      console.log(`temperature 1, top-40, seed ${seed}:\n  ${JSON.stringify(idsToText(sampled))}`);
      sampled.dispose();
    }
    tf.dispose([prompt, greedy]);
  }

  model.dispose();
}

function fmt(loss: number): string {
  return loss.toFixed(3).padStart(6);
}

/** The model's `k` most likely next tokens after `prompt`, with their probabilities. */
function topNextTokens(
  model: GptModel,
  prompt: string,
  k: number,
): { token: string; probability: number }[] {
  return tf.tidy(() => {
    const ids = textToIds(prompt);
    const logits = model.forward(ids); // [1, seqLen, vocabSize]
    const last = tf.squeeze(tf.slice(logits, [0, ids.shape[1] - 1, 0], [1, 1, -1]));
    const { values, indices } = tf.topk(tf.softmax(last), k);
    const probs = values.dataSync();
    return Array.from(indices.dataSync(), (id, i) => ({
      token: decode([id]),
      probability: probs[i] as number,
    }));
  });
}

main();
