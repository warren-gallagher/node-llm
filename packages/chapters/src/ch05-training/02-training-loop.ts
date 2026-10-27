import { type EvalPoint, GptModel, trainModel } from "@node-llm/core";
import * as tf from "@tensorflow/tfjs-node";
import {
  BOOK_CONFIG,
  idsToText,
  loadVerdict,
  makeLoaders,
  saveCheckpoint,
  textToIds,
} from "./shared";

/**
 * Chapter 5, Lesson 2: the training loop.
 *
 * Trains GPT-2 small (context 256) on "The Verdict" with the book's settings: 10 epochs,
 * AdamW with learning rate 0.0004 and weight decay 0.1, batches of 2. Prints the training
 * and validation loss every 5 steps and a sample continuation after each epoch, then saves
 * the weights for lesson 3.
 *
 * On a CPU this takes several minutes (about 5 seconds per step).
 */
function main(): void {
  const { train, val, trainTokens, valTokens } = makeLoaders(
    loadVerdict(),
    BOOK_CONFIG.contextLength,
  );
  console.log(`training tokens: ${trainTokens}, validation tokens: ${valTokens}`);
  console.log(`steps per epoch: ${train.length}`);

  const model = new GptModel(BOOK_CONFIG);
  const started = Date.now();

  trainModel(model, train, {
    epochs: 10,
    learningRate: 0.0004,
    weightDecay: 0.1,
    evalEvery: 5,
    evalBatches: 5,
    valLoader: val,
    onEval: printLosses,
    onEpochEnd: () => printSample(model, "Every effort moves you"),
  });

  const minutes = ((Date.now() - started) / 60000).toFixed(1);
  console.log(`\ntraining took ${minutes} minutes`);
  console.log("saved to", saveCheckpoint(model, "verdict-124m"));
  model.dispose();
}

function printLosses(point: EvalPoint): void {
  const step = String(point.step).padStart(3, "0");
  console.log(
    `epoch ${point.epoch + 1} (step ${step}): train loss ${point.trainLoss.toFixed(3)}, ` +
      `val loss ${point.valLoss.toFixed(3)}`,
  );
}

/** Greedy continuation of `prompt`, printed on one line. */
function printSample(model: GptModel, prompt: string): void {
  const ids = textToIds(prompt);
  const out = model.generate(ids, 50);
  console.log(`  sample: ${JSON.stringify(idsToText(out))}`);
  tf.dispose([ids, out]);
}

main();
