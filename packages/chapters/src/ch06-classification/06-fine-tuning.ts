import { classificationAccuracy, trainClassifier } from "@node-llm/core";
import { loadGpt2Classifier, makeSpamLoaders, saveClassifier } from "./shared";

/**
 * Chapter 6, lesson 6: fine-tune on the labelled messages.
 *
 * The book's settings: 5 epochs, AdamW with learning rate 5e-5 and weight decay 0.1, loss
 * checked every 50 steps on 5 batches, accuracy on 5 batches after each epoch. The learning
 * rate is about 100 times smaller than chapter 5's because the starting weights are already
 * good and only need adjusting.
 *
 * Takes roughly 15-25 minutes on a laptop CPU. Saves the fine-tuned weights for lesson 7.
 */
async function main(): Promise<void> {
  const { train, validation, test, maxLength } = await makeSpamLoaders();
  const model = await loadGpt2Classifier();
  const start = Date.now();

  trainClassifier(model, train, {
    epochs: 5,
    learningRate: 5e-5,
    weightDecay: 0.1,
    evalEvery: 50,
    evalBatches: 5,
    valLoader: validation,
    onEval: (p) =>
      console.log(
        `Ep ${p.epoch + 1} (step ${String(p.step).padStart(6, "0")}): ` +
          `train loss ${p.trainLoss.toFixed(3)}, val loss ${p.valLoss.toFixed(3)}`,
      ),
    onEpochEnd: (a) =>
      console.log(
        `training accuracy: ${pct(a.trainAccuracy)} | validation accuracy: ${pct(a.valAccuracy)}`,
      ),
  });
  console.log(`training completed in ${((Date.now() - start) / 60000).toFixed(2)} minutes`);

  console.log("\naccuracy on every message:");
  console.log(`  training    ${pct(classificationAccuracy(model, train))}`);
  console.log(`  validation  ${pct(classificationAccuracy(model, validation))}`);
  console.log(`  test        ${pct(classificationAccuracy(model, test))}`);

  console.log(`\nsaved to ${saveClassifier(model, "spam-classifier", maxLength)}`);
  model.dispose();
}

function pct(fraction: number): string {
  return `${(fraction * 100).toFixed(2)}%`;
}

main();
