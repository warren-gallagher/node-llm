import { classificationAccuracy, evaluateClassificationLoss } from "@node-llm/core";
import { loadGpt2Classifier, makeSpamLoaders } from "./shared";

/**
 * Chapter 6, lesson 5: accuracy and loss before fine-tuning.
 *
 * Accuracy (the share of messages classified correctly) is what we care about, but it can't
 * be trained directly: changing a weight slightly usually changes no prediction, so it gives
 * no gradient. Cross-entropy on the last token's two scores does, so that's what training
 * minimizes. With a random head, accuracy should be near 50%, a coin toss.
 */
async function main(): Promise<void> {
  const { train, validation, test } = await makeSpamLoaders();
  const model = await loadGpt2Classifier();

  console.log("accuracy on the first 10 batches (80 messages):");
  console.log(`  training    ${pct(classificationAccuracy(model, train, 10))}`);
  console.log(`  validation  ${pct(classificationAccuracy(model, validation, 10))}`);
  console.log(`  test        ${pct(classificationAccuracy(model, test, 10))}`);

  console.log("\nloss on the first 5 batches:");
  console.log(`  training    ${evaluateClassificationLoss(model, train, 5).toFixed(3)}`);
  console.log(`  validation  ${evaluateClassificationLoss(model, validation, 5).toFixed(3)}`);
  console.log(`  test        ${evaluateClassificationLoss(model, test, 5).toFixed(3)}`);

  model.dispose();
}

function pct(fraction: number): string {
  return `${(fraction * 100).toFixed(2)}%`;
}

main();
