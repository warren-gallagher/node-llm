import { GPT_CONFIG_124M, GptModel, trainModel } from "@node-llm/core";
import * as tf from "@tensorflow/tfjs-node";
import { idsToText, loadVerdict, makeLoaders, textToIds } from "./shared";

/**
 * Chapter 5: end-to-end training on "The Verdict", with a small model so it finishes in
 * well under a minute. Lesson 2 (`02-training-loop.ts`) runs the book's full-size setup.
 */
function main(): void {
  const config = {
    ...GPT_CONFIG_124M,
    contextLength: 64,
    embDim: 128,
    numHeads: 4,
    numLayers: 2,
  };
  const { train, val } = makeLoaders(loadVerdict(), config.contextLength, 8);
  const model = new GptModel(config);
  console.log(`parameters: ${model.numParameters().toLocaleString()}`);

  trainModel(model, train, {
    epochs: 10,
    learningRate: 0.001,
    weightDecay: 0.1,
    evalEvery: 10,
    valLoader: val,
    onEval: (p) =>
      console.log(`step ${p.step}: train ${p.trainLoss.toFixed(3)}, val ${p.valLoss.toFixed(3)}`),
  });

  const prompt = textToIds("Every effort moves you");
  const out = model.generate(prompt, 30);
  console.log(`sample: ${JSON.stringify(idsToText(out))}`);

  tf.dispose([prompt, out]);
  model.dispose();
}

main();
