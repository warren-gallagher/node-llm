import { writeFileSync } from "node:fs";
import { evaluateLoss, trainModel } from "@node-llm/core";
import {
  chosenSize,
  DATA_DIR,
  loadPretrained,
  makeInstructionLoaders,
  respond,
  saveFineTuned,
} from "./shared";

/**
 * Chapter 7, lesson 5: fine-tune on the instruction data.
 *
 * Nothing new is needed: this is chapter 5's training loop and next-token loss, run on the
 * instruction batches, with every weight trained. The book's settings: 2 epochs, AdamW with
 * learning rate 5e-5 and weight decay 0.1, loss checked every 5 steps on 5 batches.
 *
 * GPT-2 124M (the default) takes roughly half an hour on a laptop CPU; GPT-2 355M (`--355m`)
 * takes several hours. Saves the fine-tuned model for lessons 6 and 7, and the losses to
 * `data/instruction/`.
 */
async function main(): Promise<void> {
  const size = chosenSize();
  const { trainLoader, validationLoader, validation } = await makeInstructionLoaders();
  const model = await loadPretrained(size);
  const sample = validation[0];

  console.log(`fine-tuning GPT-2 ${size.toUpperCase()}: ${trainLoader.length} steps per epoch`);
  console.log(`training loss:   ${evaluateLoss(model, trainLoader, 5).toFixed(3)}`);
  console.log(`validation loss: ${evaluateLoss(model, validationLoader, 5).toFixed(3)}\n`);

  const start = Date.now();
  const history = trainModel(model, trainLoader, {
    epochs: 2,
    learningRate: 5e-5,
    weightDecay: 0.1,
    evalEvery: 5,
    evalBatches: 5,
    valLoader: validationLoader,
    onEval: (p) =>
      console.log(
        `Ep ${p.epoch + 1} (step ${String(p.step).padStart(6, "0")}): ` +
          `train loss ${p.trainLoss.toFixed(3)}, val loss ${p.valLoss.toFixed(3)} ` +
          `[${((Date.now() - start) / 60000).toFixed(1)} min]`,
      ),
    onEpochEnd: () => {
      if (sample) console.log(`\n>> ${respond(model, sample, 50)}\n`);
    },
  });
  console.log(`training completed in ${((Date.now() - start) / 60000).toFixed(2)} minutes`);

  const lossFile = new URL(`losses-${size}.json`, DATA_DIR);
  writeFileSync(lossFile, JSON.stringify(history, null, 2));
  console.log(`losses saved to ${lossFile.pathname}`);
  console.log(`model saved to ${saveFineTuned(model, size)}`);
  model.dispose();
}

main();
