import * as tf from "@tensorflow/tfjs-core";
import type { DataLoader } from "../data/dataLoader";
import type { GptModel } from "../model/gpt";

/**
 * Chapter 5: reusable training loop.
 *
 * PyTorch analog:
 *   for epoch: for batch: optimizer.zero_grad(); loss.backward(); optimizer.step()
 *
 * tfjs analog (below): `optimizer.minimize(fn, returnCost=true)` computes the loss and
 * applies gradients to every trainable variable touched by `fn` in one call.
 *
 * Gradient functions must be registered by the caller's tfjs package: `@tensorflow/tfjs-node`
 * and `@tensorflow/tfjs` do this; plain `tfjs-core` needs
 * `import "@tensorflow/tfjs-core/dist/register_all_gradients"`.
 */
export interface TrainConfig {
  epochs: number;
  learningRate: number;
  /** Evaluate train and validation loss every this many optimizer steps. */
  evalEvery?: number;
  /** Batches averaged per evaluation (from the start of each loader). Defaults to all. */
  evalBatches?: number;
  /** Data for the validation loss. Without it, only the training loss is evaluated. */
  valLoader?: DataLoader;
  /**
   * Decoupled weight decay, as in PyTorch's `AdamW`: after each step every weight shrinks by
   * `learningRate * weightDecay` of itself. Defaults to 0 (plain Adam).
   */
  weightDecay?: number;
  /** Called after each evaluation, e.g. to print progress. */
  onEval?: (point: EvalPoint) => void;
  /** Called at the end of each epoch. */
  onEpochEnd?: (epoch: number) => void;
}

export interface EvalPoint {
  epoch: number;
  step: number;
  tokensSeen: number;
  trainLoss: number;
  /** NaN when no `valLoader` was given. */
  valLoss: number;
}

export function trainModel(model: GptModel, loader: DataLoader, config: TrainConfig): EvalPoint[] {
  const optimizer = tf.train.adam(config.learningRate);
  const decay = 1 - config.learningRate * (config.weightDecay ?? 0);
  const history: EvalPoint[] = [];
  let step = 0;
  let tokensSeen = 0;

  for (let epoch = 0; epoch < config.epochs; epoch++) {
    for (const batch of loader) {
      tf.tidy(() => {
        const inputs = tf.tensor2d(batch.inputs, undefined, "int32");
        const targets = tf.tensor2d(batch.targets, undefined, "int32");

        // returnCost=true -> minimize returns the scalar loss (or null).
        const loss = optimizer.minimize(
          () => crossEntropyLoss(model, inputs, targets, true),
          true,
          model.weights,
        );
        loss?.dispose();

        if (decay !== 1) {
          for (const w of model.weights) w.assign(tf.mul(w, decay));
        }
        tokensSeen += inputs.size;
      });
      step++;

      if (config.evalEvery && step % config.evalEvery === 0) {
        const point = {
          epoch,
          step,
          tokensSeen,
          trainLoss: evaluateLoss(model, loader, config.evalBatches),
          valLoss: config.valLoader
            ? evaluateLoss(model, config.valLoader, config.evalBatches)
            : Number.NaN,
        };
        history.push(point);
        config.onEval?.(point);
      }
    }
    config.onEpochEnd?.(epoch);
  }

  optimizer.dispose();
  return history;
}

/**
 * Chapter 5, lesson 1: cross-entropy loss for next-token prediction.
 *
 * Every position in every sequence is one prediction: the logits at position i should give
 * a high probability to target token i (the input shifted by one). The loss is the average,
 * over all `batch * seqLen` predictions, of -log(probability of the correct token).
 *
 * Book analog: `torch.nn.functional.cross_entropy(logits.flatten(0, 1), targets.flatten())`.
 */
export function crossEntropyLoss(
  model: GptModel,
  inputs: tf.Tensor2D,
  targets: tf.Tensor2D,
  training = false,
): tf.Scalar {
  return tf.tidy(() => {
    const logits = model.forward(inputs, training); // [batch, seqLen, vocabSize]
    const vocabSize = logits.shape[2];

    // Flatten batch and sequence into one axis of independent predictions.
    const flatLogits = tf.reshape(logits, [-1, vocabSize]) as tf.Tensor2D;
    const flatTargets = tf.reshape(targets, [-1]) as tf.Tensor1D;
    return tf.losses.softmaxCrossEntropy(
      tf.oneHot(flatTargets, vocabSize),
      flatLogits,
    ) as tf.Scalar;
  });
}

/**
 * Chapter 5, lesson 1: average loss over a loader's batches, without training (no dropout,
 * no weight updates). Book analog: `calc_loss_loader`.
 */
export function evaluateLoss(model: GptModel, loader: DataLoader, maxBatches?: number): number {
  let total = 0;
  let count = 0;
  for (const batch of loader) {
    if (maxBatches !== undefined && count >= maxBatches) break;
    total += tf.tidy(() => {
      const inputs = tf.tensor2d(batch.inputs, undefined, "int32");
      const targets = tf.tensor2d(batch.targets, undefined, "int32");
      return crossEntropyLoss(model, inputs, targets).dataSync()[0] as number;
    });
    count++;
  }
  return count === 0 ? Number.NaN : total / count;
}
