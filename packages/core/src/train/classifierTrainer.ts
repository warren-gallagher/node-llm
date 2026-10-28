import * as tf from "@tensorflow/tfjs-core";
import type { ClassificationLoader } from "../data/classification";
import type { GptClassifier } from "../model/classifier";

/**
 * Chapter 6: training and evaluating a classifier.
 *
 * The loop is chapter 5's with three changes: the loss uses only the last token's scores
 * against one label per text, only `model.trainableWeights` are updated, and accuracy is
 * measured at the end of every epoch.
 */
export interface ClassifierTrainConfig {
  epochs: number;
  learningRate: number;
  /** Decoupled weight decay, as in PyTorch's `AdamW` (see chapter 5). Defaults to 0. */
  weightDecay?: number;
  /** Evaluate train and validation loss every this many optimizer steps, counting from 0. */
  evalEvery?: number;
  /** Batches used for each loss evaluation and each epoch's accuracy. Defaults to all. */
  evalBatches?: number;
  valLoader?: ClassificationLoader;
  onEval?: (point: ClassifierEvalPoint) => void;
  onEpochEnd?: (accuracy: EpochAccuracy) => void;
}

export interface ClassifierEvalPoint {
  epoch: number;
  step: number;
  examplesSeen: number;
  trainLoss: number;
  /** NaN when no `valLoader` was given. */
  valLoss: number;
}

export interface EpochAccuracy {
  epoch: number;
  trainAccuracy: number;
  /** NaN when no `valLoader` was given. */
  valAccuracy: number;
}

export interface ClassifierHistory {
  losses: ClassifierEvalPoint[];
  accuracies: EpochAccuracy[];
  examplesSeen: number;
}

/**
 * Chapter 6, lesson 6: fine-tunes `model.trainableWeights` on labelled batches.
 *
 * Book analog: `train_classifier_simple`. Frozen weights still take part in every forward
 * pass, but the optimizer is only given the trainable ones, so no gradients are computed
 * or applied for the rest.
 */
export function trainClassifier(
  model: GptClassifier,
  loader: ClassificationLoader,
  config: ClassifierTrainConfig,
): ClassifierHistory {
  const optimizer = tf.train.adam(config.learningRate);
  const decay = 1 - config.learningRate * (config.weightDecay ?? 0);
  const weights = model.trainableWeights;
  const history: ClassifierHistory = { losses: [], accuracies: [], examplesSeen: 0 };
  let step = -1;

  for (let epoch = 0; epoch < config.epochs; epoch++) {
    for (const batch of loader) {
      tf.tidy(() => {
        const inputs = tf.tensor2d(batch.inputs, undefined, "int32");
        const labels = tf.tensor1d(batch.labels, "int32");

        const loss = optimizer.minimize(
          () => classificationLoss(model, inputs, labels, true),
          true,
          weights,
        );
        loss?.dispose();

        if (decay !== 1) {
          for (const w of weights) w.assign(tf.mul(w, decay));
        }
      });
      history.examplesSeen += batch.labels.length;
      step++;

      if (config.evalEvery && step % config.evalEvery === 0) {
        const point = {
          epoch,
          step,
          examplesSeen: history.examplesSeen,
          trainLoss: evaluateClassificationLoss(model, loader, config.evalBatches),
          valLoss: config.valLoader
            ? evaluateClassificationLoss(model, config.valLoader, config.evalBatches)
            : Number.NaN,
        };
        history.losses.push(point);
        config.onEval?.(point);
      }
    }

    const accuracy = {
      epoch,
      trainAccuracy: classificationAccuracy(model, loader, config.evalBatches),
      valAccuracy: config.valLoader
        ? classificationAccuracy(model, config.valLoader, config.evalBatches)
        : Number.NaN,
    };
    history.accuracies.push(accuracy);
    config.onEpochEnd?.(accuracy);
  }

  optimizer.dispose();
  return history;
}

/**
 * Chapter 6, lesson 5: cross-entropy between the last token's class scores and each text's
 * label, averaged over the batch.
 *
 * Book analog: `calc_loss_batch` with `model(input_batch)[:, -1, :]`.
 */
export function classificationLoss(
  model: GptClassifier,
  inputs: tf.Tensor2D,
  labels: tf.Tensor1D,
  training = false,
): tf.Scalar {
  return tf.tidy(() => {
    const logits = model.lastTokenLogits(inputs, training); // [batch, numClasses]
    return tf.losses.softmaxCrossEntropy(tf.oneHot(labels, model.numClasses), logits) as tf.Scalar;
  });
}

/**
 * Chapter 6, lesson 5: average loss over a loader's first `maxBatches` batches (all by
 * default), without training. Book analog: `calc_loss_loader`.
 */
export function evaluateClassificationLoss(
  model: GptClassifier,
  loader: ClassificationLoader,
  maxBatches?: number,
): number {
  let total = 0;
  let count = 0;
  for (const batch of loader) {
    if (maxBatches !== undefined && count >= maxBatches) break;
    total += tf.tidy(() => {
      const inputs = tf.tensor2d(batch.inputs, undefined, "int32");
      const labels = tf.tensor1d(batch.labels, "int32");
      return classificationLoss(model, inputs, labels).dataSync()[0] as number;
    });
    count++;
  }
  return count === 0 ? Number.NaN : total / count;
}

/**
 * Chapter 6, lesson 5: the fraction of texts whose predicted class matches the label, over a
 * loader's first `maxBatches` batches (all by default). Book analog: `calc_accuracy_loader`.
 */
export function classificationAccuracy(
  model: GptClassifier,
  loader: ClassificationLoader,
  maxBatches?: number,
): number {
  let correct = 0;
  let seen = 0;
  let count = 0;
  for (const batch of loader) {
    if (maxBatches !== undefined && count >= maxBatches) break;
    const predicted = tf.tidy(() =>
      model.predict(tf.tensor2d(batch.inputs, undefined, "int32")).dataSync(),
    );
    correct += batch.labels.filter((label, i) => predicted[i] === label).length;
    seen += batch.labels.length;
    count++;
  }
  return seen === 0 ? Number.NaN : correct / seen;
}
