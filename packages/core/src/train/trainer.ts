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
 * TODO(ch05): wire up real logits + cross-entropy once GptModel.forward is implemented.
 */
export interface TrainConfig {
  epochs: number;
  learningRate: number;
  evalEvery?: number;
}

export function trainModel(model: GptModel, loader: DataLoader, config: TrainConfig): void {
  const optimizer = tf.train.adam(config.learningRate);

  for (let epoch = 0; epoch < config.epochs; epoch++) {
    for (const batch of loader) {
      tf.tidy(() => {
        const inputs = tf.tensor2d(batch.inputs, undefined, "int32");
        const targets = tf.tensor2d(batch.targets, undefined, "int32");

        // returnCost=true -> minimize returns the scalar loss (or null).
        const loss = optimizer.minimize(() => crossEntropyLoss(model, inputs, targets), true);
        loss?.dispose();
      });
    }
  }

  optimizer.dispose();
}

function crossEntropyLoss(
  _model: GptModel,
  _inputs: tf.Tensor2D,
  _targets: tf.Tensor2D,
): tf.Scalar {
  throw new Error("TODO(ch05): logits = model.forward(inputs); return softmaxCrossEntropy");
}
