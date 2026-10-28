import * as tf from "@tensorflow/tfjs-core";
import { linearWeight, project } from "./attention/utils";
import type { GptModel } from "./gpt";

/**
 * Chapter 6, lesson 4: a GPT model turned into a text classifier.
 *
 * The pretrained model's output head (one score per vocabulary token) is replaced by a much
 * smaller one with one score per class. Everything below the head is reused unchanged:
 *   token ids -> embeddings -> blocks -> final LayerNorm -> classification head
 *
 * Book analog: `model.out_head = torch.nn.Linear(emb_dim, num_classes)`. Like `nn.Linear`,
 * the new head has a bias, and both are initialized uniformly in +-1/sqrt(embDim).
 *
 * Only the last token's scores are used to classify. Because of the causal mask, the last
 * position is the only one that has attended to every token in the text.
 */
export class GptClassifier {
  private readonly headWeight: tf.Variable;
  private readonly headBias: tf.Variable;

  constructor(
    readonly base: GptModel,
    readonly numClasses: number,
  ) {
    const { embDim } = base.config;
    this.headWeight = tf.variable(linearWeight(embDim, numClasses));
    this.headBias = tf.variable(tf.randomUniform([numClasses], -(embDim ** -0.5), embDim ** -0.5));
  }

  /** The new head's weights: `[embDim, numClasses]` and `[numClasses]`. */
  get headWeights(): tf.Variable[] {
    return [this.headWeight, this.headBias];
  }

  /**
   * The weights the book fine-tunes: the last transformer block, the final LayerNorm and the
   * new head. Everything else stays frozen at its pretrained values.
   */
  get trainableWeights(): tf.Variable[] {
    const lastBlock = this.base.blocks[this.base.blocks.length - 1];
    return [...(lastBlock?.weights ?? []), ...this.base.finalNorm.weights, ...this.headWeights];
  }

  /** Scores for every position: [batch, seqLen] -> [batch, seqLen, numClasses]. */
  forward(tokenIds: tf.Tensor2D, training = false): tf.Tensor3D {
    return tf.tidy(() =>
      project(
        this.base.hiddenStates(tokenIds, training),
        this.headWeight as tf.Tensor2D,
        this.headBias as tf.Tensor1D,
      ),
    );
  }

  /** Scores for the last position only: [batch, seqLen] -> [batch, numClasses]. */
  lastTokenLogits(tokenIds: tf.Tensor2D, training = false): tf.Tensor2D {
    return tf.tidy(() => {
      const logits = this.forward(tokenIds, training);
      const seqLen = logits.shape[1];
      return tf.squeeze(tf.slice(logits, [0, seqLen - 1, 0], [-1, 1, -1]), [1]) as tf.Tensor2D;
    });
  }

  /** The predicted class for each text: [batch, seqLen] -> [batch] (int32). */
  predict(tokenIds: tf.Tensor2D): tf.Tensor1D {
    return tf.tidy(() => tf.argMax(this.lastTokenLogits(tokenIds), -1) as tf.Tensor1D);
  }

  dispose(): void {
    this.base.dispose();
    for (const w of this.headWeights) w.dispose();
  }
}
