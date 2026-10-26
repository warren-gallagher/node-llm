import * as tf from "@tensorflow/tfjs-core";
import type { LlmDataset } from "./dataset";

/** A batch as plain arrays, shape `[batchSize, maxLength]`. */
export interface ArrayBatch {
  inputs: number[][];
  targets: number[][];
}

/** A batch as int32 tensors, shape `[batchSize, maxLength]`. The caller must dispose them. */
export interface TensorBatch {
  inputs: tf.Tensor2D;
  targets: tf.Tensor2D;
}

export type Batch<AsTensors extends boolean> = AsTensors extends true ? TensorBatch : ArrayBatch;

export interface DataLoaderOptions<AsTensors extends boolean = false> {
  batchSize: number;
  /** Shuffle sample order at the start of every pass. Defaults to true. */
  shuffle?: boolean;
  /** Drop the final batch if it is smaller than `batchSize`. Defaults to true. */
  dropLast?: boolean;
  /** Yield int32 tensors instead of plain arrays. Defaults to false. */
  asTensors?: AsTensors;
}

/**
 * Chapter 2, lesson 4: batches samples from an `LlmDataset`.
 *
 * Book analog: `torch.utils.data.DataLoader`. Iterating yields one batch at a time, so
 * with `asTensors: true` tensors are only allocated for batches actually pulled. Each new
 * `for...of` is a fresh pass (epoch), reshuffled when `shuffle` is on.
 */
export class DataLoader<AsTensors extends boolean = false> {
  private readonly batchSize: number;
  private readonly shuffle: boolean;
  private readonly dropLast: boolean;
  private readonly asTensors: boolean;

  constructor(
    private readonly dataset: LlmDataset,
    options: DataLoaderOptions<AsTensors>,
  ) {
    const { batchSize, shuffle = true, dropLast = true, asTensors = false } = options;
    if (batchSize < 1) {
      throw new Error("batchSize must be positive");
    }
    this.batchSize = batchSize;
    this.shuffle = shuffle;
    this.dropLast = dropLast;
    this.asTensors = asTensors;
  }

  /** Number of batches per pass. */
  get length(): number {
    const n = this.dataset.length / this.batchSize;
    return this.dropLast ? Math.floor(n) : Math.ceil(n);
  }

  *[Symbol.iterator](): Iterator<Batch<AsTensors>> {
    const order = Array.from({ length: this.dataset.length }, (_, i) => i);
    if (this.shuffle) {
      shuffleInPlace(order);
    }

    for (let start = 0; start < order.length; start += this.batchSize) {
      const indices = order.slice(start, start + this.batchSize);
      if (this.dropLast && indices.length < this.batchSize) break;

      const samples = indices.map((i) => this.dataset.at(i));
      const inputs = samples.map((s) => s.input);
      const targets = samples.map((s) => s.target);

      if (this.asTensors) {
        const batch: TensorBatch = {
          inputs: tf.tensor2d(inputs, undefined, "int32"),
          targets: tf.tensor2d(targets, undefined, "int32"),
        };
        yield batch as Batch<AsTensors>;
      } else {
        const batch: ArrayBatch = { inputs, targets };
        yield batch as Batch<AsTensors>;
      }
    }
  }
}

/** Fisher-Yates shuffle. */
function shuffleInPlace(values: number[]): void {
  for (let i = values.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [values[i], values[j]] = [values[j] as number, values[i] as number];
  }
}
