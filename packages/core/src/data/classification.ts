/** One labelled text: its token ids (padded or truncated to a fixed length) and its class. */
export interface LabelledSample {
  input: number[];
  label: number;
}

/** A batch of labelled texts as plain arrays: `inputs` is `[batchSize, maxLength]`. */
export interface LabelledBatch {
  inputs: number[][];
  labels: number[];
}

/**
 * Chapter 6, lesson 2: labelled texts, all brought to the same length.
 *
 * Book analog: `SpamDataset`. A batch tensor needs every row to have the same length, so
 * each text is cut to `maxLength` tokens if longer, then padded with `padId` up to
 * `maxLength`. Without a `maxLength`, it is the longest text's length, so nothing is cut;
 * the validation and test sets reuse the training set's length.
 *
 * GPT-2 has no dedicated padding token; the book pads with `<|endoftext|>` (50256).
 */
export class ClassificationDataset {
  readonly maxLength: number;
  private readonly samples: LabelledSample[];

  constructor(
    encodedTexts: readonly number[][],
    labels: readonly number[],
    maxLength?: number,
    padId = 50256,
  ) {
    if (encodedTexts.length !== labels.length) {
      throw new Error(`${encodedTexts.length} texts but ${labels.length} labels`);
    }
    this.maxLength = maxLength ?? Math.max(...encodedTexts.map((ids) => ids.length));

    this.samples = encodedTexts.map((ids, i) => {
      const cut = ids.slice(0, this.maxLength);
      const padding = new Array<number>(this.maxLength - cut.length).fill(padId);
      return { input: [...cut, ...padding], label: labels[i] as number };
    });
  }

  get length(): number {
    return this.samples.length;
  }

  at(index: number): LabelledSample {
    const sample = this.samples[index];
    if (!sample) {
      throw new RangeError(`sample index ${index} out of range [0, ${this.length})`);
    }
    return sample;
  }
}

export interface ClassificationLoaderOptions {
  batchSize: number;
  /** Shuffle sample order at the start of every pass. Defaults to false. */
  shuffle?: boolean;
  /** Drop the final batch if it is smaller than `batchSize`. Defaults to false. */
  dropLast?: boolean;
}

/**
 * Chapter 6, lesson 2: batches a `ClassificationDataset`.
 *
 * The same idea as chapter 2's `DataLoader`, for samples with one label instead of a target
 * sequence. Each new `for...of` is a fresh pass (epoch), reshuffled when `shuffle` is on.
 */
export class ClassificationLoader {
  private readonly batchSize: number;
  private readonly shuffle: boolean;
  private readonly dropLast: boolean;

  constructor(
    private readonly dataset: ClassificationDataset,
    options: ClassificationLoaderOptions,
  ) {
    const { batchSize, shuffle = false, dropLast = false } = options;
    if (batchSize < 1) {
      throw new Error("batchSize must be positive");
    }
    this.batchSize = batchSize;
    this.shuffle = shuffle;
    this.dropLast = dropLast;
  }

  /** Number of batches per pass. */
  get length(): number {
    const n = this.dataset.length / this.batchSize;
    return this.dropLast ? Math.floor(n) : Math.ceil(n);
  }

  *[Symbol.iterator](): Iterator<LabelledBatch> {
    const order = Array.from({ length: this.dataset.length }, (_, i) => i);
    if (this.shuffle) {
      shuffleInPlace(order);
    }

    for (let start = 0; start < order.length; start += this.batchSize) {
      const indices = order.slice(start, start + this.batchSize);
      if (this.dropLast && indices.length < this.batchSize) break;

      const samples = indices.map((i) => this.dataset.at(i));
      yield { inputs: samples.map((s) => s.input), labels: samples.map((s) => s.label) };
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
