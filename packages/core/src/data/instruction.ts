import type { ArrayBatch } from "./dataLoader";

/** One example from an Alpaca-style instruction dataset. `input` may be empty. */
export interface InstructionEntry {
  instruction: string;
  input: string;
  output: string;
}

/**
 * Chapter 7, lesson 1: the Alpaca prompt template, up to (not including) the response.
 *
 * Book analog: `format_input`. The `### Input:` section is left out when there is no input.
 */
export function formatInput(entry: InstructionEntry): string {
  const instruction =
    "Below is an instruction that describes a task. " +
    "Write a response that appropriately completes the request." +
    `\n\n### Instruction:\n${entry.instruction}`;
  const input = entry.input ? `\n\n### Input:\n${entry.input}` : "";
  return instruction + input;
}

/** The part of the template the model learns to write: the response heading and the answer. */
export function formatResponse(entry: InstructionEntry): string {
  return `\n\n### Response:\n${entry.output}`;
}

/**
 * Chapter 7, lesson 2: instruction examples as token ids, one list per example.
 *
 * Book analog: `InstructionDataset`. Unlike chapter 6's dataset, examples are not padded
 * here: each batch is padded to its own longest example by `collateInstructions`.
 */
export class InstructionDataset {
  constructor(private readonly encodedTexts: readonly number[][]) {}

  get length(): number {
    return this.encodedTexts.length;
  }

  at(index: number): number[] {
    const ids = this.encodedTexts[index];
    if (!ids) {
      throw new RangeError(`sample index ${index} out of range [0, ${this.length})`);
    }
    return ids;
  }
}

export interface CollateOptions {
  /** Token used for the end of each text and for padding. Defaults to 50256, `<|endoftext|>`. */
  padId?: number;
  /** Target value the loss skips. Defaults to -100, as in PyTorch. */
  ignoreIndex?: number;
  /** Cut inputs and targets to this many tokens, e.g. the model's context length. */
  allowedMaxLength?: number;
}

/**
 * Chapter 7, lesson 2: pads a batch of examples and builds next-token targets.
 *
 * Book analog: `custom_collate_fn`. For each example:
 *   1. append one end-of-text token, so the model learns where a response stops;
 *   2. pad with the same token to the batch's longest example, plus that one token;
 *   3. inputs drop the last token and targets drop the first (the shift by one from chapter 5);
 *   4. in the targets, every padding token after the first end-of-text becomes `ignoreIndex`,
 *      so padding doesn't count toward the loss.
 */
export function collateInstructions(
  batch: readonly number[][],
  options: CollateOptions = {},
): ArrayBatch {
  const { padId = 50256, ignoreIndex = -100, allowedMaxLength } = options;
  const length = Math.max(...batch.map((ids) => ids.length + 1));
  const inputs: number[][] = [];
  const targets: number[][] = [];

  for (const ids of batch) {
    const padded = [...ids, ...new Array<number>(length - ids.length).fill(padId)];
    let input = padded.slice(0, -1);
    let target = padded.slice(1).map((id, i) => (i > ids.length - 1 ? ignoreIndex : id));

    if (allowedMaxLength !== undefined) {
      input = input.slice(0, allowedMaxLength);
      target = target.slice(0, allowedMaxLength);
    }
    inputs.push(input);
    targets.push(target);
  }
  return { inputs, targets };
}

export interface InstructionLoaderOptions extends CollateOptions {
  batchSize: number;
  /** Shuffle sample order at the start of every pass. Defaults to false. */
  shuffle?: boolean;
  /** Drop the final batch if it is smaller than `batchSize`. Defaults to false. */
  dropLast?: boolean;
}

/**
 * Chapter 7, lesson 3: batches an `InstructionDataset` with `collateInstructions`.
 *
 * Book analog: `DataLoader(..., collate_fn=custom_collate_fn)`. Yields the same
 * `{ inputs, targets }` batches as chapter 2's `DataLoader`, so chapter 5's `trainModel` and
 * `evaluateLoss` accept it, but each batch has its own length.
 */
export class InstructionLoader implements Iterable<ArrayBatch> {
  private readonly batchSize: number;
  private readonly shuffle: boolean;
  private readonly dropLast: boolean;
  private readonly collate: CollateOptions;

  constructor(
    private readonly dataset: InstructionDataset,
    options: InstructionLoaderOptions,
  ) {
    const { batchSize, shuffle = false, dropLast = false, ...collate } = options;
    if (batchSize < 1) {
      throw new Error("batchSize must be positive");
    }
    this.batchSize = batchSize;
    this.shuffle = shuffle;
    this.dropLast = dropLast;
    this.collate = collate;
  }

  /** Number of batches per pass. */
  get length(): number {
    const n = this.dataset.length / this.batchSize;
    return this.dropLast ? Math.floor(n) : Math.ceil(n);
  }

  *[Symbol.iterator](): Iterator<ArrayBatch> {
    const order = Array.from({ length: this.dataset.length }, (_, i) => i);
    if (this.shuffle) {
      shuffleInPlace(order);
    }

    for (let start = 0; start < order.length; start += this.batchSize) {
      const indices = order.slice(start, start + this.batchSize);
      if (this.dropLast && indices.length < this.batchSize) break;
      yield collateInstructions(
        indices.map((i) => this.dataset.at(i)),
        this.collate,
      );
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
