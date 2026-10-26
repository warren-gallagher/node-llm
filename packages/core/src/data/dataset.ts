/** One training sample: `target` is `input` shifted right by one token. */
export interface Sample {
  input: number[];
  target: number[];
}

/**
 * Chapter 2, lesson 4: sliding-window dataset over a token stream.
 *
 * Book analog: `GPTDatasetV1`. A window of `maxLength` tokens slides over the ids in steps
 * of `stride`; each window is an input, and the same window shifted by one is its target,
 * so the model learns to predict the next token at every position:
 *   ids    = [a, b, c, d, e, ...]
 *   input  = [a, b, c, d]
 *   target = [b, c, d, e]
 *
 * `stride === maxLength` gives non-overlapping chunks; a smaller stride overlaps them.
 */
export class LlmDataset {
  private readonly samples: Sample[] = [];

  constructor(ids: readonly number[], maxLength: number, stride: number) {
    if (maxLength < 1 || stride < 1) {
      throw new Error("maxLength and stride must be positive");
    }

    // Stop once there is no room left for the one-token-shifted target.
    for (let i = 0; i + maxLength < ids.length; i += stride) {
      this.samples.push({
        input: ids.slice(i, i + maxLength),
        target: ids.slice(i + 1, i + maxLength + 1),
      });
    }
  }

  get length(): number {
    return this.samples.length;
  }

  at(index: number): Sample {
    const sample = this.samples[index];
    if (!sample) {
      throw new RangeError(`sample index ${index} out of range [0, ${this.length})`);
    }
    return sample;
  }
}
