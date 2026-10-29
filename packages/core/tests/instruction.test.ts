import "@tensorflow/tfjs-backend-cpu";
// tfjs-core ships gradients separately; tfjs-node (used by the chapters) registers them itself.
import "@tensorflow/tfjs-core/dist/register_all_gradients";
import * as tf from "@tensorflow/tfjs-core";
import { beforeAll, describe, expect, it } from "vitest";
import type { GptConfig } from "../src/index";
import {
  collateInstructions,
  crossEntropyLoss,
  evaluateLoss,
  formatInput,
  formatResponse,
  GptModel,
  IGNORE_INDEX,
  InstructionDataset,
  InstructionLoader,
  trainModel,
} from "../src/index";

const TINY: GptConfig = {
  vocabSize: 20,
  contextLength: 16,
  embDim: 16,
  numHeads: 2,
  numLayers: 1,
  dropout: 0,
  qkvBias: false,
};

beforeAll(async () => {
  await tf.setBackend("cpu");
  await tf.ready();
});

describe("formatInput", () => {
  it("includes the input section only when there is an input", () => {
    const withInput = { instruction: "Fix the spelling.", input: "Ocassion", output: "Occasion" };
    expect(formatInput(withInput)).toBe(
      "Below is an instruction that describes a task. Write a response that appropriately " +
        "completes the request.\n\n### Instruction:\nFix the spelling.\n\n### Input:\nOcassion",
    );
    expect(formatInput({ ...withInput, input: "" })).not.toContain("### Input:");
    expect(formatResponse(withInput)).toBe("\n\n### Response:\nOccasion");
  });
});

describe("collateInstructions", () => {
  // The book's example batch and its printed result.
  const batch = [
    [0, 1, 2, 3, 4],
    [5, 6],
    [7, 8, 9],
  ];

  it("pads, shifts and masks padding in the targets", () => {
    const { inputs, targets } = collateInstructions(batch);
    expect(inputs).toEqual([
      [0, 1, 2, 3, 4],
      [5, 6, 50256, 50256, 50256],
      [7, 8, 9, 50256, 50256],
    ]);
    expect(targets).toEqual([
      [1, 2, 3, 4, 50256],
      [6, 50256, -100, -100, -100],
      [8, 9, 50256, -100, -100],
    ]);
  });

  it("cuts inputs and targets to allowedMaxLength", () => {
    const { inputs, targets } = collateInstructions(batch, { allowedMaxLength: 3 });
    expect(inputs.map((row) => row.length)).toEqual([3, 3, 3]);
    expect(targets[1]).toEqual([6, 50256, -100]);
  });
});

describe("InstructionLoader", () => {
  const dataset = new InstructionDataset([[1, 2], [3, 4, 5, 6], [7], [8, 9, 10]]);

  it("pads each batch to its own longest example", () => {
    const loader = new InstructionLoader(dataset, { batchSize: 2 });
    expect(loader.length).toBe(2);
    expect([...loader].map((b) => b.inputs[0]?.length)).toEqual([4, 3]);
  });
});

describe("crossEntropyLoss with ignored targets", () => {
  it("matches the loss of the same batch without the padding", () => {
    // The causal mask means padding after a token can't change that token's logits, so
    // ignoring the padded targets must give the unpadded loss exactly.
    const model = new GptModel(TINY);
    const short = collateInstructions([[1, 2, 3]], { padId: 19 });
    const padded = collateInstructions(
      [
        [1, 2, 3],
        [4, 5, 6, 7, 8, 9],
      ],
      { padId: 19 },
    );
    /** Loss on the first example of a batch only. */
    function loss(b: { inputs: number[][]; targets: number[][] }): number {
      return tf.tidy(
        () =>
          crossEntropyLoss(
            model,
            tf.tensor2d(b.inputs.slice(0, 1), undefined, "int32"),
            tf.tensor2d(b.targets.slice(0, 1), undefined, "int32"),
          ).dataSync()[0] as number,
      );
    }

    expect(padded.targets[0]).toContain(IGNORE_INDEX);
    expect(loss(padded)).toBeCloseTo(loss(short), 5);
    model.dispose();
  });
});

describe("trainModel on an InstructionLoader", () => {
  it("learns to answer with the next number in a sequence", () => {
    const model = new GptModel(TINY);
    const examples = Array.from({ length: 16 }, (_, i) => [1 + (i % 8), 2 + (i % 8), 3 + (i % 8)]);
    const loader = new InstructionLoader(new InstructionDataset(examples), {
      batchSize: 4,
      shuffle: true,
      padId: 0,
    });

    const before = evaluateLoss(model, loader);
    trainModel(model, loader, { epochs: 20, learningRate: 0.01 });
    expect(evaluateLoss(model, loader)).toBeLessThan(before / 4);
    model.dispose();
  });
});
