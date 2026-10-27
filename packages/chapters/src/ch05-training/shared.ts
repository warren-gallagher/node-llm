import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { DataLoader, GPT_CONFIG_124M, type GptConfig, GptModel, LlmDataset } from "@node-llm/core";
import * as tf from "@tensorflow/tfjs-node";
import { decode, encode } from "gpt-tokenizer/encoding/r50k_base";

/**
 * Helpers shared by the chapter 5 lessons: the training text, train/validation loaders,
 * text <-> token id conversion, and saving / loading trained weights.
 */

/** The book's chapter 5 setup: GPT-2 small, with the context shortened to 256 tokens. */
export const BOOK_CONFIG: GptConfig = { ...GPT_CONFIG_124M, contextLength: 256 };

const REPO_ROOT = new URL("../../../../", import.meta.url);

/** Edith Wharton's short story "The Verdict" (1908, public domain), as used in the book. */
export function loadVerdict(): string {
  return readFileSync(new URL("fixtures/the-verdict.txt", REPO_ROOT), "utf8");
}

export interface Loaders {
  train: DataLoader;
  val: DataLoader;
  trainTokens: number;
  valTokens: number;
}

/**
 * Splits the text 90/10 into training and validation parts, then cuts each into
 * non-overlapping `contextLength`-token windows. Only the training batches are shuffled.
 */
export function makeLoaders(text: string, contextLength: number, batchSize = 2): Loaders {
  const split = Math.floor(text.length * 0.9);
  const trainIds = encode(text.slice(0, split));
  const valIds = encode(text.slice(split));

  return {
    train: new DataLoader(new LlmDataset(trainIds, contextLength, contextLength), {
      batchSize,
      shuffle: true,
      dropLast: true,
    }),
    val: new DataLoader(new LlmDataset(valIds, contextLength, contextLength), {
      batchSize,
      shuffle: false,
      dropLast: false,
    }),
    trainTokens: trainIds.length,
    valTokens: valIds.length,
  };
}

/** "Every effort" -> int32 tensor of shape [1, numTokens]. */
export function textToIds(text: string): tf.Tensor2D {
  return tf.tensor2d([encode(text)], undefined, "int32");
}

/** First row of a [batch, numTokens] id tensor -> text. */
export function idsToText(ids: tf.Tensor2D): string {
  return decode(ids.arraySync()[0] as number[]);
}

/**
 * Saves every weight to `checkpoints/<name>/` (git-ignored): `config.json` with the model
 * configuration and each weight's shape, and `weights.bin` with the raw float32 values in
 * the same order as `model.weights`.
 */
export function saveCheckpoint(model: GptModel, name: string): string {
  const dir = new URL(`checkpoints/${name}/`, REPO_ROOT);
  mkdirSync(dir, { recursive: true });

  const shapes = model.weights.map((w) => w.shape);
  writeFileSync(new URL("config.json", dir), JSON.stringify({ config: model.config, shapes }));

  const values = new Float32Array(model.numParameters());
  let offset = 0;
  for (const w of model.weights) {
    values.set(w.dataSync() as Float32Array, offset);
    offset += w.size;
  }
  writeFileSync(new URL("weights.bin", dir), values);
  return dir.pathname;
}

/** Rebuilds a model saved by `saveCheckpoint`, or returns undefined if there is none. */
export function loadCheckpoint(name: string): GptModel | undefined {
  const dir = new URL(`checkpoints/${name}/`, REPO_ROOT);
  let saved: { config: GptConfig; shapes: number[][] };
  try {
    saved = JSON.parse(readFileSync(new URL("config.json", dir), "utf8"));
  } catch {
    return undefined;
  }

  const model = new GptModel(saved.config);
  const bytes = readFileSync(new URL("weights.bin", dir));
  const values = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);

  let offset = 0;
  for (const [i, w] of model.weights.entries()) {
    if (w.shape.join() !== saved.shapes[i]?.join()) {
      throw new Error(
        `checkpoint ${name}: weight ${i} has shape ${saved.shapes[i]}, expected ${w.shape}`,
      );
    }
    w.assign(tf.tensor(values.subarray(offset, offset + w.size), w.shape));
    offset += w.size;
  }
  return model;
}
