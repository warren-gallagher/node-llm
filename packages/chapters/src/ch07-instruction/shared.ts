import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import {
  formatInput,
  formatResponse,
  type GptModel,
  InstructionDataset,
  type InstructionEntry,
  InstructionLoader,
} from "@node-llm/core";
import * as tf from "@tensorflow/tfjs-node";
import { decode, encode } from "gpt-tokenizer/encoding/r50k_base";
import { ensureGpt2Weights, type Gpt2Size, loadGpt2 } from "../ch05-training/gpt2";
import { loadCheckpoint, saveCheckpoint } from "../ch05-training/shared";

/**
 * Helpers shared by the chapter 7 lessons: the instruction dataset and its loaders, choosing
 * the GPT-2 size, generating a response, and the fine-tuned checkpoint.
 */

const REPO_ROOT = new URL("../../../../", import.meta.url);
export const DATA_DIR = new URL("data/instruction/", REPO_ROOT); // git-ignored
const DATA_FILE = new URL("instruction-data.json", DATA_DIR);

/** The 1,100 examples the book's author prepared for this chapter. */
const DATA_URL =
  "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json";

/**
 * SHA-256 of the file, recorded when this chapter was written (2026-10). GitHub publishes no
 * checksum, so this only detects a change since then.
 */
const DATA_SHA256 = "7e09a9af353722ca15c18bc580f093f9c39d3178335a75177319cb1e1d163944";

/** `<|endoftext|>`: marks the end of a response, and pads batches. */
export const END_OF_TEXT = 50256;

/** Downloads (once, about 200 KB) and verifies the instruction dataset, then parses it. */
export async function loadInstructionData(): Promise<InstructionEntry[]> {
  if (!existsSync(DATA_FILE)) {
    const response = await fetch(DATA_URL);
    if (!response.ok) throw new Error(`download failed: HTTP ${response.status}`);
    const bytes = Buffer.from(await response.arrayBuffer());
    const actual = createHash("sha256").update(bytes).digest("hex");
    if (actual !== DATA_SHA256) {
      throw new Error(`SHA-256 mismatch: expected ${DATA_SHA256}, got ${actual}`);
    }
    mkdirSync(DATA_DIR, { recursive: true });
    writeFileSync(DATA_FILE, bytes);
    console.log(`downloaded and verified ${DATA_FILE.pathname}`);
  }
  return JSON.parse(readFileSync(DATA_FILE, "utf8")) as InstructionEntry[];
}

/**
 * The book's split, in file order without shuffling: the first 85% for training, the next
 * 10% for testing, and the remaining 5% for validation. So every split holds the same
 * examples as the book's.
 */
export async function loadInstructionSplits() {
  const data = await loadInstructionData();
  const trainEnd = Math.floor(data.length * 0.85);
  const testEnd = trainEnd + Math.floor(data.length * 0.1);
  return {
    train: data.slice(0, trainEnd),
    test: data.slice(trainEnd, testEnd),
    validation: data.slice(testEnd),
  };
}

/** The full training text of one example: prompt, then response. */
export function encodeExample(entry: InstructionEntry): number[] {
  return encode(formatInput(entry) + formatResponse(entry));
}

/** Loaders in batches of 8, each batch padded to its longest example, as in the book. */
export async function makeInstructionLoaders(batchSize = 8) {
  const splits = await loadInstructionSplits();

  function loader(entries: InstructionEntry[], train: boolean): InstructionLoader {
    return new InstructionLoader(new InstructionDataset(entries.map(encodeExample)), {
      batchSize,
      shuffle: train,
      dropLast: train,
      allowedMaxLength: 1024,
    });
  }
  return {
    ...splits,
    trainLoader: loader(splits.train, true),
    validationLoader: loader(splits.validation, false),
    testLoader: loader(splits.test, false),
  };
}

/**
 * The GPT-2 size to use. GPT-2 small (124M) by default, so every lesson finishes in minutes;
 * pass `--355m` for GPT-2 medium, the size the book uses, which follows instructions much
 * better but takes hours to fine-tune on a CPU.
 */
export function chosenSize(): Gpt2Size {
  return process.argv.includes("--355m") ? "355m" : "124m";
}

/** Pretrained GPT-2 of the chosen size, with dropout off as in the book's chapter 7. */
export async function loadPretrained(size: Gpt2Size): Promise<GptModel> {
  return loadGpt2(await ensureGpt2Weights(size), { dropout: 0 });
}

/** Folder name, under `checkpoints/`, of the fine-tuned model for a size. */
export function checkpointName(size: Gpt2Size): string {
  return `gpt2-${size}-instruct`;
}

export function saveFineTuned(model: GptModel, size: Gpt2Size): string {
  return saveCheckpoint(model, checkpointName(size));
}

/** The model fine-tuned by lesson 5, or undefined if lesson 5 hasn't been run for this size. */
export function loadFineTuned(size: Gpt2Size): GptModel | undefined {
  return loadCheckpoint(checkpointName(size));
}

/**
 * Generates a response to one example's prompt, greedily, stopping at `<|endoftext|>`.
 *
 * The model continues the prompt, so its output starts with the `### Response:` heading it
 * learned to write. Only the text after the prompt is kept, without that heading. Book analog:
 * `generate(...)` followed by `generated_text[len(input_text):].replace("### Response:", "")`.
 */
export function respond(model: GptModel, entry: InstructionEntry, maxNewTokens = 256): string {
  const prompt = encode(formatInput(entry));
  const ids = tf.tidy(() => {
    const input = tf.tensor2d([prompt], undefined, "int32");
    return model.generate(input, maxNewTokens, { eosId: END_OF_TEXT }).arraySync()[0] as number[];
  });
  return decode(ids.slice(prompt.length)).replace("### Response:", "").trim();
}
