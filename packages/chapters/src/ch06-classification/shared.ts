import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { inflateRawSync } from "node:zlib";
import {
  ClassificationDataset,
  ClassificationLoader,
  GptClassifier,
  type GptConfig,
} from "@node-llm/core";
import * as tf from "@tensorflow/tfjs-node";
import { encode } from "gpt-tokenizer/encoding/r50k_base";
import { ensureGpt2Weights, loadGpt2 } from "../ch05-training/gpt2";

/**
 * Helpers shared by the chapter 6 lessons: the SMS spam dataset, its loaders, the GPT-2
 * classifier, and saving / loading the fine-tuned weights.
 */

const REPO_ROOT = new URL("../../../../", import.meta.url);
const DATA_DIR = new URL("data/sms-spam/", REPO_ROOT); // git-ignored
const TSV = new URL("SMSSpamCollection.tsv", DATA_DIR);

/** Where the book downloads the dataset from, with its backup copy. */
const URLS = [
  "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip",
  "https://f001.backblazeb2.com/file/LLMs-from-scratch/sms%2Bspam%2Bcollection.zip",
];

/**
 * SHA-256 of the extracted `SMSSpamCollection` file, recorded when this chapter was written
 * (2026-10). UCI publishes no checksum, so this only detects a change since then.
 */
const TSV_SHA256 = "7d039a24a6083ed9ef0f806ebad56bbb976e3aeb8de05669173bfdc4996c239d";

/** The book's labels: 0 = "ham" (not spam), 1 = "spam". */
export const LABELS = ["not spam", "spam"] as const;

export interface Message {
  label: number;
  text: string;
}

/** Downloads (once, about 200 KB) and verifies the SMS Spam Collection; returns its path. */
export async function ensureSpamData(): Promise<string> {
  if (existsSync(TSV)) return TSV.pathname;

  for (const url of URLS) {
    const response = await fetch(url).catch(() => undefined);
    if (!response?.ok) {
      console.log(`download failed from ${url}, trying the next source`);
      continue;
    }
    const tsv = unzipEntry(Buffer.from(await response.arrayBuffer()), "SMSSpamCollection");
    const actual = createHash("sha256").update(tsv).digest("hex");
    if (actual !== TSV_SHA256) {
      throw new Error(`SHA-256 mismatch: expected ${TSV_SHA256}, got ${actual}`);
    }
    mkdirSync(DATA_DIR, { recursive: true });
    writeFileSync(TSV, tsv);
    console.log(`downloaded and verified ${TSV.pathname}`);
    return TSV.pathname;
  }
  throw new Error("could not download the SMS Spam Collection");
}

/**
 * Extracts one file from a zip archive. A zip ends with a central directory listing each
 * file's name, size, compression method and where its data starts.
 */
function unzipEntry(zip: Buffer, name: string): Buffer {
  const end = zip.lastIndexOf(Buffer.from([0x50, 0x4b, 0x05, 0x06])); // end-of-directory record
  let entry = zip.readUInt32LE(end + 16); // offset of the central directory

  for (let i = 0; i < zip.readUInt16LE(end + 10); i++) {
    const method = zip.readUInt16LE(entry + 10);
    const compressedSize = zip.readUInt32LE(entry + 20);
    const nameLength = zip.readUInt16LE(entry + 28);
    const extraLength = zip.readUInt16LE(entry + 30);
    const commentLength = zip.readUInt16LE(entry + 32);
    const localHeader = zip.readUInt32LE(entry + 42);
    const entryName = zip.toString("utf8", entry + 46, entry + 46 + nameLength);

    if (entryName === name) {
      const dataStart =
        localHeader + 30 + zip.readUInt16LE(localHeader + 26) + zip.readUInt16LE(localHeader + 28);
      const data = zip.subarray(dataStart, dataStart + compressedSize);
      return method === 8 ? inflateRawSync(data) : Buffer.from(data); // 8 = deflate, 0 = stored
    }
    entry += 46 + nameLength + extraLength + commentLength;
  }
  throw new Error(`${name} not found in the zip archive`);
}

/** Each line is a label ("ham" or "spam"), a tab, then the message. */
export function readMessages(path: string): Message[] {
  return readFileSync(path, "utf8")
    .split("\n")
    .filter((line) => line.includes("\t"))
    .map((line) => {
      const tab = line.indexOf("\t");
      return { label: line.slice(0, tab) === "spam" ? 1 : 0, text: line.slice(tab + 1).trimEnd() };
    });
}

/** A seeded random number generator (mulberry32), so every lesson gets the same split. */
export function seededRandom(seed: number): () => number {
  let state = seed >>> 0;
  return function next(): number {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function shuffled<T>(items: readonly T[], random: () => number): T[] {
  const out = [...items];
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [out[i], out[j]] = [out[j] as T, out[i] as T];
  }
  return out;
}

/** Keeps every spam message and the same number of randomly chosen ham messages. */
export function balance(messages: readonly Message[], random: () => number): Message[] {
  const spam = messages.filter((m) => m.label === 1);
  const ham = shuffled(
    messages.filter((m) => m.label === 0),
    random,
  ).slice(0, spam.length);
  return [...ham, ...spam];
}

/** Shuffles, then splits into training, validation and test parts (the rest). */
export function randomSplit(
  messages: readonly Message[],
  trainFraction: number,
  validationFraction: number,
  random: () => number,
): { train: Message[]; validation: Message[]; test: Message[] } {
  const all = shuffled(messages, random);
  const trainEnd = Math.floor(all.length * trainFraction);
  const validationEnd = trainEnd + Math.floor(all.length * validationFraction);
  return {
    train: all.slice(0, trainEnd),
    validation: all.slice(trainEnd, validationEnd),
    test: all.slice(validationEnd),
  };
}

/**
 * Seed for the balancing and splitting shuffles. The book uses pandas with seed 123, whose
 * shuffle can't be reproduced here, so the messages in each split differ from the book's.
 * Seed 12 is one whose longest training message is 120 tokens, the same as the book's, so
 * every batch has the book's shape.
 */
export const SPLIT_SEED = 12;

/** The book's split: balanced, then 70% training, 10% validation, 20% test. */
export async function loadSpamSplits(seed = SPLIT_SEED) {
  const random = seededRandom(seed);
  const messages = readMessages(await ensureSpamData());
  return randomSplit(balance(messages, random), 0.7, 0.1, random);
}

export interface SpamLoaders {
  train: ClassificationLoader;
  validation: ClassificationLoader;
  test: ClassificationLoader;
  /** Every text is padded or cut to this many tokens: the longest training message. */
  maxLength: number;
}

/** Token ids for each message, padded to the longest training message, in batches of 8. */
export async function makeSpamLoaders(batchSize = 8): Promise<SpamLoaders> {
  const splits = await loadSpamSplits();

  function dataset(messages: Message[], maxLength?: number): ClassificationDataset {
    return new ClassificationDataset(
      messages.map((m) => encode(m.text)),
      messages.map((m) => m.label),
      maxLength,
    );
  }
  const train = dataset(splits.train);
  const validation = dataset(splits.validation, train.maxLength);
  const test = dataset(splits.test, train.maxLength);

  return {
    train: new ClassificationLoader(train, { batchSize, shuffle: true, dropLast: true }),
    validation: new ClassificationLoader(validation, { batchSize }),
    test: new ClassificationLoader(test, { batchSize }),
    maxLength: train.maxLength,
  };
}

/** GPT-2 small with a new 2-class head, dropout off as in the book's chapter 6. */
export async function loadGpt2Classifier(
  overrides: Partial<GptConfig> = {},
): Promise<GptClassifier> {
  const base = loadGpt2(await ensureGpt2Weights(), { dropout: 0, ...overrides });
  return new GptClassifier(base, LABELS.length);
}

/** Classifies one message: cut or padded to `maxLength` tokens, as in training. */
export function classify(model: GptClassifier, text: string, maxLength: number): string {
  const ids = encode(text).slice(0, maxLength);
  const padded = [...ids, ...new Array<number>(maxLength - ids.length).fill(50256)];
  const predicted = tf.tidy(() =>
    model.predict(tf.tensor2d([padded], undefined, "int32")).dataSync(),
  );
  return LABELS[predicted[0] as number] as string;
}

/**
 * Saves only the weights fine-tuning changed (the last block, the final LayerNorm and the
 * head, about 7 million values) to `checkpoints/<name>/` (git-ignored). The other 117
 * million are GPT-2's own and are reloaded from its file, which `config.json` identifies by
 * SHA-256.
 */
export function saveClassifier(model: GptClassifier, name: string, maxLength: number): string {
  const dir = new URL(`checkpoints/${name}/`, REPO_ROOT);
  mkdirSync(dir, { recursive: true });

  const weights = model.trainableWeights;
  const meta = {
    base: "openai-community/gpt2 model.safetensors",
    baseSha256: sha256File(new URL("checkpoints/gpt2-124m/model.safetensors", REPO_ROOT)),
    maxLength,
    shapes: weights.map((w) => w.shape),
  };
  writeFileSync(new URL("config.json", dir), JSON.stringify(meta, null, 2));

  const values = new Float32Array(weights.reduce((sum, w) => sum + w.size, 0));
  let offset = 0;
  for (const w of weights) {
    values.set(w.dataSync() as Float32Array, offset);
    offset += w.size;
  }
  writeFileSync(new URL("trainable.bin", dir), values);
  return dir.pathname;
}

/** Rebuilds a classifier saved by `saveClassifier`, or returns undefined if there is none. */
export async function loadClassifier(
  name: string,
): Promise<{ model: GptClassifier; maxLength: number } | undefined> {
  const dir = new URL(`checkpoints/${name}/`, REPO_ROOT);
  if (!existsSync(new URL("config.json", dir))) return undefined;
  const meta = JSON.parse(readFileSync(new URL("config.json", dir), "utf8")) as {
    baseSha256: string;
    maxLength: number;
    shapes: number[][];
  };

  const model = await loadGpt2Classifier();
  const base = sha256File(new URL("checkpoints/gpt2-124m/model.safetensors", REPO_ROOT));
  if (base !== meta.baseSha256) {
    throw new Error(`checkpoint ${name} was fine-tuned from a different GPT-2 file`);
  }

  const bytes = readFileSync(new URL("trainable.bin", dir));
  const values = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
  let offset = 0;
  for (const [i, w] of model.trainableWeights.entries()) {
    if (w.shape.join() !== meta.shapes[i]?.join()) {
      throw new Error(`checkpoint ${name}: weight ${i} has shape ${meta.shapes[i]}`);
    }
    w.assign(tf.tensor(values.subarray(offset, offset + w.size), w.shape));
    offset += w.size;
  }
  return { model, maxLength: meta.maxLength };
}

function sha256File(url: URL): string {
  return createHash("sha256").update(readFileSync(url)).digest("hex");
}
