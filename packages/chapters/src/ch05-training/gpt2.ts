import { createHash } from "node:crypto";
import { createWriteStream, existsSync, mkdirSync, readFileSync, renameSync } from "node:fs";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";
import { GPT_CONFIG_124M, type GptConfig, GptModel } from "@node-llm/core";
import * as tf from "@tensorflow/tfjs-node";

/**
 * Downloading OpenAI's pretrained GPT-2 small weights and loading them into `GptModel`.
 *
 * The weights come from Hugging Face's copy of OpenAI's release, in the safetensors format:
 * an 8-byte header length, a JSON header naming each tensor's dtype, shape and byte range,
 * then the raw values. The download is checked against the SHA-256 Hugging Face publishes.
 */

/** GPT-2 small: the book's 124M configuration, plus the Q/K/V biases GPT-2 was trained with. */
export const GPT2_CONFIG: GptConfig = { ...GPT_CONFIG_124M, qkvBias: true };

const REPO = "openai-community/gpt2";
const FILE = "model.safetensors";
const DIR = new URL("../../../../checkpoints/gpt2-124m/", import.meta.url);
const PATH = new URL(FILE, DIR);

interface SafeTensor {
  shape: number[];
  data: Float32Array;
}

/** Downloads the weights once (about 550 MB) and verifies them; later calls reuse the file. */
export async function ensureGpt2Weights(): Promise<string> {
  if (existsSync(PATH)) return PATH.pathname;

  const expected = await publishedSha256();
  mkdirSync(DIR, { recursive: true });
  const partial = new URL(`${FILE}.partial`, DIR);

  console.log(`downloading ${REPO}/${FILE} (about 550 MB)...`);
  const response = await fetch(`https://huggingface.co/${REPO}/resolve/main/${FILE}`);
  if (!response.ok || !response.body) {
    throw new Error(`download failed: HTTP ${response.status}`);
  }
  const hash = createHash("sha256");
  const body = Readable.fromWeb(response.body as import("node:stream/web").ReadableStream);
  body.on("data", (chunk: Buffer) => hash.update(chunk));
  await pipeline(body, createWriteStream(partial));

  const actual = hash.digest("hex");
  if (actual !== expected) {
    throw new Error(`SHA-256 mismatch: expected ${expected}, got ${actual}`);
  }
  renameSync(partial, PATH);
  console.log(`verified SHA-256 ${actual}`);
  return PATH.pathname;
}

/** The SHA-256 Hugging Face lists for the file (its Git LFS object id). */
async function publishedSha256(): Promise<string> {
  const response = await fetch(`https://huggingface.co/api/models/${REPO}/tree/main`);
  const files = (await response.json()) as { path: string; lfs?: { oid: string } }[];
  const oid = files.find((f) => f.path === FILE)?.lfs?.oid;
  if (!oid) throw new Error(`no published SHA-256 for ${FILE}`);
  return oid;
}

/** Parses a safetensors file holding float32 tensors. */
export function readSafetensors(path: string): Map<string, SafeTensor> {
  const bytes = readFileSync(path);
  const headerLength = Number(bytes.readBigUInt64LE(0));
  const header = JSON.parse(bytes.subarray(8, 8 + headerLength).toString("utf8")) as Record<
    string,
    { dtype: string; shape: number[]; data_offsets: [number, number] }
  >;

  const dataStart = 8 + headerLength;
  const tensors = new Map<string, SafeTensor>();
  for (const [name, info] of Object.entries(header)) {
    if (name === "__metadata__") continue;
    if (info.dtype !== "F32") throw new Error(`${name}: unsupported dtype ${info.dtype}`);
    const [start, end] = info.data_offsets;
    // Copy into an aligned buffer: a Float32Array view needs a 4-byte-aligned offset.
    const raw = bytes.subarray(dataStart + start, dataStart + end);
    const data = new Float32Array(raw.byteLength / 4);
    new Uint8Array(data.buffer).set(raw);
    tensors.set(name, { shape: info.shape, data });
  }
  return tensors;
}

/**
 * Builds a `GptModel` with GPT-2's weights.
 *
 * GPT-2's names map onto this repo's layers as follows (per block `h.{i}`):
 *   ln_1 / ln_2        -> the block's two LayerNorms (weight = scale, bias = shift)
 *   attn.c_attn        -> query, key and value, stored side by side: split into thirds
 *   attn.c_proj        -> attention output projection
 *   mlp.c_fc / c_proj  -> feed-forward expand / contract
 * plus wte (token embedding), wpe (position embedding) and ln_f (final LayerNorm). GPT-2 has
 * no separate output head: it reuses wte, transposed ("weight tying").
 *
 * GPT-2 stores its linear weights as [dIn, dOut], the same layout as this repo, so nothing
 * needs transposing except the tied output head.
 *
 * `overrides` changes settings that don't affect the weights' shapes, such as `dropout`.
 */
export function loadGpt2(path: string, overrides: Partial<GptConfig> = {}): GptModel {
  const tensors = readSafetensors(path);
  const used = new Set<string>();

  function get(name: string): SafeTensor {
    const t = tensors.get(name);
    if (!t) throw new Error(`missing tensor ${name}`);
    used.add(name);
    return t;
  }
  function tensor(name: string): tf.Tensor {
    const t = get(name);
    return tf.tensor(t.data, t.shape);
  }

  const config = { ...GPT2_CONFIG, ...overrides };
  const model = new GptModel(config);
  const values: tf.Tensor[] = [tensor("wte.weight"), tensor("wpe.weight")];

  for (let i = 0; i < config.numLayers; i++) {
    const h = `h.${i}`;
    const [wQuery, wKey, wValue] = tf.split(tensor(`${h}.attn.c_attn.weight`), 3, 1);
    const [bQuery, bKey, bValue] = tf.split(tensor(`${h}.attn.c_attn.bias`), 3, 0);
    // Same order as TransformerBlock.weights: norm1, attention, norm2, feed-forward.
    values.push(
      tensor(`${h}.ln_1.weight`),
      tensor(`${h}.ln_1.bias`),
      wQuery as tf.Tensor,
      wKey as tf.Tensor,
      wValue as tf.Tensor,
      tensor(`${h}.attn.c_proj.weight`),
      tensor(`${h}.attn.c_proj.bias`),
      bQuery as tf.Tensor,
      bKey as tf.Tensor,
      bValue as tf.Tensor,
      tensor(`${h}.ln_2.weight`),
      tensor(`${h}.ln_2.bias`),
      tensor(`${h}.mlp.c_fc.weight`),
      tensor(`${h}.mlp.c_fc.bias`),
      tensor(`${h}.mlp.c_proj.weight`),
      tensor(`${h}.mlp.c_proj.bias`),
    );
  }
  values.push(tensor("ln_f.weight"), tensor("ln_f.bias"));
  values.push(tf.transpose(values[0] as tf.Tensor)); // tied output head: wte transposed

  if (values.length !== model.weights.length) {
    throw new Error(`expected ${model.weights.length} weight tensors, built ${values.length}`);
  }
  for (const [i, w] of model.weights.entries()) {
    const v = values[i] as tf.Tensor;
    if (v.shape.join() !== w.shape.join()) {
      throw new Error(`weight ${i}: GPT-2 shape [${v.shape}] does not fit [${w.shape}]`);
    }
    w.assign(v);
  }
  tf.dispose(values);

  // Anything left over is a tensor this mapping doesn't know about. GPT-2 files may also
  // carry each block's causal-mask buffer (attn.bias / attn.masked_bias), which isn't a weight.
  const unused = [...tensors.keys()].filter(
    (n) => !used.has(n) && !/\.attn\.(masked_)?bias$/.test(n),
  );
  if (unused.length > 0) throw new Error(`unused GPT-2 tensors: ${unused.join(", ")}`);
  return model;
}
