<div align="center">

# node-llm

**Build a Large Language Model from scratch, in Node.js and TypeScript.**

No PyTorch. No black boxes. Just tensors, attention, and a GPT you assemble yourself, one layer at a time.

[![Node.js](https://img.shields.io/badge/Node.js-%E2%89%A520-339933?logo=node.js&logoColor=white)](https://nodejs.org)
[![TypeScript](https://img.shields.io/badge/TypeScript-strict-3178C6?logo=typescript&logoColor=white)](https://www.typescriptlang.org)
[![TensorFlow.js](https://img.shields.io/badge/TensorFlow.js-engine-FF6F00?logo=tensorflow&logoColor=white)](https://github.com/tensorflow/tfjs)
[![pnpm](https://img.shields.io/badge/pnpm-workspace-F69220?logo=pnpm&logoColor=white)](https://pnpm.io)

[Read the book](https://book.humanexperiences.io) · [Run the chapters](#running-the-chapters) · [How it is laid out](#project-layout)

</div>

---

## Why this exists

Most "learn how LLMs work" material is written in Python and PyTorch. This project ports the whole journey to the language millions of developers already use every day: **JavaScript / TypeScript**.

It is the code behind my book, [**Building an LLM in NodeJS**](https://book.humanexperiences.io). Every piece is built from first principles in TypeScript, and every PyTorch idea is mapped to its JS equivalent. The result is a codebase you can read top to bottom and actually understand, backed by a book that explains every step with diagrams and worked examples.

- **No black boxes.** Tokenizer, dataset, attention, transformer block, GPT, and the training loop are all built by hand.
- **PyTorch, translated.** `.backward()` / `optim.step()` become `tf.variableGrads` / `optimizer.applyGradients`, and every lesson spells out the mapping.
- **Runnable, chapter by chapter.** Each lesson has a single entry point you can run and inspect.
- **Numerically verified.** Core ops are checked against PyTorch value dumps so the JS port behaves like the original.

> The tensor engine is [`@tensorflow/tfjs-node`](https://github.com/tensorflow/tfjs): production-grade and native-accelerated, so the same model code runs under Node (chapters) and WebGPU (in-browser demos).

## Companion book

The full written walkthrough lives at **[book.humanexperiences.io](https://book.humanexperiences.io)**: diagrams, worked examples, and a PyTorch to TypeScript mapping for every concept. The code in this repo is what the book builds toward.

## Project layout

This is a pnpm workspace with three packages:

```
packages/
  core/         the library: tokenizer, data, model, training
  chapters/     one runnable entry per book lesson (tfjs-node backend)
  playground/   interactive in-browser demos (WebGPU)
```

Inside `core`:

```
src/
  tokenizer/    hand-rolled BPE tokenizer (validated against gpt-tokenizer)
  data/         Dataset (sliding window) + DataLoader (batching / shuffling)
  model/
    attention/  simplified -> trainable self-attention -> multi-head
    block.ts    transformer block
    gpt.ts      the full GPT model
  train/         reusable training loop
```

## Requirements

- **Node.js >= 20** (developed on 22)
- **pnpm**

## Setup

```bash
pnpm install
pnpm test        # unit + numeric-parity tests
pnpm typecheck   # type-check every package
```

## Running the chapters

Each lesson is independently runnable. The granular scripts live in the `@node-llm/chapters` package:

```bash
# Chapter 2 — Tokenizer & data
pnpm --filter @node-llm/chapters ch02:1   # word tokenizer
pnpm --filter @node-llm/chapters ch02:2   # byte-pair encoding
pnpm --filter @node-llm/chapters ch02:4   # dataset + dataloader
pnpm --filter @node-llm/chapters ch02:6   # positional encoding

# Chapter 3 — Attention
pnpm --filter @node-llm/chapters ch03:1   # simplified attention (no weights)
pnpm --filter @node-llm/chapters ch03:2   # self-attention with trainable Q/K/V
pnpm --filter @node-llm/chapters ch03:3   # multi-head attention
```

Or run a whole chapter from the repo root:

```bash
pnpm ch02        # tokenizer
pnpm ch03        # attention
pnpm ch04        # full GPT forward pass
pnpm ch05:train  # training loop
pnpm playground  # launch the WebGPU demos
```

## Roadmap

| Chapter | Topic | Status |
| --- | --- | --- |
| 2 | Tokenizer, dataset, embeddings, positional encoding | Done |
| 3 | Attention: simplified, trainable, multi-head | Done |
| 4 | LayerNorm, GELU, transformer block, full GPT | In progress |
| 5 | Loss, training loop, text generation | In progress |

## Working with tfjs

A couple of things that differ from PyTorch and are worth knowing up front:

- **Manual memory.** tfjs does not garbage-collect tensors. Wrap tensor ops in `tf.tidy(() => ...)` and `dispose()` anything you keep.
- **Autograd.** `tf.variableGrads(fn)` returns `{ value, grads }`; apply them with `optimizer.applyGradients(grads)`. That pair is the tfjs stand-in for `loss.backward()` + `optimizer.step()`.

## The book

This repo is the companion code to [**Building an LLM in NodeJS**](https://book.humanexperiences.io), where each concept is explained step by step with diagrams, worked examples, and a PyTorch to TypeScript mapping. Start there if you want the full walkthrough.
