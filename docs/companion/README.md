# Companion chapters (unofficial)

The book [*Building an LLM in NodeJS*](https://book.humanexperiences.io) publishes chapters as they're written. On 2026-10-26, its Chapter 4 and Chapter 5 lessons were marked "Soon", and the matching code in this repo was a set of `TODO` stubs.

These companion chapters fill that gap until the official ones appear. Each one implements the stubs in place, so the code can be compared with the author's version, or dropped, once theirs is published. Each one also explains what the code does.

**These are not the book.** They are written independently, in their own words. They follow the lesson outline the book's site lists for each chapter, and the material it is ported from: Sebastian Raschka's [*Build a Large Language Model (From Scratch)*](https://github.com/rasbt/LLMs-from-scratch). Where the official chapter differs, the official chapter wins.

| Chapter | Lessons (as listed on the book's site) | Code | Status |
| --- | --- | --- | --- |
| [4 · GPT Model](ch04-gpt.md) | LayerNorm & GELU · Transformer block · Full GPT model | [`packages/core/src/model`](../../packages/core/src/model) · [`ch04-gpt`](../../packages/chapters/src/ch04-gpt) | Written, awaiting review |
| 5 · Training | Loss & cross-entropy · Training loop · Text generation | | Not started |

Each chapter's code is runnable on its own:

```bash
pnpm --filter @node-llm/chapters ch04:1   # LayerNorm & GELU
pnpm --filter @node-llm/chapters ch04:2   # shortcut connections + transformer block
pnpm --filter @node-llm/chapters ch04:3   # the full 124M-parameter GPT
```
