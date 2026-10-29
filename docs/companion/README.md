# Companion chapters (unofficial)

The book [*Building an LLM in NodeJS*](https://book.humanexperiences.io) publishes chapters as they're written. On 2026-10-26, its Chapter 4 and Chapter 5 lessons were marked "Soon", and the matching code in this repo was a set of `TODO` stubs.

These companion chapters fill that gap until the official ones appear. Each one implements the stubs in place, so the code can be compared with the author's version, or dropped, once theirs is published. Each one also explains what the code does.

From Chapter 6 on, the repo has no stubs to fill. Those chapters add new code beside the existing code and follow the sections of Raschka's matching chapter.

**These are not the book.** They are written independently, in their own words. They follow the lesson outline the book's site lists for each chapter, and the material it is ported from: Sebastian Raschka's [*Build a Large Language Model (From Scratch)*](https://github.com/rasbt/LLMs-from-scratch). Where the official chapter differs, the official chapter wins.

| Chapter | Lessons (as listed on the book's site) | Code | Status |
| --- | --- | --- | --- |
| [4 · GPT Model](ch04-gpt.md) | LayerNorm & GELU · Transformer block · Full GPT model | [`packages/core/src/model`](../../packages/core/src/model) · [`ch04-gpt`](../../packages/chapters/src/ch04-gpt) | Reviewed |
| [5 · Training](ch05-training.md) | Loss & cross-entropy · Training loop · Text generation · *Extra: loading GPT-2's weights* | [`packages/core/src/train`](../../packages/core/src/train) · [`ch05-training`](../../packages/chapters/src/ch05-training) | Written, awaiting review |
| [6 · Fine-tuning for classification](ch06-classification.md) | *No stubs in this repo; follows Raschka's Chapter 6:* Dataset · Padding & batches · Pretrained GPT-2 · New head & freezing · Loss & accuracy · Fine-tuning · Using the classifier | [`classifier.ts`](../../packages/core/src/model/classifier.ts) · [`classifierTrainer.ts`](../../packages/core/src/train/classifierTrainer.ts) · [`ch06-classification`](../../packages/chapters/src/ch06-classification) | Written, awaiting review |
| [7 · Fine-tuning to follow instructions](ch07-instruction.md) | *No stubs in this repo; follows Raschka's Chapter 7:* Dataset · Batching · Data loaders · Pretrained GPT-2 · Fine-tuning · Generating responses · Scoring | [`instruction.ts`](../../packages/core/src/data/instruction.ts) · [`trainer.ts`](../../packages/core/src/train/trainer.ts) · [`ch07-instruction`](../../packages/chapters/src/ch07-instruction) | Written, awaiting review |

Each chapter's code is runnable on its own:

```bash
pnpm --filter @node-llm/chapters ch04:1   # LayerNorm & GELU
pnpm --filter @node-llm/chapters ch04:2   # shortcut connections + transformer block
pnpm --filter @node-llm/chapters ch04:3   # the full 124M-parameter GPT
pnpm --filter @node-llm/chapters ch05:1   # cross-entropy loss
pnpm --filter @node-llm/chapters ch05:2   # train GPT-2 small on The Verdict (~10 min)
pnpm --filter @node-llm/chapters ch05:3   # temperature, top-k, and generating from the trained model
pnpm --filter @node-llm/chapters ch05:gpt2 # extra: download and run OpenAI's GPT-2 small (~550 MB, once)
pnpm --filter @node-llm/chapters ch06:1   # the SMS spam dataset (~200 KB, once)
pnpm --filter @node-llm/chapters ch06:4   # GPT-2 with a 2-class head
pnpm --filter @node-llm/chapters ch06:6   # fine-tune the spam classifier (~15 min on a laptop CPU)
pnpm --filter @node-llm/chapters ch06:7   # classify new messages
pnpm --filter @node-llm/chapters ch07:1   # the instruction dataset (~200 KB, once)
pnpm --filter @node-llm/chapters ch07:2   # padding, shifting and masking a batch
pnpm --filter @node-llm/chapters ch07:5   # instruction fine-tune GPT-2 124M (~25 min; add --355m for the book's size)
pnpm --filter @node-llm/chapters ch07:6   # answer the 110 test instructions
pnpm --filter @node-llm/chapters ch07:7   # score the answers with Llama 3 8B in Ollama
```
