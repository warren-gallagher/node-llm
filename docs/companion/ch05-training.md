# Chapter 5 · Training (companion)

> **Unofficial companion chapter.** Written while the book's [Chapter 5](https://book.humanexperiences.io/chapters/05-training/) lessons were marked "Soon" (as of 2026-10-27). It follows the book's lesson outline and is not the author's text. See the [companion index](README.md).

Chapter 4 ended with a complete GPT whose weights are random numbers. Give it some text and it produces a score (a *logit*) for every token in the vocabulary at every position, but the scores mean nothing yet. **Training** adjusts the weights, a little at a time, so that the correct next token gets a higher score.

The three lessons follow the cycle:

1. **Measure** how wrong the model's predictions are: the *loss*.
2. **Improve** the weights to lower the loss, over and over: the *training loop*.
3. **Use** the trained model to write text: *generation*.

```mermaid
flowchart LR
    batch["Batch of token ids<br/>inputs + targets"] --> fwd["Forward pass<br/>(chapter 4)"]
    fwd --> logits["Logits"]
    logits --> loss["Loss<br/>(lesson 1)"]
    loss --> grads["Gradients<br/>(backpropagation)"]
    grads --> update["Update weights<br/>(AdamW)"]
    update -->|"next batch (lesson 2)"| batch
```

The training text is the one Raschka's book uses: Edith Wharton's short story *The Verdict* (1908, public domain), stored at [`fixtures/the-verdict.txt`](../../fixtures/the-verdict.txt). It's 20,479 characters, or 5,145 tokens: tiny by language-model standards, so the whole chapter runs on a laptop CPU.

---

## Lesson 1 · Loss & cross-entropy

Code: [`trainer.ts`](../../packages/core/src/train/trainer.ts) (`crossEntropyLoss`, `evaluateLoss`) · Run: `pnpm --filter @node-llm/chapters ch05:1`

### From logits to the probability of the right answer

The lesson starts with two short inputs and their targets. Each target is its input shifted one token to the left, as in Chapter 2's dataset:

| Input | Target (what should come next at each position) |
| --- | --- |
| `every effort moves` | ` effort moves you` |
| `I really like` | ` really like chocolate` |

That's 6 predictions in total: 2 sequences × 3 positions. For each one, the model outputs 50,257 logits. **Softmax** turns them into probabilities that are positive and add up to 1. We then look up the probability the model gave to the *correct* token. For the untrained model, the six numbers are tiny:

```
probability of each correct token: [ '4.82e-5', '3.72e-5', '9.06e-6', '5.66e-6', '2.29e-5', '1.08e-5' ]
```

A perfect model would give each correct token a probability of 1. Training should push these numbers up.

### Why take logs, and why negate?

We want one number that summarizes how good all the predictions are. Multiplying the probabilities would underflow to zero quickly (here it's already around 10⁻²⁹). Taking the **logarithm** turns the product into a sum, which stays well-behaved, and a higher log-probability still means a better prediction. Averaging makes the number independent of how many predictions there are. Finally, we **negate** it, because optimizers *minimize*:

```
loss = −mean( log p(correct token) )      over every position in the batch
```

This is **cross-entropy loss**. The lesson computes it by hand and then with the library function, and gets the same value:

```
their logs: [ '-9.941', '-10.198', '-11.611', '-12.083', '-10.682', '-11.439' ]
mean log probability: -10.9925
cross-entropy (negated mean): 10.9925
crossEntropyLoss(): 10.9925
```

### Reading the number

| Loss | Meaning |
| --- | --- |
| 0 | Every correct token got probability 1. A perfect, and unrealistic, score. |
| ln(50,257) ≈ 10.82 | What you'd get by guessing uniformly: equal probability for every token. |
| ≈ 10.98 | This untrained model: its random weights make some tokens look more likely than others for no reason, so it does slightly *worse* than a uniform guess. |

A related number is **perplexity**, which is `exp(loss)`. You can read it as "the model is as unsure as if it were choosing among this many tokens". The untrained model's perplexity is about 59,424, a little more than the vocabulary's 50,257 tokens. A model that has learned something has a far smaller perplexity.

Raschka's untrained model also starts at about 10.98. That match isn't automatic: it depends on starting the weights the way PyTorch does, which turned out to matter for training too (see [Differences](#differences-from-raschkas-pytorch-version)).

### Training and validation data

`makeLoaders` in [`shared.ts`](../../packages/chapters/src/ch05-training/shared.ts) splits the story 90/10 by characters. The first 90% is for training. The last 10% is held out for **validation**: text the model never trains on, used only to check how well it generalizes. Each part is cut into non-overlapping 256-token windows. Batches have 2 windows each, and only the training batches are shuffled.

```
training tokens: 4612, validation tokens: 534
training loss:   10.9804
validation loss: 10.9747
```

Raschka's book reports 4,608 and 512 tokens, because it counts only the tokens that end up inside complete windows. The split itself is the same.

### Implementation notes

- [`crossEntropyLoss`](../../packages/core/src/train/trainer.ts#L105-L123) flattens the logits from `[batch, seqLen, vocab]` to `[batch × seqLen, vocab]`, so every position is one independent prediction. It then calls `tf.losses.softmaxCrossEntropy`.
- PyTorch's `cross_entropy` takes the target ids directly. The tfjs function wants *one-hot* targets instead: a row of 50,257 zeros with a single 1 at the correct id. At the book's settings that's a temporary `[512, 50257]` array, about 100 MB per batch. It's wasteful but simple, and fine at this scale.
- [`evaluateLoss`](../../packages/core/src/train/trainer.ts#L129-L142) averages the loss over a loader's batches with dropout off. It's the equivalent of Raschka's `calc_loss_loader`.

---

## Lesson 2 · The training loop

Code: [`trainer.ts`](../../packages/core/src/train/trainer.ts) (`trainModel`) · Run: `pnpm --filter @node-llm/chapters ch05:2` (about 10 minutes on a laptop CPU)

### One step

For every batch, `trainModel` does the same four things:

1. **Forward pass with dropout on**: inputs → logits → loss (lesson 1).
2. **Backpropagation**: compute, for every one of the 163 million weights, how the loss would change if that weight were nudged. This is the weight's *gradient*.
3. **Optimizer update**: move each weight a small step in the direction that lowers the loss.
4. **Weight decay**: shrink every weight very slightly toward zero.

In PyTorch these are `optimizer.zero_grad()`, `loss.backward()` and `optimizer.step()`. In tfjs, one call to `optimizer.minimize(lossFn, true, model.weights)` does steps 1–3. It records the forward pass, computes the gradients and applies them ([`trainer.ts:61-65`](../../packages/core/src/train/trainer.ts#L61-L65)). There's nothing to reset between steps, because tfjs doesn't accumulate gradients the way PyTorch does.

### The optimizer: AdamW

Plain gradient descent moves every weight by `learningRate × gradient`. **Adam** improves on this by keeping, for each weight, a running average of its recent gradients and of their squares. Weights whose gradients are consistently large get smaller steps, and weights with small but steady gradients get larger ones. That makes training much less sensitive to the exact learning rate.

**AdamW** adds *decoupled weight decay*: after each step, every weight shrinks by a fixed fraction, here `learningRate × weightDecay = 0.0004 × 0.1 = 0.00004` of itself. This discourages very large weights, which tend to mean the model is memorizing instead of generalizing. tfjs has Adam but not AdamW, so `trainModel` applies the decay itself after each update ([`trainer.ts:68-70`](../../packages/core/src/train/trainer.ts#L68-L70)). That's exactly what "decoupled" means: the decay is applied separately from the gradient step.

### The book's settings

| Setting | Value |
| --- | --- |
| Model | GPT-2 small (163M parameters), context shortened to 256 tokens |
| Optimizer | AdamW, learning rate 0.0004, weight decay 0.1 |
| Batches | 2 sequences × 256 tokens; 9 batches per epoch |
| Epochs | 10, so 90 optimizer steps in total |
| Evaluation | Every 5 steps, averaging the first 5 batches of each loader |

### What happens

The run prints the training and validation loss every 5 steps. Across the 90 steps (the numbers vary a little from run to run, because the starting weights and the batch order are random):

```mermaid
xychart-beta
    title "Loss during training on The Verdict"
    x-axis "Optimizer step" [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]
    y-axis "Cross-entropy loss" 0 --> 9
    line [8.40, 6.91, 6.21, 5.80, 5.66, 5.25, 5.17, 4.31, 4.03, 3.41, 2.90, 2.34, 1.87, 1.65, 1.23, 1.04, 0.74, 0.53]
    line [8.60, 7.32, 6.68, 6.50, 6.43, 6.53, 6.47, 6.34, 6.23, 6.20, 6.28, 6.19, 6.21, 6.24, 6.29, 6.30, 6.38, 6.41]
```

The upper line, which flattens out, is the validation loss; the lower line, which keeps falling, is the training loss. The final values were 0.53 (training) and 6.41 (validation). Raschka's book reports about 0.39 and 6.45 for the same setup.

After each epoch, the lesson prints the model's greedy continuation of "Every effort moves you":

| After epoch | Sample (first part) |
| --- | --- |
| 1 | `Every effort moves you,,,,,,,,,,,,,,,,,,,,,,,,,, the,,,,,,` |
| 3 | `Every effort moves you` followed only by line breaks |
| 5 | `Every effort moves you of the picture to have the picture--and--I had been the last of the fact, I had been…` |
| 7 | `Every effort moves you know," she was a little wild--as--I told Mrs.` |
| 10 | `Every effort moves you?" "Yes--quite insensible to the irony. She wanted him vindicated--and by me!"` |

Two things happen at once:

- **The model learns the language of the story.** Its first guesses are the most common tokens: commas and line breaks. Then come common words ("the", "I had been"), then grammatical phrases, and finally whole sentences.
- **It also starts to memorize.** The epoch-10 sample is a line from the story, word for word. The training loss keeps falling toward 0, but the validation loss stops improving at about 6.2 around step 50, then creeps up. The model is getting better at the text it trains on and no better at text it hasn't seen. This is **overfitting**. It's expected here: 163 million parameters and only about 4,600 training tokens, each seen 10 times.

On this run the best validation loss came around epoch 6–7. A real training run would keep the weights from that point (see the exercises).

### What real pre-training does differently

- **Far more data, each piece seen about once.** GPT-2 was trained on about 40 GB of web text, and today's models on trillions of tokens. With that much text, the model can't memorize it, so the training and validation losses fall together.
- **Learning-rate schedules.** The learning rate starts near zero, ramps up over the first steps ("warmup"), then gradually decreases (often following a cosine curve). This helps early stability and later fine-grained progress.
- **Gradient clipping.** Unusually large gradients are scaled down so one bad batch can't throw the weights off course.

Raschka's book covers these in an appendix. The loop here is deliberately the minimal version.

### Saving the result

Lesson 2 ends by saving the trained weights to `checkpoints/verdict-124m/`. Git ignores this folder, since the files are large. `weights.bin` holds all 163,009,536 values as raw float32, about 622 MB, and `config.json` records the model configuration and each weight's shape ([`saveCheckpoint`](../../packages/chapters/src/ch05-training/shared.ts)). Lesson 3 loads it back. This plays the role of PyTorch's `torch.save(model.state_dict())` / `load_state_dict`.

---

## Lesson 3 · Text generation

Code: [`gpt.ts`](../../packages/core/src/model/gpt.ts) (`generate`) · Run: `pnpm --filter @node-llm/chapters ch05:3` (after lesson 2)

### The generation loop

A GPT writes one token at a time ([`gpt.ts:122-145`](../../packages/core/src/model/gpt.ts#L122-L145)):

```mermaid
flowchart LR
    ids["Token ids so far"] --> crop["Keep the last<br/>contextLength tokens"]
    crop --> fwd["Forward pass"]
    fwd --> last["Logits at the<br/>last position"]
    last --> pick["Choose one token<br/>(greedy, or sample)"]
    pick --> append["Append it"]
    append -->|"repeat maxNewTokens times"| ids
```

Only the last position's logits matter, because they predict the token *after* everything so far. The model still computes logits for every earlier position, then throws them away. Real systems avoid that repeated work with a *key-value cache*. The crop step means the model can keep generating past its context length, but it forgets anything that has scrolled out of view.

How to **choose** the token is a separate decision from the model. `generate` supports three strategies through its options.

### Greedy decoding

The default (`temperature: 0`) always picks the highest-scoring token. It's deterministic: the same prompt always gives the same text. For a model trained on one short story, greedy decoding tends to reproduce passages of the story word for word, and it easily falls into loops.

### Temperature

To get variety, the model *samples* from the probabilities instead of always taking the top one. **Temperature** controls how adventurous the sampling is. The logits are divided by the temperature before softmax ([`gpt.ts:169-172`](../../packages/core/src/model/gpt.ts#L169-L172)). The first part of the lesson shows the effect on a made-up five-word vocabulary for the next word after "The cat sat on the":

| Word | T = 0.5 | T = 1 | T = 2 |
| --- | --- | --- | --- |
| mat | 0.816 | 0.611 | 0.432 |
| sofa | 0.165 | 0.275 | 0.290 |
| roof | 0.018 | 0.091 | 0.167 |
| moon | 0.001 | 0.018 | 0.075 |
| banana | 0.000 | 0.004 | 0.035 |

A temperature below 1 sharpens the distribution toward the favorite, and as it approaches 0, sampling becomes greedy. A temperature above 1 flattens it, so unlikely words like "banana" get picked more often. The lesson also draws 1,000 samples at each temperature, and the counts match these probabilities closely.

### Top-k sampling

High temperatures make the long tail of bad choices more likely. **Top-k** cuts the tail off: only the *k* highest-scoring tokens can be chosen. Every other logit is set to −∞, which softmax turns into probability 0 ([`gpt.ts:163-167`](../../packages/core/src/model/gpt.ts#L163-L167)). With *k* = 2 on the same example, only "mat" (0.690) and "sofa" (0.310) remain.

Temperature and top-k combine well: top-k removes nonsense, and temperature controls variety among what's left.

### On the trained model

After lesson 2 has saved its checkpoint, the second part of the lesson continues "Every effort moves you" for 25 tokens. Your text will differ, because training is random, but its character should be similar:

```
greedy:                    "Every effort moves you?"  "Yes--quite insensible to the irony. She wanted him vindicated--and by me!"
temp 1.4, top-25 (seed 1): "Every effort moves you?"  " he felt? I felt nervous and. She wanted him not existed he turned, and drawn the sketch
temp 1.4, top-25 (seed 2): "Every effort moves you say," she was was followed by interesting to see his tone on fellow up all not when he wouldn, that one might be
temp 1.4, top-25 (seed 3): "Every effort moves you know," was one of their "It was the exquisite Herm good-piece, on a picture--had lent herself in his
```

(Line breaks in the output are shown as spaces here.)

Greedy decoding repeats the story exactly. Sampling with temperature 1.4 and top-25 produces new combinations, built from the story's words and rhythms, that never appear in it. They're often ungrammatical, because the model has only seen one short story. The seed makes each sample reproducible: the same seed on the same weights gives the same text.

### Stopping early

`generate` also accepts `eosId`. When every sequence in the batch produces that token, generation stops, without appending it. For GPT-2's vocabulary the natural choice is 50256, `<|endoftext|>`: the separator placed between documents during training (see the [reader notes on training data](../reader-notes.md#how-would-i-train-on-a-whole-book-or-web-articles-and-what-metadata-matters)). A model trained that way learns to emit it when a text is finished.

---

## Differences from Raschka's PyTorch version

| Topic | Raschka (PyTorch) | This repo |
| --- | --- | --- |
| Optimizer | `torch.optim.AdamW` | `tf.train.adam` plus a manual decoupled weight-decay step |
| Loss | `cross_entropy(logits, target_ids)` | `softmaxCrossEntropy(oneHot(targets), logits)` |
| One training step | `zero_grad()`, `backward()`, `step()` | `optimizer.minimize(lossFn, true, model.weights)` |
| Sampling | `torch.multinomial` with a global `torch.manual_seed` | `tf.multinomial` with a per-call `seed` option |
| Saving weights | `torch.save(model.state_dict(), …)` | `saveCheckpoint`: raw float32 file + JSON shapes |
| Initialization | PyTorch defaults | The same for linear weights (uniform in ±1/√dIn) and embeddings (standard normal); biases start at 0 instead of uniform |

Starting the weights the PyTorch way matters more than it might seem. An earlier version of this repo started linear weights about 1.7 times wider (normal, standard deviation 1/√dIn). On the same lesson 2 run, that gave an untrained loss of 11.38 instead of 10.98, and a training loss of 3.50 instead of about 0.5 after 10 epochs. The model spent several epochs stuck predicting mostly "the" and commas. The shared helper is [`linearWeight`](../../packages/core/src/model/attention/utils.ts).

Raschka's chapter also loads OpenAI's pretrained GPT-2 weights into the model. The book's lesson outline for this chapter doesn't include that, so it isn't here yet. Chapter 4's design was chosen so those weights will fit.

---

## Exercises

1. **Change the sampling.** In `03-generation.ts`, try temperatures of 0.5 and 3, and `topK` of 1, 5 and 200. Which settings stay readable? Which ones copy the story?
2. **Stop at the best point.** Find the step with the lowest validation loss in lesson 2's output and change `epochs` so training stops there. Compare the samples. This is called *early stopping*.
3. **More data.** Add a second public-domain story (for example from Project Gutenberg) to the training text, with `<|endoftext|>` (id 50256) between the two, and watch how the gap between training and validation loss changes.
4. **Perplexity.** Print `Math.exp(evaluateLoss(model, val))` for the trained model in lesson 3 and compare it with the untrained value from lesson 1.
