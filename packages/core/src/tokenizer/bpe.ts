/**
 * Chapter 2: byte-pair-encoding tokenizer.
 *
 * Book analog: chapter 2 builds a simple tokenizer, then swaps in GPT-2's BPE (tiktoken).
 * Here we will hand-roll BPE to learn it, and use `gpt-tokenizer` in tests as a reference
 * oracle to verify our encode/decode matches GPT-2 exactly.
 *
 * TODO(ch02): implement train() (learn merges from a corpus), encode(), decode().
 */
export interface Tokenizer {
  encode(text: string): number[];
  decode(ids: number[]): string;
  readonly vocabSize: number;
}

export class BpeTokenizer implements Tokenizer {
  get vocabSize(): number {
    throw new Error("TODO(ch02): implement BPE vocab");
  }

  encode(_text: string): number[] {
    throw new Error("TODO(ch02): implement BPE encode");
  }

  decode(_ids: number[]): string {
    throw new Error("TODO(ch02): implement BPE decode");
  }
}
