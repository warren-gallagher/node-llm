import { existsSync, readFileSync } from "node:fs";
import { formatInput, type InstructionEntry } from "@node-llm/core";
import { chosenSize, DATA_DIR } from "./shared";

/**
 * Chapter 7, lesson 7: score the test responses with a larger model.
 *
 * Free-form answers can't be checked by exact match, so, as in the book, a larger
 * instruction-tuned model acts as the judge: Llama 3 8B, run locally by Ollama, gives each
 * response a score from 0 to 100 against the reference answer.
 *
 * Needs Ollama (https://ollama.com) running, with the model downloaded once (about 4.7 GB):
 *   ollama pull llama3
 */
const OLLAMA = "http://localhost:11434";
const JUDGE = "llama3";

interface Scored extends InstructionEntry {
  model_response: string;
}

async function main(): Promise<void> {
  const size = chosenSize();
  const file = new URL(`test-responses-${size}.json`, DATA_DIR);
  if (!existsSync(file)) {
    const flag = size === "355m" ? " --355m" : "";
    console.log(`no test responses for GPT-2 ${size} yet: run lesson 6 first (pnpm ch07:6${flag})`);
    return;
  }
  if (!(await ollamaHasModel())) return;
  const entries = JSON.parse(readFileSync(file, "utf8")) as Scored[];

  // First, three scores with the judge's reasoning.
  for (const entry of entries.slice(0, 3)) {
    console.log(`\ndataset response:\n>> ${entry.output}`);
    console.log(`\nmodel response:\n>> ${entry.model_response}`);
    console.log(`\nscore:\n>> ${await query(scorePrompt(entry))}`);
    console.log("-".repeat(40));
  }

  // Then every response, asking for the number only.
  const scores: number[] = [];
  for (const entry of entries) {
    const reply = await query(`${scorePrompt(entry)}Respond with the integer number only.`);
    const score = Number.parseInt(reply, 10);
    if (Number.isNaN(score)) console.log(`could not convert score: ${reply}`);
    else scores.push(score);
  }
  const average = scores.reduce((sum, s) => sum + s, 0) / scores.length;
  console.log(`\nnumber of scores: ${scores.length} of ${entries.length}`);
  console.log(`average score: ${average.toFixed(2)}`);
}

/** The book's judging prompt. */
function scorePrompt(entry: Scored): string {
  return (
    `Given the input \`${formatInput(entry)}\` ` +
    `and correct output \`${entry.output}\`, ` +
    `score the model response \`${entry.model_response}\`` +
    " on a scale from 0 to 100, where 100 is the best score. "
  );
}

/** One chat turn with the judge. Seed and temperature 0 make its replies repeatable. */
async function query(prompt: string): Promise<string> {
  const response = await fetch(`${OLLAMA}/api/chat`, {
    method: "POST",
    body: JSON.stringify({
      model: JUDGE,
      messages: [{ role: "user", content: prompt }],
      stream: false,
      options: { seed: 123, temperature: 0, num_ctx: 2048 },
    }),
  });
  if (!response.ok) throw new Error(`Ollama: HTTP ${response.status}`);
  const body = (await response.json()) as { message: { content: string } };
  return body.message.content.trim();
}

/** Checks that Ollama is running and has the judge model, and says what to do if not. */
async function ollamaHasModel(): Promise<boolean> {
  const response = await fetch(`${OLLAMA}/api/tags`).catch(() => undefined);
  if (!response?.ok) {
    console.log(`Ollama isn't running at ${OLLAMA}. Install it from https://ollama.com,`);
    console.log(`start it, run "ollama pull ${JUDGE}", then run this lesson again.`);
    return false;
  }
  const { models } = (await response.json()) as { models: { name: string }[] };
  if (!models.some((m) => m.name.split(":")[0] === JUDGE)) {
    console.log(`Ollama is running but doesn't have ${JUDGE}: run "ollama pull ${JUDGE}".`);
    return false;
  }
  return true;
}

main();
