import {
  balance,
  ensureSpamData,
  LABELS,
  loadSpamSplits,
  readMessages,
  SPLIT_SEED,
  seededRandom,
} from "./shared";

/**
 * Chapter 6, lesson 1: the dataset.
 *
 * The SMS Spam Collection: 5,574 text messages, each labelled "ham" (a normal message) or
 * "spam". There are far more ham messages than spam, so the book keeps every spam message
 * and an equal number of randomly chosen ham messages, then splits the result 70/10/20 into
 * training, validation and test sets.
 */
async function main(): Promise<void> {
  const messages = readMessages(await ensureSpamData());
  console.log(`${messages.length} messages`);
  printCounts(messages);

  const balanced = balance(messages, seededRandom(SPLIT_SEED));
  console.log(`\nbalanced: ${balanced.length} messages`);
  printCounts(balanced);

  const { train, validation, test } = await loadSpamSplits();
  console.log(
    `\nsplit: ${train.length} training, ${validation.length} validation, ${test.length} test`,
  );

  console.log("\nan example of each label from the training set:");
  for (const label of LABELS.keys()) {
    const m = train.find((m) => m.label === label);
    console.log(`  ${(LABELS[label] as string).padEnd(8)}  ${JSON.stringify(m?.text)}`);
  }
}

function printCounts(messages: { label: number }[]): void {
  for (const [label, name] of LABELS.entries()) {
    console.log(`  ${name.padEnd(8)}  ${messages.filter((m) => m.label === label).length}`);
  }
}

main();
