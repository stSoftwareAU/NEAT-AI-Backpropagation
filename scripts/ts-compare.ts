/**
 * Dual-run harness: NEAT-AI TypeScript/WASM accumulate + propose, same JSON
 * schema as `neat_ai_backpropagation compare`.
 *
 * Run from this repo with NEAT-AI's import map:
 *
 *   deno run -A --config ../NEAT-AI/deno.json scripts/ts-compare.ts \
 *     <creature.json> <data-dir> --max-records N --out ts-compare.json
 */
import { Creature } from "@creature";
import { createBackPropagationConfig } from "@propagate/BackPropagation.ts";
import { SparseConfig } from "@propagate/sparse/SparseConfig.ts";
import { calculateBias } from "@propagate/Bias.ts";
import { calculateWeight } from "@propagate/Weight.ts";
import { MSE } from "@costs/MSE.ts";
import { initWasmActivation } from "@wasm/WasmModuleLoader.ts";
import { findBinFiles, readRecords } from "./ts-bin.ts";

const args = parseArgs(Deno.args);
await initWasmActivation();

const creatureJson = JSON.parse(await Deno.readTextFile(args.creature));
const creature = Creature.fromJSON(creatureJson, false);
const config = createBackPropagationConfig({
  generations: 1,
  learningRate: 0.01,
  initialLearningRate: 0.01,
  learningRateStrategy: "fixed",
  maximumBiasAdjustmentScale: 10,
  maximumWeightAdjustmentScale: 10,
  limitBiasScale: 10_000,
  limitWeightScale: 100_000,
  plankConstant: 1e-7,
  disableRandomSamples: true,
  sparseRatio: 1,
  normaliseGradients: false,
  trainingMutationRate: 0.01,
  batchSize: 1_000_000_000,
  disableBiasAdjustment: false,
  disableWeightAdjustment: false,
});
const sparse = new SparseConfig(creature.exportJSON(), config);
const cost = new MSE();

const files = findBinFiles(args.trainingData);
const recordWidth = creature.input + creature.output;
let records = 0;
let mseSum = 0;
const limit = args.maxRecords ?? Number.POSITIVE_INFINITY;

outer: for (const file of files) {
  for (const rec of readRecords(file, recordWidth)) {
    if (records >= limit) break outer;
    const input = rec.subarray(0, creature.input);
    const target = rec.subarray(creature.input);
    const output = creature.activateAndTrace(input, false, sparse);
    mseSum += cost.calculate(target, output);
    creature.propagate(target, config, sparse);
    records++;
  }
}

if (records === 0) {
  throw new Error("no training records read");
}

const neurons = [];
for (const n of creature.neurons) {
  if (n.type === "input") continue;
  const ns = creature.state.node(n.index);
  neurons.push({
    uuid: n.uuid,
    biasCount: ns.count,
    totalAdjustedBias: ns.totalAdjustedBias,
    currentBias: n.bias,
    proposedBias: calculateBias(n, config),
  });
}

const synapses = [];
for (const s of creature.synapses) {
  const from = creature.neurons[s.from];
  const to = creature.neurons[s.to];
  const fromUuid = from.type === "input"
    ? `input-${from.index}`
    : from.uuid as string;
  const toUuid = to.uuid as string;
  const cs = creature.state.connection(s.from, s.to);
  synapses.push({
    fromUUID: fromUuid,
    toUUID: toUuid,
    count: cs.count,
    currentWeight: s.weight,
    proposedWeight: calculateWeight(cs, s, config),
  });
}

const dump = {
  version: "ts",
  records,
  mse: mseSum / records,
  learningRate: config.learningRate,
  neurons,
  synapses,
};

await Deno.writeTextFile(args.out, JSON.stringify(dump, null, 2) + "\n");
console.error(
  `ts-compare: records=${records} mse=${dump.mse} wrote ${args.out}`,
);

interface Args {
  creature: string;
  trainingData: string;
  maxRecords?: number;
  out: string;
}

function parseArgs(argv: string[]): Args {
  const positional: string[] = [];
  let maxRecords: number | undefined;
  let out = "ts-compare.json";
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--max-records") {
      maxRecords = Number(argv[++i]);
    } else if (a === "--out") {
      out = argv[++i];
    } else if (!a.startsWith("-")) {
      positional.push(a);
    } else {
      throw new Error(`unknown flag: ${a}`);
    }
  }
  if (positional.length < 2) {
    throw new Error(
      "usage: ts-compare.ts <creature.json> <data-dir> [--max-records N] [--out path]",
    );
  }
  return {
    creature: positional[0],
    trainingData: positional[1],
    maxRecords,
    out,
  };
}
