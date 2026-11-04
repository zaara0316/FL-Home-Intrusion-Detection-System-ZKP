const fs = require("fs");
const { buildPoseidon } = require("circomlibjs");

const INPUT_PATH = "inputs/inference_input.json";
const SCALE = 10000n;

const FIELD_MODULUS =
  21888242871839275222246405745257275088548364400416034343698204186575808495617n;

const DOMAIN = 20261104n;

function encodeSigned(magnitude, sign) {
  const mag = BigInt(magnitude);

  if (sign === 0) {
    return mag;
  }

  return (FIELD_MODULUS - mag) % FIELD_MODULUS;
}

async function main() {
  const input = JSON.parse(fs.readFileSync(INPUT_PATH, "utf8"));

  if (input.weightMag.length !== 115 || input.weightSign.length !== 115) {
    throw new Error("Expected exactly 115 model weights.");
  }

  const poseidon = await buildPoseidon();
  const F = poseidon.F;

  const encodedWeights = input.weightMag.map((mag, i) =>
    encodeSigned(mag, Number(input.weightSign[i])),
  );

  const encodedBias = encodeSigned(input.biasMag, Number(input.biasSign));

  let hash = poseidon([DOMAIN, encodedWeights[0]]);

  for (let i = 1; i < 115; i++) {
    hash = poseidon([hash, encodedWeights[i]]);
  }

  hash = poseidon([hash, encodedBias]);

  const commitment = F.toString(hash);

  input.modelCommitment = commitment;

  fs.writeFileSync(INPUT_PATH, JSON.stringify(input, null, 2));

  console.log("========================================");
  console.log("FEDNATD MODEL COMMITMENT");
  console.log("========================================");

  console.log("Number of committed weights:", encodedWeights.length);
  console.log("Bias included: Yes");
  console.log("Poseidon commitment:", commitment);

  console.log("\nUpdated:", INPUT_PATH);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
