import subprocess
import time
import csv
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

REPETITIONS = 5

INPUTS = {
    "attack": os.path.join(ROOT, "inputs", "valid_input_backup.json"),
    "benign": os.path.join(ROOT, "inputs", "inference_input.json")
}

RESULTS_DIR = os.path.join(ROOT, "results")
BUILD_DIR = os.path.join(ROOT, "build")

os.makedirs(RESULTS_DIR, exist_ok=True)


def execute(command):
    """Execute a command and return its execution time in seconds."""

    if command[0] == "npx":
        command[0] = "npx.cmd"

    start = time.perf_counter()

    result = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        shell=command[0].lower().endswith(".cmd")
    )

    elapsed = time.perf_counter() - start

    if result.returncode != 0:
        print("\nCOMMAND FAILED:")
        print(" ".join(command))
        print("\nSTDOUT:")
        print(result.stdout)
        print("\nSTDERR:")
        print(result.stderr)
        raise RuntimeError("Benchmark command failed.")

    return elapsed


rows = []

for sample_type, input_path in INPUTS.items():

    print("\n" + "=" * 65)
    print(f"BENCHMARKING: {sample_type.upper()}")
    print("=" * 65)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input not found: {input_path}")

    witness_times = []
    proving_times = []
    verification_times = []

    proof_path = os.path.join(BUILD_DIR, f"{sample_type}_benchmark_proof.json")
    public_path = os.path.join(BUILD_DIR, f"{sample_type}_benchmark_public.json")
    witness_path = os.path.join(BUILD_DIR, f"{sample_type}_benchmark_witness.wtns")

    for run in range(REPETITIONS):

        print(f"\nRun {run + 1}/{REPETITIONS}")

        # 1. Witness generation
        witness_time = execute([
            "node",
            os.path.join(BUILD_DIR, "inference_js", "generate_witness.js"),
            os.path.join(BUILD_DIR, "inference_js", "inference.wasm"),
            input_path,
            witness_path
        ])

        witness_times.append(witness_time)

        # 2. Witness correctness check
        execute([
            "npx", "snarkjs", "wtns", "check",
            os.path.join(BUILD_DIR, "inference.r1cs"),
            witness_path
        ])

        # 3. Proof generation
        prove_time = execute([
            "npx", "snarkjs", "plonk", "prove",
            os.path.join(BUILD_DIR, "inference_final.zkey"),
            witness_path,
            proof_path,
            public_path
        ])

        proving_times.append(prove_time)

        # 4. Proof verification
        verify_time = execute([
            "npx", "snarkjs", "plonk", "verify",
            os.path.join(BUILD_DIR, "verification_key.json"),
            public_path,
            proof_path
        ])

        verification_times.append(verify_time)

        print(f"Witness generation: {witness_time:.4f} seconds")
        print(f"Proof generation:   {prove_time:.4f} seconds")
        print(f"Verification:       {verify_time:.4f} seconds")

    proof_size = os.path.getsize(proof_path)
    witness_size = os.path.getsize(witness_path)

    row = {
        "Sample": sample_type,
        "Runs": REPETITIONS,
        "Avg Witness Generation (s)": sum(witness_times) / REPETITIONS,
        "Avg Proof Generation (s)": sum(proving_times) / REPETITIONS,
        "Avg Verification (s)": sum(verification_times) / REPETITIONS,
        "Proof Size (bytes)": proof_size,
        "Witness Size (bytes)": witness_size
    }

    rows.append(row)

    print("\nAVERAGES")
    print(f"Witness generation: {row['Avg Witness Generation (s)']:.4f} seconds")
    print(f"Proof generation:   {row['Avg Proof Generation (s)']:.4f} seconds")
    print(f"Verification:       {row['Avg Verification (s)']:.4f} seconds")
    print(f"Proof size:         {proof_size:,} bytes")
    print(f"Witness size:       {witness_size:,} bytes")


# Save benchmark results
csv_path = os.path.join(RESULTS_DIR, "zkp_benchmark_results.csv")

with open(csv_path, "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("\n" + "=" * 65)
print("BENCHMARK COMPLETED")
print("=" * 65)
print(f"Results saved to: {csv_path}")