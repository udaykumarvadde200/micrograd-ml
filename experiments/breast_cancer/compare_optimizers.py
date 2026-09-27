from pathlib import Path
import sys
import time

# --------------------------------------------------
# Project path
# --------------------------------------------------

project_root = Path(__file__).resolve().parents[2]

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


# --------------------------------------------------
# Import training function
# --------------------------------------------------

from train import train


# --------------------------------------------------
# Experiment configuration
# --------------------------------------------------

EPOCHS = 50

OPTIMIZERS = [
    "sgd",
    "momentum",
    "adam",
]


# --------------------------------------------------
# Run experiments
# --------------------------------------------------

results = {}

print("=" * 60)
print("OPTIMIZER COMPARISON")
print("=" * 60)

for optimizer_name in OPTIMIZERS:

    print(
        f"\n{'=' * 20} "
        f"{optimizer_name.upper()} "
        f"{'=' * 20}"
    )

    start_time = time.perf_counter()

    loss_history, accuracy = train(
        optimizer_name,
        epochs=EPOCHS
    )

    elapsed_time = time.perf_counter() - start_time

    results[optimizer_name] = {
        "loss": loss_history,
        "accuracy": accuracy,
        "time": elapsed_time,
    }


# --------------------------------------------------
# Final comparison
# --------------------------------------------------

print("\n")
print("=" * 60)
print("FINAL COMPARISON")
print("=" * 60)

print(
    f"{'Optimizer':<15}"
    f"{'Final Loss':<15}"
    f"{'Accuracy':<15}"
    f"{'Time (s)':<15}"
)

print("-" * 60)

for optimizer_name in OPTIMIZERS:

    result = results[optimizer_name]

    final_loss = result["loss"][-1]
    accuracy = result["accuracy"]
    elapsed_time = result["time"]

    print(
        f"{optimizer_name.capitalize():<15}"
        f"{final_loss:<15.6f}"
        f"{accuracy:<15.4f}"
        f"{elapsed_time:<15.2f}"
    )


# --------------------------------------------------
# Save comparison plot
# --------------------------------------------------

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))

for optimizer_name in OPTIMIZERS:

    plt.plot(
        range(1, EPOCHS + 1),
        results[optimizer_name]["loss"],
        label=optimizer_name.capitalize()
    )

plt.xlabel("Epoch")
plt.ylabel("Training Loss")

plt.title(
    "Optimizer Comparison on Breast Cancer Dataset"
)

plt.legend()
plt.grid(True)

plt.tight_layout()

output_path = (
    Path(__file__).resolve().parent
    / "optimizer_comparison.png"
)

plt.savefig(
    output_path,
    dpi=150
)

print("\nComparison plot saved to:")
print(output_path)

plt.close()