import sys
from pathlib import Path


# ==========================================
# PROJECT ROOT
# ==========================================

project_root = Path(__file__).resolve().parents[2]

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


# ==========================================
# IMPORTS
# ==========================================

from micrograd.engine import Value
from micrograd.nn import MLP
from micrograd.metrics import classification_metrics

from optimizers.sgd import SGD
from optimizers.momentum import Momentum
from optimizers.adam import Adam

from datasets.breast_cancer.preprocess import prepare_data
from datasets.batching import create_batches


# ==========================================
# BINARY CROSS ENTROPY
# ==========================================

def binary_cross_entropy(
    prediction,
    target
):
    """
    Binary Cross Entropy loss.

    Small epsilon is used to prevent
    log(0).
    """

    eps = 1e-7

    prediction = prediction + eps

    prediction = prediction / (
        1 + 2 * eps
    )

    return -(
        target * prediction.log()
        + (1 - target)
        * (1 - prediction).log()
    )


# ==========================================
# TRAINING FUNCTION
# ==========================================

def train(
    optimizer_name,
    epochs=50,
    batch_size=1
):
    """
    Train the MLP.

    Parameters
    ----------
    optimizer_name : str
        "sgd", "momentum", or "adam"

    epochs : int
        Number of training epochs.

    batch_size : int or None
        1    -> sample-by-sample training
        >1   -> mini-batch training
        None -> full-batch training

    Returns
    -------
    loss_history : list
        Average training loss per epoch.

    metrics : dict
        Accuracy, precision, recall,
        F1 score, and confusion matrix.
    """

    # ======================================
    # LOAD DATA
    # ======================================

    X_train, X_test, y_train, y_test = (
        prepare_data()
    )


    # ======================================
    # CREATE MODEL
    # ======================================

    model = MLP(
        30,
        [16, 8, 1],
        seed=42
    )


    # ======================================
    # CREATE OPTIMIZER
    # ======================================

    if optimizer_name == "sgd":

        optimizer = SGD(
            model.parameters(),
            lr=0.01
        )

    elif optimizer_name == "momentum":

        optimizer = Momentum(
            model.parameters(),
            lr=0.01,
            momentum=0.9
        )

    elif optimizer_name == "adam":

        optimizer = Adam(
            model.parameters(),
            lr=0.001
        )

    else:

        raise ValueError(
            "optimizer_name must be "
            "'sgd', 'momentum', or 'adam'"
        )


    # ======================================
    # LOSS HISTORY
    # ======================================

    loss_history = []


    # ======================================
    # TRAINING LOOP
    # ======================================

    for epoch in range(epochs):

        total_loss = 0.0
        total_samples = 0


        # ----------------------------------
        # CREATE BATCHES
        # ----------------------------------

        batches = create_batches(
            X_train,
            y_train,
            batch_size=batch_size,
            shuffle=True,
            seed=42 + epoch
        )


        # ----------------------------------
        # PROCESS EACH BATCH
        # ----------------------------------

        for X_batch, y_batch in batches:

            # ==============================
            # RESET GRADIENTS
            # ==============================

            optimizer.zero_grad()


            # ==============================
            # BATCH LOSS
            # ==============================

            batch_loss = None


            # ==============================
            # FORWARD PASS
            # ==============================

            for x, target in zip(
                X_batch,
                y_batch
            ):

                # Convert numpy features
                # into micrograd Values

                inputs = [
                    Value(float(feature))
                    for feature in x
                ]


                # Forward pass

                logits = model(inputs)


                # Convert logits to probability

                prediction = logits.sigmoid()


                # Calculate sample loss

                loss = binary_cross_entropy(
                    prediction,
                    float(target)
                )


                # Add sample loss to batch loss

                if batch_loss is None:

                    batch_loss = loss

                else:

                    batch_loss = (
                        batch_loss + loss
                    )


            # ==============================
            # AVERAGE BATCH LOSS
            # ==============================

            batch_loss = (
                batch_loss / len(X_batch)
            )


            # ==============================
            # RECORD LOSS
            # ==============================

            total_loss += (
                batch_loss.data
                * len(X_batch)
            )

            total_samples += len(X_batch)


            # ==============================
            # BACKPROPAGATION
            # ==============================

            batch_loss.backward()


            # ==============================
            # UPDATE PARAMETERS
            # ==============================

            optimizer.step()


        # ==================================
        # EPOCH LOSS
        # ==================================

        average_loss = (
            total_loss / total_samples
        )

        loss_history.append(
            average_loss
        )


        # ==================================
        # PRINT PROGRESS
        # ==================================

        if epoch % 10 == 0:

            print(
                f"{optimizer_name.upper()} "
                f"| Epoch {epoch:3d} "
                f"| Loss: {average_loss:.4f}"
            )


    # ======================================
    # EVALUATION
    # ======================================

    y_true = []
    y_pred = []


    # --------------------------------------
    # PREDICT TEST SET
    # --------------------------------------

    for x, target in zip(
        X_test,
        y_test
    ):

        inputs = [
            Value(float(feature))
            for feature in x
        ]


        # Forward pass

        logits = model(inputs)


        # Convert logit to probability

        probability = (
            logits.sigmoid().data
        )


        # Convert probability to class

        prediction = (
            1
            if probability >= 0.5
            else 0
        )


        y_true.append(
            int(target)
        )

        y_pred.append(
            prediction
        )


    # ======================================
    # CALCULATE METRICS
    # ======================================

    metrics = classification_metrics(
        y_true,
        y_pred
    )


    # ======================================
    # PRINT METRICS
    # ======================================

    print(
        f"{optimizer_name.upper()} "
        f"| Accuracy:  {metrics['accuracy']:.4f}"
    )

    print(
        f"{optimizer_name.upper()} "
        f"| Precision: {metrics['precision']:.4f}"
    )

    print(
        f"{optimizer_name.upper()} "
        f"| Recall:    {metrics['recall']:.4f}"
    )

    print(
        f"{optimizer_name.upper()} "
        f"| F1 Score:  {metrics['f1']:.4f}"
    )

    print(
        f"{optimizer_name.upper()} "
        f"| Confusion Matrix:"
    )

    print(
        metrics["confusion_matrix"]
    )


    # ======================================
    # RETURN RESULTS
    # ======================================

    return (
        loss_history,
        metrics
    )


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    # ======================================
    # TRAINING CONFIGURATION
    # ======================================

    EPOCHS = 50

    # --------------------------------------
    # Choose your training mode here
    #
    # 1     -> sample-by-sample
    # 32    -> mini-batch
    # 64    -> larger mini-batch
    # None  -> full-batch
    # --------------------------------------

    BATCH_SIZE = 32


    # ======================================
    # SGD
    # ======================================

    print("\n===== SGD =====")

    sgd_loss, sgd_metrics = train(
        "sgd",
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )


    # ======================================
    # MOMENTUM
    # ======================================

    print("\n===== MOMENTUM =====")

    momentum_loss, momentum_metrics = train(
        "momentum",
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )


    # ======================================
    # ADAM
    # ======================================

    print("\n===== ADAM =====")

    adam_loss, adam_metrics = train(
        "adam",
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )


    # ======================================
    # FINAL RESULTS
    # ======================================

    print("\n===== FINAL RESULTS =====")


    # --------------------------------------
    # SGD
    # --------------------------------------

    print("\nSGD")

    print(
        f"  Accuracy :  "
        f"{sgd_metrics['accuracy']:.4f}"
    )

    print(
        f"  Precision:  "
        f"{sgd_metrics['precision']:.4f}"
    )

    print(
        f"  Recall:     "
        f"{sgd_metrics['recall']:.4f}"
    )

    print(
        f"  F1 Score:   "
        f"{sgd_metrics['f1']:.4f}"
    )

    print(
        "  Confusion Matrix:"
    )

    print(
        sgd_metrics["confusion_matrix"]
    )


    # --------------------------------------
    # MOMENTUM
    # --------------------------------------

    print("\nMomentum")

    print(
        f"  Accuracy :  "
        f"{momentum_metrics['accuracy']:.4f}"
    )

    print(
        f"  Precision:  "
        f"{momentum_metrics['precision']:.4f}"
    )

    print(
        f"  Recall:     "
        f"{momentum_metrics['recall']:.4f}"
    )

    print(
        f"  F1 Score:   "
        f"{momentum_metrics['f1']:.4f}"
    )

    print(
        "  Confusion Matrix:"
    )

    print(
        momentum_metrics["confusion_matrix"]
    )


    # --------------------------------------
    # ADAM
    # --------------------------------------

    print("\nAdam")

    print(
        f"  Accuracy :  "
        f"{adam_metrics['accuracy']:.4f}"
    )

    print(
        f"  Precision:  "
        f"{adam_metrics['precision']:.4f}"
    )

    print(
        f"  Recall:     "
        f"{adam_metrics['recall']:.4f}"
    )

    print(
        f"  F1 Score:   "
        f"{adam_metrics['f1']:.4f}"
    )

    print(
        "  Confusion Matrix:"
    )

    print(
        adam_metrics["confusion_matrix"]
    )


    # ======================================
    # PLOT LOSS
    # ======================================

    import matplotlib.pyplot as plt


    epochs = range(
        1,
        len(sgd_loss) + 1
    )


    plt.figure(
        figsize=(8, 5)
    )


    plt.plot(
        epochs,
        sgd_loss,
        label="SGD"
    )

    plt.plot(
        epochs,
        momentum_loss,
        label="Momentum"
    )

    plt.plot(
        epochs,
        adam_loss,
        label="Adam"
    )


    plt.xlabel("Epoch")

    plt.ylabel("Training Loss")


    plt.title(
        "Optimizer Comparison "
        f"(Batch Size = {BATCH_SIZE})"
    )


    plt.legend()

    plt.grid(True)

    plt.tight_layout()


    output_path = (
        Path(__file__).resolve().parent
        / "loss_comparison.png"
    )


    plt.savefig(
        output_path,
        dpi=150
    )


    print(
        "\nLoss plot saved to:"
    )

    print(output_path)


    plt.close()