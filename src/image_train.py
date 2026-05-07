import matplotlib.pyplot as plt

import torch

from src.training_history import TrainingHistory
from src.image_generators import ImageGeneratorBaseModel

def image_train(
    model:'ImageGeneratorBaseModel',
    train_loader,
    val_loader=None,
    epochs=10,
    verbose=True,
    history_image_path:str = None,
) -> 'TrainingHistory':
    """
    Universal image generative training.

    Support:
    - VAE
    - GAN

    Return:
    - TrainingHistory
    """

    history = TrainingHistory()

    model.train()

    for epoch in range(epochs):

        train_metrics = {}
        total_train_batches = 0

        for batch in train_loader:

            if isinstance(batch, (list, tuple)):
                batch = batch[0]

            batch = batch.to(model.device)

            metrics = model.train_step(batch)

            for key, value in metrics.items():
                train_metrics[key] = (
                    train_metrics.get(key, 0.0)
                    + float(value)
                )

            total_train_batches += 1

        for key in train_metrics:
            train_metrics[key] /= total_train_batches

        epoch_metrics = {}

        for key, value in train_metrics.items():
            epoch_metrics[f"train_{key}"] = value

        if val_loader is not None:

            model.eval()

            val_metrics = {}
            total_val_batches = 0

            with torch.no_grad():

                for batch in val_loader:

                    if isinstance(batch, (list, tuple)):
                        batch = batch[0]

                    batch = batch.to(model.device)

                    if hasattr(model, "compute_loss"):

                        x_hat, mu, logvar = model(batch)

                        losses = model.compute_loss(
                            batch,
                            x_hat,
                            mu,
                            logvar,
                        )

                        metrics = {
                            key: value.item()
                            for key, value in losses.items()
                        }

                    else:

                        metrics = model.train_step(batch)

                    for key, value in metrics.items():
                        val_metrics[key] = (
                            val_metrics.get(key, 0.0)
                            + float(value)
                        )

                    total_val_batches += 1

        history.add(epoch_metrics)

        if verbose:

            metrics_text = " | ".join(
                f"{k}: {v:.4f}"
                for k, v in epoch_metrics.items()
            )

            print(
                f"Epoch [{epoch+1}/{epochs}] "
                f"{metrics_text}"
            )

    history_dict = history.to_dict()

    plt.figure(figsize=(10, 5))

    for key, values in history_dict.items():

        if "loss" in key:
            plt.plot(values, label=key)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model.name} Training History")

    plt.legend()
    plt.grid(True)
    if history_image_path is not None:
        plt.savefig(f"{history_image_path}/{model.name}-Training-History.png", dpi=300, bbox_inches="tight")
    plt.show()

    return history