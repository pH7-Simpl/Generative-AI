from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

class ImageGeneratorBaseModel(nn.Module, ABC):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.name = self.__class__.__name__
        
    def __str__(self):
        return self.name
    
    @abstractmethod
    def train_step(self, batch):
        """
        Harus return dictionary metric/loss.

        Contoh:
        {
            "loss": 0.123,
            "recon_loss": 0.100
        }
        """
        pass

    @abstractmethod
    def generate(self, num_samples: int):
        """
        Generate image baru.
        """
        pass

    def fit(
        self,
        dataloader: DataLoader,
        epochs: int = 10,
        verbose: bool = True,
    ):
        self.train()

        from training_history import TrainingHistory
        history = TrainingHistory()

        for epoch in range(epochs):

            epoch_metrics = {}
            total_batches = 0

            for batch in dataloader:

                if isinstance(batch, (list, tuple)):
                    batch = batch[0]

                batch = batch.to(self.device)

                metrics = self.train_step(batch)

                for key, value in metrics.items():
                    epoch_metrics[key] = (
                        epoch_metrics.get(key, 0.0)
                        + float(value)
                    )

                total_batches += 1

            for key in epoch_metrics:
                epoch_metrics[key] /= total_batches

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

        return history

    @torch.no_grad()
    def sample(self, num_samples: int = 16):
        self.eval()

        samples = self.generate(num_samples)

        self.train()

        return samples

class VAE(ImageGeneratorBaseModel):
    def __init__(
        self,
        input_dim: int = 784,
        latent_dim: int = 32,
        hidden_dim: int = 256,
        lr: float = 1e-3,
        device=None
    ):
        super().__init__(device=device)
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # =====================================================
        # Encoder
        #
        # Rumus:
        # z = f_theta(x)
        #
        # f_theta = encoder network
        #
        # Bagian ini mendefinisikan arsitektur encoder
        # yang mengubah image menjadi latent feature.
        # =====================================================

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # =====================================================
        # Decoder
        #
        # Rumus:
        # x_hat = g_phi(z)
        #
        # g_phi = decoder network
        #
        # Bagian ini mendefinisikan arsitektur decoder
        # yang mengubah latent vector menjadi image.
        # =====================================================

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, self.input_dim),
            nn.Sigmoid(),
        )

        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=lr,
        )

    # =====================================================
    # Encoder Forward Pass
    #
    # Rumus:
    # z = f_theta(x)
    #
    # Implementasi:
    # self.encoder(x)
    # =====================================================

    def encode(self, x):
        h = self.encoder(x)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)

        eps = torch.randn_like(std)

        z = mu + eps * std

        return z

    # =====================================================
    # Decoder Forward Pass
    #
    # Rumus:
    # x_hat = g_phi(z)
    #
    # Implementasi:
    # self.decoder(z)
    # =====================================================

    def decode(self, z):
        x_hat = self.decoder(z)
        return x_hat

    def forward(self, x):

        mu, logvar = self.encode(x)

        z = self.reparameterize(mu, logvar)

        x_hat = self.decode(z)

        return x_hat, mu, logvar

    def compute_loss(
        self,
        x,
        x_hat,
        mu,
        logvar,
    ):
        
        # =====================================================
        # Reconstruction Loss
        #
        # Rumus:
        # L = ||x - x_hat||^2
        #
        # Implementasi:
        # F.mse_loss(x_hat, x)
        # =====================================================

        recon_loss = F.mse_loss(
            x_hat,
            x,
        )
        
        # =====================================================
        # KL Divergence Loss
        #
        # Rumus:
        # D_KL(q(z|x) || p(z))
        #
        # Implementasi:
        # kl_loss = ...
        # =====================================================

        kl_loss = -0.5 * torch.mean(
            1
            + logvar
            - mu.pow(2)
            - logvar.exp()
        )

        total_loss = recon_loss + kl_loss

        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
        }

    def train_step(self, batch):

        self.optimizer.zero_grad()

        x_hat, mu, logvar = self(batch)

        losses = self.compute_loss(
            batch,
            x_hat,
            mu,
            logvar,
        )

        losses["loss"].backward()

        self.optimizer.step()

        return {
            key: value.item()
            for key, value in losses.items()
        }

    @torch.no_grad()
    def generate(self, num_samples: int):

        z = torch.randn(
            num_samples,
            self.latent_dim,
            device=self.device,
        )

        samples = self.decode(z)

        samples = samples.view(
            num_samples,
            1,
            28,
            28
        )
        samples = samples.cpu()

        return samples

    @torch.no_grad()
    def reconstruct(self, x):

        x = x.to(self.device)

        x_hat, _, _ = self(x)

        return x_hat
    
class GAN(ImageGeneratorBaseModel):
    def __init__(
        self,
        input_dim: int = 784,
        latent_dim: int = 64,
        hidden_dim: int = 256,
        lr: float = 2e-4,
        device=None
    ):
        super().__init__(device=device)
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # =====================================================
        # Generator
        #
        # Rumus:
        # G(z)
        #
        # Bagian ini mendefinisikan arsitektur generator
        # yang mengubah random noise menjadi image palsu.
        # =====================================================

        self.generator = nn.Sequential(

            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, self.input_dim),
            nn.Sigmoid(),
        )

        # =====================================================
        # Discriminator
        #
        # Rumus:
        # D(x)
        #
        # Bagian ini mendefinisikan arsitektur discriminator
        # yang membedakan image asli dan image palsu.
        # =====================================================

        self.discriminator = nn.Sequential(

            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, 1),

        )

        self.g_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=lr,
            betas=(0.5, 0.999),
        )

        self.d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=lr,
            betas=(0.5, 0.999),
        )

        self.criterion = nn.BCEWithLogitsLoss()

    def generate_fake(self, z):

        # =====================================================
        # Generator Forward Pass
        #
        # Rumus:
        # fake = G(z)
        #
        # Implementasi:
        # self.generator(z)
        # =====================================================

        fake_images = self.generator(z)

        return fake_images

    def discriminator_loss(
        self,
        real_preds,
        fake_preds,
        real_labels,
        fake_labels,
    ):

        real_loss = self.criterion(
            real_preds,
            real_labels,
        )

        fake_loss = self.criterion(
            fake_preds,
            fake_labels,
        )

        d_loss = real_loss + fake_loss

        return d_loss

    def generator_loss(
        self,
        fake_preds,
        real_labels,
    ):

        g_loss = self.criterion(
            fake_preds,
            real_labels,
        )

        return g_loss

    def train_step(self, real_images):

        batch_size = real_images.size(0)

        real_labels = torch.ones(
            batch_size,
            1,
            device=self.device,
        )

        fake_labels = torch.zeros(
            batch_size,
            1,
            device=self.device,
        )

        self.d_optimizer.zero_grad()

        # =====================================================
        # Discriminator pada image asli
        #
        # Rumus:
        # D(x)
        #
        # Implementasi:
        # self.discriminator(real_images)
        # =====================================================

        real_preds = self.discriminator(
            real_images
        )

        z = torch.randn(
            batch_size,
            self.latent_dim,
            device=self.device,
        )

        # =====================================================
        # Generator menghasilkan image palsu
        #
        # Rumus:
        # fake = G(z)
        # =====================================================

        fake_images = self.generate_fake(z)

        # =====================================================
        # Discriminator pada image palsu
        #
        # Rumus:
        # D(G(z))
        #
        # Implementasi:
        # self.discriminator(fake_images)
        # =====================================================

        fake_preds = self.discriminator(
            fake_images.detach()
        )

        d_loss = self.discriminator_loss(
            real_preds,
            fake_preds,
            real_labels,
            fake_labels,
        )

        d_loss.backward()

        self.d_optimizer.step()

        self.g_optimizer.zero_grad()

        z = torch.randn(
            batch_size,
            self.latent_dim,
            device=self.device,
        )

        fake_images = self.generate_fake(z)

        fake_preds = self.discriminator(
            fake_images
        )

        g_loss = self.generator_loss(
            fake_preds,
            real_labels,
        )

        g_loss.backward()

        self.g_optimizer.step()

        return {
            "g_loss": g_loss.item(),
            "d_loss": d_loss.item(),
        }

    @torch.no_grad()
    def generate(self, num_samples: int):

        z = torch.randn(
            num_samples,
            self.latent_dim,
            device=self.device,
        )

        samples = self.generate_fake(z)

        samples = samples.view(
            num_samples,
            1,
            28,
            28
        )

        samples = samples.cpu()

        return samples