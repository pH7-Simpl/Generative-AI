"""
BaseTextGenerator + FullAttnResLLM + LSTMLLM
Dengan Pretrained Embedder & TrainingHistory

Arsitektur:
- BaseTextGenerator: Abstract base class untuk semua text generator
- FullAttnResLLM: Transformer dengan Full Attention Residuals
- LSTMLLM: LSTM-based text generator
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import json
import os
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from src.training_history import TrainingHistory

class PretrainedEmbedder(nn.Module):
    """
    Wrapper untuk Sentence-Transformer dengan projection.

    Features:
    - Auto-detect embedding dimension dari model
    - Projection ke target dimension
    - Nearest-neighbor decode
    """
    def __init__(
        self,
        model_name: str = "LazarusNLP/all-indo-e5-small-v4",
        projection_dim: int = 256,
        device: str = "cuda",
        trust_remote_code: bool = False
    ):
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.projection_dim = projection_dim
        self.embed_dropout = nn.Dropout(0.1)

        print(f"Loading embedder: {model_name}")
        self.encoder = SentenceTransformer(
            model_name, 
            trust_remote_code=trust_remote_code, 
            device=device
        )

        self.embed_dim = self.encoder.get_sentence_embedding_dimension()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        hf_model = self.encoder[0].auto_model

        # =====================================================
        # Token Embedding
        #
        # Rumus:
        # e_i = E[x_i]
        #
        # E = embedding matrix
        #
        # Bagian ini mengubah token id menjadi
        # representasi vektor dense.
        # =====================================================

        self.token_embedding = hf_model.embeddings.word_embeddings

        for p in self.token_embedding.parameters():
            p.requires_grad = False

        self.embed_dim = self.token_embedding.embedding_dim

        # =====================================================
        # Input Projection
        #
        # Rumus:
        # h_i = W e_i + b
        #
        # Bagian ini memproyeksikan embedding pretrained
        # ke dimensi model internal.
        # =====================================================

        self.input_projection = nn.Linear(
            self.embed_dim,
            self.projection_dim
        ).to(device)

        # =====================================================
        # Output Projection
        #
        # Rumus:
        # logits = h W_embed^T
        #
        # Bagian ini mengubah hidden state kembali
        # menjadi distribusi token vocabulary.
        # =====================================================

        self.output_hidden = nn.Linear(
            self.projection_dim,
            self.embed_dim
        ).to(device)

    def tokenize(self, texts, max_length=128):

        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

        return {
            k: v.to(self.device)
            for k, v in tokens.items()
        }
    
    def encode_tokens(self, input_ids):
        emb = self.token_embedding(input_ids)
        projected = self.input_projection(emb)
        projected = self.embed_dropout(projected)
        return projected
    
    def decode_logits(self, hidden_states):
        hidden = self.output_hidden(hidden_states)

        logits = F.linear(
            hidden,
            self.token_embedding.weight
        )

        logits = logits / math.sqrt(self.embed_dim)

        return logits

    def forward(self, input_ids):
        return self.encode_tokens(input_ids)
    
    def get_config(self):
        return {
            "model_name": self.model_name,
            "projection_dim": self.projection_dim,
            "device": self.device,
        }

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    # =====================================================
    # RMS Normalization
    #
    # Rumus:
    #
    #                 x
    # y = ---------------------------
    #      sqrt(mean(x²)) + eps
    #
    # Implementasi:
    # return self.weight * x / (norm + self.eps)
    # =====================================================

    def forward(self, x):
        norm = x.norm(2, dim=-1, keepdim=True) * (x.size(-1) ** -0.5)
        return self.weight * x / (norm + self.eps)

class BaseTextGenerator(nn.Module, ABC):
    """
    Abstract base class untuk semua text generator.

    Features:
    - Auto-detect dim dari embedder.projection_dim
    - Training history tracking
    - Common training & generation logic
    """
    def __init__(
        self,
        embedder: PretrainedEmbedder,
        num_layers: int = 4,
        dropout: float = 0.1,
        device: str = "cuda"
    ):
        super().__init__()
        self.embedder = embedder
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.device = device
        self.history = TrainingHistory()

        self.dim = embedder.projection_dim
        print(f"Auto-detected dim from embedder: {self.dim}")

        self.to(device)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass - harus diimplementasikan oleh subclass"""
        pass

    @abstractmethod
    def compute_loss(
        self,
        input_emb: torch.Tensor,
        target_emb: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss - harus diimplementasikan oleh subclass"""
        pass

    def prepare_tokens(self, texts):
        full_text = " ".join(texts)

        tokens = self.embedder.tokenizer(
            full_text,
            return_tensors="pt"
        )

        return tokens["input_ids"][0]
    
    def get_config(self):
        return {
            "model_class": self.__class__.__name__,
            "embedder": self.embedder.get_config(),
            "num_layers": self.num_layers,
            "dropout": self.dropout_rate,
            "dim": self.dim,
            "device": self.device,
        }

    def train_model(
        self,
        sentences: List[str],
        num_epochs: int = 10,
        batch_size: int = 4,
        context_size: int = 5,
        learning_rate: float = 1e-4,
        print_every: int = 10,
        save_path: Optional[str] = None
    ) -> TrainingHistory:
        """Training loop umum untuk semua generator"""
        self.train()

        all_tokens = []
        for sentence in sentences:
            ids = self.embedder.tokenizer(
                sentence,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                add_special_tokens=True
            )["input_ids"][0]

            all_tokens.append(ids)

        tokens = torch.cat(all_tokens, dim=0).to(self.device)

        print(f"Total tokens: {len(tokens)}")

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.1
        )

        indices = list(range(len(tokens) - context_size - 1))

        steps_per_epoch = max(1, len(indices) // batch_size)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs * steps_per_epoch,
            eta_min=1e-6
        )

        total_start_time = time.time()

        for epoch in range(num_epochs):
            epoch_start = time.time()
            epoch_loss = 0.0
            num_batches = 0
            import random
            random.shuffle(indices)

            for i in range(0, len(indices), batch_size):

                batch_indices = indices[i:i + batch_size]

                if len(batch_indices) < batch_size:
                    continue

                input_batch = []
                target_batch = []

                for idx in batch_indices:

                    input_ids = tokens[
                        idx : idx + context_size
                    ]

                    target_ids = tokens[
                        idx + 1 : idx + context_size + 1
                    ]

                    input_batch.append(input_ids)
                    target_batch.append(target_ids)

                input_batch = torch.stack(input_batch).to(self.device)
                target_batch = torch.stack(target_batch).to(self.device)

                optimizer.zero_grad()

                loss = self.compute_loss(
                    input_batch,
                    target_batch
                )

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.parameters(),
                    1.0
                )

                optimizer.step()
                scheduler.step()

                loss_val = loss.item()

                epoch_loss += loss_val
                num_batches += 1

                if num_batches % print_every == 0:
                    print(
                        f"Epoch {epoch+1}/{num_epochs} | "
                        f"Batch {num_batches} | "
                        f"Loss: {loss_val:.6f}"
                    )

            avg_loss = epoch_loss / max(num_batches, 1)
            epoch_time = time.time() - epoch_start

            metrics = {
                "loss": avg_loss,
                "epoch_time": epoch_time,
                "learning_rate": scheduler.get_last_lr()[0],
            }

            self.history.add(metrics)

            print(f"\n=== Epoch {epoch+1}/{num_epochs} ===")
            print(f"  Average Loss: {avg_loss:.6f}")
            print(f"  Epoch Time: {epoch_time:.2f}s")
            print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.2e}")

        total_time = time.time() - total_start_time
        print(f"\n=== Training Complete ===")
        print(f"  Total Time: {total_time:.2f}s ({total_time/60:.2f}m)")
        print(f"  Final Loss: {self.history.latest().get('loss', 0):.6f}")

        if save_path is not None:
            self.save_model(save_path)

        return self.history
    
    def save_model(self, save_path: str):

        os.makedirs(save_path, exist_ok=True)

        torch.save(
            self.state_dict(),
            os.path.join(save_path, "model.pt")
        )

        with open(os.path.join(save_path, "config.json"), "w") as f:
            json.dump(self.get_config(), f, indent=4)

        with open(os.path.join(save_path, "history.json"), "w") as f:
            json.dump(self.history.to_dict(), f, indent=4)

        print(f"Model saved to: {save_path}")
    
    def generate(
        self,
        prompt: str,
        max_length: int = 20,
        temperature: float = 1.0,
        context_size: int = 32
    ):
        self.eval()

        tokens = self.embedder.tokenizer(
            prompt,
            return_tensors="pt"
        )["input_ids"].to(self.device)

        generated = tokens.clone()

        with torch.no_grad():

            for _ in range(max_length):

                input_ids = generated[:, -context_size:]

                input_emb = self.embedder.encode_tokens(input_ids)

                hidden = self.forward(input_emb)

                logits = self.embedder.decode_logits(hidden)

                # =====================================================
                # Temperature Sampling
                #
                # Rumus:
                #
                #                logits
                # p = softmax(--------------)
                #             temperature
                #
                # temperature kecil:
                # -> lebih deterministic
                #
                # temperature besar:
                # -> lebih random
                # =====================================================

                next_token_logits = logits[:, -1, :] / temperature

                # =====================================================
                # Top-K Sampling
                #
                # Rumus:
                # pilih hanya K token dengan
                # probabilitas terbesar.
                #
                # Implementasi:
                # torch.topk(...)
                # =====================================================

                top_k = 50

                values, indices = torch.topk(
                    next_token_logits,
                    top_k
                )

                probs = F.softmax(values, dim=-1)

                # =====================================================
                # Token Sampling
                #
                # Rumus:
                # token ~ Multinomial(p)
                #
                # Implementasi:
                # torch.multinomial(...)
                # =====================================================

                sampled = torch.multinomial(
                    probs,
                    num_samples=1
                )

                next_token = indices.gather(-1, sampled)

                generated = torch.cat(
                    [generated, next_token],
                    dim=1
                )

                if next_token.item() == self.embedder.tokenizer.eos_token_id:
                    break
        
        generated_text = self.embedder.tokenizer.decode(
            generated[0],
            skip_special_tokens=True
        )

        return generated_text

    def generate_batch(
        self,
        prompts: list[str],
        max_length: int = 20,
        temperature: float = 1.0,
        context_size: int = 32,
        save_path: str = None,
    ):
        results = []

        for prompt in prompts:

            generated_text = self.generate(
                prompt=prompt,
                max_length=max_length,
                temperature=temperature,
                context_size=context_size,
            )

            results.append({
                "prompt": prompt,
                "generated": generated_text,
            })

        # =====================================================
        # Save Markdown
        # =====================================================

        if save_path is not None:

            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            with open(save_path, "w", encoding="utf-8") as f:

                f.write("# Generated Text\n\n")

                for i, result in enumerate(results, start=1):

                    f.write(f"## Result {i}\n\n")
                    f.write(f"**Prompt:** {result['prompt']}\n\n")
                    f.write(f"**Generated:**\n\n")
                    f.write(f"{result['generated']}\n\n")

        return results

class FullAttnResLayer(nn.Module):
    """
    Transformer layer dengan Full Attention Residuals
    """
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # =====================================================
        # Attention Residual Query
        #
        # Rumus:
        # alpha = softmax(K q)
        #
        # q = pseudo query
        #
        # Bagian ini mempelajari bagaimana setiap
        # layer sebelumnya dikombinasikan.
        # =====================================================

        self.pseudo_query = nn.Parameter(torch.zeros(num_heads, dim))

        self.attn_res_norm = RMSNorm(dim)

        self.attn_norm = RMSNorm(dim)

        # =====================================================
        # Self Attention Projection
        #
        # Rumus:
        # Q = XW_Q
        # K = XW_K
        # V = XW_V
        #
        # Bagian ini membentuk query, key,
        # dan value untuk self-attention.
        # =====================================================

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        # =====================================================
        # Output Projection
        #
        # Rumus:
        # O = Attn(Q,K,V) W_O
        #
        # Bagian ini menggabungkan output attention.
        # =====================================================

        self.o_proj = nn.Linear(dim, dim)

        self.head_merge = nn.Linear(
            num_heads * dim,
            dim
        )

        self.mlp_norm = RMSNorm(dim)

        # =====================================================
        # Feed Forward Network
        #
        # Rumus:
        # FFN(x) = W2 GELU(W1 x)
        #
        # Bagian ini melakukan transformasi non-linear
        # setelah self-attention.
        # =====================================================

        self.mlp_up = nn.Linear(dim, 4 * dim)
        self.mlp_down = nn.Linear(4 * dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, past_outputs: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, seq, dim = x.shape

        if len(past_outputs) > 0:
            V = torch.stack(past_outputs, dim=0)
            K = self.attn_res_norm(V)

            # =====================================================
            # Attention Residual Mixing
            #
            # Rumus:
            # alpha = softmax(K q)
            #
            # h = Σ alpha_i V_i
            #
            # Implementasi:
            # torch.einsum(...)
            # =====================================================

            scores = torch.einsum('nbsd,hd->hnbs', K,self.pseudo_query)
            alpha = F.softmax(scores, dim=1)
            h = torch.einsum(
                'hnbs,nbsd->hbsd',
                alpha,
                V
            )

            h = h.permute(1, 2, 0, 3)
            h = h.reshape(batch, seq, self.num_heads * dim)

            h = self.head_merge(h)
        else:
            h = x

        h_norm = self.attn_norm(h)
        Q = self.q_proj(h_norm).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        K_seq = self.k_proj(h_norm).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        V_seq = self.v_proj(h_norm).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        
        # region Attention
        # =====================================================
        # Self Attention
        #
        # Rumus:
        #
        #                  QK^T
        # Attn(Q,K,V) = ----------- V
        #               sqrt(d_k)
        #
        # Implementasi:
        # torch.matmul(Q, K.transpose(...))
        # =====================================================
        scores = torch.matmul(Q, K_seq.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.triu(
            torch.ones(seq, seq, device=x.device),
            diagonal=1
        ).bool()

        mask = mask.unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V_seq)
        out = out.transpose(1, 2).contiguous().view(batch, seq, dim)
        out = self.o_proj(out)

        # =====================================================
        # Residual Connection
        #
        # Rumus:
        # y = x + f(x)
        #
        # Bagian ini menjaga stabilitas gradient
        # dan mempertahankan informasi lama.
        # =====================================================
        h = h + 0.5 * self.dropout(out)

        # =====================================================
        # Feed Forward Network
        #
        # Rumus:
        # FFN(x) = W2 GELU(W1 x)
        #
        # Implementasi:
        # self.mlp_down(F.gelu(...))
        # =====================================================
        h_norm = self.mlp_norm(h)
        mlp_out = self.mlp_down(F.gelu(self.mlp_up(h_norm)))
        h = h + 0.5 * self.dropout(mlp_out)

        return h, h

class FullAttnResLLM(BaseTextGenerator):
    """
    LLM dengan Full Attention Residuals
    Berdasarkan: arXiv:2603.15031v1 - Attention Residuals
    """
    def __init__(
        self,
        embedder: PretrainedEmbedder,
        num_layers: int = 6,
        num_heads: int = 4,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        device: str = "cuda"
    ):
        super().__init__(embedder, num_layers, dropout, device)

        self.num_heads = num_heads
        self.max_seq_len = max_seq_len

        # =====================================================
        # Positional Embedding
        #
        # Rumus:
        # x_pos = x + P(pos)
        #
        # P(pos) = positional embedding
        #
        # Bagian ini memberi informasi urutan token
        # kepada transformer.
        # =====================================================

        self.pos_embed = nn.Embedding(max_seq_len, self.dim).to(device)

        self.layers = nn.ModuleList([
            FullAttnResLayer(self.dim, num_heads, dropout).to(device)
            for _ in range(num_layers)
        ])

        self.final_norm = RMSNorm(self.dim).to(device)

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, dim = x.shape

        # =====================================================
        # Positional Encoding
        #
        # Rumus:
        # x = x + P(pos)
        #
        # Implementasi:
        # x + self.pos_embed(positions)
        # =====================================================

        positions = torch.arange(seq, device=x.device).unsqueeze(0)
        x = x + self.pos_embed(positions)

        x = self.dropout(x)

        past_outputs = [x.clone()]

        for layer in self.layers:
            x, vi = layer(x, past_outputs)
            past_outputs.append(vi)

        x = self.final_norm(x)
        return x

    def compute_loss(
        self,
        input_ids,
        target_ids
    ):

        input_emb = self.embedder.encode_tokens(input_ids)

        hidden = self.forward(input_emb)

        logits = self.embedder.decode_logits(hidden)

        # region Cross Entropy
        # =====================================================
        # Cross Entropy Loss
        #
        # Rumus:
        #
        # L = - Σ y log(y_hat)
        #
        # atau:
        #
        # L = -log P(token_target)
        #
        # Bagian ini menghitung error prediksi token.
        # =====================================================

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_ids.reshape(-1),
            ignore_index=self.embedder.tokenizer.pad_token_id
        )

        return loss
    
    def get_config(self):

        config = super().get_config()

        config.update({
            "num_heads": self.num_heads,
            "max_seq_len": self.max_seq_len,
        })

        return config
    
    @staticmethod
    def load_model(save_path: str, device: str = "cuda"):

        with open(os.path.join(save_path, "config.json"), "r") as f:
            config = json.load(f)

        embedder_config = config["embedder"]

        embedder = PretrainedEmbedder(
            model_name=embedder_config["model_name"],
            projection_dim=embedder_config["projection_dim"],
            device=device
        )

        model = FullAttnResLLM(
            embedder=embedder,
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            max_seq_len=config["max_seq_len"],
            dropout=config["dropout"],
            device=device
        )

        state_dict = torch.load(
            os.path.join(save_path, "model.pt"),
            map_location=device
        )

        model.load_state_dict(state_dict)

        history_path = os.path.join(save_path, "history.json")

        if os.path.exists(history_path):
            model.history = TrainingHistory.load(history_path)

        model.eval()

        print(f"Loaded model from: {save_path}")

        return model

class LSTMLLM(BaseTextGenerator):
    """
    LLM dengan LSTM
    """
    def __init__(
        self,
        embedder: PretrainedEmbedder,
        num_layers: int = 4,
        dropout: float = 0.1,
        bidirectional: bool = False,
        device: str = "cuda"
    ):
        super().__init__(embedder, num_layers, dropout, device)

        self.bidirectional = bidirectional

        # =====================================================
        # LSTM Layer
        #
        # Rumus:
        #
        # f_t = σ(W_f [h_{t-1}, x_t])
        # i_t = σ(W_i [h_{t-1}, x_t])
        # g_t = tanh(W_g [h_{t-1}, x_t])
        #
        # c_t = f_t * c_{t-1} + i_t * g_t
        #
        # h_t = o_t * tanh(c_t)
        #
        # Bagian ini memproses sequence token
        # secara recurrent.
        # =====================================================

        self.lstm = nn.LSTM(
            input_size=self.dim,
            hidden_size=self.dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        ).to(device)

        lstm_output_dim = self.dim * 2 if bidirectional else self.dim
        self.output_proj = nn.Linear(lstm_output_dim, self.dim).to(device)

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # =====================================================
        # LSTM Forward Pass
        #
        # Rumus:
        # h_t, c_t = LSTM(x_t, h_{t-1}, c_{t-1})
        #
        # Implementasi:
        # self.lstm(x)
        # =====================================================
        lstm_out, (hidden, cell) = self.lstm(x)

        # =====================================================
        # Output Projection
        #
        # Rumus:
        # y = W h + b
        #
        # Bagian ini memproyeksikan output LSTM
        # ke hidden dimension model.
        # =====================================================
        output = self.output_proj(lstm_out)

        return output

    def compute_loss(
        self,
        input_ids,
        target_ids
    ):

        input_emb = self.embedder.encode_tokens(input_ids)

        hidden = self.forward(input_emb)

        logits = self.embedder.decode_logits(hidden)

        # region Cross Entropy
        # =====================================================
        # Cross Entropy Loss
        #
        # Rumus:
        #
        # L = - Σ y log(y_hat)
        #
        # atau:
        #
        # L = -log P(token_target)
        #
        # Bagian ini menghitung error prediksi token.
        # =====================================================

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_ids.reshape(-1),
            ignore_index=self.embedder.tokenizer.pad_token_id
        )

        return loss
    
    def get_config(self):

        config = super().get_config()

        config.update({
            "bidirectional": self.bidirectional,
        })

        return config
    
    @staticmethod
    def load_model(save_path: str, device: str = "cuda"):

        with open(os.path.join(save_path, "config.json"), "r") as f:
            config = json.load(f)

        embedder_config = config["embedder"]

        embedder = PretrainedEmbedder(
            model_name=embedder_config["model_name"],
            projection_dim=embedder_config["projection_dim"],
            device=device
        )

        model = LSTMLLM(
            embedder=embedder,
            num_layers=config["num_layers"],
            bidirectional=config["bidirectional"],
            dropout=config["dropout"],
            device=device
        )

        state_dict = torch.load(
            os.path.join(save_path, "model.pt"),
            map_location=device
        )

        model.load_state_dict(state_dict)

        history_path = os.path.join(save_path, "history.json")

        if os.path.exists(history_path):
            model.history = TrainingHistory.load(history_path)

        model.eval()

        print(f"Loaded model from: {save_path}")

        return model