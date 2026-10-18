#!/usr/bin/env python3
"""
Train SASRec (Self-Attentive Sequential Recommendation) baseline model.
This implementation uses standard item IDs for sequential recommendation.
"""

import inspect
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

import wandb
from src.device_manager import DeviceManager
from src.logger import setup_logger

logger = setup_logger("train-sasrec", log_to_file=True)


@dataclass
class SASRecConfig:
    """Configuration for SASRec training."""

    # Data settings
    dataset: str = "Video_Games"  # Dataset name (Video_Games, Beauty, etc.)
    data_dir: Path = field(default_factory=lambda: Path("data"))  # Data directory path
    data_path: Optional[Path] = None  # Path to preprocessed sequences (auto-generated if None)
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints") / "sasrec")

    # Model parameters
    max_seq_length: int = 100  # Maximum sequence length
    hidden_units: int = 64  # Hidden dimension size
    num_blocks: int = 2  # Number of transformer blocks
    num_heads: int = 1  # Number of attention heads
    dropout_rate: float = 0.2  # Dropout rate

    # Training parameters
    batch_size: int = 1024  # Batch size for training
    num_epochs: int = 200  # Number of training epochs
    max_learning_rate: float = 1e-3  # Maximum learning rate (start of cosine)
    min_learning_rate: float = 1e-5  # Minimum learning rate (end of cosine)

    # Training settings
    scheduler_type: str = "cosine"  # Learning rate scheduler type ("cosine" or "cosine_with_warmup")
    warmup_steps: int = 100  # Number of warmup steps (only for cosine_with_warmup)
    warmup_start_lr: float = 1e-8  # Starting learning rate for warmup (only for cosine_with_warmup)
    steps_per_train_log: int = 100  # Log training progress every N steps
    steps_per_val_log: int = 500  # Validate and checkpoint every N steps

    def __post_init__(self):
        """Validate configuration and set computed fields."""
        # Auto-generate data path if not provided
        if self.data_path is None:
            self.data_path = self.data_dir / "output" / f"{self.dataset}_sequences_with_semantic_ids_train.parquet"

    def log_config(self):
        """Log all configuration parameters."""
        logger.info("=== SASRec Configuration ===")

        # Data settings
        logger.info("Data Settings:")
        logger.info(f"  dataset: {self.dataset}")
        logger.info(f"  data_dir: {self.data_dir}")
        logger.info(f"  data_path: {self.data_path}")
        logger.info(f"  checkpoint_dir: {self.checkpoint_dir}")

        # Model parameters
        logger.info("Model Parameters:")
        logger.info(f"  max_seq_length: {self.max_seq_length}")
        logger.info(f"  hidden_units: {self.hidden_units}")
        logger.info(f"  num_blocks: {self.num_blocks}")
        logger.info(f"  num_heads: {self.num_heads}")
        logger.info(f"  dropout_rate: {self.dropout_rate}")

        # Training parameters
        logger.info("Training Parameters:")
        logger.info(f"  batch_size: {self.batch_size}")
        logger.info(f"  num_epochs: {self.num_epochs}")
        logger.info(f"  max_learning_rate: {self.max_learning_rate}")
        logger.info(f"  min_learning_rate: {self.min_learning_rate}")

        # Training settings
        logger.info("Training Settings:")
        logger.info(f"  scheduler_type: {self.scheduler_type}")
        if self.scheduler_type == "cosine_with_warmup":
            logger.info(f"  warmup_steps: {self.warmup_steps}")
            logger.info(f"  warmup_start_lr: {self.warmup_start_lr}")
        logger.info(f"  steps_per_train_log: {self.steps_per_train_log}")
        logger.info(f"  steps_per_val_log: {self.steps_per_val_log}")

        logger.info("===========================")


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with causal mask."""

    def __init__(self, hidden_units: int, num_heads: int, dropout_rate: float):
        super().__init__()
        assert hidden_units % num_heads == 0

        # Combined QKV projection
        self.c_attn = nn.Linear(hidden_units, 3 * hidden_units)
        # Output projection
        self.c_proj = nn.Linear(hidden_units, hidden_units)
        self.attn_dropout = nn.Dropout(dropout_rate)
        self.resid_dropout = nn.Dropout(dropout_rate)
        self.num_heads = num_heads
        self.hidden_units = hidden_units

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # batch, sequence length, hidden units

        # Calculate query, key, values for all heads in batch
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.hidden_units, dim=2)

        # Reshape for multi-head attention
        k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, nh, T, hs)

        # Causal self-attention with flash attention
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.attn_dropout.p if self.training else 0.0
        )

        # Re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """Multi-layer perceptron (feed-forward network)."""

    def __init__(self, hidden_units: int, dropout_rate: float):
        super().__init__()
        self.fc1 = nn.Linear(hidden_units, hidden_units)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with pre-LN architecture."""

    def __init__(self, hidden_units: int, num_heads: int, dropout_rate: float):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_units, eps=1e-8)
        self.attn = CausalSelfAttention(hidden_units, num_heads, dropout_rate)
        self.ln_2 = nn.LayerNorm(hidden_units, eps=1e-8)
        self.ffn = MLP(hidden_units, dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN: LayerNorm -> Sub-layer -> Residual
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffn(self.ln_2(x))
        return x


class SASRec(nn.Module):
    """Self-Attentive Sequential Recommendation model."""

    def __init__(self, config: SASRecConfig, item_num: int):
        super().__init__()

        self.config = config  # Store config for later use
        self.item_num = item_num
        self.max_seq_length = config.max_seq_length
        self.hidden_units = config.hidden_units

        # Embedding layers
        self.item_emb = nn.Embedding(item_num + 1, config.hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(config.max_seq_length + 1, config.hidden_units, padding_idx=0)
        self.emb_dropout = nn.Dropout(config.dropout_rate)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(config.hidden_units, config.num_heads, config.dropout_rate)
                for _ in range(config.num_blocks)
            ]
        )

        # Final layer norm
        self.ln_f = nn.LayerNorm(config.hidden_units, eps=1e-8)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            # Don't initialize padding_idx
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for feature extraction.

        Args:
            input_ids: Item sequences [batch_size, seq_length]

        Returns:
            Hidden states [batch_size, seq_length, hidden_units]
        """
        batch_size, seq_length = input_ids.size()

        # Get item embeddings
        item_embs = self.item_emb(input_ids)
        item_embs *= self.hidden_units**0.5  # Scale by sqrt(d) as in Transformer

        # Add positional embeddings
        positions = torch.arange(1, seq_length + 1, dtype=torch.long, device=input_ids.device)
        positions = positions.unsqueeze(0).expand(batch_size, -1)
        # Mask positions where input is padding
        positions = positions * (input_ids != 0).long()
        pos_embs = self.pos_emb(positions)

        # Combine embeddings
        hidden_states = self.emb_dropout(item_embs + pos_embs)

        # Pass through transformer blocks
        for block in self.blocks:
            hidden_states = block(hidden_states)

        # Final layer norm
        hidden_states = self.ln_f(hidden_states)

        return hidden_states

    def predict(self, input_ids: torch.Tensor, candidate_ids: torch.Tensor) -> torch.Tensor:
        """
        Predict scores for candidate items.

        Args:
            input_ids: Item sequences [batch_size, seq_length]
            candidate_ids: Candidate items to score [batch_size, num_candidates]

        Returns:
            Scores for each candidate [batch_size, num_candidates]
        """
        # Get sequence representations
        hidden_states = self.forward(input_ids)  # [B, T, H]

        # Use only the last hidden state for prediction
        final_hidden = hidden_states[:, -1, :]  # [B, H]

        # Get candidate embeddings
        candidate_embs = self.item_emb(candidate_ids)  # [B, C, H]

        # Compute scores via dot product
        scores = torch.bmm(candidate_embs, final_hidden.unsqueeze(-1)).squeeze(-1)  # [B, C]

        return scores

    def training_step(
        self, input_ids: torch.Tensor, pos_ids: torch.Tensor, neg_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Training step with positive and negative items.

        Args:
            input_ids: Item sequences [batch_size, seq_length]
            pos_ids: Positive items (next items) [batch_size, seq_length]
            neg_ids: Negative items (sampled) [batch_size, seq_length]

        Returns:
            pos_logits: Scores for positive items [batch_size, seq_length]
            neg_logits: Scores for negative items [batch_size, seq_length]
        """
        # Get sequence representations
        hidden_states = self.forward(input_ids)  # [B, T, H]

        # Get item embeddings
        pos_embs = self.item_emb(pos_ids)  # [B, T, H]
        neg_embs = self.item_emb(neg_ids)  # [B, T, H]

        # Compute logits via dot product (use element-wise mult for position-wise dot products)
        pos_logits = (hidden_states * pos_embs).sum(dim=-1)  # [B, T]
        neg_logits = (hidden_states * neg_embs).sum(dim=-1)  # [B, T]

        return pos_logits, neg_logits


class SequenceDataset(Dataset):
    """Dataset for sequential recommendation."""

    def __init__(self, data_path: str, config: SASRecConfig):
        """
        Load user-item sequences from parquet file.

        File format: parquet with columns [user_id, sequence, sequence_length]
        where sequence is a list of item IDs.
        """
        self.config = config
        self.max_seq_length = config.max_seq_length

        # Load data from parquet
        logger.info(f"Loading data from {data_path}")
        df = pl.read_parquet(data_path)

        # Extract user sequences
        self.users = df["user_id"].to_list()
        sequences = df["sequence"].to_list()

        # Build user_seq dictionary and collect all items
        self.user_seq = {}
        item_set = set()

        for user, seq in zip(self.users, sequences):
            self.user_seq[user] = seq
            item_set.update(seq)

        # Map item IDs to integers
        self.item_to_id = {item: idx + 1 for idx, item in enumerate(sorted(item_set))}  # 0 is padding
        self.id_to_item = {idx: item for item, idx in self.item_to_id.items()}

        # Convert sequences to integer IDs
        for user in self.users:
            self.user_seq[user] = [self.item_to_id[item] for item in self.user_seq[user]]

        self.max_item = len(item_set)  # Maximum item ID
        self.num_users = len(self.users)
        self.num_items = len(item_set)

        # Create list of all item IDs for efficient negative sampling
        self.all_items = list(range(1, self.max_item + 1))  # All valid item IDs (excluding 0 which is padding)

        # Filter users with too few interactions (should already be filtered, but double-check)
        valid_users = []
        for u in self.users:
            if len(self.user_seq[u]) >= 3:  # Need at least 3 items (train, val, test)
                valid_users.append(u)
        self.users = valid_users

        logger.info(f"Loaded {self.num_users:,} users, {self.num_items:,} items")
        logger.info(f"After filtering: {len(self.users):,} users with >= 3 interactions")

        # Compute average sequence length
        avg_seq_len = np.mean([len(self.user_seq[u]) for u in self.users])
        logger.info(f"Average sequence length: {avg_seq_len:.2f}")

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = self.users[idx]
        seq = self.user_seq[user]

        # Return user id and full sequence
        # Train/val/test split is handled in SequentialSampler and evaluate()
        return user, seq


class SequentialSampler:
    """
    Sampler for sequential data that generates training batches.
    Each batch contains user sequences with positive and negative samples.
    Note: Tried to do this without dataloader for learning.
    """

    def __init__(self, dataset: SequenceDataset, config: SASRecConfig):
        self.dataset = dataset
        self.config = config
        self.batch_size = config.batch_size
        self.max_seq_length = config.max_seq_length
        self.max_item = dataset.max_item

        # Pre-compute which sequences are valid for training
        self.valid_user_seqs = []
        for user in dataset.users:
            seq = dataset.user_seq[user]
            if len(seq) > 2:  # Need at least 3 items
                # Store (user, seq[:-2]) for training (exclude last 2 for val/test)
                self.valid_user_seqs.append((user, seq[:-2]))

    @staticmethod
    def sample_negative_item(min_id: int, max_id_exclusive: int, seen_items: set) -> int:
        """Sample a random item ID that is not in seen_items."""
        item_id = np.random.randint(min_id, max_id_exclusive)
        while item_id in seen_items:
            item_id = np.random.randint(min_id, max_id_exclusive)
        return item_id

    def __iter__(self):
        # Shuffle at the beginning of each epoch
        indices = np.random.permutation(len(self.valid_user_seqs))

        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i : i + self.batch_size]
            batch_data = [self.valid_user_seqs[idx] for idx in batch_indices]

            # Generate batch tensors
            actual_batch_size = len(batch_data)
            seq_tensors = torch.zeros((actual_batch_size, self.max_seq_length), dtype=torch.long)
            pos_tensors = torch.zeros((actual_batch_size, self.max_seq_length), dtype=torch.long)
            neg_tensors = torch.zeros((actual_batch_size, self.max_seq_length), dtype=torch.long)

            for idx, (user, seq) in enumerate(batch_data):
                # For each training step, we predict all positions in the sequence
                # This is more efficient than sampling one position at a time
                seq_len = min(len(seq), self.max_seq_length)

                if seq_len < 1:
                    continue

                # If sequence is longer than max_seq_length, take the most recent items
                if len(seq) > self.max_seq_length:
                    seq = seq[-self.max_seq_length :]
                    seq_len = self.max_seq_length

                # Input sequence: all items in the sequence
                seq_tensors[idx, -seq_len:] = torch.tensor(seq[:seq_len])

                # Positive items: for each position i, predict item at position i+1.
                # The last position has no label (0 = masked in the loss): its next item is the user's
                # validation item, which must stay held out.
                for pos in range(seq_len - 1):
                    pos_tensors[idx, -seq_len + pos] = seq[pos + 1]

                # Sample negative items for each position
                seen_set = set(self.dataset.user_seq[user])  # Use full sequence for negative sampling
                for pos in range(seq_len):
                    neg_item = self.sample_negative_item(1, self.max_item + 1, seen_set)
                    neg_tensors[idx, -seq_len + pos] = neg_item

            yield {"input_ids": seq_tensors, "pos_ids": pos_tensors, "neg_ids": neg_tensors}

    def __len__(self):
        return (len(self.valid_user_seqs) + self.batch_size - 1) // self.batch_size


def evaluate(
    model: SASRec,
    dataset: SequenceDataset,
    mode: str = "val",
    batch_size: int = 256,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Evaluate model on validation or test set.

    Args:
        model: Trained SASRec model
        dataset: SequenceDataset instance
        mode: 'val' or 'test'
        batch_size: Batch size for evaluation
        device: Device to run evaluation on

    Returns:
        Dictionary with NDCG@10 and HR@10 metrics
    """
    model.eval()

    ndcg_sum = 0.0
    hr_sum = 0.0
    valid_users = 0

    # Process users in batches for efficiency
    users = dataset.users

    for batch_start in range(0, len(users), batch_size):
        batch_users = users[batch_start : batch_start + batch_size]
        batch_seqs = []
        batch_targets = []
        batch_valid_mask = []

        for user in batch_users:
            seq = dataset.user_seq[user]

            if mode == "val":
                # Use all but last 2 items as input, predict second-to-last
                if len(seq) < 3:
                    batch_valid_mask.append(False)
                    batch_seqs.append([])
                    batch_targets.append(0)
                    continue
                input_seq = seq[:-2]
                target = seq[-2]
            else:  # test
                # Use all but last item as input, predict last
                if len(seq) < 2:
                    batch_valid_mask.append(False)
                    batch_seqs.append([])
                    batch_targets.append(0)
                    continue
                input_seq = seq[:-1]
                target = seq[-1]

            batch_valid_mask.append(True)
            batch_seqs.append(input_seq)
            batch_targets.append(target)

        # Skip if no valid users in batch
        if not any(batch_valid_mask):
            continue

        # Prepare input tensors
        max_len = min(max(len(s) for s in batch_seqs if s), dataset.max_seq_length)
        input_tensor = torch.zeros((len(batch_users), max_len), dtype=torch.long)

        for i, seq in enumerate(batch_seqs):
            if seq and batch_valid_mask[i]:
                seq_len = min(len(seq), max_len)
                input_tensor[i, -seq_len:] = torch.tensor(seq[-seq_len:])

        input_tensor = input_tensor.to(device)

        # Generate candidates for each user
        # Target item + 100 random negative items
        candidates_list = []
        for i, (user, target) in enumerate(zip(batch_users, batch_targets)):
            if not batch_valid_mask[i]:
                candidates_list.append(torch.zeros(101, dtype=torch.long))
                continue

            candidates = [target]  # First item is always the target
            seen_items = set(dataset.user_seq[user])

            # Sample 100 negative items
            while len(candidates) < 101:
                neg_item = np.random.randint(1, dataset.max_item + 1)
                if neg_item not in seen_items:
                    candidates.append(neg_item)

            candidates_list.append(torch.tensor(candidates))

        # Stack candidates
        candidates_tensor = torch.stack(candidates_list).to(device)

        # Get predictions
        with torch.no_grad():
            # Only evaluate on valid users
            valid_indices = [i for i, valid in enumerate(batch_valid_mask) if valid]
            if not valid_indices:
                continue

            valid_input = input_tensor[valid_indices]
            valid_candidates = candidates_tensor[valid_indices]

            scores = model.predict(valid_input, valid_candidates)  # [valid_batch_size, 101]

        # Calculate metrics
        # Sort scores in descending order
        _, indices = torch.sort(scores, dim=1, descending=True)

        # Find rank of target item (which is at position 0 in candidates)
        ranks = (indices == 0).nonzero(as_tuple=True)[1].cpu().numpy() + 1  # 1-indexed ranks

        for rank in ranks:
            valid_users += 1

            # HR@10
            if rank <= 10:
                hr_sum += 1

            # NDCG@10
            if rank <= 10:
                ndcg_sum += 1 / np.log2(rank + 1)

    # Calculate average metrics
    ndcg_10 = ndcg_sum / valid_users if valid_users > 0 else 0.0
    hr_10 = hr_sum / valid_users if valid_users > 0 else 0.0

    logger.info(f"Evaluated on {valid_users:,} users")

    return {"ndcg@10": ndcg_10, "hr@10": hr_10}


def get_gradient_norm(model: nn.Module) -> float:
    """Calculate the L2 norm of gradients across all model parameters."""
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads:
        return 0.0

    # Compute norm without materializing concatenated tensor
    total_norm = torch.norm(torch.stack([torch.norm(g, 2) for g in grads]), 2)
    return total_norm.item()


def train_sasrec(
    model: SASRec, train_dataset: SequenceDataset, config: SASRecConfig, device: str = "cpu"
) -> Dict[str, float]:
    """
    Train SASRec model.

    Args:
        model: SASRec model to train
        train_dataset: Training dataset
        config: Training configuration
        device: Device to train on

    Returns:
        Dictionary with best validation metrics
    """
    model = model.to(device)

    # Apply torch.compile for faster training (CUDA only, not MPS)
    if device == "cuda":
        logger.info("Compiling model with torch.compile for faster training...")
        model = torch.compile(model)

    # Create sampler for training
    train_sampler = SequentialSampler(train_dataset, config)
    steps_per_epoch = len(train_sampler)
    total_steps = config.num_epochs * steps_per_epoch

    logger.info(f"Training for {config.num_epochs} epochs, {steps_per_epoch} steps per epoch")
    logger.info(f"Total training steps: {total_steps:,}")

    # Optimizer with fused support
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device == "cuda"
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.max_learning_rate, betas=(0.9, 0.98), fused=use_fused)

    # Learning rate scheduler
    if config.scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=config.min_learning_rate
        )
        logger.info(
            f"Cosine annealing: {config.max_learning_rate:.1e} -> {config.min_learning_rate:.1e} for {total_steps:,} steps"
        )
    elif config.scheduler_type == "cosine_with_warmup":
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=config.warmup_start_lr / config.max_learning_rate,
            total_iters=config.warmup_steps,
        )
        logger.info(
            f"Warmup: {config.warmup_start_lr:.1e} -> {config.max_learning_rate:.1e} for {config.warmup_steps:,} steps"
        )

        cosine_steps = total_steps - config.warmup_steps
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cosine_steps, eta_min=config.min_learning_rate
        )
        logger.info(
            f"Cosine annealing: {config.max_learning_rate:.1e} -> {config.min_learning_rate:.1e} for {cosine_steps:,} steps"
        )

        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[config.warmup_steps]
        )
    else:
        scheduler = None

    # Loss function
    bce_criterion = nn.BCEWithLogitsLoss()

    # Track best model
    best_val_metrics = {"ndcg@10": 0.0, "hr@10": 0.0}
    global_step = 0

    # Create progress bar for entire training (all epochs)
    pbar = tqdm(total=total_steps)

    # Training loop
    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0

        for batch_idx, batch in enumerate(train_sampler):
            t0 = time.time()
            optimizer.zero_grad()

            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            pos_ids = batch["pos_ids"].to(device)
            neg_ids = batch["neg_ids"].to(device)

            # Forward pass
            pos_logits, neg_logits = model.training_step(input_ids, pos_ids, neg_ids)

            # Calculate loss only on non-padding positions
            valid_mask = pos_ids != 0

            # Initialize loss values for logging
            loss_value = 0.0
            pos_loss_value = 0.0
            neg_loss_value = 0.0

            if valid_mask.any():
                pos_labels = torch.ones_like(pos_logits)[valid_mask]
                neg_labels = torch.zeros_like(neg_logits)[valid_mask]

                pos_loss = bce_criterion(pos_logits[valid_mask], pos_labels)
                neg_loss = bce_criterion(neg_logits[valid_mask], neg_labels)

                # Store values for logging
                pos_loss_value = pos_loss.item()
                neg_loss_value = neg_loss.item()

                loss = pos_loss + neg_loss
                loss_value = loss.item()

                # Backward pass
                loss.backward()

            # Get gradient norm for logging
            grad_norm = get_gradient_norm(model)

            # Optimizer step
            optimizer.step()

            # Step the learning rate scheduler
            if scheduler is not None:
                scheduler.step()

            # Get current learning rate for logging
            current_lr = optimizer.param_groups[0]["lr"]

            # Update metrics
            epoch_loss += loss_value
            epoch_steps += 1
            global_step += 1
            pbar.update(1)  # Update progress bar by one step

            # Time measurement
            t1 = time.time()
            batch_time_ms = (t1 - t0) * 1000
            samples_per_second = config.batch_size / (t1 - t0)

            # Logging
            if global_step == 1 or global_step % config.steps_per_train_log == 0:
                log_str = (
                    f"Step {global_step:06d} | Epoch {epoch + 1:03d}/{config.num_epochs:03d} | "
                    f"Loss: {loss_value:.4f} (pos: {pos_loss_value:.4f}, neg: {neg_loss_value:.4f}) | "
                    f"LR: {current_lr:.2e} | "
                    f"Grad: {grad_norm:.2f}"
                )
                log_str += f" | Time: {batch_time_ms:.0f}ms | Samples/s: {samples_per_second:,.0f}"
                logger.info(log_str)

                # Log to W&B
                wandb.log(
                    {
                        # Losses
                        "loss/total": loss_value,
                        "train/learning_rate": current_lr,
                        "train/gradient_norm": grad_norm,
                        "train/batch_time_ms": batch_time_ms,
                        "train/samples_per_second": samples_per_second,
                        "progress/epoch": epoch + 1,
                        "progress/step": global_step,
                    }
                )

            # Validation
            if global_step % config.steps_per_val_log == 0:
                logger.info("Running validation...")
                val_metrics = evaluate(model, train_dataset, mode="val", device=device)
                model.train()

                logger.info(
                    f"Step {global_step:06d} | Validation - NDCG@10: {val_metrics['ndcg@10']:.4f}, HR@10: {val_metrics['hr@10']:.4f}"
                )

                # Log to W&B
                wandb.log(
                    {
                        "val/ndcg@10": val_metrics["ndcg@10"],
                        "val/hr@10": val_metrics["hr@10"],
                        "progress/epoch": epoch + 1,
                        "progress/step": global_step,
                    }
                )

                # Save best model
                if val_metrics["ndcg@10"] > best_val_metrics["ndcg@10"]:
                    best_val_metrics = val_metrics
                    if config.checkpoint_dir:
                        best_path = config.checkpoint_dir / "best_model.pth"
                        torch.save(
                            {
                                "epoch": epoch,
                                "step": global_step,
                                "model_state_dict": model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                                "metrics": val_metrics,
                                "item_to_id": train_dataset.item_to_id,
                                "id_to_item": train_dataset.id_to_item,
                            },
                            best_path,
                        )
                        logger.info(f"Saved best model with NDCG@10: {val_metrics['ndcg@10']:.4f}")

                        # Save as W&B artifact
                        artifact = wandb.Artifact(
                            f"sasrec-best-{config.dataset}",
                            type="model",
                            metadata={
                                "ndcg@10": val_metrics["ndcg@10"],
                                "hr@10": val_metrics["hr@10"],
                                "epoch": epoch + 1,
                                "step": global_step,
                            },
                        )
                        artifact.add_file(str(best_path))
                        wandb.log_artifact(artifact)

    pbar.close()  # Close the progress bar
    logger.info(f"Training completed. Best validation NDCG@10: {best_val_metrics['ndcg@10']:.4f}")

    return best_val_metrics


if __name__ == "__main__":
    config = SASRecConfig()

    device_manager = DeviceManager(logger)
    device = device_manager.device

    run_name = f"sasrec-{config.dataset}-L{config.num_blocks}-H{config.hidden_units}"
    run = wandb.init(project="sasrec-experiments", name=run_name, config=config.__dict__)
    config.log_config()

    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    dataset = SequenceDataset(str(config.data_path), config)
    model = SASRec(config, item_num=dataset.max_item)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    best_metrics = train_sasrec(model=model, train_dataset=dataset, config=config, device=device)

    logger.info("Running final test evaluation...")
    test_metrics = evaluate(model, dataset, mode="test", device=device)
    logger.info(f"Test Results - NDCG@10: {test_metrics['ndcg@10']:.4f}, HR@10: {test_metrics['hr@10']:.4f}")

    wandb.log({"test/ndcg@10": test_metrics["ndcg@10"], "test/hr@10": test_metrics["hr@10"]})

    final_path = config.checkpoint_dir / "final_model.pth"
    logger.info(f"Saving final model to {final_path}")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config.__dict__,
            "val_metrics": best_metrics,
            "test_metrics": test_metrics,
            "item_to_id": dataset.item_to_id,
            "id_to_item": dataset.id_to_item,
        },
        final_path,
    )

    artifact = wandb.Artifact(
        f"sasrec-final-{config.dataset}",
        type="model",
        metadata={
            "val_ndcg@10": best_metrics["ndcg@10"],
            "val_hr@10": best_metrics["hr@10"],
            "test_ndcg@10": test_metrics["ndcg@10"],
            "test_hr@10": test_metrics["hr@10"],
        },
    )
    artifact.add_file(str(final_path))
    wandb.log_artifact(artifact)

    wandb.finish()
    logger.info("Training complete!")
