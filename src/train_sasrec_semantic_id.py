#!/usr/bin/env python3
"""
Train SASRec with Semantic IDs using sequential generation.
This implementation adapts SASRec to work with hierarchical semantic IDs.
"""

import gc
import inspect
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import wandb
from src.device_manager import DeviceManager
from src.logger import setup_logger

logger = setup_logger("train-sasrec-semantic", log_to_file=True)


@dataclass
class SemanticSASRecConfig:
    """Configuration for Semantic SASRec training."""

    # Data settings
    dataset: str = "Video_Games"  # Dataset name
    data_dir: Path = field(default_factory=lambda: Path("data"))  # Data directory path
    data_path: Optional[Path] = None  # Path to sequences (auto-generated if None)
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints") / "sasrec")

    # Semantic ID settings
    num_levels: int = 4  # Number of hierarchical levels
    codebook_size: int = 256  # Size of codebook per level
    vocab_size: Optional[int] = None  # Total vocabulary size (computed: num_levels * codebook_size)

    # Model parameters
    max_seq_length: int = 100  # Maximum sequence length in items (will be 4x in tokens)
    input_dim: int = 128  # Input embedding dimension
    head_dim: int = 64  # Dimension per attention head
    num_heads: int = 6  # Number of attention heads
    hidden_dim: Optional[int] = None  # Total hidden dimension (computed: head_dim * num_heads)
    num_blocks: int = 4  # Number of transformer blocks
    mlp_dim: int = 1024  # MLP hidden dimension in feed-forward network
    dropout_rate: float = 0.1  # Dropout rate

    # Training parameters
    batch_size: int = 1024  # Batch size for training
    gradient_accumulation_steps: int = 1  # Number of gradient accumulation steps
    num_epochs: int = 200  # Number of training epochs
    max_learning_rate: float = 1e-3  # M`aximum learning rate (start of cosine)
    min_learning_rate: float = 1e-5  # Minimum learning rate (end of cosine)
    weight_decay: float = 0.0  # Weight decay for AdamW
    l2_emb: float = 0.0  # L2 regularization for embeddings
    teacher_forcing_ratio: float = 0.9  # Probability of using teacher forcing during training

    # Training settings
    scheduler_type: str = "cosine_with_warmup"  # Learning rate scheduler type ("cosine" or "cosine_with_warmup")
    warmup_steps: int = 100  # Number of warmup steps (only for cosine_with_warmup)
    warmup_start_lr: float = 1e-8  # Starting learning rate for warmup (only for cosine_with_warmup)
    use_gradient_clipping: bool = True  # Enable gradient clipping
    gradient_clip_norm: float = 1.0  # Maximum gradient norm
    log_interval: int = 100  # Log progress every N steps
    val_interval: int = 2000  # Validate every N steps
    checkpoint_interval: int = 20  # Save checkpoint every N epochs

    # Performance optimizations
    use_compile: bool = True  # Enable torch.compile for faster training
    fused_adam: bool = True  # Use fused AdamW optimizer when available

    def __post_init__(self):
        """Validate configuration and set computed fields."""
        # Auto-generate data path if not provided
        if self.data_path is None:
            self.data_path = self.data_dir / "output" / f"{self.dataset}_sequences_with_semantic_ids_train.parquet"

        # Compute derived values
        if self.vocab_size is None:
            self.vocab_size = self.num_levels * self.codebook_size

        if self.hidden_dim is None:
            self.hidden_dim = self.head_dim * self.num_heads

    def log_config(self):
        """Log all configuration parameters."""
        logger.info("=== Semantic SASRec Configuration ===")

        # Data settings
        logger.info("Data Settings:")
        logger.info(f"  dataset: {self.dataset}")
        logger.info(f"  data_dir: {self.data_dir}")
        logger.info(f"  data_path: {self.data_path}")
        logger.info(f"  checkpoint_dir: {self.checkpoint_dir}")

        # Semantic ID settings
        logger.info("Semantic ID Settings:")
        logger.info(f"  num_levels: {self.num_levels}")
        logger.info(f"  codebook_size: {self.codebook_size}")
        logger.info(f"  vocab_size: {self.vocab_size}")

        # Model parameters
        logger.info("Model Parameters:")
        logger.info(f"  max_seq_length: {self.max_seq_length}")
        logger.info(f"  input_dim: {self.input_dim}")
        logger.info(f"  head_dim: {self.head_dim}")
        logger.info(f"  num_heads: {self.num_heads}")
        logger.info(f"  hidden_dim: {self.hidden_dim}")
        logger.info(f"  num_blocks: {self.num_blocks}")
        logger.info(f"  mlp_dim: {self.mlp_dim}")
        logger.info(f"  dropout_rate: {self.dropout_rate}")

        # Training parameters
        logger.info("Training Parameters:")
        logger.info(f"  batch_size: {self.batch_size}")
        logger.info(f"  gradient_accumulation_steps: {self.gradient_accumulation_steps}")
        logger.info(f"  effective_batch_size: {self.batch_size * self.gradient_accumulation_steps}")
        logger.info(f"  num_epochs: {self.num_epochs}")
        logger.info(f"  max_learning_rate: {self.max_learning_rate}")
        logger.info(f"  min_learning_rate: {self.min_learning_rate}")
        logger.info(f"  weight_decay: {self.weight_decay}")
        logger.info(f"  l2_emb: {self.l2_emb}")
        logger.info(f"  teacher_forcing_ratio: {self.teacher_forcing_ratio}")

        # Training settings
        logger.info("Training Settings:")
        logger.info(f"  scheduler_type: {self.scheduler_type}")
        if self.scheduler_type == "cosine_with_warmup":
            logger.info(f"  warmup_steps: {self.warmup_steps}")
            logger.info(f"  warmup_start_lr: {self.warmup_start_lr}")
        logger.info(f"  use_gradient_clipping: {self.use_gradient_clipping}")
        logger.info(f"  gradient_clip_norm: {self.gradient_clip_norm}")
        logger.info(f"  log_interval: {self.log_interval}")
        logger.info(f"  val_interval: {self.val_interval}")
        logger.info(f"  checkpoint_interval: {self.checkpoint_interval}")

        # Performance optimizations
        logger.info("Performance Optimizations:")
        logger.info(f"  use_compile: {self.use_compile}")
        logger.info(f"  fused_adam: {self.fused_adam}")

        logger.info("======================================")


def encode_semantic_id(semantic_id: str, num_levels: int = 4, codebook_size: int = 256) -> List[int]:
    """Convert semantic ID string to token indices.

    New format: <|sid_start|><|sid_52|><|sid_273|><|sid_714|><|sid_769|><|sid_end|>
    where each sid_X already contains the level offset (level * 256 + value)

    Returned ids are sid_X + 1, so they live in 1..vocab_size and 0 stays free for padding
    (otherwise <|sid_0|> would be indistinguishable from padding_idx=0).
    """
    # Extract all token IDs from the semantic ID string
    pattern = r"<\|sid_(\d+)\|>"
    matches = re.findall(pattern, semantic_id)

    if not matches:
        raise ValueError(f"Invalid semantic ID format: {semantic_id}")

    # Convert to integers (+1 to reserve 0 for padding)
    tokens = [int(match) + 1 for match in matches[:num_levels]]

    # Pad with zeros if needed (shouldn't happen with proper data)
    while len(tokens) < num_levels:
        tokens.append(0)

    return tokens


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with causal mask."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout_rate: float):
        super().__init__()
        assert hidden_dim % num_heads == 0

        # Combined QKV projection
        self.c_attn = nn.Linear(hidden_dim, 3 * hidden_dim)
        # Output projection
        self.c_proj = nn.Linear(hidden_dim, hidden_dim)
        # Regularization
        self.attn_dropout = nn.Dropout(dropout_rate)
        self.resid_dropout = nn.Dropout(dropout_rate)

        self.num_heads = num_heads
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # batch, sequence length, hidden units

        # Calculate query, key, values for all heads in batch
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.hidden_dim, dim=2)

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


class PointWiseFeedForward(nn.Module):
    """Point-wise feed-forward network."""

    def __init__(self, hidden_dim: int, mlp_dim: int, dropout_rate: float):
        super().__init__()
        # Using Conv1d as in original SASRec for efficiency
        self.conv1 = nn.Conv1d(hidden_dim, mlp_dim, kernel_size=1)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(mlp_dim, hidden_dim, kernel_size=1)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, T, C)
        # Conv1d expects (B, C, T) so we transpose
        x = x.transpose(-1, -2)
        x = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(x)))))
        x = x.transpose(-1, -2)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with pre-LN architecture."""

    def __init__(self, hidden_dim: int, num_heads: int, mlp_dim: int, dropout_rate: float):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_dim, eps=1e-8)
        self.attn = CausalSelfAttention(hidden_dim, num_heads, dropout_rate)
        self.ln_2 = nn.LayerNorm(hidden_dim, eps=1e-8)
        self.ffn = PointWiseFeedForward(hidden_dim, mlp_dim, dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN: LayerNorm -> Sub-layer -> Residual
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffn(self.ln_2(x))
        return x


class SemanticSASRec(nn.Module):
    """Self-Attentive Sequential Recommendation with Semantic IDs."""

    def __init__(self, config: SemanticSASRecConfig):
        super().__init__()

        # Store config for later use
        self.config = config

        # Extract frequently used values
        self.vocab_size = config.vocab_size
        self.num_levels = config.num_levels
        self.codebook_size = config.codebook_size
        self.max_seq_length = config.max_seq_length
        self.max_token_length = config.max_seq_length * config.num_levels
        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim

        # Token embedding with input_dim
        self.token_emb = nn.Embedding(config.vocab_size + 1, config.input_dim, padding_idx=0)

        # Position embeddings with input_dim
        self.pos_emb = nn.Embedding(self.max_token_length + 1, config.input_dim, padding_idx=0)

        # Project from input_dim to hidden_dim
        self.input_projection = nn.Linear(config.input_dim, config.hidden_dim)
        self.emb_dropout = nn.Dropout(config.dropout_rate)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(config.hidden_dim, config.num_heads, config.mlp_dim, config.dropout_rate)
                for _ in range(config.num_blocks)
            ]
        )

        # Final layer norm
        self.ln_f = nn.LayerNorm(config.hidden_dim, eps=1e-8)

        # Level-specific prediction heads for sequential generation
        self.level_heads = nn.ModuleList(
            [nn.Linear(config.hidden_dim, config.codebook_size) for _ in range(config.num_levels)]
        )

        # Context combination layers for levels 1-3
        self.context_combiners = nn.ModuleList(
            [nn.Linear(config.hidden_dim * 2, config.hidden_dim) for _ in range(config.num_levels - 1)]
        )

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

    def level_to_token_id(self, level_value: int, level: int) -> int:
        """Convert a level value and level index to token ID (+1: id 0 is reserved for padding)."""
        return level * self.codebook_size + level_value + 1

    def token_id_to_level_value(self, token_id: torch.Tensor) -> torch.Tensor:
        """Extract the level value from a token ID (removes the +1 padding shift and the level offset)."""
        return (token_id - 1) % self.codebook_size

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for encoding token sequences.

        Args:
            input_ids: Token sequences [batch_size, seq_length * num_levels]

        Returns:
            Hidden states [batch_size, seq_length * num_levels, hidden_dim]
        """
        batch_size, seq_length = input_ids.size()

        # Get token embeddings
        token_embs = self.token_emb(input_ids)
        token_embs *= self.input_dim**0.5  # Scale by sqrt(d_input)

        # Add positional embeddings
        positions = torch.arange(1, seq_length + 1, dtype=torch.long, device=input_ids.device)
        positions = positions.unsqueeze(0).expand(batch_size, -1)
        # Mask positions where input is padding
        positions = positions * (input_ids != 0).long()
        pos_embs = self.pos_emb(positions)

        # Combine embeddings and project to hidden dimension
        embeddings = self.emb_dropout(token_embs + pos_embs)
        hidden_states = self.input_projection(embeddings)

        # Pass through transformer blocks
        for block in self.blocks:
            hidden_states = block(hidden_states)

        # Final layer norm
        hidden_states = self.ln_f(hidden_states)

        return hidden_states

    def predict_next_item(
        self,
        input_ids: torch.Tensor,
        teacher_forcing: bool = True,
        target_tokens: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict the next item's semantic ID tokens sequentially.

        Args:
            input_ids: Token sequences [batch_size, seq_length * num_levels]
            teacher_forcing: Whether to use ground truth for conditioning
            target_tokens: Ground truth tokens for next item [batch_size, num_levels]

        Returns:
            Dictionary with logits for each level
        """
        # Encode the input sequence
        hidden_states = self.forward(input_ids)  # [B, T*L, H]

        # Get the representation at the last position
        # This represents the context after seeing all previous items
        last_hidden = hidden_states[:, -1, :]  # [B, H]

        predictions = {}

        # Sequential generation: predict each level conditioned on previous
        for level in range(self.num_levels):
            if level == 0:
                # Level 0: predict directly from sequence representation
                context = last_hidden
            else:
                # Levels 1-3: condition on previously predicted/true tokens
                if teacher_forcing and target_tokens is not None:
                    # Use ground truth previous levels during training
                    prev_tokens = target_tokens[:, :level]  # [B, level]
                else:
                    # Use predicted tokens during inference
                    prev_tokens = self._sample_from_predictions(predictions, level)

                # Encode previous level tokens
                prev_embeds = self.token_emb(prev_tokens)  # [B, level, input_dim]
                # Project to hidden dimension and average pool
                prev_embeds_projected = self.input_projection(prev_embeds)  # [B, level, H]
                prev_context = prev_embeds_projected.mean(dim=1)  # [B, H]

                # Combine with sequence context
                combined = torch.cat([last_hidden, prev_context], dim=-1)  # [B, 2*H]
                context = self.context_combiners[level - 1](combined)  # [B, H]

            # Predict current level
            logits = self.level_heads[level](context)  # [B, codebook_size]
            predictions[f"logits_l{level}"] = logits

        return predictions

    def _sample_from_predictions(self, predictions: Dict[str, torch.Tensor], up_to_level: int) -> torch.Tensor:
        """Sample tokens from predictions for levels 0 to up_to_level-1."""
        sampled_tokens = []

        for level in range(up_to_level):
            logits = predictions[f"logits_l{level}"]
            # Greedy sampling (could use top-k or nucleus sampling)
            tokens = torch.argmax(logits, dim=-1)  # [B]
            # Add level offset to get actual token IDs
            tokens = self.level_to_token_id(tokens, level)
            sampled_tokens.append(tokens)

        return torch.stack(sampled_tokens, dim=1)  # [B, up_to_level]

    def score_candidates(self, input_ids: torch.Tensor, candidate_tokens_batch: torch.Tensor) -> torch.Tensor:
        """
        Fast batch evaluation for ranking candidates.
        Processes all candidates in a single forward pass.

        Args:
            input_ids: Token sequences [batch_size, seq_length * num_levels]
            candidate_tokens_batch: Pre-tokenized candidates [batch_size, num_candidates, num_levels]

        Returns:
            scores: Log probability scores [batch_size, num_candidates]
        """
        batch_size, num_candidates, _ = candidate_tokens_batch.shape

        # 1. Encode sequences once
        hidden_states = self.forward(input_ids)  # [B, T*L, H]
        last_hidden = hidden_states[:, -1, :]  # [B, H]

        # 2. Flatten candidates for batch processing
        flat_candidates = candidate_tokens_batch.view(-1, self.num_levels)  # [B*C, num_levels]

        # 3. Compute scores for all levels
        all_log_probs = []

        # Level 0: Direct scoring from sequence representation
        # Expand last_hidden to match flattened candidates
        context_l0 = last_hidden.unsqueeze(1).expand(-1, num_candidates, -1).reshape(-1, self.hidden_dim)
        logits_l0 = self.level_heads[0](context_l0)  # [B*C, codebook_size]

        # Get target tokens for level 0
        targets_l0 = self.token_id_to_level_value(flat_candidates[:, 0])
        log_probs_l0 = F.log_softmax(logits_l0, dim=-1)
        scores_l0 = log_probs_l0.gather(1, targets_l0.unsqueeze(1)).squeeze(1)
        all_log_probs.append(scores_l0.view(batch_size, num_candidates))

        # Levels 1-3: Condition on previous levels
        for level in range(1, self.num_levels):
            # Get embeddings for all previous levels
            prev_tokens = flat_candidates[:, :level]  # [B*C, level]
            prev_embeds = self.token_emb(prev_tokens)  # [B*C, level, input_dim]
            # Project to hidden dimension and average pool
            prev_embeds_projected = self.input_projection(prev_embeds)  # [B*C, level, H]
            prev_context = prev_embeds_projected.mean(dim=1)  # [B*C, H]

            # Combine with sequence context
            expanded_last_hidden = last_hidden.unsqueeze(1).expand(-1, num_candidates, -1).reshape(-1, self.hidden_dim)
            combined = torch.cat([expanded_last_hidden, prev_context], dim=-1)  # [B*C, 2*H]
            context = self.context_combiners[level - 1](combined)  # [B*C, H]

            # Get logits for current level
            logits = self.level_heads[level](context)  # [B*C, codebook_size]

            # Score the target tokens
            targets = self.token_id_to_level_value(flat_candidates[:, level])
            log_probs = F.log_softmax(logits, dim=-1)
            scores = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            all_log_probs.append(scores.view(batch_size, num_candidates))

        # 4. Sum log probabilities across all levels
        total_scores = torch.stack(all_log_probs, dim=-1).sum(dim=-1)  # [B, C]

        return total_scores

    def training_step(
        self,
        input_ids: torch.Tensor,
        pos_tokens: torch.Tensor,
        neg_tokens: torch.Tensor,
        teacher_forcing_ratio: float = 0.9,
    ) -> Dict[str, torch.Tensor]:
        """
        Training step with positive and negative semantic ID tokens.

        Args:
            input_ids: Input token sequences [batch_size, seq_length * num_levels]
            pos_tokens: Positive next item tokens [batch_size, num_levels]
            neg_tokens: Negative item tokens [batch_size, num_levels]
            teacher_forcing_ratio: Probability of using teacher forcing

        Returns:
            Dictionary with losses for each level
        """
        use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio

        # Predict next item tokens
        pos_predictions = self.predict_next_item(
            input_ids, teacher_forcing=use_teacher_forcing, target_tokens=pos_tokens
        )
        neg_predictions = self.predict_next_item(
            input_ids, teacher_forcing=use_teacher_forcing, target_tokens=neg_tokens
        )

        losses = {}

        # Calculate loss for each level
        for level in range(self.num_levels):
            pos_logits = pos_predictions[f"logits_l{level}"]  # [B, codebook_size]
            neg_logits = neg_predictions[f"logits_l{level}"]  # [B, codebook_size]

            # Extract the correct token for this level (remove offset)
            pos_target = self.token_id_to_level_value(pos_tokens[:, level])
            neg_target = self.token_id_to_level_value(neg_tokens[:, level])

            # Cross-entropy loss for positive tokens
            pos_loss = F.cross_entropy(pos_logits, pos_target)

            # For negative samples, we want low probability for the negative token
            # We can use the negative log-likelihood of NOT predicting the negative token
            neg_probs = F.softmax(neg_logits, dim=-1)
            neg_token_probs = neg_probs.gather(1, neg_target.unsqueeze(1)).squeeze(1)
            neg_loss = -torch.log(1 - neg_token_probs + 1e-8).mean()

            losses[f"loss_l{level}"] = pos_loss + neg_loss

        return losses


class SemanticSequenceDataset(Dataset):
    """Dataset for semantic ID sequences with pre-tokenization."""

    def __init__(self, config: SemanticSASRecConfig):
        """
        Load semantic ID sequences from parquet file.

        File format: parquet with columns [user_id, semantic_id_sequence, semantic_id_sequence_length]
        where semantic_id_sequence is a list of semantic IDs like:
        ["<|sid_start|><|sid_112|><|sid_291|><|sid_570|><|sid_768|><|sid_end|>", "<|sid_start|><|sid_92|><|sid_448|><|sid_572|><|sid_768|><|sid_end|>", ...]
        """
        self.config = config
        self.max_seq_length = config.max_seq_length
        self.num_levels = config.num_levels
        self.codebook_size = config.codebook_size
        self.max_token_length = config.max_seq_length * config.num_levels

        # Load data from parquet
        logger.info(f"Loading data from {config.data_path}")
        df = pl.read_parquet(str(config.data_path))

        # Extract user sequences
        self.users = df["user_id"].to_list()
        semantic_sequences = df["semantic_id_sequence"].to_list()

        logger.info("Pre-tokenizing all sequences...")

        # Create user token sequences directly
        self.user_seq_tokens = {}
        self.user_seq_lengths = {}  # Store original sequence lengths for reference

        for user, seq in zip(self.users, semantic_sequences):
            # Create flattened token sequence
            token_seq = []
            for sid in seq:
                tokens = encode_semantic_id(sid, self.num_levels, self.codebook_size)
                token_seq.extend(tokens)
            self.user_seq_tokens[user] = token_seq
            self.user_seq_lengths[user] = len(seq)

        self.num_users = len(self.users)

        # Filter users with too few interactions
        valid_users = []
        for u in self.users:
            if self.user_seq_lengths[u] >= 3:  # Need at least 3 items
                valid_users.append(u)
        self.users = valid_users

        logger.info(f"Loaded {self.num_users:,} users")
        logger.info(f"After filtering: {len(self.users):,} users with >= 3 interactions")

        # Compute average sequence length
        avg_seq_len = np.mean([self.user_seq_lengths[u] for u in self.users])
        logger.info(f"Average sequence length: {avg_seq_len:.2f} items")

        # Create candidate pool for evaluation
        self._create_candidate_pool()

    def _create_candidate_pool(self):
        """Create a pool of unique item token sequences for negative sampling during evaluation."""
        # Collect all unique item token sequences
        unique_items = set()
        for user in self.users:
            token_seq = self.user_seq_tokens[user]
            # Each item is num_levels tokens
            for i in range(0, len(token_seq), self.num_levels):
                item_tokens = tuple(token_seq[i : i + self.num_levels])
                unique_items.add(item_tokens)

        # Convert to list for indexing
        self.candidate_pool = list(unique_items)
        self.num_unique_items = len(self.candidate_pool)

        # Create reverse mapping from item tuples to indices for O(1) lookups
        self.item_to_index = {item: idx for idx, item in enumerate(self.candidate_pool)}

        logger.info(f"Created candidate pool with {self.num_unique_items:,} unique items")

    def get_user_seen_indices(self, user):
        """Get set of candidate pool indices for items seen by a user. Uses O(1) lookups via the item_to_index mapping."""
        seen_indices = set()
        token_seq = self.user_seq_tokens[user]
        for i in range(0, len(token_seq), self.num_levels):
            item_tokens = tuple(token_seq[i : i + self.num_levels])
            # O(1) lookup instead of O(n) search
            if item_tokens in self.item_to_index:
                seen_indices.add(self.item_to_index[item_tokens])
        return seen_indices

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = self.users[idx]
        token_seq = self.user_seq_tokens[user]  # Pre-tokenized sequence
        seq_length = self.user_seq_lengths[user]  # Original sequence length

        return user, token_seq, seq_length


class SemanticTrainingDataset(Dataset):
    """Training dataset that returns pre-computed training samples."""

    def __init__(self, base_dataset: SemanticSequenceDataset):
        self.base_dataset = base_dataset
        self.max_seq_length = base_dataset.max_seq_length
        self.num_levels = base_dataset.num_levels
        self.max_token_length = base_dataset.max_token_length

        # Pre-compute valid training sequences. The last two items of every user are held out
        # (evaluate() uses item -2 as the val target and item -1 as the test target), so a training
        # target may be at most item -3: end_idx in [2, seq_length - 3].
        self.training_samples = []
        for user in base_dataset.users:
            seq_length = base_dataset.user_seq_lengths[user]
            if seq_length > 4:  # Need >= 2 history items + 1 target + 2 held-out items
                for end_idx in range(2, seq_length - 2):
                    self.training_samples.append((user, end_idx))

        logger.info(f"Created {len(self.training_samples):,} training samples")

    def __len__(self):
        return len(self.training_samples)

    def __getitem__(self, idx):
        user, end_idx = self.training_samples[idx]

        # Calculate actual sequence length for this sample
        train_seq_length = min(end_idx, self.max_seq_length)

        # Get pre-tokenized sequence
        full_token_seq = self.base_dataset.user_seq_tokens[user]
        start_token_idx = max(0, (end_idx - train_seq_length) * self.num_levels)
        end_token_idx = end_idx * self.num_levels
        input_tokens = full_token_seq[start_token_idx:end_token_idx]

        # Positive item tokens (next item)
        pos_token_start = end_idx * self.num_levels
        pos_token_end = pos_token_start + self.num_levels
        pos_tokens = full_token_seq[pos_token_start:pos_token_end]

        return {"input_tokens": input_tokens, "pos_tokens": pos_tokens, "seq_length": train_seq_length, "user": user}


def collate_semantic_batch(samples):
    """Collate function for semantic training data with in-batch negative sampling."""
    batch_size = len(samples)
    max_token_length = max(len(s["input_tokens"]) for s in samples)
    num_levels = len(samples[0]["pos_tokens"])

    # Pre-allocate tensors
    input_ids = torch.zeros((batch_size, max_token_length), dtype=torch.long)
    pos_tokens = torch.zeros((batch_size, num_levels), dtype=torch.long)
    neg_tokens = torch.zeros((batch_size, num_levels), dtype=torch.long)

    # Collect positive tokens for in-batch sampling
    positive_token_lists = []

    for i, sample in enumerate(samples):
        token_len = len(sample["input_tokens"])
        input_ids[i, -token_len:] = torch.tensor(sample["input_tokens"])
        pos_tokens[i] = torch.tensor(sample["pos_tokens"])

        # Track positive tokens for negative sampling
        positive_token_lists.append(sample["pos_tokens"])

    # In-batch negative sampling
    for i in range(batch_size):
        # Get valid negatives: other samples' positives
        valid_neg_indices = [j for j in range(batch_size) if j != i]

        if valid_neg_indices:
            # Randomly sample a negative from valid candidates
            neg_idx = np.random.choice(valid_neg_indices)
            neg_tokens[i] = torch.tensor(positive_token_lists[neg_idx])
        else:
            # Fallback: if batch size is 1, use own positive shifted
            # This shouldn't happen in practice with reasonable batch sizes
            neg_tokens[i] = pos_tokens[i]

    return {"input_ids": input_ids, "pos_tokens": pos_tokens, "neg_tokens": neg_tokens}


def evaluate(
    model: SemanticSASRec,
    dataset: SemanticSequenceDataset,
    mode: str = "val",
    batch_size: int = 512,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Evaluate model on validation or test set.

    Args:
        model: Trained SemanticSASRec model
        dataset: SemanticSequenceDataset instance
        mode: 'val' or 'test'
        batch_size: Batch size for evaluation
        device: Device to run evaluation on

    Returns:
        Dictionary with NDCG@10 and HR@10 metrics
    """
    model.eval()

    # Pre-allocate metrics arrays for efficiency
    all_ranks = []

    # Process users in batches with progress bar
    users = dataset.users
    num_batches = (len(users) + batch_size - 1) // batch_size

    # Create progress bar for evaluation
    pbar = tqdm(
        range(0, len(users), batch_size), total=num_batches, desc=f"Evaluating ({mode})", unit="batch", leave=False
    )

    for batch_start in pbar:
        batch_end = min(batch_start + batch_size, len(users))
        batch_users = users[batch_start:batch_end]

        # Prepare batch data with vectorized operations
        batch_data = []
        for user in batch_users:
            seq_length = dataset.user_seq_lengths[user]
            token_seq = dataset.user_seq_tokens[user]

            # Determine input/target split based on mode
            if mode == "val":
                if seq_length < 3:
                    continue
                split_idx = seq_length - 2
            else:  # test
                if seq_length < 2:
                    continue
                split_idx = seq_length - 1

            # Extract input and target tokens
            input_end = split_idx * dataset.num_levels
            input_tokens = token_seq[:input_end]
            target_start = split_idx * dataset.num_levels
            target_tokens = token_seq[target_start : target_start + dataset.num_levels]

            batch_data.append({"user": user, "input_tokens": input_tokens, "target_tokens": target_tokens})

        if not batch_data:
            continue

        # Create input tensor batch
        max_len = min(max(len(d["input_tokens"]) for d in batch_data), dataset.max_token_length)

        input_tensor = torch.zeros((len(batch_data), max_len), dtype=torch.long, device=device)
        for i, data in enumerate(batch_data):
            tokens = data["input_tokens"]
            if len(tokens) > max_len:
                tokens = tokens[-max_len:]  # Keep most recent
            input_tensor[i, -len(tokens) :] = torch.tensor(tokens, dtype=torch.long, device=device)

        # Pre-compute seen items mapping for batch (using fast O(1) lookups)
        batch_seen_indices = []
        for data in batch_data:
            seen_indices = dataset.get_user_seen_indices(data["user"])
            batch_seen_indices.append(seen_indices)

        # Batch candidate generation and scoring
        with torch.no_grad():
            batch_candidates = []

            for i, data in enumerate(batch_data):
                target_tokens = data["target_tokens"]
                seen_indices = batch_seen_indices[i]

                # Create candidate list: target + 100 negatives
                candidates = [target_tokens]

                # Efficient negative sampling
                valid_indices = list(set(range(len(dataset.candidate_pool))) - seen_indices)

                if len(valid_indices) >= 100:
                    neg_indices = np.random.choice(valid_indices, size=100, replace=False)
                else:
                    neg_indices = np.random.choice(valid_indices, size=100, replace=True)

                candidates.extend([dataset.candidate_pool[idx] for idx in neg_indices])
                batch_candidates.append(candidates)

            # Convert all candidates to tensor at once (more efficient)
            max_candidates = max(len(c) for c in batch_candidates)
            candidate_tensor = torch.zeros(
                (len(batch_data), max_candidates, dataset.num_levels), dtype=torch.long, device=device
            )

            # Pre-convert to numpy for faster tensor creation
            for i, candidates in enumerate(batch_candidates):
                if candidates:
                    # Convert list of tuples to numpy array first, then to tensor
                    candidates_array = np.array(candidates, dtype=np.int64)
                    candidate_tensor[i, : len(candidates)] = torch.from_numpy(candidates_array).to(device)

            # Score all candidates in batch
            scores = model.score_candidates(input_tensor, candidate_tensor)

            # Calculate ranks for each user
            for i in range(len(batch_data)):
                # Target is always at position 0
                user_scores = scores[i, : len(batch_candidates[i])]
                rank = (user_scores[0] < user_scores).sum().item() + 1
                all_ranks.append(rank)

        # Clean up batch tensors to free memory
        del input_tensor, candidate_tensor, scores
        if device == "cuda":
            torch.cuda.empty_cache()

        # Update progress bar with current metrics
        pbar.set_postfix(
            {"users_evaluated": len(all_ranks), "avg_rank": f"{np.mean(all_ranks) if all_ranks else 0:.1f}"}
        )

    # Calculate metrics using vectorized operations
    ranks = np.array(all_ranks)
    valid_users = len(ranks)

    if valid_users == 0:
        return {"ndcg@10": 0.0, "hr@10": 0.0}

    # HR@10: percentage of ranks <= 10
    hr_10 = (ranks <= 10).mean()

    # NDCG@10: normalized discounted cumulative gain
    ndcg_scores = np.where(ranks <= 10, 1.0 / np.log2(ranks + 1), 0.0)
    ndcg_10 = ndcg_scores.mean()

    logger.info(f"Evaluated on {valid_users:,} users")

    # Final cleanup
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return {"ndcg@10": ndcg_10, "hr@10": hr_10}


def get_gradient_norm(model: nn.Module) -> float:
    """Calculate the L2 norm of gradients across all model parameters."""
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads:
        return 0.0

    # Compute norm without materializing concatenated tensor
    total_norm = torch.norm(torch.stack([torch.norm(g, 2) for g in grads]), 2)
    return total_norm.item()


def train_semantic_sasrec(
    model: SemanticSASRec, train_dataset: SemanticSequenceDataset, config: SemanticSASRecConfig, device: str = "cpu"
) -> Dict[str, float]:
    """
    Train SemanticSASRec model.

    Args:
        model: SemanticSASRec model to train
        train_dataset: Training dataset
        config: Training configuration
        device: Device to train on

    Returns:
        Dictionary with best validation metrics
    """
    model = model.to(device)

    # Create training dataset and DataLoader
    training_dataset = SemanticTrainingDataset(train_dataset)

    # Use DataLoader with multiple workers
    train_loader = DataLoader(
        training_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4 if device != "mps" else 0,  # MPS doesn't support multi-process data loading
        pin_memory=(device == "cuda"),
        prefetch_factor=2 if device != "mps" else None,
        persistent_workers=False,  # Disabled to prevent memory accumulation
        collate_fn=collate_semantic_batch,
    )

    steps_per_epoch = len(train_loader)
    total_steps = config.num_epochs * steps_per_epoch

    logger.info(f"Training for {config.num_epochs} epochs, {steps_per_epoch} steps per epoch")
    logger.info(f"Total training steps: {total_steps:,}")
    logger.info(f"DataLoader workers: {train_loader.num_workers}")

    # Optimizer with fused support
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device == "cuda" and config.fused_adam
    if use_fused:
        logger.info("Using fused AdamW optimizer")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.max_learning_rate,
        betas=(0.9, 0.98),  # As in original SASRec
        weight_decay=config.weight_decay,
        fused=use_fused,
    )

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

    # Track best model
    best_val_metrics = {"ndcg@10": 0.0}
    global_step = 0

    # Calculate total steps for progress bar
    steps_per_epoch = len(train_loader)
    total_training_steps = config.num_epochs * steps_per_epoch

    # Create progress bar for entire training (all epochs)
    pbar = tqdm(total=total_training_steps)

    # Training loop
    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0

        for batch_idx, batch in enumerate(train_loader):
            # Accumulate gradients
            if batch_idx % config.gradient_accumulation_steps == 0:
                t0 = time.time()
                optimizer.zero_grad()
                loss_accum = 0.0

            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            pos_tokens = batch["pos_tokens"].to(device)
            neg_tokens = batch["neg_tokens"].to(device)

            # Forward pass
            losses = model.training_step(input_ids, pos_tokens, neg_tokens, config.teacher_forcing_ratio)

            # Combine losses from all levels
            total_loss = sum(losses.values()) / len(losses)

            # Add L2 regularization for embeddings
            if config.l2_emb > 0:
                emb_loss = config.l2_emb * model.token_emb.weight.norm(2)
                total_loss = total_loss + emb_loss

            # Scale loss for gradient accumulation
            total_loss = total_loss / config.gradient_accumulation_steps
            loss_accum += total_loss.item() * config.gradient_accumulation_steps

            # Backward pass
            total_loss.backward()

            # Optimizer step after accumulating gradients
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                # Get gradient norm before clipping
                grad_norm_before = get_gradient_norm(model)

                # Gradient clipping
                if config.use_gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
                    grad_norm_after = get_gradient_norm(model)
                else:
                    grad_norm_after = grad_norm_before

                # Optimizer step
                optimizer.step()

                # Step the learning rate scheduler
                if scheduler is not None:
                    scheduler.step()

                # Get current learning rate for logging
                current_lr = optimizer.param_groups[0]["lr"]

                # Update metrics
                epoch_loss += loss_accum
                epoch_steps += 1
                global_step += 1
                pbar.update(1)  # Update progress bar by one step

                # Time measurement
                t1 = time.time()
                batch_time_ms = (t1 - t0) * 1000
                samples_per_second = (config.batch_size * config.gradient_accumulation_steps) / (t1 - t0)

                # Logging
                if global_step == 1 or global_step % config.log_interval == 0:
                    log_str = (
                        f"Step {global_step:06d} | Epoch {epoch + 1:03d}/{config.num_epochs:03d} | "
                        f"Loss: {loss_accum:.4f} | "
                        f"LR: {current_lr:.2e} | "
                        f"Grad norm: {grad_norm_before:.2f}"
                    )
                    if config.use_gradient_clipping and grad_norm_before != grad_norm_after:
                        log_str += f" -> {grad_norm_after:.2f}"
                    log_str += f" | Time: {batch_time_ms:.0f}ms | Samples/s: {samples_per_second:,.0f}"

                    logger.info(log_str)

                    # Log to W&B
                    wandb_log = {
                        "loss/total": loss_accum,
                        "train/learning_rate": current_lr,
                        "train/gradient_norm": grad_norm_before,
                        "train/batch_time_ms": batch_time_ms,
                        "train/samples_per_second": samples_per_second,
                        "progress/epoch": epoch + 1,
                        "progress/step": global_step,
                    }

                    # Log individual level losses
                    for level in range(config.num_levels):
                        if f"loss_l{level}" in losses:
                            wandb_log[f"loss/level_{level}"] = losses[f"loss_l{level}"].item()

                    wandb.log(wandb_log)

                # Validation
                if global_step % config.val_interval == 0:
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
                    if val_metrics["ndcg@10"] > best_val_metrics.get("ndcg@10", 0.0):
                        best_val_metrics = val_metrics.copy()  # Make a copy before cleanup
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
                                },
                                best_path,
                            )
                            logger.info(f"Saved best model with NDCG@10: {val_metrics['ndcg@10']:.4f}")

                            # Save as W&B artifact
                            artifact = wandb.Artifact(
                                f"semantic-sasrec-best-{config.dataset}",
                                type="model",
                                metadata=val_metrics,
                            )
                            artifact.add_file(str(best_path))
                            wandb.log_artifact(artifact)

                    # Clean up memory after evaluation
                    del val_metrics
                    gc.collect()
                    if device == "cuda":
                        torch.cuda.empty_cache()

        # Periodic checkpoint
        if config.checkpoint_dir and (epoch + 1) % config.checkpoint_interval == 0:
            checkpoint_path = config.checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                },
                checkpoint_path,
            )
            logger.info(f"Saved checkpoint to {checkpoint_path}")

            # Clean up memory after checkpointing
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

    pbar.close()  # Close the progress bar
    logger.info(f"Training completed. Best validation NDCG@10: {best_val_metrics.get('ndcg@10', 0.0):.4f}")

    return best_val_metrics


if __name__ == "__main__":
    config = SemanticSASRecConfig()

    device_manager = DeviceManager(logger)
    device = device_manager.device

    run_name = f"semantic-sasrec-{config.dataset}-L{config.num_blocks}-H{config.num_heads}-D{config.hidden_dim}"
    run = wandb.init(project="sasrec-experiments", name=run_name, config=config.__dict__)
    config.log_config()

    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    dataset = SemanticSequenceDataset(config)

    model = SemanticSASRec(config)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    if device == "cuda" and config.use_compile:
        logger.info("Compiling model with torch.compile for faster training...")
        model = torch.compile(model)

    best_metrics = train_semantic_sasrec(model=model, train_dataset=dataset, config=config, device=device)

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
        },
        final_path,
    )

    artifact = wandb.Artifact(
        f"semantic-sasrec-final-{config.dataset}",
        type="model",
        metadata={
            "val_ndcg@10": best_metrics.get("ndcg@10", 0.0),
            "val_hr@10": best_metrics.get("hr@10", 0.0),
            "test_ndcg@10": test_metrics["ndcg@10"],
            "test_hr@10": test_metrics["hr@10"],
        },
    )
    artifact.add_file(str(final_path))
    wandb.log_artifact(artifact)

    wandb.finish()
    logger.info("Training complete!")
