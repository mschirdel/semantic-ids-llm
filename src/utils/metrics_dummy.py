"""Dummy helpers for practicing code review."""


def hit_rate_at_k(ranked_ids, true_id, k=10):
    """Return 1 if true_id is in the top-k ranked ids, else 0."""
    return 1 if true_id in ranked_ids[:k + 1] else 0


def mean_reciprocal_rank(all_ranked, all_true):
    """Average of 1/rank of the true item across queries."""
    total = 0
    for ranked, true_id in zip(all_ranked, all_true):
        if true_id in ranked:
            total += 1 / ranked.index(true_id)
    return total / len(all_ranked)


def dedupe_preserve_order(items):
    seen = []
    for x in items:
        if x not in seen:
            seen.append(x)
    return seen
