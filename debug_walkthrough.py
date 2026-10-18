#!/usr/bin/env python3
"""
Step-by-step debug walkthrough of the whole semantic-ID pipeline on a tiny toy dataset.

Every step is an independent function that reads the previous steps' output from data/debug/,
so you can run (or debug) one at a time, in order:

    1   prep data             notebooks 01-05   raw records -> item_context + user sequences
    2   embed items           embed_items.py    text -> 1024-d vectors (Qwen3-Embedding-0.6B)
    3   RQ-VAE                train_rqvae.py    vectors -> 3 hierarchical codes
    4   semantic IDs          notebooks 06-07   codes -> unique SIDs, sequences -> SIDs, train/val split
    5   extend tokenizer      finetune_..._vocab.py   add <|rec|>, <|sid_start|>, <|sid_end|>, <|sid_0..1023|>
    6   LLM training data     notebook 08       Type A-F instruction data -> chat conversations
    7   prepare model         finetune_..._vocab.py   resize embeddings, freeze, one loss + backward (no step)
    8   stage-1 mini training finetune_..._vocab.py   ~300 real optimizer steps on a TINY Qwen3
    9   inference             demo.ipynb        prompt -> generate -> SIDs -> titles (+ constrained decoding)
    10  SASRec baseline       train_sasrec.py   item-ID sequential recommender
    11  SASRec on SIDs        train_sasrec_semantic_id.py   same task, hierarchical SID tokens
    12  compare               notebooks 09-10   Hit@K / NDCG@K side by side

    uv run python debug_walkthrough.py 3          # run one step
    uv run python debug_walkthrough.py all        # run 1..12
    uv run python debug_walkthrough.py 3 --pause  # drop into pdb at every pause() point

In VS Code use the "Walkthrough: step N" launch configs and set breakpoints anywhere.
Nothing needs a GPU. Step 8 trains a ~20M-param random Qwen3 (same architecture as Qwen3-8B,
just small); the full Qwen3-8B stage 1 / stage 2 finetuning is NOT run.
Optional: `9 --real-mapping` downloads the author's public item mapping; `9 --real-model` runs the
author's finetuned 8B model (needs ~16GB of memory).
"""

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

DEBUG_DIR = Path("data") / "debug"
EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
LLM_NAME = "Qwen/Qwen3-8B"  # only the tokenizer + config are downloaded
DEVICE = "cpu"

# Real pipeline: 3 RQ-VAE levels x 256 codes (+ 1 collision code). Toy: 3 levels x 8 codes.
# The token vocabulary still uses a stride of 256 per level, exactly like the real thing.
TOY_CODEBOOK_SIZE = 8
SID_STRIDE = 256
NUM_SID_TOKENS = 1024  # <|sid_0|> .. <|sid_1023|>

# Notebook 08's system prompt (the one the training conversations use)
SYSTEM_PROMPT = (
    "You are a helpful AI assistant that understands and works with semantic IDs for product recommendations. "
    "Semantic IDs are hierarchical identifiers in the format "
    "<|sid_start|><|sid_0|><|sid_256|><|sid_512|><|sid_768|><|sid_end|> "
    "that encode product relationships and categories."
)

PAUSE = False
REAL_MAPPING = False
REAL_MODEL = False

# ----------------------------------------------------------------------------
# Toy catalogue: 4 themes x 6 items (so similar items should end up sharing SID prefixes)
# ----------------------------------------------------------------------------
ITEMS = [
    # racing
    ("Mario Kart 8 Deluxe - Nintendo Switch", "Kart racing with Mario characters, items, drifting and local multiplayer."),
    ("Need for Speed Heat - PlayStation 4", "Open-world street racing by day and night, car tuning and police chases."),
    ("Forza Horizon 5 - Xbox Series X", "Open-world racing festival in Mexico with hundreds of licensed cars."),
    ("Sonic & All-Stars Racing Transformed - PS3", "Arcade kart racer where cars transform on land, sea and air."),
    ("Gran Turismo 7 - PlayStation 5", "Realistic driving simulator with a huge car collection and tuning."),
    ("Burnout Paradise Remastered - Xbox One", "Arcade racing with crashes, stunts and an open city to explore."),
    # role playing
    ("Dragon Quest Heroes II - PlayStation 4", "Action RPG with party combat, monsters and a classic fantasy story."),
    ("Nights of Azure - PlayStation 4", "Action RPG with demon hunting, character progression and fast combat."),
    ("Final Fantasy X/X-2 HD Remaster - PS Vita", "Japanese RPG with turn-based battles, summons and an epic story."),
    ("The Witcher 3: Wild Hunt - Xbox One", "Open-world fantasy RPG about a monster hunter, quests and choices."),
    ("Persona 5 Royal - PlayStation 4", "Turn-based RPG about student thieves, social links and dungeons."),
    ("Fire Emblem: Three Houses - Nintendo Switch", "Tactical RPG with classroom life, army battles and branching story."),
    # cute / animals / kids
    ("Animal Crossing: New Leaf - Nintendo 3DS", "Cute life simulation where you run a village of animal neighbors."),
    ("Nintendogs + Cats - Nintendo 3DS", "Raise and play with adorable puppies and kittens on your handheld."),
    ("Disney Infinity Starter Pack - Xbox 360", "Toy-box adventure with Disney characters, figures and sandbox play."),
    ("Kirby's Epic Yarn - Nintendo Wii", "Charming yarn-themed platformer starring Kirby, easy and family friendly."),
    ("Yoshi's Crafted World - Nintendo Switch", "Cute craft-themed platformer with Yoshi, collectibles and co-op."),
    ("Pokemon Let's Go Pikachu - Nintendo Switch", "Catch and train adorable Pokemon with a simple family-friendly adventure."),
    # shooters / action
    ("Halo 4 - Xbox 360", "Sci-fi first-person shooter starring Master Chief with campaign and multiplayer."),
    ("Call of Duty: Modern Warfare - PlayStation 4", "Military first-person shooter with intense campaign and online play."),
    ("Mass Effect - Xbox 360", "Sci-fi action RPG shooter with squad combat and space exploration."),
    ("DOOM Eternal - PC", "Fast, brutal first-person shooter against demons with upgrades."),
    ("Star Wars Battlefront II - Xbox One", "Sci-fi multiplayer shooter in the Star Wars universe."),
    ("Fallout: New Vegas - Ultimate Edition", "Post-apocalyptic open-world shooter RPG with factions and choices."),
]
THEME_NAMES = ["racing", "rpg", "cute", "shooter"]
THEMES = [t for t in THEME_NAMES for _ in range(6)]
THEME_GENRE = {"racing": "Racing", "rpg": "RPG", "cute": "Family, Simulation", "shooter": "Shooter"}
THEME_BLURB = {
    "racing": "Includes multiple tracks, vehicle upgrades and online leaderboards for competitive play.",
    "rpg": "Features a large world, deep character customization and dozens of hours of content.",
    "cute": "A relaxing, colorful experience that the whole family can enjoy together at home.",
    "shooter": "Includes a story campaign, weapon customization and several online multiplayer modes.",
}
THEME_FEATURES = {
    "racing": ["Multiple race modes", "Online multiplayer"],
    "rpg": ["Large open world", "Deep character progression"],
    "cute": ["Family friendly", "Easy to pick up and play"],
    "shooter": ["Campaign and multiplayer", "Weapon customization"],
}
PLATFORM_WORDS = ["Nintendo Switch", "Nintendo 3DS", "Nintendo Wii", "PlayStation 4", "PlayStation 5", "PS Vita", "PS3",
                  "Xbox Series X", "Xbox One", "Xbox 360", "PC"]

# Items with bad metadata (the real data has plenty): they appear in user sequences but get filtered out.
JUNK_ITEMS = [
    ("B0JUNK0001", "Controller", ["Black."]),  # title too short, description too short
    ("B0JUNK0002", "Unknown Bundle Pack Misc Edition", []),  # no description
]
MIN_TITLE_CHARS = 10  # real notebook 01 uses 20; toy titles like "DOOM Eternal - PC" are shorter
MIN_DESC_CHARS = 100
MIN_SEQ_LEN = 3
MAX_SEQ_LEN = 100


def banner(text: str):
    print("\n" + "=" * 100 + f"\n{text}\n" + "=" * 100)


def pause(label: str):
    """Breakpoint helper: only stops when --pause is passed (or you set a breakpoint yourself)."""
    print(f"\n[pause] {label}")
    if PAUSE:
        breakpoint()  # inspect any local variable, then `c` to continue


def sid_token(level: int, code: int) -> str:
    return f"<|sid_{level * SID_STRIDE + code}|>"


def sid_string(codes: list[int]) -> str:
    return "<|sid_start|>" + "".join(sid_token(l, c) for l, c in enumerate(codes)) + "<|sid_end|>"


def save_json(obj, name: str):
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    (DEBUG_DIR / name).write_text(json.dumps(obj, indent=1))


def load_json(name: str):
    return json.loads((DEBUG_DIR / name).read_text())


def load_items() -> list[dict]:
    return load_json("items.json")


def load_rqvae_inputs() -> torch.Tensor:
    """Embeddings -> RQ-VAE input.

    TOY-ONLY preprocessing: raw Qwen embeddings share a big common component (all cosines ~0.7),
    so with just 24 items the toy RQ-VAE collapses every item onto ONE code. Mean-centring and
    rescaling removes that. The real pipeline feeds raw embeddings; with ~66k items the
    reconstruction loss gives the codebooks enough signal to spread out on their own.
    """
    x = torch.load(DEBUG_DIR / "embeddings.pt")["embeddings"]
    return (x - x.mean(dim=0, keepdim=True)) / x.std()


def toy_rqvae_config():
    from src.train_rqvae import RQVAEConfig

    return RQVAEConfig(
        item_embedding_dim=1024,
        codebook_embedding_dim=32,
        codebook_quantization_levels=3,
        codebook_size=TOY_CODEBOOK_SIZE,
    )


# ----------------------------------------------------------------------------
# Step 1: raw records -> items + sequences   (notebooks 01-05)
# ----------------------------------------------------------------------------
TITLE_CLEAN_PROMPT = """Gemini, you are a data cleaning expert who rewrites product titles to be clear, concise, and well-formatted.
... (10 numbered rules: remove promo text, fix capitalisation, keep brand/model, <200 chars ...)
Product title to clean:
<title>
{title}
</title>
Return ONLY the cleaned title within <clean_title> tags."""


def build_raw_items() -> list[dict]:
    """Records shaped like Amazon Reviews 2023 `meta_Video_Games.jsonl` (description/features are LISTS)."""
    raw = []
    for i, (title, desc) in enumerate(ITEMS):
        theme = THEMES[i]
        raw.append({
            "parent_asin": f"B0TOY{i:04d}", "title": title, "theme": theme,
            "description": [desc, THEME_BLURB[theme]], "features": THEME_FEATURES[theme],
            "categories": ["Video Games", THEME_GENRE[theme].split(",")[0]], "main_category": "Video Games",
            "store": "Toy Store", "average_rating": 4.5, "rating_number": 100 + i, "price": "29.99",
        })
    for asin, title, desc in JUNK_ITEMS:
        raw.append({
            "parent_asin": asin, "title": title, "theme": None, "description": desc, "features": [],
            "categories": [], "main_category": "Video Games", "store": "", "average_rating": None,
            "rating_number": None, "price": None,
        })
    return raw


def build_item_context(r: dict) -> str:
    """Notebook 05 cell 16: title + description + category + (Gemini-extracted) metadata."""
    platform = next((p for p in PLATFORM_WORDS if p in r["title"]), "N/A")
    genre = THEME_GENRE[r["theme"]]
    return (
        f"Product: {r['title']}\nDescription: {r['description_text']}\nCategory: {r['categories_text']}"
        f"\nProduct Type: Game\nPlatform: {platform}\nGenre: {genre}\nBrand: Unknown\nMultiplayer: Unknown\nFranchise: N/A"
    )


def build_raw_sequences(rng: random.Random, all_asins: list[str]) -> list[dict]:
    """Rows shaped like the 5-core `timestamp_w_his` csv: user_id, history (space-joined), parent_asin (target).
    Users mostly follow a chain inside one theme (so there is something to learn), with some noise."""
    rows = []
    junk = [a for a, _, _ in JUNK_ITEMS]
    for n in range(800):
        theme = rng.randrange(4)
        idx = rng.randrange(6)
        length = rng.randint(4, 9)
        seq = []
        for _ in range(length):
            seq.append(f"B0TOY{theme * 6 + idx:04d}")
            r = rng.random()
            if r < 0.6:
                idx = (idx + 1) % 6  # next item in the theme "chain"
            elif r < 0.9:
                idx = rng.randrange(6)
            else:
                theme, idx = rng.randrange(4), rng.randrange(6)  # jump to another theme
            if rng.random() < 0.04:
                seq.append(rng.choice(junk))
        user = f"U{n:04d}"
        rows.append({"user_id": user, "history": " ".join(seq[:-1]), "parent_asin": seq[-1]})
        if n % 20 == 0 and len(seq) > 4:  # the real csv has several rows per user with growing histories
            rows.append({"user_id": user, "history": " ".join(seq[:-3]), "parent_asin": seq[-3]})
    for n in range(800, 815):  # very short users that end up too short once junk items are removed
        rows.append({"user_id": f"U{n:04d}", "history": f"B0TOY{rng.randrange(24):04d} {rng.choice(junk)}",
                     "parent_asin": rng.choice(junk)})
    return rows


def step1_prep_data():
    banner("STEP 1: raw Amazon-style records -> items (with item_context) + user sequences")
    rng = random.Random(0)

    # 1a. Items. Notebook 01: join list fields to text, then keep only items with a real title + description.
    raw = build_raw_items()
    for r in raw:
        r["description_text"] = " ".join(r["description"])
        r["features_text"] = " ".join(r["features"])
        r["categories_text"] = " > ".join(r["categories"]) or r["main_category"]
    print(f"raw items: {len(raw)}; description is a LIST: {raw[0]['description']}")
    kept = [r for r in raw if len(r["title"]) > MIN_TITLE_CHARS and len(r["description_text"]) > MIN_DESC_CHARS]
    dropped = [r["parent_asin"] for r in raw if r not in kept]
    print(f"kept {len(kept)} items with title > {MIN_TITLE_CHARS} chars and description > {MIN_DESC_CHARS} chars; dropped {dropped}")
    pause("raw / kept: compare what the filters removed")

    # 1b. Notebooks 02-04 (NOT run): Gemini rewrites titles and descriptions and extracts platform/genre/brand.
    print("\nNotebooks 02-04 call the Gemini API for every item. The title-cleaning prompt looks like:\n")
    print(TITLE_CLEAN_PROMPT.format(title=kept[0]["title"]))
    print("\n(skipped here: the toy text is already clean, and platform/genre come from the title / theme)")

    # 1c. Notebook 05: build the text that will be embedded
    for r in kept:
        r["item_context"] = build_item_context(r)
    print("\nitem_context for item 0 (this exact string is what step 2 embeds):\n" + kept[0]["item_context"])
    save_json([{k: r[k] for k in ("parent_asin", "title", "theme", "description_text", "features_text",
                                  "categories_text", "main_category", "item_context")} for r in kept], "items.json")

    # 1d. Sequences. Notebook 01: history + target -> one list per user
    rows = build_raw_sequences(rng, [r["parent_asin"] for r in raw])
    print(f"\nraw sequence rows: {len(rows)}  (example: {rows[0]})")
    best = {}
    for r in rows:  # keep the row with the longest history for each user
        n = len(r["history"].split())
        if r["user_id"] not in best or n > len(best[r["user_id"]]["history"].split()):
            best[r["user_id"]] = r
    print(f"after de-duplicating users (keep longest history): {len(best)} users")
    valid = {r["parent_asin"] for r in kept}
    seqs = {}
    n_removed_items = 0
    for u, r in best.items():
        seq = r["history"].split() + [r["parent_asin"]]
        filtered = [a for a in seq if a in valid]  # drop items without usable metadata
        n_removed_items += len(seq) - len(filtered)
        if len(filtered) >= MIN_SEQ_LEN:
            seqs[u] = filtered[-MAX_SEQ_LEN:]  # keep the most recent items
    print(f"removed {n_removed_items} junk-item occurrences; users with >= {MIN_SEQ_LEN} items left: {len(seqs)}")
    lens = [len(s) for s in seqs.values()]
    print(f"sequence length: min {min(lens)}, max {max(lens)}, mean {sum(lens) / len(lens):.1f}")
    print("example:", next(iter(seqs.items())))
    save_json(seqs, "sequences.json")
    print(f"\nSaved items.json, sequences.json -> {DEBUG_DIR}")


# ----------------------------------------------------------------------------
# Step 2: item text -> embedding   (src/tokenize_items.py + src/embed_items.py)
# ----------------------------------------------------------------------------
def step2_embed():
    from transformers import AutoModel, AutoTokenizer

    from src.embed_items import generate_embeddings, last_token_pool
    from src.tokenize_items import get_detailed_instruct

    banner("STEP 2: embed item_context with Qwen3-Embedding-0.6B")
    items = load_items()

    task = "Given a product description, generate a semantic embedding that captures its key features and characteristics"
    instructed = [get_detailed_instruct(task, it["item_context"]) for it in items]
    print("Text that actually goes into the model (item 0):\n" + instructed[0])

    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
    enc = tokenizer(instructed, padding=True, truncation=True, max_length=512, return_tensors="pt")
    print(f"\npadding_side={tokenizer.padding_side}  input_ids={tuple(enc['input_ids'].shape)}")
    pause("tokenized batch: inspect enc['input_ids'] / enc['attention_mask']")

    model = AutoModel.from_pretrained(EMBED_MODEL).to(DEVICE).eval()
    with torch.no_grad():
        hidden = model(**{k: v.to(DEVICE) for k, v in enc.items()}).last_hidden_state
    print(f"last_hidden_state: {tuple(hidden.shape)}  (batch, tokens, hidden)")

    # Pool: last REAL token's hidden state, truncate to 1024, L2-normalise (dot product == cosine)
    pooled = last_token_pool(hidden, enc["attention_mask"].to(DEVICE))
    emb = F.normalize(pooled[:, :1024], p=2, dim=1)
    print(f"pooled: {tuple(pooled.shape)}; norms after L2-normalise: {emb.norm(dim=1)[:3].tolist()}")
    pause("pooled / emb: one 1024-d vector per item")

    ref = generate_embeddings(model, DEVICE, {k: v for k, v in enc.items()}, target_dim=1024)
    print("matches src.embed_items.generate_embeddings:", torch.allclose(emb.cpu(), torch.from_numpy(ref), atol=1e-4))

    sims = emb @ emb.T
    print("\nNearest neighbour of each item (cosine):")
    for i, it in enumerate(items):
        s = sims[i].clone()
        s[i] = -1
        j = int(s.argmax())
        print(f"  [{it['theme']:7s}] {it['title'][:42]:42s} -> {items[j]['title'][:42]:42s} ({s[j]:.3f}) [{items[j]['theme']}]")

    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"asins": [it["parent_asin"] for it in items], "embeddings": emb.cpu()}, DEBUG_DIR / "embeddings.pt")
    print(f"\nSaved {tuple(emb.shape)} embeddings -> {DEBUG_DIR / 'embeddings.pt'}")


# ----------------------------------------------------------------------------
# Step 3: embeddings -> RQ-VAE   (src/train_rqvae.py)
# ----------------------------------------------------------------------------
def step3_rqvae():
    from torch.utils.data import DataLoader, TensorDataset

    from src.train_rqvae import RQVAE

    banner("STEP 3: RQ-VAE (encoder -> 3 residual quantizers -> decoder)")
    torch.manual_seed(0)
    items = load_items()
    x = load_rqvae_inputs()
    model = RQVAE(toy_rqvae_config())
    print(model)
    pause("model built: model.encoder, model.vq_layers[i].embedding.weight (codebooks), model.decoder")

    # K-means init: put each level's codebook at cluster centres of the (residual) latents
    model.kmeans_init(DataLoader(TensorDataset(x), batch_size=len(x)), DEVICE)

    # Tiny training loop (the real one adds a scheduler, validation, W&B and dead-code resets).
    # Loss = reconstruction MSE + codebook loss + 0.25 * commitment loss, per level.
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    for step in range(1, 401):
        x_recon, indices, ld = model(x)
        opt.zero_grad()
        ld["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step in (1, 50, 100, 200, 300, 400):
            uniq = model.calculate_unique_ids_proportion(torch.stack(indices, dim=-1))
            usage = [f"{u:.2f}" for u in model.calculate_codebook_usage()]
            print(
                f"step {step:3d} loss={ld['loss'].item():.4f} recon={ld['recon_loss'].item():.4f} "
                f"vq={ld['vq_loss'].item():.4f} unique_ids={uniq:.2f} codebook_usage/level={usage}"
            )

    # Walk ONE item through the quantizers by hand, so you can see the residual shrink.
    model.eval()
    item = 0
    with torch.no_grad():
        z = model.encode(x[item : item + 1])
        print(f"\nItem {item}: {items[item]['title']}")
        print(f"latent z: shape {tuple(z.shape)}, norm {z.norm().item():.3f}")
        residual, total, chosen = z, torch.zeros_like(z), []
        for level, vq in enumerate(model.vq_layers):
            dists = torch.cdist(residual, vq.embedding.weight)
            idx, quantized = vq.find_nearest_codes(residual)
            chosen.append(int(idx))
            print(f"\n level {level}: distances to codes = {[round(d, 2) for d in dists[0].tolist()]}")
            print(f"          nearest code = {int(idx)}   residual norm {residual.norm():.3f} -> ", end="")
            residual = residual - quantized
            total = total + quantized
            print(f"{residual.norm():.3f}")
            pause(f"level {level} done: residual, quantized, total")
        recon = model.decode(total)
        cos = F.cosine_similarity(recon, x[item : item + 1]).item()
        print(f"\ncodes by hand = {chosen}; decode(sum of 3 code vectors) vs original embedding cosine = {cos:.3f}")
        assert chosen == model.encode_to_semantic_ids(x[item : item + 1])[0].tolist()
        print("matches model.encode_to_semantic_ids ✓")

    torch.save(model.state_dict(), DEBUG_DIR / "rqvae.pt")
    print(f"\nSaved -> {DEBUG_DIR / 'rqvae.pt'}")


# ----------------------------------------------------------------------------
# Step 4: RQ-VAE codes -> final semantic IDs; sequences -> SIDs; split   (notebooks 06-07)
# ----------------------------------------------------------------------------
def step4_semantic_ids():
    import polars as pl

    from src.train_rqvae import RQVAE

    banner("STEP 4: collect semantic IDs (3 RQ-VAE codes + collision code -> SID tokens)")
    items = load_items()
    x = load_rqvae_inputs()
    model = RQVAE(toy_rqvae_config())
    model.load_state_dict(torch.load(DEBUG_DIR / "rqvae.pt"))
    model.eval()

    codes = model.encode_to_semantic_ids(x).tolist()
    pause("codes: list of [c0, c1, c2] per item")

    # Items sharing all 3 codes get a 4th code 0, 1, 2... in original order, so every full ID is unique.
    seen = Counter()
    full = []
    for c in codes:
        key = tuple(c)
        full.append(list(key) + [seen[key]])
        seen[key] += 1
    n_collide = sum(1 for v in seen.values() if v > 1)
    print(f"{len(seen)} unique 3-code IDs out of {len(codes)} items; {n_collide} colliding groups")
    assert len({tuple(f) for f in full}) == len(full)
    if n_collide == 0:
        print("(no collisions in this toy run, so every 4th code is 0. With 66k real items many items collide;")
        print(" try TOY_CODEBOOK_SIZE = 2 and rerun steps 3-4 to force collisions and watch the 4th code count up.)")

    # Level L, code c  ->  <|sid_{L*256 + c}|>
    table = {}
    print(f"\n{'theme':8s} {'title':44s} codes        sid string")
    for i in sorted(range(len(codes)), key=lambda i: full[i]):
        it = items[i]
        s = sid_string(full[i])
        table[it["parent_asin"]] = {"title": it["title"], "codes": full[i], "sid": s, "theme": it["theme"]}
        print(f"{it['theme']:8s} {it['title'][:42]:44s} {str(full[i]):12s} {s}")
    pause("table: sorted by code, so items with the same first code(s) sit together")

    print("\nLevel-0 code per theme (similar items should share it):")
    for theme in THEME_NAMES:
        print(f"  {theme:8s}", Counter(full[i][0] for i in range(len(full)) if items[i]["theme"] == theme).most_common())
    with torch.no_grad():
        recon = model.decode_from_semantic_ids(torch.tensor(codes))
    print(f"mean cosine(decode(codes), original embedding) = {F.cosine_similarity(recon, x).mean():.3f}")
    save_json(table, "semantic_ids.json")

    # Notebook 07: swap every ASIN in every user sequence for its SID, then split users 90/10.
    seqs = load_json("sequences.json")
    rows = [
        {"user_id": u, "sequence": s, "sequence_length": len(s),
         "semantic_id_sequence": [table[a]["sid"] for a in s], "semantic_id_sequence_length": len(s)}
        for u, s in seqs.items()
    ]
    print(f"\nexample sequence: {rows[0]['sequence']}\n  as SIDs: {rows[0]['semantic_id_sequence'][:2]} ...")
    random.Random(42).shuffle(rows)
    n_val = int(len(rows) * 0.1)
    val, train = rows[:n_val], rows[n_val:]
    pl.DataFrame(train).write_parquet(DEBUG_DIR / "sequences_train.parquet")
    pl.DataFrame(val).write_parquet(DEBUG_DIR / "sequences_val.parquet")
    print(f"split users: {len(train)} train / {len(val)} val  -> sequences_{{train,val}}.parquet")


# ----------------------------------------------------------------------------
# Step 5: teach the tokenizer about SID tokens   (finetune_qwen3_8b_vocab.py: extend_tokenizer)
# ----------------------------------------------------------------------------
def new_token_list() -> list[str]:
    return ["<|rec|>", "<|sid_start|>", "<|sid_end|>"] + [f"<|sid_{i}|>" for i in range(NUM_SID_TOKENS)]


def step5_tokenizer():
    from transformers import AutoConfig, AutoTokenizer

    banner("STEP 5: extend the Qwen3-8B tokenizer with <|rec|>, <|sid_start|>, <|sid_end|>, <|sid_0..1023|>")
    table = load_json("semantic_ids.json")
    one_sid = next(iter(table.values()))["sid"]

    tok = AutoTokenizer.from_pretrained(LLM_NAME)
    n_before = len(tok)
    emb_rows = AutoConfig.from_pretrained(LLM_NAME).vocab_size
    print(f"len(tokenizer)={n_before:,}  but the model's embedding matrix has {emb_rows:,} rows")
    print("  (HF pads the matrix; extend_tokenizer() first shrinks the matrix to len(tokenizer) so the two line up)")

    before = tok.tokenize(one_sid)
    print(f"\nBEFORE: {one_sid}\n  -> {len(before)} tokens: {before}")

    # Unsloth's add_new_tokens() == tokenizer.add_tokens + resize embeddings + mean-init the new rows.
    # Here we do the tokenizer half (step 7 does the model half).
    n_added = tok.add_tokens(new_token_list(), special_tokens=True)
    print(f"\nadded {n_added} tokens; len(tokenizer) {n_before:,} -> {len(tok):,}")
    pause("tok is extended: try tok.convert_tokens_to_ids('<|sid_768|>')")

    after = tok.tokenize(one_sid)
    ids = tok.convert_tokens_to_ids(after)
    print(f"\nAFTER:  -> {len(after)} tokens: {after}")
    print(f"        ids: {ids}   (new ids start at {n_before:,})")
    assert tok.convert_tokens_to_ids("<|sid_0|>") == n_before + 3  # after the 3 special tokens

    sids = [v["sid"] for v in table.values()]
    convo = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"User's last purchases: {sids[0]}, {sids[1]}. Next:\n<|rec|>"},
        {"role": "assistant", "content": sids[2]},
    ]
    text = tok.apply_chat_template(convo, tokenize=False, add_generation_prompt=False, enable_thinking=False)
    print("\nChat-template text:\n" + text)
    print("per-token view:", [tok.decode([i]) for i in tok(text)["input_ids"]])

    tok.save_pretrained(DEBUG_DIR / "tokenizer")
    print(f"\nSaved extended tokenizer -> {DEBUG_DIR / 'tokenizer'}")


# ----------------------------------------------------------------------------
# Step 6: build the LLM training data   (notebook 08)
# ----------------------------------------------------------------------------
def gen_type_a(rows):
    """SemanticID -> text."""
    out = []
    for r in rows:
        sid = r["semantic_id"]
        out.append({"instruction": f"Product {sid} has title:", "output": r["title"], "type": "sid_to_title"})
        if len(r["description_text"]) > 50:
            out.append({"instruction": f"Item {sid} is described as:", "output": r["description_text"], "type": "sid_to_description"})
        if r["categories_text"]:
            out.append({"instruction": f"Product {sid} belongs to category:", "output": r["categories_text"], "type": "sid_to_category"})
        out.append({"instruction": f"Tell me about product {sid}:",
                    "output": f"Title: {r['title']} Description: {r['description_text']} Category: {r['main_category']}", "type": "sid_to_context"})
        if len(r["features_text"]) > 20:
            out.append({"instruction": f"Product {sid} features:", "output": r["features_text"], "type": "sid_to_features"})
    return out


def gen_type_b(rows):
    """Text -> SemanticID."""
    out = []
    for r in rows:
        sid, words = r["semantic_id"], r["title"].split()
        out.append({"instruction": f'The product "{r["title"]}" has SemanticID:', "output": sid, "type": "title_to_sid"})
        if len(words) >= 3:
            out.append({"instruction": f'A product titled "{" ".join(words)}" has SemanticID:', "output": sid, "type": "title_prefix_to_sid"})
        if len(words) > 6:
            s = len(words) // 2 - 2
            out.append({"instruction": f'A product with "...{" ".join(words[s:s + 4])}..." in its title has SemanticID:', "output": sid, "type": "title_contains_to_sid"})
        d = r["description_text"]
        if len(d) > 250:
            out.append({"instruction": f'A product described as "{d[:250]}..." has SemanticID:', "output": sid, "type": "description_to_sid"})
        for i in (50, 100, 200):
            if len(d) > i + 50 and len(d[i:i + 50].strip()) > 10:
                out.append({"instruction": f'A product with description containing "...{d[i:i + 50]}..." has SemanticID:', "output": sid, "type": "description_contains_to_sid"})
        if len(r["features_text"]) > 20:
            out.append({"instruction": f'A {r["main_category"]} product with features "{r["features_text"]}" has SemanticID:', "output": sid, "type": "features_to_sid"})
    return out


def gen_type_c(seq_rows):
    """User history -> next item. Every split point of every sequence."""
    out = []
    for seq in seq_rows:
        for sp in range(2, len(seq)):
            hist, target = seq[:sp], seq[sp]
            if len(hist) >= 3:
                out.append({"instruction": f"Based on recent purchases: {', '.join(hist[-3:])}, next item:", "output": target, "type": "seq_last_3"})
            if len(hist) >= 5:
                out.append({"instruction": f"Purchase sequence: {', '.join(hist[-5:])}. Predict next:", "output": target, "type": "seq_last_5"})
            out.append({"instruction": f"User's last purchases: {', '.join(hist[-2:])}. Next:", "output": target, "type": "seq_last_2"})
    return out


def gen_type_d(rows):
    """Prefix (hierarchy) understanding. NB: notebook 08 writes these prefixes as '<sid><SID_n>' (an older format that is
    not in the vocabulary); here they use the real <|sid_n|> tokens."""
    out = []
    g0 = defaultdict(list)
    g01 = defaultdict(list)
    for r in rows:
        g0[r["codes"][0]].append(r)
        g01[(r["codes"][0], r["codes"][1])].append(r)
    for c0, grp in g0.items():
        if len(grp) >= 5:
            cat = Counter(r["main_category"] for r in grp).most_common(1)[0][0]
            out.append({"instruction": f"Products starting with <|sid_start|>{sid_token(0, c0)} are typically:", "output": f"{cat} products", "type": "prefix_category"})
            out.append({"instruction": f"Examples of products with prefix <|sid_start|>{sid_token(0, c0)}:", "output": ", ".join(r["semantic_id"] for r in grp[:3]), "type": "prefix_examples"})
    for (c0, c1), grp in g01.items():
        pref = f"<|sid_start|>{sid_token(0, c0)}{sid_token(1, c1)}"
        if len(grp) >= 3:
            out.append({"instruction": f"Products starting with {pref} are:", "output": f"{grp[0]['main_category']} items", "type": "prefix_2level_category"})
        if len(grp) > 1:
            for i, base in enumerate(grp):
                others = [o["semantic_id"] for j, o in enumerate(grp) if j != i]
                out.append({"instruction": f"Products similar to {base['semantic_id']} include:", "output": ", ".join(others[:5]), "type": "similar_products"})
    return out


def gen_type_e(rows, seq_rows, max_users=100):
    """Co-purchase patterns. (Capped to `max_users` sequences here; the notebook uses all of them.)"""
    info = {r["semantic_id"]: r for r in rows}
    out = []
    for seq in seq_rows[:max_users]:
        for i in range(len(seq) - 1):
            for j in range(i + 1, min(i + 5, len(seq))):
                a, b = seq[i], seq[j]
                out.append({"instruction": f"A user who bought {a} ({info[a]['title']}) might also buy:", "output": b, "type": "copurchase_forward"})
                out.append({"instruction": f"Before buying {b} ({info[b]['title']}), users often buy:", "output": a, "type": "copurchase_backward"})
    trans = Counter()
    for seq in seq_rows:
        cats = [info[s]["main_category"] for s in seq]
        trans.update(zip(cats[:-1], cats[1:]))
    for (c1, c2), n in trans.items():
        if n >= 5:
            out.append({"instruction": f"After buying from {c1}, users often buy from:", "output": c2, "type": "category_transition"})
    return out


def gen_type_f(seq_rows):
    out = []
    for length in (3, 5, 10, 20):
        for seq in [s for s in seq_rows if len(s) >= length][:10]:
            out.append({"instruction": f"Example of a {length}+ item purchase sequence:", "output": ", ".join(seq[:length]), "type": "sequence_length_example"})
    pairs = Counter()
    for seq in seq_rows:
        for i in range(len(seq) - 1):
            for j in range(i + 1, min(i + 3, len(seq))):
                pairs[tuple(sorted([seq[i], seq[j]]))] += 1
    for (a, b), n in pairs.most_common(500):
        if n >= 3:
            out.append({"instruction": f"Items frequently bought together: {a} and:", "output": b, "type": "frequently_bought_together"})
    return out


def step6_llm_data():
    import polars as pl
    from transformers import AutoTokenizer

    banner("STEP 6: build the LLM training conversations (Types A-F)")
    table = load_json("semantic_ids.json")
    items = {it["parent_asin"]: it for it in load_items()}
    rows = [{**items[a], "semantic_id": t["sid"], "codes": t["codes"]} for a, t in table.items()]
    train_seqs = pl.read_parquet(DEBUG_DIR / "sequences_train.parquet")["semantic_id_sequence"].to_list()
    print(f"{len(rows)} items with SIDs, {len(train_seqs)} training sequences (validation users are NOT used to build data)")

    types = {
        "A  SID -> text": gen_type_a(rows), "B  text -> SID": gen_type_b(rows), "C  history -> next item": gen_type_c(train_seqs),
        "D  prefix / hierarchy": gen_type_d(rows), "E  co-purchase": gen_type_e(rows, train_seqs), "F  patterns": gen_type_f(train_seqs),
    }
    for name, samples in types.items():
        print(f"\n--- Type {name}: {len(samples):,} samples; subtypes: {dict(Counter(s['type'] for s in samples).most_common(4))}")
        s = random.Random(1).choice(samples)
        print(f"  IN : {s['instruction'][:160]}\n  OUT: {s['output'][:160]}")
    pause("types: dict of lists; each sample is {instruction, output, type}")

    # Wrap each sample as a 3-turn chat (system / user / assistant). No de-duplication on purpose.
    all_samples = [s for v in types.values() for s in v]
    convs = [{"conversations": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": s["instruction"]},
                                {"role": "assistant", "content": s["output"]}], "type": s["type"]} for s in all_samples]
    random.Random(42).shuffle(convs)
    n_train = int(0.95 * len(convs))
    train, val = convs[:n_train], convs[n_train:]
    print(f"\ntotal conversations: {len(convs):,}  -> train {len(train):,} / val {len(val):,}")

    # Token lengths per sample type, using the extended tokenizer from step 5
    tok = AutoTokenizer.from_pretrained(DEBUG_DIR / "tokenizer")
    by_type = defaultdict(list)
    for c in train:
        if len(by_type[c["type"]]) < 20:
            text = tok.apply_chat_template(c["conversations"], tokenize=False, enable_thinking=False)
            by_type[c["type"]].append(len(tok(text)["input_ids"]))
    print("\navg tokens per conversation (20 per subtype):")
    for t, ls in sorted(by_type.items()):
        print(f"  {t:28s} {sum(ls) / len(ls):6.0f}")

    n_sid_notation = sum("<sid>" in s["output"] for s in all_samples)
    print(f"\noutputs containing the old '<sid>' notation: {n_sid_notation}. Notebook 08's validator only inspects outputs "
          f"containing '<sid>', so with the current <|sid_n|> format it checks nothing.")
    save_json(train, "llm_train.json")
    save_json(val, "llm_val.json")
    print(f"Saved llm_train.json / llm_val.json -> {DEBUG_DIR}")


# ----------------------------------------------------------------------------
# Step 7: prepare the model for stage-1 training, without training   (extend_tokenizer + prepare_model)
# ----------------------------------------------------------------------------
def build_stage1_model(tok, verbose: bool = False):
    """Tiny Qwen3 (real vocab size, untied lm_head, 2 small layers) prepared exactly like stage 1:
    shrink padded matrix -> grow to extended tokenizer -> mean-init new rows -> freeze all but embeddings + lm_head."""
    from transformers import AutoConfig, AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

    orig_len = len(AutoTokenizer.from_pretrained(LLM_NAME))
    padded_rows = AutoConfig.from_pretrained(LLM_NAME).vocab_size
    torch.manual_seed(0)
    cfg = Qwen3Config(
        vocab_size=padded_rows, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, tie_word_embeddings=False,
    )
    model = Qwen3ForCausalLM(cfg)
    emb, head = model.get_input_embeddings(), model.get_output_embeddings()
    if verbose:
        print(f"initial: embed_tokens {tuple(emb.weight.shape)}, lm_head {tuple(head.weight.shape)}")
    model.resize_token_embeddings(orig_len)
    model.resize_token_embeddings(len(tok))
    emb, head = model.get_input_embeddings(), model.get_output_embeddings()
    if verbose:
        print(f"after resize: embed_tokens {tuple(emb.weight.shape)}, lm_head {tuple(head.weight.shape)}, new rows = {len(tok) - orig_len}")
    with torch.no_grad():  # mean init, as Unsloth's add_new_tokens does per the script's comment
        for w in (emb.weight, head.weight):
            w[orig_len:] = w[:orig_len].mean(dim=0, keepdim=True)
    for p in model.parameters():
        p.requires_grad = False
    emb.weight.requires_grad = True
    head.weight.requires_grad = True
    return model, orig_len


def step7_prepare_model():
    from transformers import AutoTokenizer

    banner("STEP 7: resize embeddings, freeze everything but embeddings, compute one loss (no training)")
    tok = AutoTokenizer.from_pretrained(DEBUG_DIR / "tokenizer")
    model, old_rows = build_stage1_model(tok, verbose=True)
    emb, head = model.get_input_embeddings(), model.get_output_embeddings()
    a, b = emb.weight[old_rows], emb.weight[old_rows + 500]
    print(f"cosine(two different new-token rows) = {F.cosine_similarity(a, b, dim=0).item():.3f}  (identical rows)")
    pause("model resized + mean-initialised + frozen")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\ntrainable params: {trainable:,} / {total:,} ({100 * trainable / total:.1f}%) [tiny model]")
    print(f"for Qwen3-8B (hidden 4096): 2 x {len(tok):,} x 4096 = {2 * len(tok) * 4096:,} trainable params")
    print("(whole matrices are unfrozen, not only the new rows: the old rows get updated too)")

    # One real training example from step 6
    conv = next(c for c in load_json("llm_train.json") if c["type"] == "seq_last_3")
    text = tok.apply_chat_template(conv["conversations"], tokenize=False, add_generation_prompt=False, enable_thinking=False)
    ids = tok(text, return_tensors="pt")["input_ids"]
    print(f"\nexample ({conv['type']}): {ids.shape[1]} tokens, {int((ids >= old_rows).sum())} of them are new tokens")
    out = model(input_ids=ids, labels=ids)  # SFT on the chat text: labels = input_ids (loss over every token)
    print(f"loss on random tiny model = {out.loss.item():.2f}  (ln(vocab) = {torch.log(torch.tensor(float(len(tok)))).item():.2f})")
    out.loss.backward()  # one backward pass to SEE the gradients; we never call optimizer.step() in this step
    pause("out.loss.backward() done: inspect model.*.grad")

    print("\nParams with gradients:", [n for n, p in model.named_parameters() if p.grad is not None])
    print(f"embed_tokens rows with gradient: {int((emb.weight.grad.abs().sum(dim=1) > 0).sum())}  "
          f"(only the {len(set(ids[0].tolist()))} unique tokens in the example)")
    print(f"lm_head rows with gradient:      {int((head.weight.grad.abs().sum(dim=1) > 0).sum()):,}  (softmax touches every vocabulary row)")


# ----------------------------------------------------------------------------
# Step 8: stage-1 mini training on the tiny model   (finetune_qwen3_8b_vocab.py: train_embeddings, scaled down)
# ----------------------------------------------------------------------------
STAGE1_STEPS = 300
STAGE1_BATCH = 4
STAGE1_LR = 1e-2  # real stage 1 uses 1e-3 on the 8B; the toy model is tiny, so go faster
STAGE1_MAX_LEN = 192


def encode_convs(tok, convs):
    texts = [tok.apply_chat_template(c["conversations"], tokenize=False, add_generation_prompt=False, enable_thinking=False) for c in convs]
    enc = tok(texts, padding=True, truncation=True, max_length=STAGE1_MAX_LEN, return_tensors="pt")
    labels = enc["input_ids"].clone()
    labels[enc["attention_mask"] == 0] = -100
    return enc["input_ids"], enc["attention_mask"], labels


def mean_pairwise_cosine(w: torch.Tensor) -> float:
    n = w.shape[0]
    wn = F.normalize(w, dim=1)
    return float(((wn.sum(0).norm() ** 2 - n) / (n * (n - 1))))


@torch.no_grad()
def eval_stage1(model, tok, convs, old_rows):
    model.eval()
    tot, n, sid_tot, sid_n, sid_hit = 0.0, 0, 0.0, 0, 0
    for i in range(0, len(convs), 8):
        ids, mask, labels = encode_convs(tok, convs[i:i + 8])
        logits = model(input_ids=ids, attention_mask=mask).logits[:, :-1]
        tgt = labels[:, 1:]
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), ignore_index=-100, reduction="none").view_as(tgt)
        valid = tgt != -100
        is_sid = valid & (tgt >= old_rows + 3)  # <|sid_k|> tokens
        tot += ce[valid].sum().item(); n += int(valid.sum())
        sid_tot += ce[is_sid].sum().item(); sid_n += int(is_sid.sum())
        sid_hit += int((logits.argmax(-1)[is_sid] == tgt[is_sid]).sum())
    model.train()
    return tot / n, sid_tot / max(sid_n, 1), sid_hit / max(sid_n, 1)


def step8_stage1_train():
    from transformers import AutoTokenizer

    banner("STEP 8: stage-1 mini training (embeddings + lm_head only) on a tiny Qwen3")
    tok = AutoTokenizer.from_pretrained(DEBUG_DIR / "tokenizer")
    model, old_rows = build_stage1_model(tok)
    emb, head = model.get_input_embeddings(), model.get_output_embeddings()

    # A balanced sample: up to 40 conversations per subtype, so rare types are not drowned out by Type C / E
    train_all, per_type, train = load_json("llm_train.json"), Counter(), []
    random.Random(0).shuffle(train_all)
    for c in train_all:
        if per_type[c["type"]] < 40:
            per_type[c["type"]] += 1
            train.append(c)
    val = load_json("llm_val.json")[:96]
    print(f"train conversations: {len(train)} (from {len(train_all):,}), val: {len(val)}")

    opt = torch.optim.AdamW([emb.weight, head.weight], lr=STAGE1_LR, weight_decay=0.01)
    new_rows_before = emb.weight[old_rows + 3:old_rows + 3 + NUM_SID_TOKENS].detach().clone()
    old_before = emb.weight[:old_rows].detach().clone()
    l, sl, acc = eval_stage1(model, tok, val, old_rows)
    print(f"\nBEFORE: val loss {l:.2f} | loss on <|sid_k|> tokens {sl:.2f} | top-1 acc on sid tokens {acc:.3f}")
    print(f"        mean pairwise cosine between the 1024 sid rows = {mean_pairwise_cosine(new_rows_before):.3f}  (1.0 = all identical)")
    pause("before training: model, opt, train")

    model.train()
    rng = random.Random(0)
    for step in range(1, STAGE1_STEPS + 1):
        batch = rng.sample(train, STAGE1_BATCH)
        ids, mask, labels = encode_convs(tok, batch)
        loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
        loss.backward()
        opt.step()
        opt.zero_grad()
        if step in (1, 25, 50, 100, 150, 200, 250, STAGE1_STEPS):
            print(f"step {step:3d} train loss {loss.item():.3f}")

    l, sl, acc = eval_stage1(model, tok, val, old_rows)
    print(f"\nAFTER:  val loss {l:.2f} | loss on <|sid_k|> tokens {sl:.2f} | top-1 acc on sid tokens {acc:.3f}")
    new_rows_after = emb.weight[old_rows + 3:old_rows + 3 + NUM_SID_TOKENS].detach()
    print(f"        mean pairwise cosine between the 1024 sid rows = {mean_pairwise_cosine(new_rows_after):.3f}  (they have moved apart)")
    print(f"        mean |change| per old-vocab row = {(emb.weight[:old_rows].detach() - old_before).abs().mean():.5f} "
          f"vs new rows {(new_rows_after - new_rows_before).abs().mean():.5f}")
    print("\nThe transformer layers stayed frozen and random, so this tiny model can't truly reason about items, and much of that")
    print("accuracy is easy tokens (the 4th code is almost always <|sid_768|>, the format is rigid, and the 24-item catalogue is")
    print("small). What you can see is the mechanism: the new rows start identical, get pulled apart by gradients, and the")
    print("loss on SID tokens falls.")
    print("Stage 2 (finetune_qwen3_8b_full.py, not covered here) then unfreezes the whole model.")
    torch.save(model.state_dict(), DEBUG_DIR / "tiny_stage1.pt")
    print(f"Saved -> {DEBUG_DIR / 'tiny_stage1.pt'}")


# ----------------------------------------------------------------------------
# Step 9: inference   (demo.ipynb)
# ----------------------------------------------------------------------------
def parse_sid_levels(sid: str) -> list[str]:
    return re.findall(r"<\|sid_\d+\|>", sid.replace("<|sid_start|>", "").replace("<|sid_end|>", ""))


def sid_to_titles(sid: str, mapping: list[dict]) -> dict:
    """demo.ipynb map_semantic_id_to_titles: exact 4-token match, else fall back to a 3/2/1-token prefix."""
    levels = parse_sid_levels(sid)
    exact = [m["title"] for m in mapping if m["semantic_id"] == sid]
    if exact:
        return {"match": "exact", "titles": exact}
    for depth in range(min(3, len(levels)), 0, -1):
        prefix = "<|sid_start|>" + "".join(levels[:depth])
        hits = [m["title"] for m in mapping if m["semantic_id"].startswith(prefix)]
        if hits:
            return {"match": f"prefix L{depth}", "titles": hits}
    return {"match": "none", "titles": []}


def replace_sids_with_titles(text: str, mapping: list[dict]) -> str:
    def repl(m):
        r = sid_to_titles(m.group(0), mapping)
        return f'"{r["titles"][0]}" ({r["match"]})' if r["titles"] else "[Unknown Item]"
    return re.sub(r"<\|sid_start\|>(?:<\|sid_\d+\|>)+<\|sid_end\|>", repl, text)


def build_sid_prefix_fn(tok, valid_sids: list[str], prompt_len: int):
    """Constrained decoding (NOT in this repo): only allow token sequences that spell a real item's SID.
    Enforces <|sid_start|> t0 t1 t2 t3 <|sid_end|> with every prefix present in the catalogue."""
    start, end, eos = tok.convert_tokens_to_ids("<|sid_start|>"), tok.convert_tokens_to_ids("<|sid_end|>"), tok.convert_tokens_to_ids("<|im_end|>")
    items = [tok.convert_tokens_to_ids(parse_sid_levels(s)) for s in valid_sids]

    def allowed(batch_id, input_ids):
        gen = input_ids[prompt_len:].tolist()
        if not gen:
            return [start]
        if end in gen:
            return [eos]
        body = gen[1:]  # after <|sid_start|>
        if len(body) == len(items[0]):
            return [end]
        return sorted({it[len(body)] for it in items if it[:len(body)] == body})

    return allowed


def step9_inference():
    from transformers import AutoTokenizer, Qwen3ForCausalLM

    banner("STEP 9: inference — prompt -> generate -> SIDs -> titles")
    table = load_json("semantic_ids.json")
    mapping = [{"semantic_id": v["sid"], "title": v["title"]} for v in table.values()]
    if REAL_MAPPING:
        from datasets import load_dataset

        ds = load_dataset("eugeneyan/video-games-semantic-ids-mapping", split="train")
        mapping = [{"semantic_id": r["semantic_id"], "title": r["title"]} for r in ds]
        print(f"loaded the author's real mapping: {len(mapping):,} items; columns {ds.column_names}")
    print(f"mapping rows: {len(mapping):,}; example: {mapping[0]}")

    # 9a. SID <-> title mapping with prefix fallback
    sid = mapping[0]["semantic_id"]
    print(f"\nexact:   {sid}\n  -> {sid_to_titles(sid, mapping)}")
    levels = parse_sid_levels(sid)
    fake = "<|sid_start|>" + "".join(levels[:3]) + "<|sid_999|><|sid_end|>"  # right 3-token prefix, wrong 4th code
    print(f"\nunknown 4th code: {fake}\n  -> {sid_to_titles(fake, mapping)}   (falls back to a prefix: similar items)")
    pause("mapping helpers: this is how the demo turns model output back into product names")

    # 9b. The prompt the model sees
    tok = AutoTokenizer.from_pretrained(DEBUG_DIR / "tokenizer")
    sids = [m["semantic_id"] for m in mapping[:3]]
    user_msg = f"Based on recent purchases: {', '.join(sids[:2])}, next item:\n<|rec|>"
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_msg}]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    print("\nprompt (add_generation_prompt=True leaves the assistant turn open):\n" + prompt)
    print("readable:", replace_sids_with_titles(user_msg, mapping))

    # 9c. Generate with the tiny stage-1 model: mechanics only, the content is not meaningful
    model, _ = build_stage1_model(tok)
    model.load_state_dict(torch.load(DEBUG_DIR / "tiny_stage1.pt"))
    model.eval()
    inputs = tok(prompt, return_tensors="pt")
    plen = inputs["input_ids"].shape[1]
    valid_sids = [v["sid"] for v in table.values()]
    gen_kwargs = dict(max_new_tokens=7, eos_token_id=tok.convert_tokens_to_ids("<|im_end|>"), pad_token_id=tok.pad_token_id)
    with torch.no_grad():
        greedy = model.generate(**inputs, do_sample=False, **gen_kwargs)
        print(f"\ngreedy: {tok.decode(greedy[0, plen:], skip_special_tokens=False)}")
        print("  ->", replace_sids_with_titles(tok.decode(greedy[0, plen:], skip_special_tokens=False), mapping))
        # Sampling 40 completions per mode shows what constrained decoding buys you
        for name, extra in (("free sampling", {}), ("constrained to real SIDs",
                            {"prefix_allowed_tokens_fn": build_sid_prefix_fn(tok, valid_sids, plen)})):
            torch.manual_seed(0)
            out = model.generate(**inputs, do_sample=True, temperature=1.0, top_k=0, num_return_sequences=40, **gen_kwargs, **extra)
            texts = [tok.decode(o[plen:], skip_special_tokens=False).replace("<|im_end|>", "") for o in out]
            exact = sum(t in valid_sids for t in texts)
            print(f"\n{name}: {exact}/40 completions are exactly a catalogue SID. First 3:")
            for t in texts[:3]:
                print(f"  {t}\n    -> {replace_sids_with_titles(t, mapping)}")
    print("\nA free-sampled completion can be malformed or belong to no item (the demo then falls back to a prefix match or prints"
          "\n[Unknown Item]). Constrained decoding makes every completion a real item; the content is still random here.")

    # 9d. The real thing
    if REAL_MODEL:
        run_real_model(sids)
    else:
        print("\nTo chat with the author's finetuned 8B model: `9 --real-model` (needs ~16GB memory), or see demo.ipynb.")


def run_real_model(sids):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    repo = "eugeneyan/semantic-id-qwen3-8b-video-games"
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(repo, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.bfloat16, device_map=device)
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "I like animal and cute games.\n<|rec|>"}]
    inputs = tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True), return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=64, temperature=0.3, do_sample=True)
    print(tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=False))


# ----------------------------------------------------------------------------
# Shared by steps 10-12: toy sequential-recommendation eval
# ----------------------------------------------------------------------------
EVAL_K = 5


def metrics_from_ranks(ranks: list[int], k: int = EVAL_K) -> dict:
    hit = sum(r <= k for r in ranks) / len(ranks)
    ndcg = sum(1 / torch.log2(torch.tensor(r + 1.0)).item() for r in ranks if r <= k) / len(ranks)
    return {f"hit@{k}": hit, f"ndcg@{k}": ndcg}


def val_users() -> list[tuple[str, list[str]]]:
    import polars as pl

    df = pl.read_parquet(DEBUG_DIR / "sequences_val.parquet")
    return [(u, s) for u, s in zip(df["user_id"], df["sequence"]) if len(s) >= 2]


# ----------------------------------------------------------------------------
# Step 10: SASRec baseline on plain item IDs   (src/train_sasrec.py, notebook 09)
# ----------------------------------------------------------------------------
def step10_sasrec():
    from src.train_sasrec import SASRec, SASRecConfig, SequenceDataset, SequentialSampler

    banner("STEP 10: SASRec baseline (item-ID sequential recommender)")
    torch.manual_seed(0)
    import numpy as np

    np.random.seed(0)
    cfg = SASRecConfig(data_path=DEBUG_DIR / "sequences_train.parquet", max_seq_length=20, hidden_units=32,
                       num_blocks=2, num_heads=1, dropout_rate=0.2, batch_size=128)
    ds = SequenceDataset(str(cfg.data_path), cfg)  # maps each ASIN to an integer id; 0 is padding
    sampler = SequentialSampler(ds, cfg)
    titles = {it["parent_asin"]: it["title"] for it in load_items()}
    print(f"items: {ds.num_items}; item_to_id example: {dict(list(ds.item_to_id.items())[:3])}")

    # What one training row looks like: at each position the target is the NEXT item, plus a random negative.
    batch = next(iter(sampler))
    row = 0
    n = int((batch["input_ids"][row] != 0).sum())
    print(f"\nTraining row {row} (left-padded to {cfg.max_seq_length}; showing the last {min(n, 6)} positions):")
    print(f"{'input item':44s} {'positive (next)':44s} negative")
    for p in range(cfg.max_seq_length - min(n, 6), cfg.max_seq_length):
        name = lambda i: "<pad>" if i == 0 else titles[ds.id_to_item[i]][:42]
        print(f"{name(int(batch['input_ids'][row, p])):44s} {name(int(batch['pos_ids'][row, p])):44s} {name(int(batch['neg_ids'][row, p]))}")
    # Check what the LAST position is trained to predict, for one user.
    one = SequentialSampler(ds, cfg)
    one.valid_user_seqs = sampler.valid_user_seqs[:1]
    user, trimmed = one.valid_user_seqs[0]
    full = ds.user_seq[user]
    last_label = int(next(iter(one))["pos_ids"][0, -1])
    print(f"\nUser {user}: full sequence has {len(full)} items; training input = first {len(trimmed)} (seq[:-2]).")
    print(f"  label at the final training position = {last_label}; the user's 2nd-to-last item (the 'val' target) = {full[-2]}; last item (the 'test' target) = {full[-1]}")
    if last_label == full[-2]:
        print("  -> LEAK: the sampler trains on this user's validation item, so a 'val' metric on these users would be optimistic.")
    else:
        print("  -> the final position is unlabeled (0 = masked in the loss), so the validation and test items stay held out.")
        print("     (before the fix, this label was the validation item.) This walkthrough also evaluates on entirely held-out users.")
    pause("batch: input_ids / pos_ids / neg_ids, all [B, T]")

    model = SASRec(cfg, ds.num_items)
    print(f"\nparams: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.98))
    bce = torch.nn.BCEWithLogitsLoss()
    model.train()
    for epoch in range(1, 41):
        tot, nb = 0.0, 0
        for b in sampler:
            pos_logits, neg_logits = model.training_step(b["input_ids"], b["pos_ids"], b["neg_ids"])
            m = b["pos_ids"] != 0  # ignore padding
            loss = bce(pos_logits[m], torch.ones_like(pos_logits[m])) + bce(neg_logits[m], torch.zeros_like(neg_logits[m]))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot, nb = tot + loss.item(), nb + 1
        if epoch in (1, 5, 10, 20, 30, 40):
            print(f"epoch {epoch:2d} loss {tot / nb:.4f}")

    # Evaluate on held-out users: history = all but last, target = last item. Rank it among every item the user has not seen.
    model.eval()
    ranks, lens = [], []
    n_cands = []
    for _, seq in val_users():
        hist = [ds.item_to_id[a] for a in seq[:-1]][-cfg.max_seq_length:]
        target = ds.item_to_id[seq[-1]]
        seen = set(ds.item_to_id[a] for a in seq[:-1]) - {target}  # users can re-buy; the target must stay a candidate
        cands = [i for i in range(1, ds.num_items + 1) if i not in seen]
        with torch.no_grad():
            scores = model.predict(torch.tensor([hist]), torch.tensor([cands]))[0]
        ranks.append(int((scores > scores[cands.index(target)]).sum()) + 1)
        lens.append(len(seq))
        n_cands.append(len(cands))
    res = metrics_from_ranks(ranks)
    print(f"\nheld-out users: {len(ranks)}; {res}")
    save_json({"model": "SASRec (item IDs)", **res, "ranks": ranks, "lens": lens, "n_cands": n_cands}, "metrics_sasrec.json")


# ----------------------------------------------------------------------------
# Step 11: SASRec on semantic IDs   (src/train_sasrec_semantic_id.py, notebook 10)
# ----------------------------------------------------------------------------
def step11_semantic_sasrec():
    from torch.utils.data import DataLoader

    from src.train_sasrec_semantic_id import (SemanticSASRec, SemanticSASRecConfig, SemanticSequenceDataset,
                                              SemanticTrainingDataset, collate_semantic_batch, encode_semantic_id)

    banner("STEP 11: SASRec on semantic IDs (each item = 4 tokens, predicted level by level)")
    import numpy as np

    torch.manual_seed(0)
    np.random.seed(0)
    cfg = SemanticSASRecConfig(data_path=DEBUG_DIR / "sequences_train.parquet", max_seq_length=20, input_dim=32,
                               head_dim=16, num_heads=2, num_blocks=2, mlp_dim=64, dropout_rate=0.1, batch_size=128,
                               use_compile=False)
    ds = SemanticSequenceDataset(cfg)  # pre-tokenises: each SID string -> 4 token ids; sequence length is 4x items
    train_ds = SemanticTrainingDataset(ds)  # one sample per (user, prefix end): history prefix -> next item
    loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_semantic_batch)
    titles = {v["sid"]: v["title"] for v in load_json("semantic_ids.json").values()}

    batch = next(iter(loader))
    print(f"batch: input_ids {tuple(batch['input_ids'].shape)} (items x 4 tokens), pos_tokens {tuple(batch['pos_tokens'].shape)}, neg_tokens = another sample's positive")
    row = batch["input_ids"][0]
    n_tok = int((row != 0).sum())
    print(f"row 0 token ids (non-pad): {row[row != 0].tolist()}  -> {n_tok // 4} items\n  target next item tokens: {batch['pos_tokens'][0].tolist()}")
    pause("batch: input_ids / pos_tokens / neg_tokens")

    model = SemanticSASRec(cfg)
    print(f"params: {sum(p.numel() for p in model.parameters()):,}")
    print("Prediction heads: one 256-way classifier per level; level k>0 also sees the embeddings of levels < k (teacher forcing).")
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    model.train()
    for epoch in range(1, 31):
        tot, nb = 0.0, 0
        for b in loader:
            losses = model.training_step(b["input_ids"], b["pos_tokens"], b["neg_tokens"], cfg.teacher_forcing_ratio)
            loss = sum(losses.values()) / len(losses)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot, nb = tot + loss.item(), nb + 1
        if epoch in (1, 5, 10, 20, 30):
            print(f"epoch {epoch:2d} loss {tot / nb:.4f}  (per level: {[round(v.item(), 2) for v in losses.values()]})")

    # Same protocol as step 10: rank the held-out last item among every item the user has not seen,
    # scoring a candidate by the sum of its 4 per-level log-probabilities.
    sid_items = list(titles)  # every catalogue SID
    cand_tokens = {s: encode_semantic_id(s, cfg.num_levels, cfg.codebook_size) for s in sid_items}
    val = []
    import polars as pl

    dfv = pl.read_parquet(DEBUG_DIR / "sequences_val.parquet")
    for u, s_ in zip(dfv["user_id"], dfv["semantic_id_sequence"]):
        if len(s_) >= 2:
            val.append((u, s_))
    model.eval()
    ranks, lens, n_cands = [], [], []
    for _, seq in val:
        hist = [t for s in seq[:-1][-cfg.max_seq_length:] for t in cand_tokens[s]]
        seen = set(seq[:-1]) - {seq[-1]}  # users can re-buy; the target must stay a candidate
        cands = [s for s in sid_items if s not in seen]
        ct = torch.tensor([[cand_tokens[s] for s in cands]])
        with torch.no_grad():
            scores = model.score_candidates(torch.tensor([hist]), ct)[0]
        ranks.append(int((scores > scores[cands.index(seq[-1])]).sum()) + 1)
        lens.append(len(seq))
        n_cands.append(len(cands))
    res = metrics_from_ranks(ranks)
    print(f"\nheld-out users: {len(ranks)}; {res}")

    # Token ids are <|sid_k|> id + 1, so 0 is reserved for padding (padding_idx=0). Before this was fixed, <|sid_0|> was id 0
    # and looked like padding: its embedding was frozen at zero and its position embedding masked out.
    first_ids = sorted({t[0] for t in cand_tokens.values()})
    sid0 = encode_semantic_id("<|sid_start|><|sid_0|><|sid_256|><|sid_512|><|sid_768|><|sid_end|>")
    norms = model.token_emb(torch.tensor([0, sid0[0]])).norm(dim=-1).tolist()
    print(f"\nfirst-token ids in this catalogue: {first_ids}; ids for <|sid_0|><|sid_256|><|sid_512|><|sid_768|>: {sid0}")
    print(f"|token_emb(0)| = {norms[0]:.3f} (padding only) vs |token_emb(id of <|sid_0|>)| = {norms[1]:.3f} (a real, trainable token)")
    assert 0 not in sum(cand_tokens.values(), []) and min(sid0) >= 1
    lvl = model.token_id_to_level_value(torch.tensor(sid0))
    assert lvl.tolist() == [0, 0, 0, 0] and [model.level_to_token_id(0, l) for l in range(4)] == sid0
    print("level_to_token_id / token_id_to_level_value round-trip ✓")
    save_json({"model": "SASRec (semantic IDs)", **res, "ranks": ranks, "lens": lens, "n_cands": n_cands}, "metrics_semantic.json")


# ----------------------------------------------------------------------------
# Step 12: compare   (notebooks 09-10)
# ----------------------------------------------------------------------------
def step12_compare():
    banner("STEP 12: compare the recommenders (held-out users, rank among all unseen items)")
    a, b = load_json("metrics_sasrec.json"), load_json("metrics_semantic.json")
    k = EVAL_K
    rand_hit = sum(min(1.0, k / c) for c in a["n_cands"]) / len(a["n_cands"])
    print(f"{'model':28s} {'Hit@' + str(k):>8s} {'NDCG@' + str(k):>8s}")
    print(f"{'random':28s} {rand_hit:8.3f} {'-':>8s}   (avg {sum(a['n_cands']) / len(a['n_cands']):.1f} candidates per user)")
    for m in (a, b):
        print(f"{m['model']:28s} {m[f'hit@{k}']:8.3f} {m[f'ndcg@{k}']:8.3f}")
    print(f"\nby history length ({len(a['ranks'])} users — expect noisy numbers):")
    for lo, hi in ((2, 4), (5, 6), (7, 20)):
        row = f"  len {lo}-{hi if hi < 20 else '+'}:"
        for m in (a, b):
            rs = [r for r, n in zip(m["ranks"], m["lens"]) if lo <= n <= hi]
            row += f"  {m['model'].split('(')[1][:-1]}: " + (f"hit {sum(r <= k for r in rs) / len(rs):.2f} (n={len(rs)})" if rs else "n/a")
        print(row)
    print("\nBoth beat random by a lot because toy users walk a chain inside one theme. The item-ID model wins here, plausibly")
    print("because a 24-item catalogue is easy to memorise as plain IDs; the semantic-ID model's pitch is scale (shared structure")
    print("across 66k items) and that is not tested by this toy. The real comparison is notebooks 09-10 on the full data.")
    print("The LLM from steps 5-9 is evaluated differently: by generating and checking the recommendations.")


STEPS = {
    "1": step1_prep_data, "2": step2_embed, "3": step3_rqvae, "4": step4_semantic_ids, "5": step5_tokenizer,
    "6": step6_llm_data, "7": step7_prepare_model, "8": step8_stage1_train, "9": step9_inference,
    "10": step10_sasrec, "11": step11_semantic_sasrec, "12": step12_compare,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=[*STEPS, "all"])
    parser.add_argument("--pause", action="store_true", help="call breakpoint() at each pause() point")
    parser.add_argument("--real-mapping", action="store_true", help="step 9: use the author's HF item mapping")
    parser.add_argument("--real-model", action="store_true", help="step 9: run the author's finetuned 8B model")
    args = parser.parse_args()
    PAUSE, REAL_MAPPING, REAL_MODEL = args.pause, args.real_mapping, args.real_model
    for key in STEPS if args.step == "all" else [args.step]:
        STEPS[key]()
