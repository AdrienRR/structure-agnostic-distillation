#!/usr/bin/env python
"""Precompute the cross-modal teacher: BGE-large embeddings of ImageNet captions.

Reads the webdataset shards of the captioned ImageNet of Degeorge et al. (2025)
(one ``<n{wnid}_{imgid}>.txt`` caption per training image), embeds each caption with
BAAI/bge-large-en-v1.5 (CLS pooling + L2 normalisation), and writes, after --merge,
``embeddings_fp16.npy`` + ``key_to_index.json`` into --out (= TEXT_EMB_ROOT).

Multi-process: each rank embeds shards[rank::world] (rank/world from torchrun's
RANK/WORLD_SIZE or SLURM_PROCID/SLURM_NTASKS), then run once more with --merge.
"""
import os, glob, tarfile, argparse, json
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

MODEL = "BAAI/bge-large-en-v1.5"


def rank_world():
    rank = os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0))
    world = os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1))
    return int(rank), int(world)


def embed_shards(args):
    rank, world = rank_world()
    local = int(os.environ.get("LOCAL_RANK", 0))
    dev = f"cuda:{local}" if torch.cuda.is_available() and "LOCAL_RANK" in os.environ else \
        ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(MODEL)
    mdl = AutoModel.from_pretrained(MODEL, torch_dtype=torch.float16).to(dev).eval()

    shards = sorted(glob.glob(os.path.join(args.data, "*.tar")))[rank::world]
    os.makedirs(args.out, exist_ok=True)
    keys, texts = [], []

    def flush(keys, texts, embs):
        for i in range(0, len(texts), args.batch):
            enc = tok(texts[i:i + args.batch], padding=True, truncation=True,
                      max_length=512, return_tensors="pt").to(dev)
            with torch.no_grad():
                e = mdl(**enc).last_hidden_state[:, 0]
                e = torch.nn.functional.normalize(e.float(), dim=1).half().cpu().numpy()
            embs.append(e)

    embs = []
    n = 0
    for si, sh in enumerate(shards):
        with tarfile.open(sh) as t:
            for m in t:
                if m.name.endswith(".txt"):
                    keys.append(os.path.splitext(os.path.basename(m.name))[0])
                    texts.append(t.extractfile(m).read().decode("utf-8", "ignore").strip())
        if len(texts) >= 20000:               # embed in chunks to bound memory
            flush([], texts, embs); n += len(texts); texts = []
        if rank == 0:
            print(f"[rank0] {si+1}/{len(shards)} shards, {n+len(texts)} captions", flush=True)
    if texts:
        flush([], texts, embs); n += len(texts)
    E = np.concatenate(embs, 0) if embs else np.zeros((0, 1024), np.float16)
    assert len(keys) == len(E), (len(keys), len(E))
    np.savez(os.path.join(args.out, f"rank{rank:03d}.npz"),
             keys=np.array(keys), emb=E)
    print(f"[rank {rank}] wrote {len(keys)} embeddings", flush=True)


def merge(args):
    files = sorted(glob.glob(os.path.join(args.out, "rank*.npz")))
    all_keys, all_emb = [], []
    for f in files:
        d = np.load(f, allow_pickle=True)
        all_keys.append(d["keys"]); all_emb.append(d["emb"])
    keys = np.concatenate(all_keys); emb = np.concatenate(all_emb, 0)
    # dedup (webdataset shards are disjoint, but guard anyway)
    _, idx = np.unique(keys, return_index=True)
    keys, emb = keys[idx], emb[idx]
    np.save(os.path.join(args.out, "embeddings_fp16.npy"), emb)
    json.dump({k: i for i, k in enumerate(keys.tolist())},
              open(os.path.join(args.out, "key_to_index.json"), "w"))
    print(f"merged {len(keys)} unique keys, emb {emb.shape} -> {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="directory of captioned-ImageNet .tar shards")
    p.add_argument("--out", required=True, help="output directory (use as TEXT_EMB_ROOT)")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--merge", action="store_true")
    args = p.parse_args()
    (merge if args.merge else embed_shards)(args)
