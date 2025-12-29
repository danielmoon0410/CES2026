# /mnt/data/scorer_v2.py
# SBERT-based person–asset graph builder
# - All weights initialized to 0
# - Edges are created ONLY via user events
# - No polarity (only positive accumulation)
# - Optional propagation via asset–asset similarity

from __future__ import annotations
import json, os, argparse, re
from collections import defaultdict
from embedding_backend import make_embeddings

# =======================
# Paths
# =======================
BASE_DIR = "/home/daniel/Desktop/CES2026/CES2026/data"
ASSETS_JSON = os.path.join(BASE_DIR, "assets.json")
PEOPLE_JSON = os.path.join(BASE_DIR, "people.json")
EMBED_JSON  = os.path.join(BASE_DIR, "embeddings.json")
WEIGHT_JSON = os.path.join(BASE_DIR, "weights.json")

# =======================
# Hyperparameters
# =======================
ALPHA = 0.3   # direct edge update strength
BETA  = 0.6   # propagation scaling

# =======================
# Data loading
# =======================
def load_data():
    with open(ASSETS_JSON, "r", encoding="utf-8") as f:
        assets = json.load(f)["assets"]
    with open(PEOPLE_JSON, "r", encoding="utf-8") as f:
        people = json.load(f)["people"]
    return people, assets

# =======================
# Token helpers
# =======================
def tokset(items):
    s = set()
    for x in items:
        if not x:
            continue
        for w in re.split(r"[^a-z0-9\.\-\&]+", x.lower()):
            if w:
                s.add(w)
    return s

def build_indexes(people, assets):
    p_index, a_index = {}, {}

    for p in people:
        names = [p.get("name","")] + p.get("aliases",[]) + p.get("descriptors",[])
        p_index[p["entity_id"]] = {
            "nameset": tokset(names),
            "name": p.get("name","").lower()
        }

    for a in assets:
        names = [a.get("name",""), a.get("symbol","")] + a.get("keywords",[])
        a_index[a["asset_id"]] = {
            "nameset": tokset(names),
            "name": a.get("name","").lower(),
            "symbol": a.get("symbol","").lower()
        }

    return p_index, a_index

# =======================
# Math
# =======================
def cosine(a, b):
    return sum(x*y for x, y in zip(a, b))

# =======================
# Initialization
# =======================
def init_weights(people):
    # Every person starts with NO edges
    weights = {}
    for p in people:
        weights[p["entity_id"]] = {}
    return weights

# =======================
# Persistence
# =======================
def persist_embeddings(p_emb, a_emb, meta):
    out = {
        "people": {pid: {"embedding": v} for pid, v in p_emb.items()},
        "assets": {aid: {"embedding": v} for aid, v in a_emb.items()},
        "meta": meta
    }
    with open(EMBED_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f)

def persist_weights(weights):
    with open(WEIGHT_JSON, "w", encoding="utf-8") as f:
        json.dump(weights, f)

def load_store():
    with open(EMBED_JSON, "r", encoding="utf-8") as f:
        emb = json.load(f)
    with open(WEIGHT_JSON, "r", encoding="utf-8") as f:
        w = json.load(f)
    return emb, w

# =======================
# Resolution
# =======================
def resolve_person(q, p_index):
    q = q.lower().strip()
    if q in p_index:
        return q
    for pid, info in p_index.items():
        if q == info["name"] or q in info["nameset"]:
            return pid
    for pid, info in p_index.items():
        if any(q in n for n in info["nameset"]):
            return pid
    return None

def resolve_asset(q, a_index):
    q = q.lower().strip()
    if q in a_index:
        return q
    for aid, info in a_index.items():
        if q == info["symbol"] or q == info["name"] or q in info["nameset"]:
            return aid
    for aid, info in a_index.items():
        if q in info["symbol"] or q in info["name"] or any(q in n for n in info["nameset"]):
            return aid
    return None

# =======================
# Event update (NO polarity)
# =======================
def event_update(person_q, asset_q, strength):
    people, assets = load_data()
    p_index, a_index = build_indexes(people, assets)
    emb, w = load_store()

    a_emb = {aid: emb["assets"][aid]["embedding"] for aid in emb["assets"]}

    pid = resolve_person(person_q, p_index)
    aid = resolve_asset(asset_q, a_index)
    if not pid or not aid:
        raise SystemExit(f"unresolved: person={person_q}, asset={asset_q}")

    # --- direct edge ---
    prev = w.get(pid, {}).get(aid, 0.0)
    neww = min(1.0, prev + ALPHA * strength)
    w.setdefault(pid, {})[aid] = round(neww, 6)

    # --- propagation ---
    target_vec = a_emb[aid]
    for aid2, vec in a_emb.items():
        if aid2 == aid:
            continue
        sim = max(0.0, cosine(target_vec, vec))
        if sim <= 0.0:
            continue

        delta = ALPHA * BETA * strength * sim
        prev2 = w.get(pid, {}).get(aid2, 0.0)
        new2 = min(1.0, prev2 + delta)
        if new2 > prev2:
            w.setdefault(pid, {})[aid2] = round(new2, 6)

    persist_weights(w)
    print(f"updated: {person_q} → {asset_q} (strength={strength})")

# =======================
# CLI
# =======================
def cmd_init(embed, model_name, dim):
    people, assets = load_data()
    p_emb, a_emb, meta = make_embeddings(
        people,
        assets,
        method=embed,
        model_name=model_name,
        dim=dim
    )

    weights = init_weights(people)
    persist_embeddings(p_emb, a_emb, meta)
    persist_weights(weights)

    print(f"initialized. method={meta['method']} dim={meta['dim']} model={meta['model']}")

# =======================
# Entry point
# =======================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp_init = sub.add_parser("init")
    sp_init.add_argument("--embed", choices=["sbert"], default="sbert")
    sp_init.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    sp_init.add_argument("--dim", type=int, default=384)

    sp_event = sub.add_parser("event")
    sp_event.add_argument("--person", required=True)
    sp_event.add_argument("--asset", required=True)
    sp_event.add_argument("--strength", type=float, default=1.0)

    args = ap.parse_args()

    if args.cmd == "init":
        cmd_init(args.embed, args.model, args.dim)
    elif args.cmd == "event":
        event_update(args.person, args.asset, args.strength)
