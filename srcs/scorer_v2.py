# /mnt/data/scorer_v2.py
# Same scoring, now accepts --embed hash|sbert|hf|e5 and benefits from templates+IDF+domain tags.
from __future__ import annotations
import json, os, argparse, re
from collections import defaultdict
from embedding_backend import make_embeddings

BASE_DIR = "./data"
ASSETS_JSON = os.path.join(BASE_DIR, "assets.json")
PEOPLE_JSON = os.path.join(BASE_DIR, "people.json")
EMBED_JSON = os.path.join(BASE_DIR, "embeddings.json")
WEIGHT_JSON = os.path.join(BASE_DIR, "weights.json")

DIRECT_LINK_BONUS = 0.6
SIM_BASE_WEIGHT  = 0.4
ALPHA            = 0.3
BETA             = 0.6

def load_data():
    with open(ASSETS_JSON, "r", encoding="utf-8") as f:
        assets = json.load(f)["assets"]
    with open(PEOPLE_JSON, "r", encoding="utf-8") as f:
        people = json.load(f)["people"]
    return people, assets

def tokset(items):
    s = set()
    for x in items:
        if not x: continue
        for w in re.split(r"[^a-z0-9\.\-\&]+", x.lower()):
            if w: s.add(w)
    return s

def build_indexes(people, assets):
    p_index, a_index = {}, {}
    for p in people:
        names = [p.get("name","")] + p.get("aliases",[]) + p.get("descriptors",[])
        p_index[p["entity_id"]] = {"nameset": tokset(names), "name": p.get("name","").lower()}
    for a in assets:
        names = [a.get("name",""), a.get("symbol","")] + a.get("keywords",[])
        a_index[a["asset_id"]] = {"nameset": tokset(names),
                                  "name": a.get("name","").lower(),
                                  "symbol": a.get("symbol","").lower()}
    return p_index, a_index

def cosine(a, b):
    return sum(x*y for x,y in zip(a,b))

def has_direct_link(person_nameset, asset_nameset):
    if person_nameset & asset_nameset:
        return True
    joined = " ".join(sorted(asset_nameset))
    return any((len(n) >= 3 and n in joined) for n in person_nameset)

def init_weights(people, assets, p_emb, a_emb, p_index, a_index):
    weights = defaultdict(dict)
    for p in people:
        pid = p["entity_id"]
        pname = p_index[pid]["nameset"]
        pv = p_emb[pid]
        for a in assets:
            aid = a["asset_id"]
            av = a_emb[aid]
            base = max(0.0, cosine(pv, av)) * SIM_BASE_WEIGHT
            if has_direct_link(pname, a_index[aid]["nameset"]):
                base = min(1.0, base + DIRECT_LINK_BONUS)
            if base > 0.0:
                weights[pid][aid] = round(base, 6)
    return weights

def persist_embeddings(p_emb, a_emb, meta):
    out = {"people": {pid: {"embedding": v} for pid, v in p_emb.items()},
           "assets": {aid: {"embedding": v} for aid, v in a_emb.items()},
           "meta": meta}
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

def resolve_person(q, p_index):
    q = q.lower().strip()
    if q in p_index: return q
    for pid, info in p_index.items():
        if q == info["name"] or q in info["nameset"]:
            return pid
    for pid, info in p_index.items():
        if any(q in n for n in info["nameset"]):
            return pid
    return None

def resolve_asset(q, a_index):
    q = q.lower().strip()
    if q in a_index: return q
    for aid, info in a_index.items():
        if q == info["symbol"] or q == info["name"] or q in info["nameset"]:
            return aid
    for aid, info in a_index.items():
        if q in info["symbol"] or q in info["name"] or any(q in n for n in info["nameset"]):
            return aid
    return None

def event_update(person_q, asset_q, polarity, strength):
    people, assets = load_data()
    p_index, a_index = build_indexes(people, assets)
    emb, w = load_store()
    p_emb = {pid: emb["people"][pid]["embedding"] for pid in emb["people"]}
    a_emb = {aid: emb["assets"][aid]["embedding"] for aid in emb["assets"]}

    pid = resolve_person(person_q, p_index)
    aid = resolve_asset(asset_q, a_index)
    if not pid or not aid:
        raise SystemExit(f"unresolved: person={person_q} id={pid}, asset={asset_q} id={aid}")

    sign = 1.0 if polarity.lower().startswith("pos") else -1.0

    prev = w.get(pid, {}).get(aid, 0.0)
    neww = max(0.0, min(1.0, prev + ALPHA*sign*strength))
    w.setdefault(pid, {})[aid] = round(neww, 6)

    target_vec = a_emb[aid]
    for aid2, vec in a_emb.items():
        if aid2 == aid: continue
        sim = max(0.0, cosine(target_vec, vec))
        if sim <= 0.0: continue
        delta = ALPHA * BETA * sign * strength * sim
        prev2 = w.get(pid, {}).get(aid2, 0.0)
        new2 = max(0.0, min(1.0, prev2 + delta))
        if new2 > 0.0:
            w.setdefault(pid, {})[aid2] = round(new2, 6)

    persist_weights(w)
    print(f"updated: {person_q} → {asset_q} ({polarity}, {strength})")

def cmd_init(embed, model_name, dim):
    people, assets = load_data()
    p_emb, a_emb, meta = make_embeddings(people, assets, method=embed, model_name=model_name, dim=dim)
    p_index, a_index = build_indexes(people, assets)
    weights = init_weights(people, assets, p_emb, a_emb, p_index, a_index)
    persist_embeddings(p_emb, a_emb, meta)
    persist_weights(weights)
    print(f"initialized. method={meta['method']} dim={meta['dim']} model={meta['model']}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp_init = sub.add_parser("init")
    sp_init.add_argument("--embed", choices=["hash","sbert","hf","e5"], default="hash")
    sp_init.add_argument("--model", default="intfloat/e5-base-v2")
    sp_init.add_argument("--dim", type=int, default=1024, help="only for hash mode")

    sp_event = sub.add_parser("event")
    sp_event.add_argument("--person", required=True)
    sp_event.add_argument("--asset", required=True)
    sp_event.add_argument("--polarity", choices=["positive","negative"], required=True)
    sp_event.add_argument("--strength", type=float, default=1.0)

    args = ap.parse_args()
    if args.cmd == "init":
        cmd_init(args.embed, args.model, args.dim)
    elif args.cmd == "event":
        event_update(args.person, args.asset, args.polarity, args.strength)
