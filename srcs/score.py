# /mnt/data/scorer.py
# python3 scorer.py init
# python3 scorer.py event --person "jensen huang" --asset "samsung" --polarity positive --strength 1.0
# python3 scorer.py event --person "jensen huang" --asset "nvda" --polarity positive --strength 1.0
# outputs: /mnt/data/embeddings.json, /mnt/data/weights.json

import json, math, os, argparse, re, hashlib
from collections import defaultdict

BASE_DIR = "./data"
ASSETS_JSON = os.path.join(BASE_DIR, "assets.json")
PEOPLE_JSON = os.path.join(BASE_DIR, "people.json")
EMBED_JSON = os.path.join(BASE_DIR, "embeddings.json")
WEIGHT_JSON = os.path.join(BASE_DIR, "weights.json")

DIM = 512
DIRECT_LINK_BONUS = 0.6
SIM_BASE_WEIGHT = 0.4
ALPHA = 0.3
BETA = 0.6

def load_data():
    with open(ASSETS_JSON, "r", encoding="utf-8") as f:
        assets = json.load(f)["assets"]
    with open(PEOPLE_JSON, "r", encoding="utf-8") as f:
        people = json.load(f)["people"]
    return people, assets

def tok(s: str):
    s = s.lower()
    s = re.sub(r"[^a-z0-9\+\-\&\. ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    parts = []
    for w in s.split():
        parts.append(w)
        if len(w) > 3:
            parts.append(w[:-1])
    return parts

def fe_hash(token: str, dim=DIM):
    h = int(hashlib.blake2b(token.encode("utf-8"), digest_size=8).hexdigest(), 16)
    i = h % dim
    sign = 1.0 if ((h >> 8) & 1) == 0 else -1.0
    return i, sign

def embed_from_text(texts, dim=DIM):
    v = [0.0]*dim
    ct = 0
    for t in texts:
        for w in tok(t):
            i, sgn = fe_hash(w, dim)
            v[i] += sgn
            ct += 1
    if ct == 0:
        return v
    norm = math.sqrt(sum(x*x for x in v)) or 1.0
    return [x/norm for x in v]

def cosine(a, b):
    return sum(x*y for x,y in zip(a,b))

def build_indexes(people, assets):
    p_index = {}
    for p in people:
        names = [p.get("name","")]
        names += p.get("aliases",[])
        names += p.get("descriptors",[])
        p_index[p["entity_id"]] = {
            "nameset": set([n.lower() for n in names if n]),
            "name": p.get("name","").lower()
        }
    a_index = {}
    for a in assets:
        names = [a.get("name",""), a.get("symbol","")]
        names += a.get("keywords",[])
        a_index[a["asset_id"]] = {
            "nameset": set([n.lower() for n in names if n]),
            "name": a.get("name","").lower(),
            "symbol": a.get("symbol","").lower()
        }
    return p_index, a_index

def make_embeddings(people, assets):
    p_emb, a_emb = {}, {}
    for p in people:
        texts = [p.get("name","")] + p.get("aliases",[]) + p.get("descriptors",[])
        p_emb[p["entity_id"]] = embed_from_text(texts)
    for a in assets:
        texts = [a.get("name",""), a.get("symbol","")] + a.get("keywords",[])
        a_emb[a["asset_id"]] = embed_from_text(texts)
    return p_emb, a_emb

def has_direct_link(person_nameset, asset_nameset):
    if person_nameset & asset_nameset:
        return True
    # name containment like "jensen huang" inside keywords string
    joined = " ".join(sorted(asset_nameset))
    for n in person_nameset:
        if len(n) >= 3 and n in joined:
            return True
    return False

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

def persist_embeddings(p_emb, a_emb, p_index, a_index):
    out = {
        "people": {pid: {"embedding": v} for pid, v in p_emb.items()},
        "assets": {aid: {"embedding": v} for aid, v in a_emb.items()},
        "meta": {"dim": DIM}
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

def resolve_person(q, people, p_index):
    q = q.lower().strip()
    # by id or exact name/alias
    if q in p_index:
        return q
    for pid, info in p_index.items():
        if q == info["name"] or q in info["nameset"]:
            return pid
    # substring fallback
    for pid, info in p_index.items():
        for n in info["nameset"]:
            if q in n:
                return pid
    return None

def resolve_asset(q, assets, a_index):
    q = q.lower().strip()
    # by id or symbol or exact name/keyword
    if q in a_index:
        return q
    for aid, info in a_index.items():
        if q == info["symbol"] or q == info["name"] or q in info["nameset"]:
            return aid
    # substring fallback
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

    pid = resolve_person(person_q, people, p_index)
    aid = resolve_asset(asset_q, assets, a_index)
    if not pid or not aid:
        raise SystemExit(f"unresolved: person={person_q} id={pid}, asset={asset_q} id={aid}")

    sign = 1.0 if polarity.lower().startswith("pos") else -1.0
    # direct edge
    prev = w.get(pid, {}).get(aid, 0.0)
    neww = max(0.0, min(1.0, prev + ALPHA*sign*strength))
    w.setdefault(pid, {})[aid] = round(neww, 6)

    # chain to similar assets
    target_vec = a_emb[aid]
    for aid2, vec in a_emb.items():
        if aid2 == aid:
            continue
        sim = max(0.0, cosine(target_vec, vec))  # only reinforce by positive similarity
        if sim <= 0.0:
            continue
        delta = ALPHA * BETA * sign * strength * sim
        prev2 = w.get(pid, {}).get(aid2, 0.0)
        new2 = max(0.0, min(1.0, prev2 + delta))
        if new2 > 0.0:
            w.setdefault(pid, {})[aid2] = round(new2, 6)

    persist_weights(w)
    print(f"updated: person={pid} asset={aid} polarity={polarity} strength={strength}")

def cmd_init():
    people, assets = load_data()
    p_index, a_index = build_indexes(people, assets)
    p_emb, a_emb = make_embeddings(people, assets)
    weights = init_weights(people, assets, p_emb, a_emb, p_index, a_index)
    persist_embeddings(p_emb, a_emb, p_index, a_index)
    persist_weights(weights)
    print(f"initialized embeddings and weights. people={len(p_emb)} assets={len(a_emb)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp_init = sub.add_parser("init")

    sp_event = sub.add_parser("event")
    sp_event.add_argument("--person", required=True)
    sp_event.add_argument("--asset", required=True)
    sp_event.add_argument("--polarity", choices=["positive","negative"], required=True)
    sp_event.add_argument("--strength", type=float, default=1.0)

    args = ap.parse_args()
    if args.cmd == "init":
        cmd_init()
    elif args.cmd == "event":
        event_update(args.person, args.asset, args.polarity, args.strength)
