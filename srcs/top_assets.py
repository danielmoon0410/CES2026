# /mnt/data/top_assets.py
# python3 top_assets.py --person "jensen huang" --k 10

import json, os, argparse, re

BASE_DIR = "./data"
WEIGHT_JSON = os.path.join(BASE_DIR, "weights.json")
ASSETS_JSON = os.path.join(BASE_DIR, "assets.json")
PEOPLE_JSON = os.path.join(BASE_DIR, "people.json")

def load():
    with open(WEIGHT_JSON,"r",encoding="utf-8") as f:
        W = json.load(f)
    with open(ASSETS_JSON,"r",encoding="utf-8") as f:
        A = {a["asset_id"]: a for a in json.load(f)["assets"]}
    with open(PEOPLE_JSON,"r",encoding="utf-8") as f:
        P = {p["entity_id"]: p for p in json.load(f)["people"]}
    return W, A, P

def resolve_person(q, P):
    ql = q.lower().strip()
    # id
    if ql in P: return ql
    # exact name or alias
    for pid, p in P.items():
        names = set([p.get("name","").lower()] + [a.lower() for a in p.get("aliases",[])])
        if ql in names: return pid
    # substring fallback
    for pid, p in P.items():
        ns = " ".join([p.get("name","")] + p.get("aliases",[]) + p.get("descriptors",[])).lower()
        if ql in ns: return pid
    return None

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--person", required=True)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    W, A, P = load()
    pid = resolve_person(args.person, P)
    if not pid:
        raise SystemExit(f"person not found: {args.person}")

    edges = W.get(pid, {})
    items = sorted(edges.items(), key=lambda kv: kv[1], reverse=True)[:args.k]
    for aid, w in items:
        a = A.get(aid, {})
        symbol = a.get("symbol","").lower()
        name = a.get("name","").lower()
        print(f"{aid}\t{symbol}\t{name}\t{w:.6f}")
