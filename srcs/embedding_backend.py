# /mnt/data/embedding_backend.py
# Embedding backends with signal boosting:
#  - "hash": domain-augmented hashed bag-of-tokens (+IDF weights)
#  - "sbert": Sentence-Transformers encoder with templates + domain tags
#  - "hf": HuggingFace Transformers encoder with templates + domain tags
#  - "e5": Retrieval-tuned encoder (e.g., intfloat/e5-base-v2) with proper prefixing
#
# Signal boosters implemented:
#   1) Structured templates per entity type (person vs asset)
#   2) Domain tag injection with weights
#   3) Acronym/ticker expansion
#   4) Corpus IDF weighting (auto-computed from provided people/assets)
#
# Dependencies per mode:
#   hash: none
#   sbert: sentence-transformers stack
#   hf/e5: torch + transformers
from __future__ import annotations
import re, math, hashlib
from typing import Dict, List, Tuple

# -------- domain tags and weights --------
DOMAIN_TAGS = {
    "hbm": ["hbm3e","hbm3","hbm","dram","memory","stacked"],
    "semiconductor": ["semiconductor","foundry","fab","euv","asml","5nm","3nm","cows","cowoS","2.5d","chiplet","cxl"],
    "gpu": ["gpu","cuda","tensor core","inference","training","nvlink","blackwell","hopper","h100","b200"],
    "cpu": ["cpu","x86","arm","server","datacenter","epyc","xeon","grace"],
    "ai": ["genai","llm","transformer","embedding","retrieval","mlops"],
    "macro": ["tariff","sanction","subsidy","10y","fed funds","yield","treasury","cpi","ppi"],
    "energy": ["opec","oil","brent","wti","production cut","supply"],
    "mobile": ["android","ios","handset","modem","rf","camera sensor"]
}
TAG_WEIGHT = 2.0  # repeats high-signal tags

# -------- helpers --------
def _clean(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _tok_iter(s: str):
    for w in re.split(r"[^a-z0-9\.\-\&]+", s.lower()):
        if w: yield w

def _acronyms(s: str) -> List[str]:
    # crude acronym expansion: "samsung electronics" -> ["se"]
    words = [w for w in re.split(r"[^a-z]+", s.lower()) if w]
    if len(words) >= 2:
        return ["".join(w[0] for w in words)]
    return []

def _augment_domain(text: str) -> List[str]:
    t = text.lower()
    tags = []
    for toks in DOMAIN_TAGS.values():
        if any(k in t for k in toks):
            tags.extend(toks)
    # weight by repetition
    weighted = []
    for z in tags:
        weighted.extend([z] * int(TAG_WEIGHT))
    return weighted

# -------- templating --------
def build_person_text(p: dict) -> str:
    name = p.get("name","")
    aliases = ", ".join(p.get("aliases",[]))
    desc = ", ".join(p.get("descriptors",[]))
    core = f"person: {name}. aliases: {aliases}. role/keywords: {desc}."
    tags = " ".join(_augment_domain(core))
    return _clean(f"{core} {tags}")

def build_asset_text(a: dict) -> str:
    name = a.get("name","")
    sym = a.get("symbol","")
    kw = ", ".join(a.get("keywords",[]))
    acr = " ".join(_acronyms(name))
    core = f"asset: {name}. ticker: {sym}. keywords: {kw}. acronyms: {acr}."
    tags = " ".join(_augment_domain(core))
    return _clean(f"{core} {tags}")

# -------- IDF weighting (computed on the fly from the full corpus) --------
def compute_idf(all_texts: List[str]) -> Dict[str, float]:
    # simple df and idf
    N = max(1, len(all_texts))
    df = {}
    for t in all_texts:
        seen = set(_tok_iter(t))
        for w in seen: df[w] = df.get(w, 0) + 1
    idf = {w: math.log((N + 1) / (c + 1)) + 1.0 for w, c in df.items()}  # smooth
    return idf

def apply_idf_weights(tokens: List[str], idf: Dict[str,float]) -> List[Tuple[str, float]]:
    return [(w, idf.get(w, 1.0)) for w in tokens]

# -------- hashing embedder (with IDF) --------
def _fe_hash(token: str, dim: int):
    import hashlib as _hh
    h = int(_hh.blake2b(token.encode("utf-8"), digest_size=8).hexdigest(), 16)
    i = h % dim
    sign = 1.0 if ((h >> 8) & 1) == 0 else -1.0
    return i, sign

def _embed_hash_from_tokens(weighted: List[Tuple[str,float]], dim: int) -> List[float]:
    v = [0.0]*dim
    for w, wt in weighted:
        i, s = _fe_hash(w, dim)
        v[i] += s * wt
    norm = math.sqrt(sum(x*x for x in v)) or 1.0
    return [x/norm for x in v]

# -------- SBERT / HF / E5 encoders --------
_ST_MODEL = None
def _embed_sbert(text: str, model_name: str) -> List[float]:
    global _ST_MODEL
    if _ST_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _ST_MODEL = SentenceTransformer(model_name)
    vec = _ST_MODEL.encode(text, normalize_embeddings=True)
    return vec.tolist()

_HF_CACHE = {}
def _embed_hf(text: str, model_name: str) -> List[float]:
    import torch
    if _HF_CACHE.get("name") != model_name:
        from transformers import AutoTokenizer, AutoModel
        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModel.from_pretrained(model_name)
        mdl.eval()
        _HF_CACHE["name"] = model_name
        _HF_CACHE["tok"] = tok
        _HF_CACHE["mdl"] = mdl
    tok, mdl = _HF_CACHE["tok"], _HF_CACHE["mdl"]
    enc = tok(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        out = mdl(**enc)
        x = out.last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1)
        v = (x * m).sum(dim=1) / m.sum(dim=1).clamp_min(1)
        v = torch.nn.functional.normalize(v, p=2, dim=1)
    return v[0].tolist()

def _embed_e5(text: str, model_name: str) -> List[float]:
    # e5 expects prefix; we treat entities as passages
    return _embed_hf(f"passage: {text}", model_name)

# -------- public API --------
def make_embeddings(people: List[dict], assets: List[dict],
                    method: str = "hash",
                    model_name: str = "intfloat/e5-base-v2",
                    dim: int = 1024):
    method = method.lower()
    if method not in {"hash","sbert","hf","e5"}:
        raise ValueError("method must be 'hash','sbert','hf','e5'")

    # build templated texts and a shared IDF
    p_text = {p["entity_id"]: build_person_text(p) for p in people}
    a_text = {a["asset_id"]:  build_asset_text(a)  for a in assets}
    idf = compute_idf(list(p_text.values()) + list(a_text.values()))

    p_emb, a_emb = {}, {}

    if method == "hash":
        for pid, t in p_text.items():
            toks = list(_tok_iter(t))
            weighted = apply_idf_weights(toks, idf)
            p_emb[pid] = _embed_hash_from_tokens(weighted, dim)
        for aid, t in a_text.items():
            toks = list(_tok_iter(t))
            weighted = apply_idf_weights(toks, idf)
            a_emb[aid] = _embed_hash_from_tokens(weighted, dim)
        meta = {"method":"hash","dim":dim,"model":None}
        return p_emb, a_emb, meta

    if method == "sbert":
        for pid, t in p_text.items(): p_emb[pid] = _embed_sbert(t, model_name)
        for aid, t in a_text.items(): a_emb[aid] = _embed_sbert(t, model_name)
    elif method == "hf":
        for pid, t in p_text.items(): p_emb[pid] = _embed_hf(t, model_name)
        for aid, t in a_text.items(): a_emb[aid] = _embed_hf(t, model_name)
    else:  # e5
        for pid, t in p_text.items(): p_emb[pid] = _embed_e5(t, model_name)
        for aid, t in a_text.items(): a_emb[aid] = _embed_e5(t, model_name)

    dim_eff = len(next(iter(a_emb.values()))) if a_emb else dim
    meta = {"method":method,"dim":dim_eff,"model":model_name}
    return p_emb, a_emb, meta
