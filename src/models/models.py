from .itransformer import Model as ITransformer
try:
    from .patchst import Model as PatchTST
except Exception:
    PatchTST = None
try:
    from .deepar import Model as DeepAR
except Exception:
    DeepAR = None

REGISTRY = {
    "itransformer": ITransformer,
    "patchtst": PatchTST,
    "deepar": DeepAR,
}

def build_model(name: str, cfg):
    name = name.lower()
    if name not in REGISTRY or REGISTRY[name] is None:
        raise ValueErrror(f"Unknown or unavailable model: {name}")
    return REGISTRY[name](cfg)