"""Ex-CD: Exposure-Aware Debiased Cognitive Diagnosis with Exposure-Stratified DOA.

``run`` / ``RunConfig`` are lazily exposed so that the torch-free submodules
(``excd.data``, ``excd.metrics``) can be imported without pulling in torch.
"""

__all__ = ["RunConfig", "run"]


def __getattr__(name):
    if name in ("RunConfig", "run"):
        from excd.train import RunConfig, run

        return {"RunConfig": RunConfig, "run": run}[name]
    raise AttributeError(f"module 'excd' has no attribute {name!r}")
