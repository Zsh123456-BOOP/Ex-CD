"""CertCD: Instance-Level Identifiability Certificates with Calibrated Abstention for CD.

Builds on the validated excd CDM/data/DOA layer. Lazy export so the torch-free certificate /
calibration / selective modules import without pulling torch.
"""

__all__ = ["RunConfig", "run"]


def __getattr__(name):
    if name in ("RunConfig", "run"):
        from certcd.run import RunConfig, run

        return {"RunConfig": RunConfig, "run": run}[name]
    raise AttributeError(f"module 'certcd' has no attribute {name!r}")
