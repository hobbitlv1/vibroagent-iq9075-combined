"""Training entry points are imported lazily to keep source-only tools usable."""

__all__ = ["run_alignment_training", "run_lora_training", "run_pretraining"]

def __getattr__(name):
    if name == "run_pretraining":
        from .pretrain import run_pretraining
        return run_pretraining
    if name in {"run_alignment_training", "run_lora_training"}:
        from .lm_train import run_alignment_training, run_lora_training
        return {"run_alignment_training": run_alignment_training, "run_lora_training": run_lora_training}[name]
    raise AttributeError(name)
