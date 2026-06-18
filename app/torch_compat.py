def allow_trusted_pyannote_checkpoints():
    try:
        import torch
        from omegaconf.dictconfig import DictConfig
        from omegaconf.listconfig import ListConfig
    except ImportError:
        return

    add_safe_globals = getattr(torch.serialization, "add_safe_globals", None)
    if add_safe_globals is None:
        return

    add_safe_globals([DictConfig, ListConfig])
