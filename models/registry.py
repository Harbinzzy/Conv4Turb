from importlib import import_module


_MODEL_FACTORIES = {
    "edsr": ("models.edsr", "init_edsr"),
    "edsr4d": ("models.edsr4d", "init_edsr4d"),
    "rcan": ("models.rcan", "init_rcan"),
    "rcan4d": ("models.rcan4d", "init_rcan4d"),
}

_SUPPORTED_SIZES = {
    "edsr": ("0.5M", "0.8M", "1.4M", "2.7M", "5M", "11M", "17M", "35M"),
    "edsr4d": ("0.5M", "5M"),
    "rcan": ("0.5M", "0.8M", "1.4M", "2.7M", "5M", "11M", "17M", "50M"),
    "rcan4d": ("0.5M", "0.8M", "17M"),
}

MODEL_NAMES = tuple(_MODEL_FACTORIES)


def create_model(model_name, approx_param, upscale):
    try:
        module_name, factory_name = _MODEL_FACTORIES[model_name]
    except KeyError as exc:
        choices = ", ".join(MODEL_NAMES)
        raise ValueError(f"Unknown model {model_name!r}; choose from {choices}") from exc

    supported_sizes = _SUPPORTED_SIZES[model_name]
    if supported_sizes is not None and approx_param not in supported_sizes:
        choices = ", ".join(supported_sizes)
        raise ValueError(
            f"{model_name} does not support approx_param={approx_param!r}; "
            f"choose from {choices}"
        )

    factory = getattr(import_module(module_name), factory_name)
    return factory(approx_param=approx_param, upscale=upscale)
