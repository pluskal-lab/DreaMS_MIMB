MODEL_REGISTRY: dict[str, type] = {}

def register_model(name: str):
    """
    Use as:

      @register_model("lit_classifier")
      class LitClassifier(...):
          ...
    """
    def decorator(cls):
        if name in MODEL_REGISTRY:
            raise KeyError(f"Duplicate model name: {name}")
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator