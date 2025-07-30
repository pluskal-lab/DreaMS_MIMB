from .dreams_classifier import DreamsClassifier
from .lit_classifier     import LitClassifier

MODEL_REGISTRY = {
    'dreams_classifier': DreamsClassifier,
    'lit_classifier':    LitClassifier,
    # add future models here...
}