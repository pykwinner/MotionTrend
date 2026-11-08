from .sequence_classification import (MSRPSTModel, NTUPSTModel,
                                      build_model, build_ntu_model, load_checkpoint)
from .semantic_segmentation import (SynthiaPSTSegmentation, build_synthia_model,
                           load_checkpoint as load_synthia_checkpoint,
                           load_parent_only as load_synthia_backbone)

MoTrend = MSRPSTModel
__all__ = ["MSRPSTModel", "MoTrend", "build_model", "load_checkpoint", "NTUPSTModel", "build_ntu_model",
           "SynthiaPSTSegmentation", "build_synthia_model", "load_synthia_checkpoint", "load_synthia_backbone"]
