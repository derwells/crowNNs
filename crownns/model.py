import torchvision

from torchvision.models.detection.fcos import FCOS
from deepforest.model import create_anchor_generator


def load_backbone():
    """A torch vision FCOS model"""
    backbone = torchvision.models.detection.fcos_resnet50_fpn(pretrained=True)

    return backbone


def create_model(num_classes, nms_thresh, score_thresh, backbone=None):
    """Create an FCOS model
    Args:
        num_classes (int): number of classes in the model
        nms_thresh (float): non-max suppression threshold for intersection-over-union [0,1]
        score_thresh (float): minimum prediction score to keep during prediction  [0,1]
    Returns:
        model: nn.Module
    """
    if not backbone:
        fcos = load_backbone()
        backbone = fcos.backbone
    model = FCOS(
        backbone=backbone,
        num_classes=num_classes,
    )
    model.nms_thresh = nms_thresh
    model.score_thresh = score_thresh

    return model
