import logging

log = logging.getLogger(__name__)

_TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    log.info('PyTorch not available for histology models')


class PatchClassifier:
    """Wrapper for tile/patch-level classification models."""

    def __init__(self, model_path=None, num_classes=4, device='cpu'):
        self.model_path = model_path
        self.num_classes = num_classes
        self.device = device
        self.model = None

    def load(self):
        if not _TORCH_AVAILABLE:
            raise RuntimeError('PyTorch required')

        from torchvision.models import resnet50
        self.model = resnet50(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)

        if self.model_path:
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()
        return self

    def predict(self, tile_tensor):
        if self.model is None:
            raise RuntimeError('Model not loaded')

        with torch.no_grad():
            if not isinstance(tile_tensor, torch.Tensor):
                tile_tensor = torch.from_numpy(tile_tensor).float()
            if tile_tensor.dim() == 3:
                tile_tensor = tile_tensor.unsqueeze(0)
            tile_tensor = tile_tensor.to(self.device)
            output = self.model(tile_tensor)
            probs = torch.softmax(output, dim=1)
            return probs.cpu().numpy()
