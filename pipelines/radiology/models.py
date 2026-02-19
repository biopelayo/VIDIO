import logging

log = logging.getLogger(__name__)

_TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    log.info('PyTorch not available for radiology models')


class UNet3DWrapper:
    """Wrapper for 3D U-Net segmentation models."""

    def __init__(self, model_path=None, in_channels=1, out_channels=2, device='cpu'):
        self.model_path = model_path
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.device = device
        self.model = None

    def load(self):
        if not _TORCH_AVAILABLE:
            raise RuntimeError('PyTorch required')

        try:
            from monai.networks.nets import UNet
            self.model = UNet(
                spatial_dims=3,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
            )
        except ImportError:
            log.warning('MONAI not available, 3D U-Net requires MONAI')
            raise

        if self.model_path:
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()
        return self

    def predict(self, volume_tensor):
        if self.model is None:
            raise RuntimeError('Model not loaded')

        with torch.no_grad():
            if not isinstance(volume_tensor, torch.Tensor):
                volume_tensor = torch.from_numpy(volume_tensor).float()
            if volume_tensor.dim() == 3:
                volume_tensor = volume_tensor.unsqueeze(0).unsqueeze(0)
            elif volume_tensor.dim() == 4:
                volume_tensor = volume_tensor.unsqueeze(0)
            volume_tensor = volume_tensor.to(self.device)
            output = self.model(volume_tensor)
            return torch.argmax(output, dim=1).squeeze().cpu().numpy()
