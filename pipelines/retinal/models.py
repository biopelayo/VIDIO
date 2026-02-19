import logging

log = logging.getLogger(__name__)

_TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    log.info('PyTorch not available. DL models will not be loaded.')


class RetinalModelWrapper:
    """
    Wrapper for retinal disease classification models.

    Supports loading pre-trained weights for:
    - EfficientNet B4 (multi-label retinal disease)
    - U-Net (vessel segmentation)
    - Vision Transformer (disease classification)
    """

    def __init__(self, model_path=None, architecture='efficientnet_b4', device='cpu'):
        self.model_path = model_path
        self.architecture = architecture
        self.device = device
        self.model = None

    def load(self):
        if not _TORCH_AVAILABLE:
            raise RuntimeError('PyTorch is required for DL models')

        if self.architecture == 'efficientnet_b4':
            self.model = self._build_efficientnet()
        elif self.architecture == 'unet':
            self.model = self._build_unet()
        else:
            raise ValueError(f'Unknown architecture: {self.architecture}')

        if self.model_path:
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()
        return self

    def predict(self, img_tensor):
        if self.model is None:
            raise RuntimeError('Model not loaded. Call load() first.')

        with torch.no_grad():
            if not isinstance(img_tensor, torch.Tensor):
                img_tensor = torch.from_numpy(img_tensor).float()
            if img_tensor.dim() == 3:
                img_tensor = img_tensor.unsqueeze(0)
            img_tensor = img_tensor.to(self.device)
            output = self.model(img_tensor)
            return output.cpu().numpy()

    def _build_efficientnet(self):
        try:
            from torchvision.models import efficientnet_b4
            model = efficientnet_b4(weights=None)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, 8)
            return model
        except Exception as ex:
            log.error(f'Failed to build EfficientNet B4: {ex}')
            raise

    def _build_unet(self):
        try:
            from monai.networks.nets import UNet
            model = UNet(
                spatial_dims=2,
                in_channels=1,
                out_channels=2,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
            )
            return model
        except ImportError:
            log.warning('MONAI not available, using basic U-Net')
            return self._build_basic_unet()

    def _build_basic_unet(self):
        if not _TORCH_AVAILABLE:
            raise RuntimeError('PyTorch required')

        class BasicUNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.enc1 = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU())
                self.pool1 = nn.MaxPool2d(2)
                self.enc2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
                self.pool2 = nn.MaxPool2d(2)
                self.bottleneck = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.Conv2d(128, 128, 3, padding=1), nn.ReLU())
                self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
                self.dec2 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
                self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
                self.dec1 = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU())
                self.out = nn.Conv2d(32, 2, 1)

            def forward(self, x):
                e1 = self.enc1(x)
                e2 = self.enc2(self.pool1(e1))
                b = self.bottleneck(self.pool2(e2))
                d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
                d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
                return self.out(d1)

        return BasicUNet()
