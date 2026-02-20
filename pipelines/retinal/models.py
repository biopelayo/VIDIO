"""
Retinal Deep Learning Model Wrappers
======================================

Abstraction layer for loading, configuring, and running deep learning
models used in the retinal pipeline.

This module provides :class:`RetinalModelWrapper`, a unified interface
that supports multiple architectures:

+-------------------+----------------+-------------------------------------+
| Architecture key  | Task           | Implementation                      |
+===================+================+=====================================+
| ``efficientnet_b4``| Multi-label   | ``torchvision.models.efficientnet_b4``|
|                   | disease        | with 8-class output head            |
|                   | classification |                                     |
+-------------------+----------------+-------------------------------------+
| ``unet``          | Vessel         | MONAI ``UNet`` (preferred) or       |
|                   | segmentation   | :class:`BasicUNet` fallback         |
+-------------------+----------------+-------------------------------------+

Dependency Strategy
-------------------
* **PyTorch** is an optional dependency.  If it is not installed, the
  module logs an informational message and the wrapper raises
  ``RuntimeError`` when :meth:`~RetinalModelWrapper.load` is called.
* **MONAI** is a secondary optional dependency.  If MONAI is not
  available, the U-Net architecture falls back to a minimal two-level
  :class:`BasicUNet` implemented in pure PyTorch.
* **torchvision** is required only for the ``efficientnet_b4``
  architecture.

Usage
-----
>>> wrapper = RetinalModelWrapper(
...     model_path='weights/retinal_efficientnet.pth',
...     architecture='efficientnet_b4',
...     device='cuda',
... )
>>> wrapper.load()
>>> predictions = wrapper.predict(image_tensor)
"""
import logging

log = logging.getLogger(__name__)

# --- Optional PyTorch import ---
# The retinal pipeline can operate entirely with classical (non-DL)
# methods; PyTorch is only needed when deep-learning models are used.
_TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    log.info('PyTorch not available. DL models will not be loaded.')


class RetinalModelWrapper:
    """
    Unified wrapper for retinal deep learning models.

    This class encapsulates the full lifecycle of a retinal DL model:
    architecture construction, weight loading, device placement, and
    inference.  It provides a consistent ``load() -> predict()`` API
    regardless of the underlying architecture.

    Supported architectures:

    * ``'efficientnet_b4'`` -- EfficientNet B4 from torchvision with the
      classifier head replaced by a ``Linear(in_features, 8)`` layer for
      multi-label retinal disease classification (see
      :meth:`_build_efficientnet` for the 8-class rationale).
    * ``'unet'`` -- MONAI 2D U-Net for binary vessel segmentation (2
      output channels: background + vessel).  Falls back to
      :meth:`_build_basic_unet` if MONAI is not installed.

    Attributes
    ----------
    model_path : str or None
        Filesystem path to a ``state_dict`` checkpoint (``.pth`` file).
        If ``None``, the model is initialised with random weights
        (useful for architecture inspection or fine-tuning from scratch).
    architecture : str
        Architecture identifier.  One of ``'efficientnet_b4'`` or
        ``'unet'``.
    device : str
        PyTorch device string (``'cpu'``, ``'cuda'``, ``'cuda:0'``,
        etc.).
    model : torch.nn.Module or None
        The instantiated model.  ``None`` until :meth:`load` is called.

    Notes
    -----
    **Architecture choice rationale** :

    * EfficientNet B4 was chosen over B0--B3 because retinal images are
      high-resolution (typically 2048 x 1536) and contain subtle features
      (microaneurysms as small as 20 microns / ~3 px).  B4's larger
      receptive field and higher capacity improve sensitivity to these
      small features compared to lighter variants, while remaining
      tractable for single-GPU inference.
    * U-Net is the de facto standard for biomedical image segmentation.
      The MONAI implementation adds advanced features (instance
      normalisation, residual connections) while the basic fallback
      ensures the pipeline works in minimal-dependency environments.

    Examples
    --------
    >>> wrapper = RetinalModelWrapper(
    ...     model_path='weights/retinal_efficientnet.pth',
    ...     architecture='efficientnet_b4',
    ...     device='cuda',
    ... )
    >>> wrapper.load()
    >>> predictions = wrapper.predict(image_tensor)
    """

    def __init__(self, model_path=None, architecture='efficientnet_b4', device='cpu'):
        """
        Initialise the model wrapper.

        Parameters
        ----------
        model_path : str or None, optional
            Path to a PyTorch ``state_dict`` checkpoint.  If ``None``,
            the model will be initialised with random weights.
        architecture : {'efficientnet_b4', 'unet'}, optional
            Architecture to instantiate.  Default ``'efficientnet_b4'``.
        device : str, optional
            PyTorch device for model placement and inference.
            Default ``'cpu'``.
        """
        self.model_path = model_path
        self.architecture = architecture
        self.device = device
        self.model = None

    def load(self):
        """
        Build the model architecture and optionally load pre-trained weights.

        The method performs three steps in order:

        1. **Architecture construction** -- dispatches to the appropriate
           builder method based on ``self.architecture``.
        2. **Weight loading** -- if ``self.model_path`` is set, loads the
           ``state_dict`` from disk using ``torch.load`` with
           ``map_location=self.device`` so that GPU checkpoints can be
           loaded on CPU and vice versa.
        3. **Device placement and eval mode** -- moves the model to
           ``self.device`` and sets it to evaluation mode (disables
           dropout and batch-norm running statistics updates).

        Returns
        -------
        RetinalModelWrapper
            ``self``, to allow fluent chaining:
            ``wrapper = RetinalModelWrapper(...).load()``.

        Raises
        ------
        RuntimeError
            If PyTorch is not installed.
        ValueError
            If ``self.architecture`` is not a recognised architecture key.

        Notes
        -----
        ``torch.load(map_location=self.device)`` handles the common
        scenario where a model is trained on GPU but deployed on CPU
        (or on a different GPU index).  Without ``map_location``, loading
        a CUDA checkpoint on a CPU-only machine would raise an error.
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError('PyTorch is required for DL models')

        # --- Architecture dispatch ---
        if self.architecture == 'efficientnet_b4':
            self.model = self._build_efficientnet()
        elif self.architecture == 'unet':
            self.model = self._build_unet()
        else:
            raise ValueError(f'Unknown architecture: {self.architecture}')

        # --- Weight loading ---
        # map_location ensures GPU checkpoints can be loaded on CPU.
        if self.model_path:
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)

        # --- Device placement and eval mode ---
        self.model.to(self.device)
        self.model.eval()
        return self

    def predict(self, img_tensor):
        """
        Run forward inference on the loaded model.

        Accepts either a PyTorch tensor or a NumPy array and returns a
        NumPy array of predictions.  The input is automatically
        converted, batched, and moved to the correct device.

        Parameters
        ----------
        img_tensor : torch.Tensor or numpy.ndarray
            Input image(s).  Accepted shapes:

            * ``(C, H, W)`` -- single image; will be unsqueezed to
              ``(1, C, H, W)`` (a batch of one).
            * ``(B, C, H, W)`` -- batch of images; used as-is.

            For ``efficientnet_b4``: ``C=3`` (RGB), ``H=W=380``
            (EfficientNet B4 default input size, though any size is
            accepted by the convolutional backbone).

            For ``unet``: ``C=1`` (greyscale), ``H`` and ``W`` should
            be divisible by 16 (four 2x downsampling steps).

            If a NumPy array is provided, it is converted to a float32
            tensor.

        Returns
        -------
        numpy.ndarray
            Model output moved back to CPU and converted to NumPy.

            * ``efficientnet_b4`` : shape ``(B, 8)`` -- raw logits for
              8 disease classes.  Apply ``sigmoid`` for multi-label
              probabilities.
            * ``unet`` : shape ``(B, 2, H, W)`` -- per-pixel logits for
              background (channel 0) and vessel (channel 1).  Apply
              ``argmax(dim=1)`` for the binary vessel mask.

        Raises
        ------
        RuntimeError
            If :meth:`load` has not been called yet.

        Notes
        -----
        Inference is wrapped in ``torch.no_grad()`` to disable gradient
        computation, reducing memory usage and improving speed.  The
        output tensor is moved to CPU before converting to NumPy to
        avoid CUDA-to-NumPy errors.
        """
        if self.model is None:
            raise RuntimeError('Model not loaded. Call load() first.')

        with torch.no_grad():
            # Accept NumPy arrays by converting to float32 tensor.
            if not isinstance(img_tensor, torch.Tensor):
                img_tensor = torch.from_numpy(img_tensor).float()
            # Add batch dimension if a single image is provided.
            if img_tensor.dim() == 3:
                img_tensor = img_tensor.unsqueeze(0)
            # Move to the model's device (CPU or GPU).
            img_tensor = img_tensor.to(self.device)
            output = self.model(img_tensor)
            # Move back to CPU for NumPy conversion.
            return output.cpu().numpy()

    def _build_efficientnet(self):
        """
        Construct an EfficientNet B4 model with an 8-class output head.

        The standard EfficientNet B4 classifier outputs 1000 classes
        (ImageNet).  Here the final ``Linear`` layer is replaced with
        ``Linear(in_features, 8)`` for multi-label retinal disease
        classification.

        Returns
        -------
        torch.nn.Module
            EfficientNet B4 model with modified classifier head.

        Raises
        ------
        Exception
            Re-raised if model construction fails (e.g. torchvision not
            installed or incompatible version).

        Notes
        -----
        **Why 8 output classes?**  The output head is designed for
        ODIR-like (Ocular Disease Intelligent Recognition) multi-label
        classification tasks.  The 8 classes correspond to:

        0. Normal
        1. Diabetic retinopathy (DR)
        2. Glaucoma
        3. Cataract
        4. Age-related macular degeneration (AMD)
        5. Hypertensive retinopathy
        6. Pathological myopia
        7. Other diseases / abnormalities

        Each class is independent (multi-label, not multi-class), so the
        model output should be passed through ``torch.sigmoid`` (not
        ``softmax``) to obtain per-class probabilities.

        **weights=None** :  The model is initialised with random weights.
        Pre-trained ImageNet weights could be loaded separately for
        transfer learning, but are not used by default to keep the
        download-free setup simple.
        """
        try:
            from torchvision.models import efficientnet_b4
            model = efficientnet_b4(weights=None)
            # Replace the 1000-class ImageNet head with an 8-class
            # multi-label retinal disease head.
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, 8)
            return model
        except Exception as ex:
            log.error(f'Failed to build EfficientNet B4: {ex}')
            raise

    def _build_unet(self):
        """
        Construct a U-Net model for binary vessel segmentation.

        Attempts to use the MONAI library's ``UNet`` implementation,
        which includes instance normalisation and residual connections
        for improved training stability.  If MONAI is not installed,
        falls back to :meth:`_build_basic_unet`.

        Returns
        -------
        torch.nn.Module
            U-Net model ready for weight loading.

        Notes
        -----
        **Channel configuration** :

        * ``in_channels=1`` -- single-channel greyscale input (the
          preprocessed green channel or OCT slice).
        * ``out_channels=2`` -- two-class output (background, vessel).
          A 2-channel output with cross-entropy loss is preferred over
          a single-channel output with binary cross-entropy because it
          allows easy extension to multi-class segmentation (e.g.
          arteries vs. veins) without changing the architecture.
        * ``channels=(16, 32, 64, 128, 256)`` -- five resolution levels.
          The encoder progressively downsamples by 2x at each level
          (``strides=(2, 2, 2, 2)``), giving a total downsampling
          factor of 16x.  This is sufficient to capture both fine
          capillaries (1-3 px wide) at the top levels and large-scale
          vessel tree topology at the bottleneck.
        * ``strides=(2, 2, 2, 2)`` -- four 2x downsampling steps
          (one fewer than the number of channel levels, as the first
          level operates at full resolution).

        **MONAI vs. basic U-Net** :  MONAI adds instance normalisation
        (more stable than batch norm for small batch sizes typical in
        medical imaging), residual connections (easing gradient flow
        in deeper networks), and flexible n-dimensional support.  The
        basic fallback is a minimal two-level U-Net suitable for
        inference but less robust for training.
        """
        try:
            from monai.networks.nets import UNet
            model = UNet(
                spatial_dims=2,         # 2D retinal images
                in_channels=1,          # Single greyscale channel
                out_channels=2,         # Binary: background + vessel
                channels=(16, 32, 64, 128, 256),  # 5-level encoder
                strides=(2, 2, 2, 2),   # 4 downsampling steps (16x total)
            )
            return model
        except ImportError:
            log.warning('MONAI not available, using basic U-Net')
            return self._build_basic_unet()

    def _build_basic_unet(self):
        """
        Construct a minimal two-level U-Net as a MONAI fallback.

        This is a lightweight U-Net implementation using only core
        PyTorch modules (``nn.Conv2d``, ``nn.MaxPool2d``,
        ``nn.ConvTranspose2d``).  It is designed as a fallback when
        MONAI is not available, ensuring the pipeline can still perform
        vessel segmentation in minimal-dependency environments.

        Returns
        -------
        torch.nn.Module
            A ``BasicUNet`` instance.

        Raises
        ------
        RuntimeError
            If PyTorch is not installed.

        Notes
        -----
        **Architecture** :  The BasicUNet has two encoder levels, a
        bottleneck, and two decoder levels with skip connections:

        ::

            Input (1 ch)
              |
            [enc1: 1->32, Conv3x3-ReLU x2]  ──────────────┐
              |                                             |
            [pool1: MaxPool2x2]                             |
              |                                             |
            [enc2: 32->64, Conv3x3-ReLU x2]  ────────┐     |
              |                                       |     |
            [pool2: MaxPool2x2]                       |     |
              |                                       |     |
            [bottleneck: 64->128, Conv3x3-ReLU x2]   |     |
              |                                       |     |
            [up2: ConvTranspose 128->64, 2x2]         |     |
              |                                       |     |
            [concat with enc2]  <─────────────────────┘     |
              |                                             |
            [dec2: 128->64, Conv3x3-ReLU x2]                |
              |                                             |
            [up1: ConvTranspose 64->32, 2x2]                |
              |                                             |
            [concat with enc1]  <───────────────────────────┘
              |
            [dec1: 64->32, Conv3x3-ReLU x2]
              |
            [out: Conv1x1, 32->2]
              |
            Output (2 ch: background + vessel)

        **Channel progression** : 1 -> 32 -> 64 -> 128 -> 64 -> 32 -> 2.
        This is deliberately shallow (only 4x total downsampling) to
        keep the model lightweight for CPU inference while still
        capturing enough context for vessel segmentation.

        **Compared to the MONAI U-Net** :  The basic version lacks
        instance normalisation (making it sensitive to input intensity
        scale), residual connections (harder to train from scratch), and
        has only 2 levels vs. MONAI's 5 levels (smaller receptive field).
        It is adequate for inference with pre-trained weights but may
        require more careful training setup than the MONAI variant.
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError('PyTorch required')

        class BasicUNet(nn.Module):
            """
            Minimal two-level U-Net for binary vessel segmentation.

            A pure-PyTorch fallback implementation with skip connections
            and transposed-convolution upsampling.  Input must be
            single-channel greyscale; output is 2-channel (background
            + vessel logits).

            See :meth:`RetinalModelWrapper._build_basic_unet` for the
            full architecture diagram and design rationale.
            """

            def __init__(self):
                super().__init__()
                # --- Encoder level 1: 1 -> 32 channels ---
                self.enc1 = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU())
                self.pool1 = nn.MaxPool2d(2)
                # --- Encoder level 2: 32 -> 64 channels ---
                self.enc2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
                self.pool2 = nn.MaxPool2d(2)
                # --- Bottleneck: 64 -> 128 channels ---
                self.bottleneck = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.Conv2d(128, 128, 3, padding=1), nn.ReLU())
                # --- Decoder level 2: upsample 128->64, then concat with enc2 (128 ch) -> 64 ---
                self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
                self.dec2 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
                # --- Decoder level 1: upsample 64->32, then concat with enc1 (64 ch) -> 32 ---
                self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
                self.dec1 = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU())
                # --- Output: 1x1 conv to 2 classes (background + vessel) ---
                self.out = nn.Conv2d(32, 2, 1)

            def forward(self, x):
                """
                Forward pass with skip connections.

                Parameters
                ----------
                x : torch.Tensor
                    Input tensor of shape ``(B, 1, H, W)``.  H and W
                    should be divisible by 4 (two 2x pooling steps).

                Returns
                -------
                torch.Tensor
                    Output logits of shape ``(B, 2, H, W)``.
                """
                # Encoder path with feature map caching for skip connections.
                e1 = self.enc1(x)                          # (B, 32, H, W)
                e2 = self.enc2(self.pool1(e1))             # (B, 64, H/2, W/2)
                b = self.bottleneck(self.pool2(e2))        # (B, 128, H/4, W/4)
                # Decoder path: upsample + skip-concatenate + convolve.
                d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))  # (B, 64, H/2, W/2)
                d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1)) # (B, 32, H, W)
                return self.out(d1)                        # (B, 2, H, W)

        return BasicUNet()
