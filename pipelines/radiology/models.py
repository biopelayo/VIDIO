"""
Deep Learning Models for Radiology
====================================

Provides :class:`UNet3DWrapper`, a convenience wrapper around MONAI's
3D U-Net architecture for volumetric medical image segmentation.

The module is designed for **optional** usage -- PyTorch and MONAI are
imported lazily and their absence is handled gracefully:

* If **PyTorch** is unavailable, the module-level flag
  ``_TORCH_AVAILABLE`` is set to ``False`` and :meth:`UNet3DWrapper.load`
  raises ``RuntimeError``.
* If **MONAI** is unavailable, :meth:`load` raises ``ImportError`` at
  model-instantiation time (PyTorch itself may still be present for
  other uses).

Architecture
------------
The default 3D U-Net uses the following encoder-decoder layout::

    Encoder channels : 16 -> 32 -> 64 -> 128 -> 256
    Strides          :    2     2     2      2
    Decoder          : symmetric (MONAI default)

This yields approximately 3.9 M parameters, suitable for moderate-
resolution brain volumes (e.g. 128x128x128 or 192x192x192).

Dependencies
------------
- **torch** (>= 1.10) : Tensor operations and autograd.
- **monai** (>= 1.0) : ``monai.networks.nets.UNet`` architecture.
"""

import logging

log = logging.getLogger(__name__)

# -- Lazy PyTorch import ---------------------------------------------------
_TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    log.info('PyTorch not available for radiology models')


class UNet3DWrapper:
    """
    Wrapper for MONAI's 3D U-Net volumetric segmentation model.

    Encapsulates model construction, weight loading, device placement,
    and inference behind a simple ``load`` / ``predict`` interface.

    The wrapper does **not** subclass ``torch.nn.Module`` -- it holds
    the model as an attribute and manages the numpy-to-tensor
    conversion, dimension unsqueezing, and argmax post-processing that
    are otherwise boilerplate in every inference script.

    Parameters
    ----------
    model_path : str or None, optional
        Path to a ``state_dict`` file (``.pt`` / ``.pth``).  If
        ``None``, the model is initialised with random weights
        (useful for architecture testing).
    in_channels : int, optional
        Number of input channels (default ``1`` for single-modality
        greyscale volumes).  Set to ``4`` for multi-sequence MRI
        (T1 + T2 + FLAIR + DWI stacked along the channel axis).
    out_channels : int, optional
        Number of output segmentation classes (default ``2`` for
        binary foreground/background).
    device : str, optional
        PyTorch device string (default ``'cpu'``).  Use ``'cuda'`` or
        ``'cuda:0'`` for GPU inference.

    Attributes
    ----------
    model_path : str or None
        Path supplied at construction.
    in_channels : int
        Input channel count.
    out_channels : int
        Output class count.
    device : str
        Target device string.
    model : monai.networks.nets.UNet or None
        The underlying MONAI model, populated after :meth:`load` is
        called.  ``None`` before loading.

    Examples
    --------
    >>> wrapper = UNet3DWrapper(model_path='brain_seg.pt', device='cuda')
    >>> wrapper.load()
    >>> seg_map = wrapper.predict(volume_array)  # (D, H, W) np.ndarray
    >>> seg_map.shape
    (D, H, W)

    See Also
    --------
    monai.networks.nets.UNet : Underlying architecture.
    pipelines.radiology.segmentation.segment_lesions_3d :
        Threshold-based alternative that does not require a trained
        model.
    """

    def __init__(self, model_path=None, in_channels=1, out_channels=2, device='cpu'):
        self.model_path = model_path
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.device = device
        self.model = None

    def load(self):
        """
        Instantiate the 3D U-Net architecture and load weights.

        Architecture configuration:

        * ``spatial_dims=3`` -- full volumetric convolutions.
        * ``channels=(16, 32, 64, 128, 256)`` -- five encoder levels
          with progressively increasing feature maps.  The small
          initial channel count (16) keeps memory usage manageable
          for whole-brain volumes.
        * ``strides=(2, 2, 2, 2)`` -- each encoder level halves the
          spatial resolution (via strided convolutions).

        If ``model_path`` was provided at construction, the saved
        ``state_dict`` is loaded with ``map_location=self.device``
        (enabling transparent CPU/GPU portability).  The model is
        placed on ``self.device`` and set to eval mode (disabling
        dropout and batch-norm running stats updates).

        Returns
        -------
        UNet3DWrapper
            ``self``, to allow method chaining:
            ``wrapper = UNet3DWrapper(...).load()``.

        Raises
        ------
        RuntimeError
            If PyTorch is not installed.
        ImportError
            If MONAI is not installed (PyTorch alone is not
            sufficient).

        Notes
        -----
        * Weights are loaded via ``torch.load`` which uses ``pickle``
          internally.  Only load weights from **trusted** sources.
        * ``model.eval()`` is called automatically; there is no need
          to call it again before :meth:`predict`.
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError('PyTorch required')

        try:
            from monai.networks.nets import UNet
            self.model = UNet(
                spatial_dims=3,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                channels=(16, 32, 64, 128, 256),    # 5-level encoder
                strides=(2, 2, 2, 2),                # 4 down-sampling steps
            )
        except ImportError:
            log.warning('MONAI not available, 3D U-Net requires MONAI')
            raise

        # Load pre-trained weights if a path was provided
        if self.model_path:
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()
        return self

    def predict(self, volume_tensor):
        """
        Run forward inference and return an integer segmentation map.

        Handles the common dimension-mismatch issues that arise when
        passing raw volumes to a model that expects a 5D tensor
        ``(N, C, D, H, W)``:

        * **3D input** ``(D, H, W)`` -- unsqueezed twice to add batch
          and channel dimensions: ``(1, 1, D, H, W)``.
        * **4D input** ``(C, D, H, W)`` -- unsqueezed once to add the
          batch dimension: ``(1, C, D, H, W)``.
        * **5D input** ``(N, C, D, H, W)`` -- passed through as-is.

        NumPy arrays are automatically converted to ``torch.Tensor``
        (``float32``).  Inference runs inside ``torch.no_grad()`` to
        disable gradient tracking and reduce memory usage.

        The raw model output has shape ``(N, out_channels, D, H, W)``
        (one logit per class).  ``argmax(dim=1)`` collapses the class
        dimension into an integer label per voxel, and the result is
        squeezed back to ``(D, H, W)`` and returned as a NumPy array.

        Parameters
        ----------
        volume_tensor : numpy.ndarray or torch.Tensor
            Input volume.  Accepted shapes:

            - ``(D, H, W)`` -- single-channel volume.
            - ``(C, D, H, W)`` -- multi-channel volume.
            - ``(N, C, D, H, W)`` -- batched multi-channel volume.

        Returns
        -------
        numpy.ndarray
            Integer segmentation map of shape ``(D, H, W)`` with
            values in ``[0, out_channels)``.  Dtype is platform-
            dependent (typically ``int64``).

        Raises
        ------
        RuntimeError
            If :meth:`load` has not been called yet.

        Notes
        -----
        * For large volumes that exceed GPU memory, consider slicing
          along the depth axis and predicting in overlapping patches,
          then stitching results.
        * The ``argmax`` post-processing is a hard assignment; for
          probability maps, access ``self.model(tensor)`` directly
          and apply ``softmax(dim=1)``.
        """
        if self.model is None:
            raise RuntimeError('Model not loaded')

        with torch.no_grad():
            # -- Ensure tensor type ---------------------------------------------
            if not isinstance(volume_tensor, torch.Tensor):
                volume_tensor = torch.from_numpy(volume_tensor).float()

            # -- Unsqueeze to 5D (N, C, D, H, W) if necessary ------------------
            if volume_tensor.dim() == 3:
                # (D, H, W) -> (1, 1, D, H, W)
                volume_tensor = volume_tensor.unsqueeze(0).unsqueeze(0)
            elif volume_tensor.dim() == 4:
                # (C, D, H, W) -> (1, C, D, H, W)
                volume_tensor = volume_tensor.unsqueeze(0)

            volume_tensor = volume_tensor.to(self.device)
            output = self.model(volume_tensor)

            # Collapse class logits -> integer labels, remove batch dim
            return torch.argmax(output, dim=1).squeeze().cpu().numpy()
