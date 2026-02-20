"""
Deep-learning patch classifier for histopathology tiles.

This module provides :class:`PatchClassifier`, a thin wrapper around a
torchvision ResNet-50 backbone adapted for tile-level tissue classification
in histopathology whole-slide images.

Architecture
------------
* **Backbone** : ``torchvision.models.resnet50`` (50-layer residual
  network, ~25.6 M parameters).  ImageNet pre-trained weights are *not*
  loaded by default -- the model expects a custom checkpoint trained on
  histopathology tiles.
* **Head** : The final fully-connected layer (``model.fc``) is replaced
  with a new ``nn.Linear`` mapping from 2048 features to *num_classes*
  outputs.
* **Output** : A softmax probability distribution over the classes.

Default Class Mapping (num_classes=4)
-------------------------------------
===  ============
Idx  Label
===  ============
0    tumour
1    normal
2    stroma
3    background
===  ============

The exact mapping depends on the training data and label encoding used
when the checkpoint was produced.

Optional Dependency
-------------------
PyTorch and torchvision are **optional** dependencies.  If they are not
installed, importing this module succeeds but :meth:`PatchClassifier.load`
raises ``RuntimeError``.  This allows the rest of the histology package
(which relies on OpenCV / NumPy only) to function without PyTorch.
"""

import logging

log = logging.getLogger(__name__)

# Graceful optional import: PyTorch is only needed for the DL classifier.
_TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    log.info('PyTorch not available for histology models')


class PatchClassifier:
    """
    ResNet-50-based tile / patch classifier for histopathology images.

    This class manages the full lifecycle of the classification model:
    instantiation, weight loading, device placement, and inference.  It
    is designed for **inference only** (``model.eval()`` + ``torch.no_grad``).

    Parameters
    ----------
    model_path : str or pathlib.Path, optional
        Path to a ``.pth`` or ``.pt`` checkpoint file containing a
        ``state_dict`` for the modified ResNet-50.  If ``None``, the
        model is initialised with random weights (useful for smoke-testing
        the inference pipeline).
    num_classes : int, optional
        Number of output classes.  Default ``4`` (tumour, normal, stroma,
        background).  Must match the checkpoint if one is loaded.
    device : str, optional
        PyTorch device string.  Default ``'cpu'``.  Use ``'cuda'`` or
        ``'cuda:0'`` for GPU inference.

    Attributes
    ----------
    model : torch.nn.Module or None
        The underlying ResNet-50 model, or ``None`` before :meth:`load`
        is called.
    model_path : str or None
        Stored checkpoint path.
    num_classes : int
        Number of output classes.
    device : str
        Target device.

    Examples
    --------
    >>> clf = PatchClassifier(model_path='weights/histo_resnet50.pth',
    ...                       device='cuda')
    >>> clf.load()
    >>> probs = clf.predict(tile_tensor)  # shape (1, 4)
    >>> predicted_class = probs.argmax(axis=1)[0]
    """

    def __init__(self, model_path=None, num_classes=4, device='cpu'):
        """
        Store configuration; the model is **not** created until
        :meth:`load` is called.

        Parameters
        ----------
        model_path : str or None
            See class docstring.
        num_classes : int
            See class docstring.
        device : str
            See class docstring.
        """
        self.model_path = model_path
        self.num_classes = num_classes
        self.device = device
        self.model = None

    def load(self):
        """
        Build the ResNet-50 model, load weights, and set to eval mode.

        Steps performed:

        1. Instantiate ``resnet50(weights=None)`` (no ImageNet init).
        2. Replace the final FC layer with
           ``nn.Linear(2048, num_classes)``.
        3. If ``model_path`` was provided, load the ``state_dict`` from
           the checkpoint file.  ``map_location=self.device`` ensures
           tensors are loaded directly to the target device, avoiding a
           GPU-to-CPU copy when the checkpoint was saved on a different
           device.
        4. Move the model to ``self.device``.
        5. Call ``model.eval()`` to disable dropout and switch
           batch-normalisation layers to running-statistics mode.

        Returns
        -------
        PatchClassifier
            ``self``, to allow fluent chaining:
            ``clf = PatchClassifier(...).load()``.

        Raises
        ------
        RuntimeError
            If PyTorch is not installed.
        FileNotFoundError
            If ``model_path`` does not exist (propagated from
            ``torch.load``).
        RuntimeError
            If the checkpoint's ``state_dict`` keys do not match the
            model architecture (e.g. wrong ``num_classes``).
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError('PyTorch required')

        from torchvision.models import resnet50
        # Instantiate backbone without pre-trained weights.
        self.model = resnet50(weights=None)
        # Replace the 1000-class ImageNet head with our custom head.
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)

        if self.model_path:
            # map_location ensures we load directly to the target device.
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        # Eval mode: disable dropout, use running BN statistics.
        self.model.eval()
        return self

    def predict(self, tile_tensor):
        """
        Run forward inference on a single tile or batch of tiles.

        Parameters
        ----------
        tile_tensor : torch.Tensor or numpy.ndarray
            Input tile(s).  Accepted shapes:

            * ``(C, H, W)`` -- single tile (will be unsqueezed to batch
              dimension automatically).
            * ``(N, C, H, W)`` -- batch of *N* tiles.

            If a NumPy array is provided it is converted to a float32
            ``torch.Tensor``.  Channel order must be ``(C, H, W)`` --
            i.e. channels-first, which is the default for torchvision
            transforms.

        Returns
        -------
        numpy.ndarray
            Softmax probability matrix of shape ``(N, num_classes)``.
            Each row sums to 1.0.  The column index corresponds to the
            class index (see class-level docstring for the default
            mapping).

        Raises
        ------
        RuntimeError
            If :meth:`load` has not been called yet (``self.model`` is
            ``None``).

        Notes
        -----
        * Inference runs inside ``torch.no_grad()`` to disable gradient
          computation, reducing memory usage and improving speed.
        * The output probabilities are moved to CPU before conversion to
          NumPy, so this method works correctly regardless of whether the
          model is on GPU.
        """
        if self.model is None:
            raise RuntimeError('Model not loaded')

        with torch.no_grad():
            # Accept NumPy arrays transparently.
            if not isinstance(tile_tensor, torch.Tensor):
                tile_tensor = torch.from_numpy(tile_tensor).float()
            # Add batch dimension if a single tile is provided.
            if tile_tensor.dim() == 3:
                tile_tensor = tile_tensor.unsqueeze(0)
            tile_tensor = tile_tensor.to(self.device)
            output = self.model(tile_tensor)
            # Softmax over the class dimension to get probabilities.
            probs = torch.softmax(output, dim=1)
            return probs.cpu().numpy()
