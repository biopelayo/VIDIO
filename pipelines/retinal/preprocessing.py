"""
Retinal Image Preprocessing Functions
======================================

Modality-specific preprocessing routines for the three retinal imaging
modalities supported by VIDIO:

* **Fundus photography** (:func:`preprocess_fundus`) -- green channel
  extraction, CLAHE contrast enhancement, bilateral edge-preserving
  smoothing.
* **Optical Coherence Tomography** (:func:`preprocess_oct`) -- greyscale
  conversion, min-max normalisation, bilateral smoothing with a wider
  colour sigma to handle speckle noise.
* **Slit-lamp photography** (:func:`preprocess_slit_lamp`) -- greyscale
  conversion, CLAHE with large tile grid, Gaussian smoothing for fine
  texture analysis of crystalline lens opacities.

All functions accept a NumPy array and return a single-channel float32
array in the [0, 255] value range.  The returned array is ready for the
segmentation stage of the pipeline.

Standalone vs. Pipeline Usage
-----------------------------
These functions are also used by :class:`RetinalPipeline.preprocess`,
which inlines the fundus path directly for speed.  The standalone
functions here are provided so that individual preprocessing steps can
be tested, composed, or used in notebook-based exploratory workflows.
"""
import numpy as np

from core.intensity import clahe_enhance, normalize_minmax
from core.filtering import bilateral_filter, gaussian_filter


def extract_green_channel(img):
    """
    Extract the green channel from a colour retinal image.

    The green channel (index 1 in BGR / RGB arrays) provides the best
    contrast for retinal vessel visualisation because oxyhaemoglobin
    (HbO2) and deoxyhaemoglobin (Hb) have their peak absorption near
    540 nm, which falls squarely in the green band of the visible
    spectrum.  Vessels therefore appear maximally dark against the
    lighter fundus background in this channel.

    Additionally, the green channel has the highest signal-to-noise
    ratio in typical CMOS/CCD fundus cameras, making it the preferred
    input for Laplacian-based edge detection and morphological
    segmentation.

    Parameters
    ----------
    img : numpy.ndarray
        Input image.  Accepted shapes:

        * ``(H, W, C)`` with ``C >= 3`` -- colour image; channel 1
          (green) is extracted.
        * ``(H, W)`` -- already greyscale; returned as float32 without
          modification.

    Returns
    -------
    numpy.ndarray
        Float32 single-channel array of shape ``(H, W)``.

    Notes
    -----
    This function does *not* distinguish between BGR and RGB ordering;
    it always takes ``img[:, :, 1]``.  For both OpenCV (BGR) and PIL
    (RGB) conventions, index 1 corresponds to the green channel.
    """
    if len(img.shape) == 3 and img.shape[2] >= 3:
        return img[:, :, 1].astype(np.float32)
    return img.astype(np.float32)


def enhance_vessels(img, clip_limit=3.0):
    """
    Enhance retinal vessel contrast using CLAHE.

    Contrast-Limited Adaptive Histogram Equalisation (CLAHE) is applied
    to compensate for the uneven illumination field inherent in fundus
    photography (bright near the optic disc, progressively darker
    toward the peripheral retina).  By equalising histograms within
    local tiles, CLAHE normalises contrast across the entire field of
    view without the over-amplification artefacts of global histogram
    equalisation.

    Parameters
    ----------
    img : numpy.ndarray
        Single-channel float32 image (typically the green channel
        extracted by :func:`extract_green_channel`).
    clip_limit : float, optional
        Contrast amplification limit per tile.  Default is **3.0**, which
        is slightly higher than the common default of 2.0 to boost
        visibility of faint vessels and early-stage lesions in
        low-contrast peripheral regions.  Values above 4.0 tend to
        introduce visible noise amplification.

    Returns
    -------
    numpy.ndarray
        CLAHE-enhanced float32 image.

    Notes
    -----
    The ``grid_size=(8, 8)`` parameter divides the image into 64 tiles.
    For a typical 2048 x 1536 fundus image each tile is approximately
    256 x 192 pixels -- large enough for stable histogram estimation
    but small enough to correct local illumination gradients.
    """
    enhanced = clahe_enhance(img, clip_limit=clip_limit, grid_size=(8, 8))
    return enhanced


def preprocess_fundus(img):
    """
    Full preprocessing chain for colour fundus photography images.

    Applies the three-step fundus preprocessing pipeline:

    1. **Green channel extraction** -- isolate the band with highest
       haemoglobin absorption contrast (:func:`extract_green_channel`).
    2. **CLAHE enhancement** -- normalise local contrast across the
       uneven illumination field (:func:`enhance_vessels`).
    3. **Bilateral filtering** -- suppress sensor noise while preserving
       sharp vessel walls and lesion boundaries.

    Parameters
    ----------
    img : numpy.ndarray
        Raw colour fundus image of shape ``(H, W, C)`` with ``C >= 3``,
        or a greyscale image of shape ``(H, W)``.

    Returns
    -------
    numpy.ndarray
        Preprocessed single-channel float32 image ready for
        segmentation.

    Notes
    -----
    Bilateral filter parameter choices:

    * ``d=0`` -- OpenCV automatically computes the kernel diameter from
      ``sigma_space``, which is more robust than hard-coding a diameter
      that may not suit all resolutions.
    * ``sigma_color=2.0`` -- deliberately tight colour sigma so that
      only pixels with very similar intensity are averaged together.
      This prevents smoothing across vessel edges (where the intensity
      jump can be 20-50 grey levels) while still averaging within
      homogeneous tissue.
    * ``sigma_space=5.0`` -- moderate spatial extent (~5 px Gaussian
      falloff) provides effective noise suppression without blurring
      fine capillaries.

    See Also
    --------
    extract_green_channel : Step 1 -- channel isolation.
    enhance_vessels : Step 2 -- CLAHE contrast enhancement.
    """
    # Step 1: isolate the green channel for optimal vessel contrast.
    green = extract_green_channel(img)
    # Step 2: CLAHE to normalise uneven retinal illumination.
    enhanced = enhance_vessels(green)
    # Step 3: bilateral filter to suppress noise, preserve edges.
    filtered = bilateral_filter(enhanced, d=0, sigma_color=2.0, sigma_space=5.0)
    return filtered


def preprocess_oct(img):
    """
    Preprocess an Optical Coherence Tomography (OCT) image.

    OCT images are cross-sectional views of the retina acquired by
    interferometric measurement of back-scattered near-infrared light.
    They exhibit a characteristic speckle noise pattern and typically
    have a much wider dynamic range than fundus photographs.

    The preprocessing chain:

    1. **Greyscale conversion** -- average all channels if the input is
       multi-channel (some OCT exports save pseudo-colour images).
    2. **Min-max normalisation** to [0, 255] -- maps the full dynamic
       range of the scan to the uint8-compatible domain expected by
       downstream processing.
    3. **Bilateral filtering** with a wider colour sigma
       (``sigma_color=3.0``) than the fundus pipeline to handle OCT
       speckle noise, which creates larger intensity fluctuations than
       typical sensor noise.

    Parameters
    ----------
    img : numpy.ndarray
        Raw OCT image.  Shape ``(H, W, C)`` (pseudo-colour export) or
        ``(H, W)`` (native greyscale).

    Returns
    -------
    numpy.ndarray
        Single-channel float32 image normalised to [0, 255], with
        speckle noise suppressed.

    Notes
    -----
    **Why min-max normalisation instead of CLAHE?**  OCT B-scans have a
    naturally layered structure with alternating bright (high-reflectance)
    and dark (low-reflectance) bands corresponding to retinal layers.
    CLAHE would over-enhance the transitions between layers and could
    create artificial contrast artefacts at layer boundaries, interfering
    with thickness measurement algorithms.  Simple min-max normalisation
    preserves the relative reflectance profile while bringing all scans
    to a common intensity scale.

    **sigma_color = 3.0** (vs. 2.0 for fundus) accounts for the larger
    intensity variance caused by OCT speckle.  A tighter sigma would
    leave too much speckle un-smoothed; a wider sigma would blur the
    retinal layer boundaries that are critical for segmentation.
    """
    # Convert pseudo-colour OCT exports to greyscale.
    if len(img.shape) == 3:
        gray = np.mean(img, axis=2).astype(np.float32)
    else:
        gray = img.astype(np.float32)
    # Normalise the wide OCT dynamic range to [0, 255].
    normalized = normalize_minmax(gray, 0, 255)
    # Bilateral filter with wider colour sigma for speckle suppression.
    filtered = bilateral_filter(normalized, d=0, sigma_color=3.0, sigma_space=5.0)
    return filtered


def preprocess_slit_lamp(img):
    """
    Preprocess a slit-lamp photograph of the anterior segment / lens.

    Slit-lamp images capture the crystalline lens and anterior chamber
    at high magnification, often under slit-beam illumination that
    creates a narrow, very bright strip against a dark background.
    These images are of particular interest for Alzheimer's disease
    biomarker research, where amyloid-beta deposits in the crystalline
    lens (supranuclear cataracts) may serve as an early indicator.

    The preprocessing chain:

    1. **Greyscale conversion** -- average colour channels.
    2. **CLAHE with large tile grid** -- ``grid_size=(16, 16)`` creates
       256 tiles for fine-grained local contrast correction.
    3. **Gaussian smoothing** -- light 3x3 Gaussian to suppress fine
       sensor noise without blurring the small opacity features of
       interest.

    Parameters
    ----------
    img : numpy.ndarray
        Raw slit-lamp photograph.  Shape ``(H, W, C)`` or ``(H, W)``.

    Returns
    -------
    numpy.ndarray
        Single-channel float32 image with enhanced local contrast and
        light smoothing.

    Notes
    -----
    **Why grid_size = (16, 16) instead of (8, 8)?**  Slit-lamp images
    have extreme local contrast variation: the slit beam region is
    near-saturated while the surrounding iris/pupil area is very dark.
    A finer tile grid (256 tiles vs. 64) ensures that CLAHE can
    normalise contrast *within* the slit beam and *within* the dark
    region independently, revealing subtle lens opacities that would
    otherwise be washed out by coarser tiles that straddle the
    bright/dark boundary.

    **clip_limit = 2.0** (lower than the 3.0 used for fundus) is chosen
    because slit-lamp images have inherently higher local contrast
    within the beam; a higher clip limit would over-amplify the already-
    bright beam region, potentially saturating subtle opacity details.

    **Gaussian vs. bilateral** :  Unlike fundus images where sharp vessel
    edges must be preserved, slit-lamp images of the lens have smooth,
    gradually varying opacities.  A simple 3x3 Gaussian is sufficient
    for noise suppression and computationally cheaper; bilateral
    filtering's edge-preservation benefit is not needed here.
    """
    # Convert to greyscale if multi-channel.
    if len(img.shape) == 3:
        gray = np.mean(img, axis=2).astype(np.float32)
    else:
        gray = img.astype(np.float32)
    # CLAHE with fine tile grid (16x16 = 256 tiles) for the extreme
    # contrast variation in slit-beam illumination.
    enhanced = clahe_enhance(gray, clip_limit=2.0, grid_size=(16, 16))
    # Light Gaussian smoothing to suppress sensor noise.
    filtered = gaussian_filter(enhanced, ksize=3)
    return filtered
