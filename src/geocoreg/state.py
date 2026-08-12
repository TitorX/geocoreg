from dataclasses import dataclass
from typing import Any, Tuple, Optional
import numpy as np

@dataclass(frozen=True)
class InitContext:
    """Read-only context established in Phase 0."""
    ref: Any  # RasterInfo
    mov: Any  # RasterInfo
    ct_mov2ref: Any
    ct_ref2mov: Any
    overlap_bbox: Tuple[float, float, float, float]
    ref_crop: Tuple[int, int, int, int]
    mov_crop: Tuple[int, int, int, int]
    gt_mov_overlap: Tuple[float, float, float, float, float, float]
    gsd_ratio_x: float
    gsd_ratio_y: float
    ref_overlap_data: np.ndarray
    mov_overlap_data: np.ndarray

@dataclass
class DisplacementState:
    """State containing the dense displacement map and associated fields.

    D/F/C are defined on the MOV overlap pixel grid (shape follows mov_crop,
    geotransform is gt_mov_overlap). D[..., 0]/D[..., 1] are x_geo/y_geo
    displacements in MOV CRS: corrected_mov_geo - original_mov_geo.
    """
    D: np.ndarray  # (H, W, 2) float64
    F: np.ndarray  # (H, W) uint8
    C: np.ndarray  # (H, W) float32

    def copy(self) -> 'DisplacementState':
        """Create a deep copy to avoid state mutation side-effects."""
        return DisplacementState(
            D=self.D.copy(),
            F=self.F.copy(),
            C=self.C.copy(),
        )
