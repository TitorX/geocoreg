from abc import ABC, abstractmethod
import numpy as np

class FeatureMatcher(ABC):
    @abstractmethod
    def match(self, img_ref: np.ndarray, img_mov: np.ndarray) -> tuple:
        """
        Match two image patches and return correspondence info.

        Args:
            img_ref: (H, W) or (H, W, C) uint8 array representing the reference patch.
            img_mov: (H, W) or (H, W, C) uint8 array representing the moving patch.

        Returns:
            A tuple of 6 elements:
            - kp_ref: List of cv2.KeyPoint for reference image.
            - kp_mov: List of cv2.KeyPoint for moving image.
            - good_matches: List of cv2.DMatch mapping kp_ref to kp_mov.
            - ratios: List of floats representing ratio scores or distance scores.
            - inlier_mask: (N,) boolean numpy array indicating which matches are inliers.
            - reproj_errors: (N,) float numpy array with reprojection errors of the inliers.
        """
        pass

    @abstractmethod
    def match_backward(self, img_ref: np.ndarray, img_mov: np.ndarray) -> tuple:
        """
        Reverse match from moving to reference. Used for forward-backward consistency check.
        """
        pass
