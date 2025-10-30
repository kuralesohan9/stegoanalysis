# hugo.py (Modified for Web Interface)

import numpy as np
import cv2
import logging
from scipy.signal import convolve2d

# Import the base class from the modified AES+LSB file
from AES_LSB import UniversalSteganography 

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("HugoSteganography")

class HugoSteganography(UniversalSteganography):
    def __init__(self, payload=0.4, gamma=1.0, sigma=1.0):
        super().__init__(payload)
        self.gamma = gamma
        self.sigma = sigma
        self.srm_filters = self._get_srm_filters()

    def _get_srm_filters(self):
        # High-pass filters from the SRM model
        q = [4.0, 12.0, 2.0]
        filter1 = np.array([[0,0,0,0,0],[0,0,0,0,0],[0,1,-2,1,0],[0,0,0,0,0],[0,0,0,0,0]], dtype=float) / q[0]
        filter2 = np.array([[-1,2,-1],[2,-4,2],[-1,2,-1]], dtype=float) / q[1]
        filter3 = np.array([[0,0,0,0,0],[0,-1,2,-1,0],[0,2,-4,2,0],[0,-1,2,-1,0],[0,0,0,0,0]], dtype=float) / q[2]
        return [filter1, filter2, filter3]

    def calculate_costs(self, image: np.ndarray) -> np.ndarray:
        """
        Overrides the base method to calculate costs using HUGO's logic.
        """
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image

        gray_image = gray_image.astype(np.float64)

        # Calculate residuals for each filter
        residuals = [convolve2d(gray_image, f, mode="same", boundary="symm") for f in self.srm_filters]
        
        # Aggregate residuals
        total_residual = np.zeros_like(gray_image)
        for res in residuals:
            total_residual += np.abs(res)

        # Calculate costs - higher residual means lower cost to embed
        costs = 1.0 / (self.sigma + total_residual ** self.gamma)

        # Normalize costs to ensure stability
        costs[np.isinf(costs)] = 1.0
        if costs.max() - costs.min() > 1e-9:
             costs = (costs - costs.min()) / (costs.max() - costs.min())
        
        return costs