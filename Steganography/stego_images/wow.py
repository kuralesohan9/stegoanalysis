# wow.py (Modified for Web Interface)

import numpy as np
import cv2
import logging
import pywt # PyWavelets library

# Import the base class from the modified AES+LSB file
from AES_LSB import UniversalSteganography

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("WowSteganography")

class WowSteganography(UniversalSteganography):
    def __init__(self, payload=0.4, wavelet="db1", levels=3):
        super().__init__(payload)
        self.wavelet = wavelet
        self.levels = levels

    def calculate_costs(self, image: np.ndarray) -> np.ndarray:
        """
        Overrides the base method to calculate costs using Wavelet domain (WOW).
        """
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image

        original_shape = gray_image.shape

        # Decompose image using Discrete Wavelet Transform (DWT)
        coeffs = pywt.wavedec2(gray_image, self.wavelet, level=self.levels)

        # Calculate costs in wavelet domain (inverse of absolute coefficient values)
        # We don't modify the approximation coefficients (coeffs[0])
        new_coeffs = [coeffs[0]] 
        for i in range(1, len(coeffs)):
            cH, cV, cD = coeffs[i] # Horizontal, Vertical, Diagonal details
            cost_H = 1.0 / (1.0 + np.abs(cH))
            cost_V = 1.0 / (1.0 + np.abs(cV))
            cost_D = 1.0 / (1.0 + np.abs(cD))
            new_coeffs.append((cost_H, cost_V, cost_D))

        # Reconstruct the cost map from the wavelet domain
        costs = pywt.waverec2(new_coeffs, self.wavelet)

        # Crop to original size as reconstruction can sometimes add padding
        costs = costs[: original_shape[0], : original_shape[1]]

        # Normalize costs to a [0, 1] range
        if costs.max() - costs.min() > 1e-9:
            costs = (costs - costs.min()) / (costs.max() - costs.min())

        return costs