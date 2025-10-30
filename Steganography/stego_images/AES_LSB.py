# AES+LSB.py (Modified for Web Interface)

import numpy as np
import cv2
import json
import zlib
import pickle
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad
import logging
from typing import Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("UniversalSteganography")

class UniversalSteganography:
    def __init__(self, payload=0.4, encryption_enabled=True, compression_enabled=True):
        self.payload = payload
        self.encryption_enabled = encryption_enabled
        self.compression_enabled = compression_enabled

    # ========= DATA PREP / RECOVERY =========
    def _prepare_data(self, data: Any, key: Optional[bytes] = None) -> bytes:
        metadata = {
            "timestamp": np.datetime64("now").astype(str),
            "data_type": self._detect_data_type(data),
            "version": "1.0",
        }

        if isinstance(data, str):
            data_bytes = data.encode("utf-8")
        elif isinstance(data, (dict, list)):
            data_bytes = json.dumps(data).encode("utf-8")
        elif isinstance(data, bytes):
            data_bytes = data
        else:
            try:
                data_bytes = pickle.dumps(data)
            except Exception:
                raise ValueError(f"Cannot serialize data type: {type(data)}")

        metadata["data_length"] = len(data_bytes)
        metadata_json = json.dumps(metadata).encode("utf-8")
        metadata_length = len(metadata_json).to_bytes(4, "big")

        full_payload = metadata_length + metadata_json + data_bytes
        crc = zlib.crc32(full_payload)
        payload_with_crc = crc.to_bytes(4, "big") + full_payload

        if self.compression_enabled:
            payload_with_crc = zlib.compress(payload_with_crc, level=9)

        if self.encryption_enabled and key:
            payload_with_crc = self._encrypt_data(payload_with_crc, key)

        return payload_with_crc

    def _detect_data_type(self, data: Any) -> str:
        if isinstance(data, str): return "text"
        if isinstance(data, (dict, list)): return "json"
        if isinstance(data, bytes): return "binary"
        return "object"

    def _encrypt_data(self, data: bytes, key: bytes) -> bytes:
        nonce = get_random_bytes(16)
        cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
        ciphertext, tag = cipher.encrypt_and_digest(data)
        return nonce + tag + ciphertext

    # ========= CORE STEGANOGRAPHY =========
    def calculate_costs(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(image, 100, 200)
        costs = 1.0 - (edges / 255.0)
        costs = np.clip(costs + 0.1, 0.1, 1.0)
        return costs

    def embed(self, cover: np.ndarray, data: Any, key: Optional[bytes] = None) -> np.ndarray:
        prepared_data = self._prepare_data(data, key)
        binary_data = "".join([format(byte, "08b") for byte in prepared_data])
        data_bits = [int(b) for b in binary_data]
        n_bits = len(data_bits)

        costs = self.calculate_costs(cover)
        h, w = costs.shape
        capacity = int(self.payload * (h * w))
        if n_bits > capacity:
            raise ValueError(f"Data too large: {n_bits} bits needed, only {capacity} available")

        stego = cover.copy().astype(np.int32)
        flat_costs = costs.flatten()
        pixel_indices = np.argsort(flat_costs)[::-1]

        ch = 0  # blue channel
        bit_idx = 0
        for pix_pos in pixel_indices:
            if bit_idx >= n_bits: break
            row, col = pix_pos // w, pix_pos % w
            current_val = int(cover[row, col, ch])
            current_bit = current_val & 1
            target_bit = data_bits[bit_idx]
            if current_bit != target_bit:
                stego[row, col, ch] = current_val - 1 if current_val % 2 != 0 else current_val + 1
            bit_idx += 1
            
        return np.clip(stego, 0, 255).astype(np.uint8)

    # ========= FILE HELPER (FOR SINGLE FILE) =========
    def embed_file(self, cover_path: str, output_path: str, data: Any, key: Optional[bytes] = None) -> bool:
        try:
            cover = cv2.imread(cover_path)
            if cover is None:
                raise ValueError("Could not load cover image")
            stego = self.embed(cover, data, key)
            success = cv2.imwrite(output_path, stego)
            if not success:
                raise ValueError("Failed to save stego image")
            logger.info(f"Successfully embedded data into {output_path}")
            return True
        except Exception as e:
            logger.error(f"Embedding failed for {cover_path}: {str(e)}")
            return False