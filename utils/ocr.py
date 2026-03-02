import os
import cv2
import numpy as np
import logging
import base64
from dotenv import load_dotenv
from mistralai import Mistral

load_dotenv()

class OCR:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.api_key = os.getenv("MISTRAL_API_KEY")
        self.client = Mistral(api_key=self.api_key)

    def _is_blurry(self, img, size: int = 60, blur_thresh: float = 32.0) -> bool:
        (h, w) = img.shape
        (cX, cY) = (int(w / 2.0), int(h / 2.0))
        fft = np.fft.fft2(img)
        fftShift = np.fft.fftshift(fft)
        fftShift[cY - size:cY + size, cX - size:cX + size] = 0
        fftShift = np.fft.ifftshift(fftShift)
        recon = np.fft.ifft2(fftShift)
        magnitude = 20 * np.log(np.abs(recon))
        return np.mean(magnitude) <= blur_thresh

    def _sharpen_image(self, img):
        blurred = cv2.GaussianBlur(img, (7, 7), sigmaX=3)
        return cv2.addWeighted(img, 3.5, blurred, -2.5, 0)

    def extract_text(self, image_bytes: bytes) -> tuple[str, float]:
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                raise ValueError("Invalid image data")
            
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if self._is_blurry(gray_img):
                gray_img = self._sharpen_image(gray_img)

            success, buffer = cv2.imencode('.jpeg', gray_img)
            if not success:
                raise ValueError("Failed to encode processed image to JPEG format.")
            
            base64_encoded = base64.b64encode(buffer.tobytes()).decode('utf-8')
            base64_url = f"data:image/jpeg;base64,{base64_encoded}"

            self.logger.info("Sending image to Mistral OCR...")
            ocr_response = self.client.ocr.process(
                model="mistral-ocr-latest",
                document={
                    "type": "image_url",
                    "image_url": base64_url,
                },
            )

            full_text = ""
            for page in ocr_response.pages:
                full_text += page.markdown + "\n"

            return full_text.strip(), 1.0
        
        except Exception as e:
            self.logger.error(f"Mistral OCR Extraction failed: {e}")
            raise e