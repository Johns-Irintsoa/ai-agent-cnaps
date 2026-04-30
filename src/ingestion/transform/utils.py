import easyocr
import fitz  # PyMuPDF
import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_path
from PIL import Image

# ---------------------------------------------------------------------------
# Convert PDF to OpenCV images + OCR fallback (Tesseract ou EasyOCR)
# ---------------------------------------------------------------------------

def pdf_to_OpenCV(pdf_path):
    """
    Convertit chaque page d'un PDF en une liste d'images formatées pour OpenCV.
    """
    # Conversion du PDF en liste d'images PIL
    # thread_count=4 permet d'accélérer le traitement sur CPU
    images = convert_from_path(pdf_path, dpi=300) 
    
    cv_images = []
    for img in images:
        # Conversion PIL -> NumPy (format OpenCV)
        open_cv_image = np.array(img)
        # Conversion RGB vers BGR (standard OpenCV)
        open_cv_image = open_cv_image[:, :, ::-1].copy()
        cv_images.append(open_cv_image)
        
    return cv_images

def preprocess_image(cv_img):
    """
    Applique un prétraitement pour améliorer la précision de Tesseract sur CPU.
    """
    # Passage en nuances de gris
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    
    # Seuil adaptatif pour gérer les scans de mauvaise qualité
    processed_img = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    return processed_img

