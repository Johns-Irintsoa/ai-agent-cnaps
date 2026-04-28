import os
from pathlib import Path

# Configuration des dossiers temporaires internes au Docker
CACHE_DIR = "/tmp/.cache"
os.environ["XDG_CACHE_HOME"] = CACHE_DIR
os.environ["HF_HOME"] = CACHE_DIR
os.makedirs(CACHE_DIR, exist_ok=True)

# Monkey Patch pour RapidOCR (évite le Permission Denied sur .onnx)
try:
    import rapidocr_onnxruntime
    from rapidocr_onnxruntime import utils
    
    new_model_path = Path(CACHE_DIR) / "rapidocr_models"
    os.makedirs(new_model_path, exist_ok=True)

    # Patch uniquement si la classe ModelPath existe (v1.3.1 et moins)
    if hasattr(utils, 'ModelPath'):
        utils.ModelPath.models_path = new_model_path
    
    # Pour les versions plus récentes, on utilise la variable d'env
    os.environ["RAPIDOCR_CACHE"] = str(new_model_path)
    
except (ImportError, AttributeError):
    pass

print(f"--- Infrastructure initialisée : Cache redirigé vers {CACHE_DIR} ---")