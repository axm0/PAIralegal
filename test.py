import pytesseract
import torch
import transformers

pytesseract.pytesseract.tesseract_cmd = r'C:\Developer\Tools\Tesseract-OCR\tesseract.exe'

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
try:
    version = pytesseract.get_tesseract_version()
    print("✓ Tesseract version:", version)
except Exception as e:
    print("✗ Tesseract error:", str(e))
print("Transformers version:", transformers.__version__)
