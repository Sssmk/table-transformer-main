import fitz  # PyMuPDF
import os
import json
import pytesseract
import gc
from PIL import Image, ImageEnhance

def preprocess_image_for_ocr(img):
    """
    Image enhancement for OCR: Grayscale -> Contrast
    """
    img = img.convert('L')
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)
    return img

def process_pdf_to_data(pdf_path, output_dir):
    """
    Convert PDF to Images & Text (Auto-detect per page)
    
    Logic:
    1. If page has text > 100 chars AND No images -> Native Mode (Fastest, Clear)
    2. If page has text > 100 chars BUT Has images -> OCR Mode (To capture image-based tables)
    3. If page has little text -> OCR Mode (Scanned document)
    """
    img_dir = os.path.join(output_dir, "images")
    words_dir = os.path.join(output_dir, "words")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(words_dir, exist_ok=True)

    doc = fitz.open(pdf_path)
    
    print(f"=== Processing PDF | Total Pages: {len(doc)} ===")
    
    file_pairs = []

    for page_num, page in enumerate(doc):
        # -------------------------------------------------
        # 1. Smart Detection Strategy (Per-Page)
        # -------------------------------------------------
        
        # A. Analyze Page Features
        text = page.get_text()
        char_count = len(text)
        images_on_page = page.get_images() # List of images on this page
        has_images = len(images_on_page) > 0
        
        # B. Decision Logic
        use_ocr = True 
        mode_label = "OCR"
        
        if char_count > 100 and not has_images:
            # Lots of text & No images -> Pure Native PDF
            use_ocr = False
            mode_label = "Native (Text)"
        elif char_count > 100 and has_images:
            # Lots of text & Has images -> Hybrid Page (Text + Image Tables)
            # Must use OCR to catch tables inside images
            use_ocr = True
            mode_label = "OCR (Hybrid)"
        else:
            # Little text -> Scanned Document
            use_ocr = True
            mode_label = "OCR (Scanned)"
            
        print(f"  >> Page {page_num + 1}: [{mode_label}] (Chars:{char_count}, Imgs:{len(images_on_page)})", end="\r")
        
        # -------------------------------------------------
        # 2. Smart Zoom (Prevent Memory Crash)
        # -------------------------------------------------
        default_pix = page.get_pixmap()
        
        # Only zoom in if using OCR and the image is small
        if use_ocr and default_pix.width < 1500:
            zoom = 2.0 
        else:
            zoom = 1.0 
            
        mat = fitz.Matrix(zoom, zoom)
        
        # A. Save Image
        pix = page.get_pixmap(matrix=mat)
        page_img_name = f"page_{page_num:03d}.jpg"
        img_path = os.path.join(img_dir, page_img_name)
        pix.save(img_path)
        pix = None # Free memory immediately
        
        words_data = []
        
        # B. Extract Content
        if not use_ocr:
            # === Native Mode ===
            for w in page.get_text("words"):
                words_data.append({
                    "bbox": [w[0]*zoom, w[1]*zoom, w[2]*zoom, w[3]*zoom], 
                    "text": w[4]
                })
        else:
            # === OCR Mode ===
            try:
                raw_img = Image.open(img_path)
                ocr_img = preprocess_image_for_ocr(raw_img)
                
                # Use English engine (eng) + PSM 6 (Block of text)
                custom_config = r'--oem 3 --psm 6'
                
                data = pytesseract.image_to_data(
                    ocr_img, 
                    lang='eng', 
                    config=custom_config, 
                    output_type=pytesseract.Output.DICT
                )
                
                for i in range(len(data['text'])):
                    text = data['text'][i].strip()
                    if text:
                        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                        words_data.append({
                            "text": text,
                            "bbox": [x, y, x+w, y+h]
                        })
                
                raw_img.close()
                ocr_img.close()
            except Exception as e:
                print(f"\n  [OCR Error] Page {page_num+1}: {e}")

        # C. Save JSON
        json_path = os.path.join(words_dir, f"page_{page_num:03d}_words.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({"words": words_data}, f, ensure_ascii=False)
            
        file_pairs.append((img_path, json_path))
        
        # Force Garbage Collection
        gc.collect()

    return file_pairs