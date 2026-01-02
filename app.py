import streamlit as st
import os
import sys
import tempfile
import json
import zipfile
import io
import pandas as pd
from PIL import Image

# Import utils
from utils import process_pdf_to_data

# Setup paths for model inference
sys.path.append(os.path.join(os.getcwd(), "src"))
sys.path.append(os.path.join(os.getcwd(), "detr"))

try:
    from inference import TableExtractionPipeline
except ImportError as e:
    st.error(f"Environment Error: {e}")
    st.stop()

# ================= Config =================
DET_MODEL = "pubtables1m_detection_detr_r18.pth"
STR_MODEL = "TATR-v1.1-All-msft.pth"
DEVICE = "cuda" # Change to "cpu" if no GPU is available
# ==========================================

st.set_page_config(page_title="PDF Table Extractor", layout="wide")

@st.cache_resource
def load_pipeline():
    if not os.path.exists(DET_MODEL) or not os.path.exists(STR_MODEL):
        st.error(f"Model files missing! Please check the root directory.")
        return None
    return TableExtractionPipeline(
        det_config_path='src/detection_config.json',
        det_model_path=DET_MODEL,
        det_device=DEVICE,
        str_config_path='src/structure_config.json',
        str_model_path=STR_MODEL,
        str_device=DEVICE
    )

def merge_cross_page_tables(csv_results):
    if not csv_results: return []
    tables_data = []
    
    # 1. Parse CSVs
    for fname, content in csv_results:
        try:
            parts = fname.replace('.csv', '').split('_')
            page_num = int(parts[1])
            df = pd.read_csv(io.StringIO(content))
            tables_data.append({'fname': fname, 'page': page_num, 'df': df})
        except:
            pass
            
    tables_data.sort(key=lambda x: x['page'])
    merged_results = []
    
    # 2. Merge Logic
    i = 0
    while i < len(tables_data):
        current = tables_data[i]
        if i + 1 < len(tables_data):
            next_table = tables_data[i+1]
            # Heuristics: Consecutive page + Same columns + Same headers
            is_consecutive = (next_table['page'] == current['page'] + 1)
            is_same_cols = (len(current['df'].columns) == len(next_table['df'].columns))
            headers_match = (list(current['df'].columns) == list(next_table['df'].columns))
            
            if is_consecutive and is_same_cols and headers_match:
                merged_df = pd.concat([current['df'], next_table['df']], ignore_index=True)
                current['df'] = merged_df
                i += 1 
                continue 

        merged_results.append((current['fname'], current['df'].to_csv(index=False)))
        i += 1

    return merged_results

def main():
    st.title("Smart PDF Table Extractor")
    
    st.markdown("""
    Upload your PDF file. The system will automatically detect the best extraction method (OCR or Native) for each page.
    """)
    
    # Load Model
    pipe = load_pipeline()
    if not pipe: return

    uploaded_file = st.file_uploader("Upload PDF File", type=["pdf"])

    if uploaded_file:
        with tempfile.TemporaryDirectory() as temp_root:
            pdf_path = os.path.join(temp_root, uploaded_file.name)
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            if st.button("Start Extraction", type="primary"):
                progress_bar = st.progress(0)
                status = st.empty()

                # Step 1: Preprocessing
                status.info("Phase 1: Analyzing Document Pages...")
                try:
                    # No need to pass force_ocr_global anymore
                    file_pairs = process_pdf_to_data(pdf_path, temp_root)
                except Exception as e:
                    st.error(f"Preprocessing Failed: {e}")
                    return
                
                progress_bar.progress(30)

                # Step 2: Inference
                status.info("Phase 2: Detecting Tables...")
                csv_results = []
                total_pages = len(file_pairs)

                for i, (img_path, json_path) in enumerate(file_pairs):
                    try:
                        image = Image.open(img_path).convert("RGB")
                        with open(json_path, 'r') as f:
                            tokens = json.load(f).get('words', [])

                        # Fill missing metadata
                        for idx, token in enumerate(tokens):
                            if 'span_num' not in token: token['span_num'] = idx
                            if 'line_num' not in token: token['line_num'] = 0
                            if 'block_num' not in token: token['block_num'] = 0

                        # Run Inference
                        extracted_tables = pipe.extract(
                            image, tokens,
                            out_objects=False, out_cells=False, out_html=False, out_csv=True
                        )

                        for table_idx, table in enumerate(extracted_tables):
                            if 'csv' in table and table['csv'] and table['csv'][0]:
                                filename = f"Page_{i+1:02d}_Table_{table_idx+1:02d}.csv"
                                csv_results.append((filename, table['csv'][0]))
                        
                        image.close()
                    except Exception as e:
                        print(f"Error on page {i}: {e}")

                    # Memory Management
                    import gc
                    gc.collect()
                    
                    # Update Progress
                    current_prog = 30 + int(70 * (i + 1) / total_pages)
                    progress_bar.progress(min(current_prog, 100))

                # Step 3: Export
                progress_bar.progress(100)
                if csv_results:
                    final_results = merge_cross_page_tables(csv_results)
                    status.success(f"Success! Extracted {len(csv_results)} tables (Merged to {len(final_results)}).")
                    
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, "w") as zf:
                        for fname, content in final_results:
                            zf.writestr(fname, str(content))
                    
                    st.download_button(
                        "Download CSV Results (ZIP)",
                        data=zip_buffer.getvalue(),
                        file_name="tables.zip",
                        mime="application/zip"
                    )
                else:
                    status.warning("No tables detected.")

if __name__ == "__main__":
    main()