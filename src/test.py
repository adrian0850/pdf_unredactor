import streamlit as st
import cv2
import numpy as np
import fitz  # PyMuPDF
from PIL import Image, ImageFont
import io
import logging
import os
import re
import statistics
import time
import functools

from streamlit_image_coordinates import streamlit_image_coordinates

import config_loader as cfg

logger = logging.getLogger(__name__)
_FONT = fitz.Font("helv")

def _tesseract_available():
    try:
        import pytesseract  # noqa: F401
        return True
    except Exception:
        return False

# --- CORE COMPUTER VISION LOGIC ---

def pdf_to_image(pdf_data, page_num, dpi=300):
    """Rasterize a PDF page to a high-res OpenCV image."""
    logger.debug(f"Rasterizing PDF page {page_num} at {dpi} DPI")
    doc = fitz.open(stream=pdf_data, filetype="pdf")
    page = doc[page_num]
    pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def image_bytes_to_bgr(image_bytes):
    """Load image bytes into an OpenCV BGR image."""
    logger.debug("Loading image bytes into BGR array")
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    rgb = np.array(pil_img)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

def extract_words_from_pdf(pdf_data, page_num, dpi=300):
    """Extract words and spans from PDF and map coordinates into pixel space."""
    doc = fitz.open(stream=pdf_data, filetype="pdf")
    page = doc[page_num]
    scale = dpi / 72.0
    words = page.get_text("words")
    extracted = []
    for w in words:
        x0, y0, x1, y1, text = w[0], w[1], w[2], w[3], w[4]
        if not text.strip():
            continue
        extracted.append({
            "text": text,
            "x0": x0 * scale,
            "y0": y0 * scale,
            "x1": x1 * scale,
            "y1": y1 * scale
        })
    span_blocks = page.get_text("dict").get("blocks", [])
    spans = []
    for block in span_blocks:
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "").strip()
                if not text:
                    continue
                bbox = span.get("bbox", [0, 0, 0, 0])
                spans.append({
                    "text": text,
                    "font": span.get("font", "helv"),
                    "size": span.get("size", 12.0),
                    "x0": bbox[0] * scale,
                    "y0": bbox[1] * scale,
                    "x1": bbox[2] * scale,
                    "y1": bbox[3] * scale
                })
    logger.info(f"PDF text extraction returned {len(extracted)} words and {len(spans)} spans")
    return extracted, spans

def extract_words_from_image(image_bgr):
    """Extract words from image using Tesseract OCR if available."""
    try:
        import pytesseract
    except Exception as exc:
        logger.warning(f"OCR unavailable: {exc}")
        return []

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    data = pytesseract.image_to_data(rgb, output_type=pytesseract.Output.DICT)
    extracted = []
    for i in range(len(data.get("text", []))):
        text = data["text"][i].strip()
        if not text:
            continue
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        extracted.append({
            "text": text,
            "x0": x,
            "y0": y,
            "x1": x + w,
            "y1": y + h
        })
    logger.info(f"OCR text extraction returned {len(extracted)} words")
    return extracted

def detect_pixel_redactions(image):
    """Find solid black rectangles using pixel analysis."""
    logger.debug("Starting pixel redaction detection")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Threshold to find absolute black (redactions are usually 0,0,0)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV)
    
    # Use morphology to close any small gaps or noise in the scan
    kernel = np.ones((5,5), np.uint8)
    morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    found_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Filter by size: must be bigger than a single character, but not the whole page
        if 20 < w < 2000 and 10 < h < 100:
            found_boxes.append({'x': x, 'y': y, 'w': w, 'h': h})
            logger.debug(f"Found candidate box at ({x},{y}) size {w}x{h}")

    logger.info(f"Detected {len(found_boxes)} redaction boxes")
    return found_boxes, thresh

def _get_line_spans(spans, target):
    return [
        s for s in spans
        if abs((s["y0"] + s["y1"]) / 2 - (target["y"] + target["h"] / 2)) < target["h"] * 0.6
    ]

def _get_text_position(target, left_word, line_spans, font_size_px):
    text_x = int(left_word["x1"]) + 2 if left_word else target["x"]
    if line_spans:
        text_y = int(line_spans[0]["y0"])
    else:
        text_y = int(target["y"] + target["h"] / 2 - font_size_px / 2)
    return text_x, text_y

def _render_candidate_mask_roi(thresh_img, roi, text_x, text_y, candidate_text, font):
    from PIL import ImageDraw

    x0, y0, x1, y1 = roi
    roi_h = y1 - y0
    roi_w = x1 - x0
    mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
    pil_mask = Image.fromarray(mask)
    draw = ImageDraw.Draw(pil_mask)
    draw.text((text_x - x0, text_y - y0), candidate_text, fill=255, font=font)
    return np.array(pil_mask)

def _artifact_score(thresh_img, box, candidate_text, font, text_x, text_y, pad=5, return_debug=False):
    logger.debug(f"Analyzing artifacts for candidate: {candidate_text}")
    h, w = thresh_img.shape[:2]
    x, y, bw, bh = box["x"], box["y"], box["w"], box["h"]

    roi_x0 = max(0, x - 20)
    roi_y0 = max(0, y - pad - 5)
    roi_x1 = min(w, x + bw + 20)
    roi_y1 = min(h, y + bh + pad + 5)
    roi = (roi_x0, roi_y0, roi_x1, roi_y1)

    predicted = _render_candidate_mask_roi(thresh_img, roi, text_x, text_y, candidate_text, font)
    actual = thresh_img[roi_y0:roi_y1, roi_x0:roi_x1]

    zone_mask = np.zeros_like(actual, dtype=bool)
    top_y0 = max(roi_y0, y - pad) - roi_y0
    top_y1 = max(roi_y0, y) - roi_y0
    bot_y0 = min(roi_y1, y + bh) - roi_y0
    bot_y1 = min(roi_y1, y + bh + pad) - roi_y0
    if top_y1 > top_y0:
        zone_mask[top_y0:top_y1, :] = True
    if bot_y1 > bot_y0:
        zone_mask[bot_y0:bot_y1, :] = True

    pred_zone = (predicted > 0) & zone_mask
    actual_zone = (actual > 0) & zone_mask

    pred_count = int(pred_zone.sum())
    actual_count = int(actual_zone.sum())
    overlap = int((pred_zone & actual_zone).sum())

    precision = overlap / pred_count if pred_count else 0.0
    recall = overlap / actual_count if actual_count else 0.0
    if precision + recall:
        score = (2.0 * precision * recall) / (precision + recall)
    else:
        score = 0.0

    if not return_debug:
        return score, None, None

    debug = np.zeros((actual.shape[0], actual.shape[1], 3), dtype=np.uint8)
    debug[actual_zone] = (255, 0, 0)
    debug[pred_zone] = (0, 0, 255)
    debug[pred_zone & actual_zone] = (0, 255, 0)

    metrics = {
        "precision": precision,
        "recall": recall,
        "overlap": overlap,
        "pred_pixels": pred_count,
        "actual_pixels": actual_count,
    }
    return score, debug, metrics

def analyze_artifacts(thresh_img, box, candidate_text, font, text_x, text_y):
    """Score overlap between predicted glyph leakage and actual leaks."""
    score, _, _ = _artifact_score(
        thresh_img, box, candidate_text, font, text_x, text_y, return_debug=False
    )
    return score

def validate_post_redaction_alignment(candidate, right_word, target, font_scale, span_scale, font_size_pt, font_name):
    """Check if text after redaction aligns with candidate."""
    if not right_word or not candidate:
        return 1.0
    
    try:
        font = fitz.Font(font_name)
    except Exception:
        font = _FONT
    
    candidate_width = font.text_length(candidate, fontsize=font_size_pt) * span_scale
    predicted_end_x = target["x"] + candidate_width
    actual_start_x = right_word["x0"]
    gap = abs(predicted_end_x - actual_start_x)
    tolerance = target["h"] * 0.3
    
    if gap <= tolerance:
        return 1.0
    else:
        return max(0.0, 1.0 - (gap / (target["w"] * 2)))

def estimate_font_scale(words, spans):
    """Estimate pixel-per-font-unit scale from PDF words/spans."""
    scales = []
    for span in spans:
        text = span["text"]
        if not text:
            continue
        pixel_width = max(1.0, span["x1"] - span["x0"])
        font_name = span.get("font", "helv")
        font_size = span.get("size", 12.0)
        try:
            font = fitz.Font(font_name)
        except Exception:
            font = _FONT
        base_width = font.text_length(text, fontsize=font_size) or 1.0
        scales.append(pixel_width / base_width)

    if not scales:
        for w in words:
            text = w["text"]
            if not text:
                continue
            pixel_width = max(1.0, w["x1"] - w["x0"])
            base_width = _FONT.text_length(text, fontsize=1) or 1.0
            scales.append(pixel_width / base_width)
    if not scales:
        return 1.0
    return statistics.median(scales)

def find_line_words(words, box):
    """Return words on the same line as the box based on y-center proximity."""
    box_center_y = box["y"] + box["h"] / 2.0
    tolerance = max(8.0, box["h"] * 0.6)
    line_words = [
        w for w in words
        if abs(((w["y0"] + w["y1"]) / 2.0) - box_center_y) <= tolerance
    ]
    return sorted(line_words, key=lambda w: w["x0"])

def score_candidates_by_alignment(
    candidates,
    box,
    words,
    font_scale,
    font_name="helv",
    font_size_pt=None,
    scale_is_per_point=False,
    thresh_img=None,
    use_artifacts=False,
    spans=None,
):
    """Score candidates by how well they fit between nearest left/right words."""
    line_words = find_line_words(words, box)
    left_words = [w for w in line_words if w["x1"] <= box["x"]]
    right_words = [w for w in line_words if w["x0"] >= box["x"] + box["w"]]
    left = left_words[-1] if left_words else None
    right = right_words[0] if right_words else None
    line_spans = _get_line_spans(spans, box) if spans else []

    expected_gap = max(1.0, box["w"])

    matches = []
    font_path = _resolve_font_path(font_name)
    for c in candidates:
        try:
            font = fitz.Font(font_name)
        except Exception:
            font = _FONT
        if scale_is_per_point and font_size_pt:
            cand_width = font.text_length(c, fontsize=font_size_pt) * font_scale
        else:
            cand_width = font.text_length(c, fontsize=1) * font_scale
        diff = abs(cand_width - expected_gap)
        confidence = max(0.0, 100.0 - (diff / expected_gap * 100.0))
        artifact_score = None
        if use_artifacts and thresh_img is not None:
            if scale_is_per_point and font_size_pt:
                font_size_px = max(1, int(font_size_pt * font_scale))
            else:
                font_size_px = max(1, int(box["h"] * 0.8))
            try:
                render_font = ImageFont.truetype(font_path, font_size_px)
            except Exception:
                render_font = ImageFont.load_default()
            text_x, text_y = _get_text_position(box, left, line_spans, font_size_px)
            artifact_score = analyze_artifacts(
                thresh_img, box, c, render_font, text_x, text_y
            )
            confidence = (confidence * 0.7) + (artifact_score * 100.0 * 0.3)
        
        post_align_score = validate_post_redaction_alignment(
            c, right, box, font_scale, (line_scale if scale_is_per_point else font_scale), 
            font_size_pt or 12, font_name
        )
        confidence = confidence * post_align_score
        matches.append({
            "Candidate": c,
            "Text Width": cand_width,
            "Confidence": confidence,
            "Expected Gap": expected_gap
            ,
            "Artifact Score": artifact_score
        })
    matches.sort(key=lambda m: m["Confidence"], reverse=True)
    return matches, left, right

@functools.lru_cache(maxsize=256)
def _resolve_font_path(font_name):
    """Best-effort mapping from PDF font names to installed Windows fonts."""
    name = (font_name or "").lower()
    normalized = re.sub(r"[^a-z0-9]+", "", name)
    fonts_dir = "C:/Windows/Fonts"
    try:
        for fname in os.listdir(fonts_dir):
            lower = fname.lower()
            if not lower.endswith((".ttf", ".otf", ".ttc")):
                continue
            if normalized and normalized in re.sub(r"[^a-z0-9]+", "", lower):
                return f"{fonts_dir}/{fname}"
    except Exception:
        pass

    is_bold = "bold" in name or name.endswith("bd")
    if "cour" in name:
        return "C:/Windows/Fonts/courbd.ttf" if is_bold else "C:/Windows/Fonts/cour.ttf"
    if "times" in name:
        return "C:/Windows/Fonts/timesbd.ttf" if is_bold else "C:/Windows/Fonts/times.ttf"
    if "helv" in name or "helvetica" in name or "arial" in name:
        return "C:/Windows/Fonts/arialbd.ttf" if is_bold else "C:/Windows/Fonts/arial.ttf"
    return "C:/Windows/Fonts/arial.ttf"

def _get_line_font_info(spans, target):
    """Get line font name, size, and pixel-per-point scale using span widths."""
    line_spans = _get_line_spans(spans, target)
    spans_for_scale = line_spans or spans
    font_name = spans_for_scale[0].get("font", "helv") if spans_for_scale else "helv"
    font_size_pt = int(spans_for_scale[0].get("size", 12)) if spans_for_scale else 12

    scales = []
    for span in spans_for_scale:
        text = span.get("text", "")
        if not text.strip():
            continue
        pixel_width = max(1.0, span["x1"] - span["x0"])
        size = span.get("size", 0.0)
        if size <= 0:
            continue
        try:
            font = fitz.Font(span.get("font", "helv"))
        except Exception:
            font = _FONT
        base_width = font.text_length(text, fontsize=size) or 1.0
        scales.append(pixel_width / base_width)

    scale = statistics.median(scales) if scales else 1.0
    return font_name, font_size_pt, scale, line_spans

def render_text_with_candidate(image_rgb, target, left_word, right_word, candidate, font_scale, spans):
    """Overlay the candidate text into the redaction gap using detected font/size."""
    from PIL import ImageDraw, ImageFont

    pil_img = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(pil_img)

    font_name, font_size_pt, line_scale, line_spans = _get_line_font_info(spans, target)
    scale = line_scale if line_spans else font_scale
    font_size_px = max(1, int(font_size_pt * scale))

    font_path = _resolve_font_path(font_name)
    try:
        font = ImageFont.truetype(font_path, font_size_px)
    except Exception:
        font = ImageFont.load_default()

    text_x, text_y = _get_text_position(target, left_word, line_spans, font_size_px)

    draw.text((text_x, text_y), candidate, fill=(0, 0, 255), font=font)

    return np.array(pil_img)

def render_complete_line_with_candidate(image_rgb, target, left_word, right_word, candidate, font_scale, spans, words):
    """Render the complete reconstructed line with candidate filled in, validating post-redaction alignment."""
    from PIL import ImageDraw, ImageFont

    pil_img = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(pil_img)

    font_name, font_size_pt, line_scale, line_spans = _get_line_font_info(spans, target)
    scale = line_scale if line_spans else font_scale
    font_size_px = max(1, int(font_size_pt * scale))
    font_path = _resolve_font_path(font_name)
    try:
        font = ImageFont.truetype(font_path, font_size_px)
    except Exception:
        font = ImageFont.load_default()

    line_words = [
        w for w in words
        if abs(((w["y0"] + w["y1"]) / 2) - (target["y"] + target["h"] / 2)) < target["h"] * 0.6
    ]
    line_words = sorted(line_words, key=lambda w: w["x0"])

    _, text_y = _get_text_position(target, left_word, line_spans, font_size_px)

    for w in line_words:
        if w["x0"] >= target["x"] and w["x1"] <= target["x"] + target["w"]:
            continue
        else:
            draw.text((int(w["x0"]), text_y), w["text"], fill=(0, 0, 0), font=font)

    draw.text((int(target["x"]), text_y), candidate, fill=(0, 0, 255), font=font)

    return np.array(pil_img)

def expand_candidate_names(full_names):
    """Expand full names into individual components for matching."""
    expanded = []
    for full_name in full_names:
        expanded.append(full_name)  # Keep full name
        parts = full_name.split()
        for part in parts:
            if part not in expanded and len(part) > 2:  # Skip short fragments
                expanded.append(part)
        if len(parts) > 1:
            for spaces in (2, 3):
                expanded.append((" " * spaces).join(parts))
    return list(dict.fromkeys(expanded))  # Remove duplicates while preserving order

# --- STREAMLIT GUI ---

st.set_page_config(layout="wide")
st.title("🔬 Pixel-Level Redaction Forensic System")

uploaded_file = st.file_uploader("Upload Document (PDF or Image)", type=["pdf", "png", "jpg"])

if uploaded_file:
    logger.info(f"Uploaded file: {uploaded_file.name}")
    ocr_mode = getattr(cfg, "OCR_MODE", "auto").lower()
    ocr_available = _tesseract_available()
    # 1. Ingestion
    t0 = time.perf_counter()
    file_bytes = uploaded_file.read()
    is_pdf = uploaded_file.type == "application/pdf" or uploaded_file.name.lower().endswith(".pdf")
    if is_pdf:
        img = pdf_to_image(file_bytes, 0)
        words, spans = extract_words_from_pdf(file_bytes, 0)
    else:
        img = image_bytes_to_bgr(file_bytes)
        words = []
        spans = []
    t_ingest = time.perf_counter() - t0

    # OCR fallback or force
    t1 = time.perf_counter()
    if ocr_mode == "force" and ocr_available:
        words = extract_words_from_image(img)
    elif ocr_mode == "auto" and (not words) and ocr_available:
        words = extract_words_from_image(img)
    t_ocr = time.perf_counter() - t1
    
    # 2. Pixel Detection
    t2 = time.perf_counter()
    boxes, thresh_img = detect_pixel_redactions(img)
    t_detect = time.perf_counter() - t2

    with st.sidebar:
        st.subheader("Pipeline Status")
        st.write("✅ Ingestion")
        st.write("✅ Rasterization" if is_pdf else "✅ Image Decode")
        st.write("✅ Redaction Detection")
        st.progress(min(len(boxes) / 20, 1.0))
        st.caption(f"Detected {len(boxes)} candidate redactions")
        st.caption(f"OCR mode: {ocr_mode} | Available: {'yes' if ocr_available else 'no'}")
        st.metric("Ingestion", f"{t_ingest*1000:.0f} ms")
        st.metric("Text Extraction", f"{t_ocr*1000:.0f} ms")
        st.metric("Detection", f"{t_detect*1000:.0f} ms")
        if not ocr_available and ocr_mode in ("auto", "force"):
            st.warning("Tesseract not available. Install it to enable OCR for scanned docs.")

        with st.expander("Debug Views", expanded=False):
            st.image(thresh_img, caption="Threshold Mask", use_container_width=True)
            st.write("Detected boxes (first 10):")
            st.json(boxes[:10])
    
    col1, col2 = st.columns([2, 1])
    
    show_text_overlay = st.sidebar.checkbox("Show text overlay", value=False)
    show_complete_line = st.sidebar.checkbox("Show complete line", value=False)

    with col2:
        st.subheader("Redaction Metadata")
        if boxes:
            candidates = cfg.NAMES if hasattr(cfg, "NAMES") else ["Sarah Kellen", "Steven Bannon", "Leslie Groff"]
            candidates_expanded = expand_candidate_names(candidates)
            
            if "selected_box" not in st.session_state:
                st.session_state.selected_box = 0
            
            selected_idx = st.session_state.selected_box
            target = boxes[selected_idx]
            logger.debug(f"Selected box {selected_idx}: {target}")
            t3 = time.perf_counter()
            font_scale = estimate_font_scale(words, spans)
            line_font_name, line_font_size_pt, line_scale, line_spans = _get_line_font_info(spans, target)
            scale_for_line = line_scale if line_spans else font_scale
            matches, left_word, right_word = score_candidates_by_alignment(
                candidates_expanded,
                target,
                words,
                scale_for_line,
                line_font_name,
                line_font_size_pt,
                bool(line_spans),
                thresh_img,
                True,
                spans,
            )
            t_score = time.perf_counter() - t3
            best_fit_name = matches[0]["Candidate"] if matches else "Unknown"
            best_confidence = matches[0]["Confidence"] if matches else 0.0
            if "preview_idx" not in st.session_state:
                st.session_state.preview_idx = 0
            preview_idx = st.session_state.preview_idx
            preview_candidate = matches[preview_idx]["Candidate"] if preview_idx < len(matches) else best_fit_name
            if matches:
                st.write("### Top Candidates (Click to preview)")
                for i, m in enumerate(matches[:10]):
                    conf = m["Confidence"]
                    bar = "█" * int(conf / 10) + "░" * (10 - int(conf / 10))
                    is_selected = (i == preview_idx)
                    style = "color: #00ff00; font-weight: bold;" if is_selected else ""
                    if st.write(f"<a id='cand_{i}' style='{style}'>{i+1}. **{m['Candidate']}** {bar} {conf:.1f}%</a>", unsafe_allow_html=True):
                        pass
                    if st.button(f"Preview", key=f"cand_btn_{selected_idx}_{i}"):
                        st.session_state.preview_idx = i
                        st.rerun()
            logger.info(f"Best fit candidate: {best_fit_name} ({best_confidence:.1f}%)")
            
            # Show a zoomed-in "Pixel View" of the redaction and its edges
            zoom_y0 = max(0, target["y"] - 10)
            zoom_y1 = min(img.shape[0], target["y"] + target["h"] + 10)
            zoom_x0 = max(0, target["x"] - 10)
            zoom_x1 = min(img.shape[1], target["x"] + target["w"] + 10)
            zoom_base = img[zoom_y0:zoom_y1, zoom_x0:zoom_x1]
            if left_word and right_word and preview_candidate:
                overlay_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                overlay_rgb = render_text_with_candidate(
                    overlay_rgb, target, left_word, right_word, preview_candidate, font_scale, spans
                )
                overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
                zoom = overlay_bgr[zoom_y0:zoom_y1, zoom_x0:zoom_x1]
            else:
                zoom = zoom_base
            st.image(zoom, caption=f"Pixel Zoom (Width: {target['w']}px)")
            
            st.write(f"**Exact Pixel Width:** `{target['w']}`")
            st.write(f"**Exact Pixel Height:** `{target['h']}`")
            st.metric("Best Fit", best_fit_name)
            
            if matches:
                top_match = matches[0]
                st.metric("Top Confidence", f"{top_match['Confidence']:.1f}%")
                with st.expander("📊 Scoring Breakdown", expanded=True):
                    st.write(f"**Expected Width (Gap):** {top_match['Expected Gap']:.1f} px")
                    st.write(f"**Candidate Text Width:** {top_match['Text Width']:.1f} px")
                    st.write(f"**Pixel Difference:** {abs(top_match['Text Width'] - top_match['Expected Gap']):.1f} px")
                    st.write(f"**Confidence:** {top_match['Confidence']:.1f}%")
                    if best_confidence < 80:
                        st.warning(f"⚠️ Confidence below 80% - result may be unreliable")
                    else:
                        st.success(f"✅ High confidence match!")
            
            if left_word and right_word:
                st.caption(f"Anchored between: '{left_word['text']}' ... '{right_word['text']}'")
            st.metric("Scoring", f"{t_score*1000:.0f} ms")

            detected_font = line_font_name
            detected_size_pt = line_font_size_pt
            detected_size_px = max(1, int(detected_size_pt * scale_for_line))
            detected_font_path = _resolve_font_path(detected_font)
            st.caption(
                f"Detected font: {detected_font} | {detected_size_pt} pt (~{detected_size_px}px)"
            )
            st.caption(f"Font path: {detected_font_path}")
            
            st.divider()
            
            if show_text_overlay and left_word and right_word:
                if show_complete_line:
                    overlay_img = render_complete_line_with_candidate(
                        cv2.cvtColor(img, cv2.COLOR_BGR2RGB), target, left_word, right_word, 
                        preview_candidate, font_scale, spans, words
                    )
                else:
                    overlay_img = render_text_with_candidate(
                        cv2.cvtColor(img, cv2.COLOR_BGR2RGB), target, left_word, right_word,
                        preview_candidate, font_scale, spans
                    )
                st.image(overlay_img, caption="Text overlay preview", use_container_width=True)

            with st.expander("Artifact match", expanded=False):
                if left_word and preview_candidate:
                    try:
                        artifact_font_size_px = max(1, int(detected_size_pt * scale_for_line))
                        artifact_font = ImageFont.truetype(detected_font_path, artifact_font_size_px)
                    except Exception:
                        artifact_font = ImageFont.load_default()
                        artifact_font_size_px = max(1, int(target["h"] * 0.8))
                    line_spans = _get_line_spans(spans, target)
                    text_x, text_y = _get_text_position(
                        target, left_word, line_spans, artifact_font_size_px
                    )
                    a_score, debug_img, metrics = _artifact_score(
                        thresh_img,
                        target,
                        preview_candidate,
                        artifact_font,
                        text_x,
                        text_y,
                        return_debug=True,
                    )
                    st.caption(
                        f"Artifact score: {a_score:.2f} | precision {metrics['precision']:.2f} | recall {metrics['recall']:.2f}"
                    )
                    st.image(debug_img, caption="Artifacts: red=actual, blue=predicted, green=overlap")
                else:
                    st.caption("Need a selected candidate and anchors to compute artifacts.")
            for i, match in enumerate(matches[:10]):
                conf = match['Confidence']
                bar = "█" * int(conf / 10) + "░" * (10 - int(conf / 10))
                st.write(f"{i+1}. **{match['Candidate']}** {bar} {conf:.1f}%")
                logger.debug(
                    f"Candidate scored: {match['Candidate']} width={match['Text Width']:.1f} confidence={conf:.1f}%"
                )

    with col1:
        st.subheader("Visual Analysis")
        # Draw detected boxes with selection highlighting and best fit names on all redactions
        display_img = img.copy()
        
        # Calculate best fits for ALL boxes for overlay
        all_best_fits = {}
        for b_idx, b in enumerate(boxes):
            b_font_name, b_font_size_pt, b_line_scale, b_line_spans = _get_line_font_info(spans, b)
            b_scale = b_line_scale if b_line_spans else font_scale
            b_matches, _, _ = score_candidates_by_alignment(
                candidates_expanded,
                b,
                words,
                b_scale,
                b_font_name,
                b_font_size_pt,
                bool(b_line_spans),
                thresh_img,
                True,
                spans,
            )
            if b_matches:
                all_best_fits[b_idx] = b_matches[0]["Candidate"]
        
        for b_idx, b in enumerate(boxes):
            color = (0, 255, 0) if b_idx == selected_idx else (100, 150, 0)
            thickness = 3 if b_idx == selected_idx else 2
            cv2.rectangle(display_img, (b['x'], b['y']), (b['x']+b['w'], b['y']+b['h']), color, thickness)

        if boxes and 'target' in locals() and left_word and right_word and preview_candidate:
            display_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
            if show_complete_line:
                display_rgb = render_complete_line_with_candidate(
                    display_rgb, target, left_word, right_word, preview_candidate, font_scale, spans, words
                )
            else:
                display_rgb = render_text_with_candidate(
                    display_rgb, target, left_word, right_word, preview_candidate, font_scale, spans
                )
            display_img = cv2.cvtColor(display_rgb, cv2.COLOR_RGB2BGR)

        if boxes and 'target' in locals() and left_word and right_word:
            lx = int(left_word["x1"])
            rx = int(right_word["x0"])
            cy = int(target["y"] + target["h"] / 2)
            cv2.line(display_img, (lx, cy - 12), (lx, cy + 12), (0, 0, 255), 2)
            cv2.line(display_img, (rx, cy - 12), (rx, cy + 12), (0, 0, 255), 2)
            cv2.putText(display_img, left_word["text"], (lx + 4, cy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(display_img, right_word["text"], (rx + 4, cy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Interactive click detection on image
        st.write("### Click on a redaction to select it")
        display_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        display_pil = Image.fromarray(display_rgb)
        zoom_level = st.slider("Zoom", min_value=0.5, max_value=2.5, value=1.0, step=0.1)
        max_width = 1200
        scale = 1.0
        if display_pil.width > max_width:
            scale = max_width / display_pil.width
            display_pil = display_pil.resize(
                (int(display_pil.width * scale), int(display_pil.height * scale))
            )
        if zoom_level != 1.0:
            scale *= zoom_level
            display_pil = display_pil.resize(
                (int(display_pil.width * zoom_level), int(display_pil.height * zoom_level))
            )
        coords = streamlit_image_coordinates(display_pil)
        
        if coords:
            x, y = coords["x"], coords["y"]
            if scale != 1.0:
                x = int(x / scale)
                y = int(y / scale)
            # Find which box was clicked
            for b_idx, b in enumerate(boxes):
                if b['x'] <= x <= b['x'] + b['w'] and b['y'] <= y <= b['y'] + b['h']:
                    st.session_state.selected_box = b_idx
                    st.rerun()