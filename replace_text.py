"""OCR Text Detection with Font Attribute Extraction

This standalone module uses EasyOCR to detect text in an image and augments
each detected region with additional font-related characteristics extracted
via classical image-processing heuristics:

• font_size      – approximated from the bounding-box height
• bold           – boolean derived from median stroke width vs size
• italic         – boolean based on estimated slant angle
• slant_angle    – measured angle in degrees (positive = right slanted)
• color_hsv      – median HSV foreground colour in the text region
• stroke_width   – median stroke width in pixels (useful downstream)

If a detected EasyOCR token appears to contain multiple fonts (e.g. substring
with very different stroke width or slant) the region is sub-divided so that
attributes are returned per sub-region.  This gives robust results without
needing to retrain the CNN for every attribute.

Example usage:
    handler = DetectTextHandler()
    result = handler.process({}, {"input_image": "sample.jpg"})
    for r in result["outputs"]["regions"]:
        print(r)

The code is dependency-light: only OpenCV, NumPy and EasyOCR are required.
"""
from __future__ import annotations

import os
from typing import Dict, List, Tuple
import re
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import easyocr
import logging
logging.basicConfig(level=logging.DEBUG)

# ===== Font Detection Heuristic Thresholds (TUNE HERE) =====
SERIF_EDGE_SCORE_THRESHOLD = 0.05  # Default: 0.05
BOLD_STROKE_WIDTH_THRESHOLD = 3.0  # Default: 3.0
ITALIC_SLANT_ANGLE_THRESHOLD = 5.0 # Default: 8.0

# (You can tune these at the top of the file for predictable output)


# Torch-specific imports guarded later
try:
    import torch
    import torch.nn as nn
    from torchvision import transforms
except ImportError:
    torch = None  # type: ignore

# Optional: only import torch if available for family classifier fallback
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    TORCH_AVAILABLE = False

# -------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------

def _enhance_for_ocr(img_bgr: np.ndarray) -> np.ndarray:
    """Apply a light-weight enhancement pipeline (CLAHE + sharpening) to make
    characters stand out before sending to the OCR engine.

    The pipeline is intentionally simple so that it does not slow the whole
    processing down, yet noticeably improves contrast for faint strokes which
    EasyOCR sometimes confuses (e.g. *e*→*c*, *P*→*F*).
    """
    # Convert to LAB and apply CLAHE on the L-channel
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab_enh = cv2.merge((l2, a, b))
    enhanced = cv2.cvtColor(lab_enh, cv2.COLOR_LAB2BGR)

    # Optional: unsharp masking to crisp edges
    blur = cv2.GaussianBlur(enhanced, (0, 0), 2)
    sharp = cv2.addWeighted(enhanced, 1.5, blur, -0.5, 0)
    return sharp

def _median_stroke_width(mask: np.ndarray) -> float:
    """Estimate median stroke width of foreground strokes in *binary* mask.

    Instead of skeletonisation (which is brittle without `cv2.ximgproc`), we
    use a run-length approach:
      1. For several evenly spaced horizontal scan-lines, find contiguous runs
         of foreground pixels (value > 0).
      2. Collect their lengths; the median of these lengths is a good proxy
         for stroke width in pixels.
    """
    if mask is None or mask.size == 0 or mask.sum() == 0:
        return 0.0

    # Ensure binary 0/1 for simplicity
    bin_mask = (mask > 0).astype(np.uint8)
    h, _ = bin_mask.shape

    sample_rows = np.linspace(0, h - 1, num=min(15, h), dtype=int)
    run_lengths: List[int] = []

    for r in sample_rows:
        row = bin_mask[r]
        # Find indices where value changes
        diff = np.diff(np.concatenate([[0], row, [0]]))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        lengths = ends - starts
        run_lengths.extend(lengths.tolist())

    if not run_lengths:
        return 0.0
    return float(np.median(run_lengths))


def _simple_thinning(img: np.ndarray) -> np.ndarray:
    """Fallback 8-neighbour iterative thinning if ximgproc unavailable."""
    thinned = img.copy()
    prev = np.zeros_like(thinned)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        eroded = cv2.erode(thinned, kernel)
        temp = cv2.dilate(eroded, kernel)
        temp = cv2.subtract(thinned, temp)
        skel = cv2.bitwise_or(prev, temp)
        thinned = eroded.copy()
        if cv2.countNonZero(thinned) == 0:
            break
        prev = skel.copy()
    return prev


def _estimate_slant(mask: np.ndarray) -> float:
    """Estimate slant from the central body of the character to avoid serifs."""
    if mask is None or mask.size == 0:
        return 0.0

    h, w = mask.shape
    y0, y1 = int(0.2*h), int(0.8*h)       # ignore top/bottom serifs
    body = mask[y0:y1, :]

    angles = []

    # 1) Hough lines on edges
    edges = cv2.Canny(body, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=max(10, min(body.shape)//4))
    if lines is not None:
        for rho, theta in lines[:,0]:
            deg = (theta - np.pi/2)*180/np.pi
            if -45 < deg < 45:
                angles.append(deg)

    # 2) Contour‐based ellipse fitting
    contours, _ = cv2.findContours(body, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > 30 and len(cnt)>=5:
            ellipse = cv2.fitEllipse(cnt)
            ang = ellipse[2] - 90
            if -45 < ang < 45:
                angles.extend([ang]*2)   # weight contour more

    # 3) Moment‐based skew
    M = cv2.moments(body)
    if M['m00'] != 0:
        mu20 = M['mu20']/M['m00']
        mu02 = M['mu02']/M['m00']
        mu11 = M['mu11']/M['m00']
        if abs(mu20-mu02) > 1e-3:
            ma = 0.5 * np.arctan2(2*mu11, mu20-mu02)*180/np.pi
            if -45 < ma < 45:
                angles.extend([ma]*3)   # weight moment highest

    if not angles:
        return 0.0
    return float(np.median(angles))




def _extract_color_hsv(region_bgr: np.ndarray) -> Tuple[int, int, int]:
    """Return median HSV of foreground pixels using simple OTSU binarisation."""
    gray = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    hsv = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2HSV)
    fg_pixels = hsv[thresh > 0]
    if fg_pixels.size == 0:
        return 0, 0, 0
    median = np.median(fg_pixels, axis=0)
    return tuple(int(x) for x in median)


# -------------------------------------------------------------
# Main handler
# -------------------------------------------------------------


class DetectTextHandler:
    def __init__(self):
        self.reader = easyocr.Reader(["en"], gpu=True)

        # ---------------- font model ----------------
        # Remove CNN-based font model. Use heuristic font family estimation instead.
        self.font_model = None
        self.font_mapping = None

    def _estimate_font_family_heuristic(self, region_img_bgr, stroke_width, slant_angle):
        """
        Estimate font family using classical heuristics:
        - Serif vs sans-serif: edge analysis (serif_score)
        - Bold: stroke width
        - Italic: slant angle
        All thresholds are defined at the top of the file for predictability.
        Logs all decisions for tuning.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(region_img_bgr, cv2.COLOR_BGR2GRAY)
        # Edge detection for serif analysis
        edges = cv2.Canny(gray, 80, 200)
        h, w = edges.shape
        # Count small horizontal/vertical lines at ends (serif proxy)
        serif_score = (
            np.sum(edges[:, :3]) + np.sum(edges[:, -3:]) + np.sum(edges[:3, :]) + np.sum(edges[-3:, :])
        ) / (h * w)
        
        is_serif = serif_score > SERIF_EDGE_SCORE_THRESHOLD
        is_bold = stroke_width > BOLD_STROKE_WIDTH_THRESHOLD
        is_italic = abs(slant_angle) > ITALIC_SLANT_ANGLE_THRESHOLD

        # Debug logging for every font assignment decision
        logging.debug(
            f"Font heuristic: serif_score={serif_score:.4f} (>{SERIF_EDGE_SCORE_THRESHOLD}), "
            f"stroke_width={stroke_width:.2f} (>{BOLD_STROKE_WIDTH_THRESHOLD}), "
            f"slant_angle={slant_angle:.2f} (>{ITALIC_SLANT_ANGLE_THRESHOLD}) => "
            f"is_serif={is_serif}, is_bold={is_bold}, is_italic={is_italic}"
        )

        # Decide font
        if is_serif:
            if is_bold and is_italic:
                font = "Times-BoldItalic"
            elif is_bold:
                font = "Times-Bold"
            elif is_italic:
                font = "Times-Italic"
            else:
                font = "Times New Roman"
        else:
            if is_bold and is_italic:
                font = "Arial-BoldItalic"
            elif is_bold:
                font = "Arial-Bold"
            elif is_italic:
                font = "Arial-Italic"
            else:
                font = "Arial"
        logging.debug(f"Assigned font: {font}")
        return font



    # ----------------------- public API -----------------------
    def process(self, inputs: Dict, config: Dict) -> Dict:
        img_path = config["input_image"]
        min_conf = int(config.get("min_confidence", 30))
        regions, lines = self._detect_with_easyocr(img_path, min_conf)

        debug_path = self._save_debug_image(img_path, regions)

        # Optional replacement stage
        replacements = config.get("replacements")
        replaced_path = None
        if replacements:
            replaced_path = self._apply_replacements(img_path, regions, replacements)


        return {
            "status": "success",
            "outputs": {
                "regions": regions,
                "lines": lines,
                "debug_image": debug_path,
                "replaced_image": replaced_path,
            },
        }

    # -------------------- internal helpers --------------------
    def _detect_with_easyocr(self, img_path: str, min_conf: int, binarize: bool = False):
        # Read & enhance image first – this often boosts character separability
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot load image: {img_path}")
        enhanced = _enhance_for_ocr(img_bgr)

        # Optional: apply binarization after enhancement (before OCR)
        if binarize:
            gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            enhanced = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            logging.debug("Applied binarization before OCR.")

        # Tuned EasyOCR invocation – beam search decoder + slight contrast
        results = self.reader.readtext(
            enhanced,
            detail=1,
            paragraph=False,
            min_size=10,
            decoder="beamsearch",
            beamWidth=10,
            contrast_ths=0.1,
            adjust_contrast=0.7,
        )
        img_bgr = cv2.imread(img_path)
        regions: List[Dict] = []
        lines: List[Dict] = []

        for bbox, text, conf in results:
            if conf * 100 < min_conf:
                continue
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            x_min, y_min, x_max, y_max = min(xs), min(ys), max(xs), max(ys)
            full_region = {
                "text": text,
                "bbox": (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)),
                "confidence": int(conf * 100),
            }

            # Analyse font differences inside this region
            sub_regions = self._split_if_multi_font(full_region, img_bgr)
            for sr in sub_regions:
                self._add_text_attributes(sr, img_bgr)
                regions.append(sr)
            # For line-level view we use original easyOCR boxes
            self._add_text_attributes(full_region, img_bgr)
            lines.append(full_region)

        # Resolve vertical overlaps (stacked boxes) by shrinking lower boxes vertically
        regions = self._shrink_vertical_overlaps(regions)
        lines = self._shrink_vertical_overlaps(lines)
        return regions, lines

    # ---------------------------------------------------------
    def _split_if_multi_font(self, region: Dict, img_bgr: np.ndarray) -> List[Dict]:
        """Split region into smaller ones if stroke-width variance suggests mixed fonts."""
        text = region["text"]
        if len(text.strip().split()) <= 1:
            return [region]
        tokens = text.split()
        token_boxes = self._approx_word_boxes(region["bbox"], text, tokens)

        sw_values = []
        for tb in token_boxes:
            x1, y1, x2, y2 = tb
            crop = img_bgr[y1:y2, x1:x2]
            mask = _binarise(crop)
            sw_values.append(_median_stroke_width(mask))
        if max(sw_values) - min(sw_values) < 1.5:  # uniform font
            return [region]
        # Otherwise split
        sub = []
        for t, box in zip(tokens, token_boxes):
            x1, y1, x2, y2 = box
            sub.append({
                "text": t,
                "bbox": (x1, y1, x2 - x1, y2 - y1),
                "confidence": region["confidence"],
            })
        return sub
    def _detect_baseline(self, mask: np.ndarray) -> int:
        """Detect text baseline for better positioning."""
        if mask is None or mask.size == 0:
            return 0
        
        h, w = mask.shape
        
        # Find horizontal projection (sum of pixels in each row)
        horizontal_proj = np.sum(mask, axis=1)
        
        # Find the bottom-most significant pixel row (baseline area)
        significant_rows = np.where(horizontal_proj > w * 0.1)[0]  # At least 10% coverage
        
        if len(significant_rows) == 0:
            return h // 4  # Default to 25% from bottom
        
        # Baseline is typically at 75-85% from top for most fonts
        baseline = int(significant_rows[-1] * 0.85)
        return min(baseline, h - 1)

    def _get_text_metrics(self, text: str, font, draw_context) -> Dict:
        """Get accurate text metrics for positioning."""
        # Get bounding box of the text
        bbox = draw_context.textbbox((0, 0), text, font=font)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        
        # Get ascent and descent
        ascent, descent = font.getmetrics()
        
        return {
            'width': width,
            'height': height,
            'ascent': ascent,
            'descent': descent,
            'baseline_offset': ascent
        }
    # ---------------------------------------------------------
    def _add_text_attributes(self, region: Dict, img_bgr: np.ndarray):
        """Enhanced attribute extraction with better italic detection."""
        x, y, w, h = region["bbox"]
        crop = img_bgr[y : y + h, x : x + w]
        mask = _binarise(crop)

        # Size (approx pts at 72 dpi baseline)
        region["font_size"] = h

        # Stroke width / weight
        sw = _median_stroke_width(mask)
        region["stroke_width"] = round(sw, 2)
        # Improved bold detection with size normalization
        region["bold"] = sw / max(h, 1) > 0.08 and sw > 1.5

        # Enhanced slant / italic detection
        angle = _estimate_slant(mask)
        region["slant_angle"] = round(angle, 2)
        # Lower threshold for better italic detection
        region["italic"] = abs(angle) > 5.0

        # Color extraction
        hsv = _extract_color_hsv(crop)
        region["color_hsv"] = hsv

        # Add baseline detection for better positioning
        region["baseline_offset"] = self._detect_baseline(mask)

        # Font family via CNN if available
        if self.font_model is not None and torch is not None:
            try:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                pil_img = Image.fromarray(gray)  # mode 'L'
                inp = self._preprocess(pil_img).unsqueeze(0)  # 1x1x64x64
                with torch.no_grad():
                    logits = self.font_model(inp)
                    probs = torch.softmax(logits, dim=1)
                    pred_idx = int(torch.argmax(probs, dim=1).item())
                    region["font_prob"] = float(probs[0, pred_idx])
                if self.font_mapping and str(pred_idx) in self.font_mapping:
                    region["font_family"] = self.font_mapping[str(pred_idx)]
                else:
                    region["font_family"] = pred_idx
            except Exception as e:
                region["font_family"] = None

    # ---------------------------------------------------------
    def _save_debug_image(self, img_path: str, regions: List[Dict]) -> str:
        # Ensure all regions have font_family and font_prob using heuristic
        for reg in regions:
            if 'font_family' not in reg or reg['font_family'] == 'Unknown':
                x, y, w, h = reg["bbox"]
                img = cv2.imread(img_path)
                region_img_bgr = img[y:y+h, x:x+w]
                stroke_width = reg.get("stroke_width", 1.0)
                slant_angle = reg.get("slant_angle", 0.0)
                reg['font_family'] = self._estimate_font_family_heuristic(region_img_bgr, stroke_width, slant_angle)
            if 'font_prob' not in reg:
                reg['font_prob'] = 1.0
        """Draw rectangles + labels over detected regions for quick visual check."""
        img = cv2.imread(img_path)
        for reg in regions:
            if re.match(r'^\d{4} [I1]{4} \d{4}$', reg['text']):
                reg['text'] = reg['text'].replace('I', '1')
            x, y, w, h = reg["bbox"]
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
            label = f"{reg['font_family']} ({reg.get('font_prob',0):.2f})"
            cv2.putText(img, label, (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
        # Save to output directory
        output_dir = os.path.join(os.path.dirname(img_path), 'output')
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(img_path))[0] + "_debug.png"
        out_path = os.path.join(output_dir, base_name)
        cv2.imwrite(out_path, img)
        return out_path

    def _apply_replacements(self, img_path: str, regions: List[Dict], replacements: Dict[str, str]) -> str:
        """Inpaint and overwrite text that matches `replacements` dict."""
        img_bgr = cv2.imread(img_path)
        # Work largest→smallest to avoid overlaps
        targets = [r for r in regions if r["text"] in replacements]
        targets.sort(key=lambda r: r["bbox"][2]*r["bbox"][3], reverse=True)
        for reg in targets:
            x, y, w, h = reg["bbox"]
            # Inpaint original text
            mask = np.zeros(img_bgr.shape[:2], np.uint8)
            mask[y:y+h, x:x+w] = 255
            img_bgr = cv2.inpaint(img_bgr, mask, 3, cv2.INPAINT_TELEA)
            # Prepare replacement text
            # Map font_family index to actual font family name using font_mapping.json
            # Heuristic font family estimation
            x, y, w, h = reg["bbox"]
            region_img_bgr = img_bgr[y:y+h, x:x+w]
            stroke_width = reg.get("stroke_width", 1.0)
            slant_angle = reg.get("slant_angle", 0.0)
            font_name = self._estimate_font_family_heuristic(region_img_bgr, stroke_width, slant_angle)
            reg['font_family'] = font_name  # Always add to region dict for downstream use
            reg['font_prob'] = 1.0  # Heuristic, so set to 1.0 or leave as default
            pt = max(12, int(reg["font_size"] * 76/ 96))
            # Courier special handling for .ttc and style
            if font_name.startswith("Courier"):
                # Map style to ttc index
                bold = reg.get("bold", False)
                italic = reg.get("italic", False)
                courier_ttc = "/System/Library/Fonts/Courier.ttc"
                # Index: 0=Regular, 1=Bold, 2=Oblique, 3=BoldOblique
                if bold and italic:
                    ttc_index = 3
                elif bold:
                    ttc_index = 1
                elif italic:
                    ttc_index = 2
                else:
                    ttc_index = 0
                try:
                    font = ImageFont.truetype(courier_ttc, pt, index=ttc_index)
                except Exception:
                    font = ImageFont.load_default()
            else:
                # Mapping from font names in font_mapping.json to actual font files and indices
                font_file_map = {
                    "Arial": "/Users/sithijshetty/Desktop/Poster_or_Graphic_Design/Fonts copy/Supplemental/Arial.ttf",
            "Arial-Bold": "/Users/sithijshetty/Desktop/Poster_or_Graphic_Design/Fonts copy/Supplemental/Arial Bold.ttf",
            "Arial-Italic": "/Users/sithijshetty/Desktop/Poster_or_Graphic_Design/Fonts copy/Supplemental/Arial Italic.ttf",
            "Arial-BoldItalic": "/Users/sithijshetty/Desktop/Poster_or_Graphic_Design/Fonts copy/Supplemental/Arial Bold Italic.ttf",
            "Times New Roman": "/Users/sithijshetty/Desktop/Poster_or_Graphic_Design/Fonts copy/Supplemental/Times New Roman.ttf",
            "Times-Bold": "/Users/sithijshetty/Desktop/Poster_or_Graphic_Design/Fonts copy/Supplemental/Times New Roman Bold.ttf",
            "Times-Italic": "/Users/sithijshetty/Desktop/Poster_or_Graphic_Design/Fonts copy/Supplemental/Times New Roman Italic.ttf",
            "Times-BoldItalic": "/Users/sithijshetty/Desktop/Poster_or_Graphic_Design/Fonts copy/Supplemental/Times New Roman Bold Italic.ttf",
            "Courier": "/Users/sithijshetty/Desktop/Poster_or_Graphic_Design/Fonts copy/Supplemental/Courier New.ttf",
            "Courier-Bold": "/Users/sithijshetty/Desktop/Poster_or_Graphic_Design/Fonts copy/Supplemental/Courier New Bold.ttf",
            "Courier-Italic": "/Users/sithijshetty/Desktop/Poster_or_Graphic_Design/Fonts copy/Supplemental/Courier New Italic.ttf",
            "Courier-BoldItalic":"/Users/sithijshetty/Desktop/Poster_or_Graphic_Design/Fonts copy/Supplemental/Courier New Bold Italic.ttf",
                }
                font_info = font_file_map.get(font_name)
                if font_info is None:
                    logging.error(f"No font file mapping for detected font '{font_name}'. Skipping replacement for this region.")
                    continue
                simulate_bold = False
                # Handle TTC index fonts (e.g., Courier, Times)
                simulate_italic = False
                if isinstance(font_info, tuple):
                    font_file, ttc_index = font_info
                    try:
                        font = ImageFont.truetype(font_file, pt, index=ttc_index)
                        logging.debug(f"Loaded TTC font: {font_file} (index {ttc_index}), size: {pt}")
                    except Exception as e:
                        logging.error(f"Failed to load TTC font {font_file} (index {ttc_index}): {e}. Skipping region.")
                        continue
                else:
                    font_file = font_info
                    if reg.get("bold"):
                        bold_candidate = font_file.replace(".ttf", "-Bold.ttf")
                        if os.path.exists(bold_candidate):
                            logging.debug(f"Using bold font variant: {bold_candidate}")
                            font_file = bold_candidate
                        else:
                            logging.debug("Simulating bold (manual thickening)")
                            simulate_bold = True  # will thicken manually
                    if reg.get("italic") and os.path.exists(font_file.replace(".ttf", "-Italic.ttf")):
                        italic_candidate = font_file.replace(".ttf", "-Italic.ttf")
                        logging.debug(f"Using italic font variant: {italic_candidate}")
                        font_file = italic_candidate
                    elif reg.get("italic"):
                        logging.debug("Simulating italic (shear transform)")
                        simulate_italic = True  # will shear manually
                    try:
                        font = ImageFont.truetype(font_file, pt)
                        logging.debug(f"Loaded font: {font_file}, size: {pt}")
                    except Exception as e:
                        logging.error(f"Failed to load font {font_file}: {e}. Skipping region.")
                        continue
            hsv = np.uint8([[reg["color_hsv"]]])
            rgb = tuple(int(c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0][0])
            # Draw text via PIL
            pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil)
            text_val = replacements[reg["text"]]
            if 'simulate_bold' in locals() and simulate_bold:
                logging.debug(f"Drawing text '{text_val}' at ({x},{y}) with simulated bold.")
                offsets = [(0,0), (1,0), (0,1), (1,1)]
                for dx, dy in offsets:
                    draw.text((x+dx, y+dy), text_val, font=font, fill=rgb)
                img_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            elif 'simulate_italic' in locals() and simulate_italic:
                logging.debug(f"Drawing text '{text_val}' at ({x},{y}) with simulated italic (shear transform).")
                text_img = Image.new('RGBA', pil.size, (0,0,0,0))
                text_draw = ImageDraw.Draw(text_img)
                text_draw.text((x, y), text_val, font=font, fill=rgb+(255,))
                shear = -0.3
                matrix = (1, shear, 0, 0, 1, 0)
                sheared = text_img.transform(text_img.size, Image.AFFINE, matrix, resample=Image.BICUBIC)
                pil = pil.convert('RGBA')
                pil.alpha_composite(sheared)
                img_bgr = cv2.cvtColor(np.array(pil.convert('RGB')), cv2.COLOR_RGB2BGR)
            else:
                logging.debug(f"Drawing text '{text_val}' at ({x},{y}) with normal style with font size {pt} and font family {font_name}")
                draw.text((x, y), text_val, font=font, fill=rgb)
                img_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        # Save to output directory
        output_dir = os.path.join(os.path.dirname(img_path), 'output')
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(img_path))[0] + "_replaced.png"
        out_path = os.path.join(output_dir, base_name)
        cv2.imwrite(out_path, img_bgr)
        return out_path

    @staticmethod
    def _approx_word_boxes(bbox: Tuple[int, int, int, int], text: str, tokens: List[str]):
        """Given the parent bbox and token list, roughly split horizontally by char ratio."""
        x, y, w, h = bbox
        total_chars = len(text)
        boxes = []
        cur_x = x
        for tok in tokens:
            ratio = len(tok) / total_chars
            tok_w = max(1, int(ratio * w))
            boxes.append((cur_x, y, cur_x + tok_w, y + h))
            cur_x += tok_w
        # Ensure last box ends exactly at parent right edge
        if boxes:
            last = boxes[-1]
            boxes[-1] = (last[0], last[1], x + w, last[3])
        return boxes


# ---------------------------------------------------------
    def _shrink_vertical_overlaps(self, regs: List[Dict]) -> List[Dict]:
        """Shrink vertically stacked boxes so they do not overlap or touch.
        If a lower box overlaps or touches the one above (and shares horizontal intersection),
        BOTH boxes are shrunk equally from their adjacent edges to ensure a 1-pixel gap between them.
        """
        regs = sorted(regs, key=lambda r: (r["bbox"][1], r["bbox"][0]))  # top-to-bottom
        for i in range(len(regs)):
            xi, yi, wi, hi = regs[i]["bbox"]
            for j in range(i + 1, len(regs)):
                xj, yj, wj, hj = regs[j]["bbox"]
                # require horizontal intersection to treat as stacked in same column
                if xi + wi <= xj or xj + wj <= xi:
                    continue
                # vertical overlap or touch amount
                overlap = (yi + hi) - yj
                if overlap >= 0:  # overlap or just-touch
                    gap = 1  # always leave a 1-pixel gap
                    total_shrink = overlap + gap
                    shrink_upper = total_shrink // 2
                    shrink_lower = total_shrink - shrink_upper

                    # Shrink upper box from bottom
                    new_hi = max(1, hi - shrink_upper)
                    # Shrink lower box from top
                    new_yj = yj + shrink_lower
                    new_hj = max(1, hj - shrink_lower)

                    regs[i]["bbox"] = (xi, yi, wi, new_hi)
                    regs[j]["bbox"] = (xj, new_yj, wj, new_hj)
        return regs

# -------------------------------------------------------------
# Utility
# -------------------------------------------------------------

def _binarise(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return th


# -------------------------------------------------------------
# Command-line helper for quick testing
# -------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    import argparse, json

    parser = argparse.ArgumentParser(description="Detect text + attributes")
    parser.add_argument("--input_image", help="Input image path")
    parser.add_argument("--min_confidence", type=int, default=30)
    parser.add_argument("--replacements_file", help="JSON file with original→new text mapping")
    args = parser.parse_args()

    handler = DetectTextHandler()
    repl = None
    if args.replacements_file:
        with open(args.replacements_file) as f:
            repl = json.load(f)
    res = handler.process({}, {"input_image": args.input_image,
                               "min_confidence": args.min_confidence,
                               "replacements": repl})
    print(json.dumps(res, indent=2))