# Poster or Graphic Design OCR Project

![Individual Node Architecture Diagram](architecture.png)
*Individual Node Architecture - Poster/Graphic Design System*

This project provides tools for automated text detection, template management, and QR code overlay for poster or graphic design workflows. It is designed to help you quickly replace, theme, and enhance text in images and SVG templates, and to overlay QR codes on generated graphics.

## Features
- **OCR Text Detection**: Detects and extracts text regions from images using EasyOCR and heuristic font analysis.
- **Template Selection & Theming**: Select SVG templates and apply color themes, with automatic SVG-to-PNG conversion.
- **QR Code Overlay**: Overlay QR codes on images with flexible positioning, scaling, and opacity.
- **Organized Output**: All outputs are saved in an `output/` directory for easy access.

## Requirements
Install all dependencies with:
```bash
pip install -r requirements.txt
```

## Usage

---

## Node Documentation

### 1. Select Template and Color Theme Node (`SelectTemplateHandler`)

**Purpose:**  
Selects an SVG template and applies a color theme, optionally converting the result to PNG.

**Input:**  
- No direct file input required; operates on template names and theme names.

**Configuration Options:**
| Option           | Type    | Description                                                      | Example                  |
|------------------|---------|------------------------------------------------------------------|--------------------------|
| `template_name`  | string  | Template to use (relative to `/templates/` dir)                  | `"business/flyer2"`      |
| `theme_name`     | string  | Theme to apply (from themes JSON for template)                   | `"blue"`                 |
| `convert_to_png` | bool    | If `True`, output will include PNG conversion                    | `True`                   |
| `output_format`  | string  | `"svg"` or `"png"` (optional, legacy, use `convert_to_png` now)  | `"svg"`                  |

**Output Format:**
```json
{
  "status": "success",
  "outputs": {
    "output_path": "output/template_flyer2_blue_<uuid>.svg",
    "theme_name": "blue",
    "png_path": "output/template_flyer2_blue_<uuid>.png" // if PNG requested
  }
}
```

---

### 2. Replace Text Node (`ReplaceTextHandler`)

**Purpose:**  
Detects text in an image, analyzes font properties, and optionally replaces specified text regions.

**Input:**  
- `input_image`: Path to the image (PNG, JPG, etc.)

**Configuration Options:**
| Option             | Type   | Description                                                | Example                   |
|--------------------|--------|------------------------------------------------------------|---------------------------|
| `input_image`      | string | Path to image file                                         | `"output/template.png"`   |
| `replacements`     | dict   | Mapping of original text → new text                        | `{ "Name XXXX": "John Doe" }`|
| `min_confidence`   | int    | Minimum OCR confidence (0–100)                             | `40`                      |

**Font Mapping Example:**
```python
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
    "Courier-BoldItalic": "/Users/sithijshetty/Desktop/Poster_or_Graphic_Design/Fonts copy/Supplemental/Courier New Bold Italic.ttf",
}
```

**Output Format:**
```json
{
  "status": "success",
  "outputs": {
    "regions": [ ... ],        // List of detected text regions with attributes
    "lines": [ ... ],          // Line-level OCR results
    "debug_image": "output/sample_debug.png",
    "replaced_image": "output/sample_replaced.png" // Only if replacements applied
  }
}
```

---

### 3. Add QR Code Node (`QRCodeHandler` / `overlay_qr_image_handler.py`)

**Purpose:**  
Overlays a QR code onto an image at a specified position, scale, and opacity.

**Input:**  
- `image_path`: Path to the base image (typically PNG)
- `qr_data` or `qr_image_path`: Data to encode, or path to a pre-generated QR image

**Configuration Options:**
| Option           | Type    | Description                                                      | Example                |
|------------------|---------|------------------------------------------------------------------|------------------------|
| `image_path`     | string  | Path to base image                                               | `"output/sample.png"`  |
| `qr_data`        | string  | Data to encode as QR (mutually exclusive with `qr_image_path`)   | `"https://example.com"`|
| `qr_image_path`  | string  | Path to existing QR code image                                   | `"qr.png"`             |
| `position`       | string  | Where to place QR: `"top_left"`, `"bottom_right"`, etc.          | `"top_right"`          |
| `offset`         | string  | Pixel offset as tuple string: `"(10,10)"`                        | `"(20,0)"`             |
| `qr_scale`       | float   | Relative size of QR code (1.0 = original)                        | `0.5`                  |
| `opacity`        | float   | QR code opacity (0.0–1.0)                                        | `0.8`                  |

**Output Format:**
```json
{
  "status": "success",
  "outputs": {
    "final_image": "output/sample_with_qr.png"
  }
}
```

---

### 1. OCR Text Detection & Replacement
Detect text and optionally replace text in an image:
```bash
python replace_text.py --input_image path/to/image.jpg --replacements_file my_replacements.json
```

### 2. Template Selection and Theming
Apply a theme to a template and convert to PNG:
```bash
python select_template.py --template_name business/flyer2 --theme_name blue --convert_to_png True
```

### 3. Overlay QR Code on Image
Overlay a QR code onto an image:
```bash
python overlay_qr_image_handler.py --image_path path/to/output.png --qr_image_path path/to/qr.png --position top_right --qr_scale 0.5 --opacity 0.9 --offset "(10,10)"
```
- `--offset` format: "(x,y)" or "x,y" (e.g., "(20,0)" for horizontal displacement)

## Directory Structure
```
Poster_or_Graphic_Design/
├── overlay_qr_image_handler.py
├── replace_text.py
├── select_template.py
├── requirements.txt
├── .gitignore
├── README.md
├── output/
├── templates/
│   ├── business/
│   ├── invitations/
│   ├── events/
│   └── themes/
└── samples/
```

## Tips
- All outputs (debug images, replaced images, themed SVG/PNGs, QR overlays) are saved in the `output/` folder.
- You can add new templates and themes in the `templates/` directory.
- The `.gitignore` is set up to keep your repository clean.
