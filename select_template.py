import logging
import os
import uuid
import json
from pathlib import Path
from shutil import copyfile
import xml.etree.ElementTree as ET
import re
from copy import deepcopy
import sys
import argparse

# PNG conversion: check which method is available (cairosvg preferred)
try:
    from cairosvg import svg2png
    CAIRO_AVAILABLE = True
    WAND_AVAILABLE = False
    INKSCAPE_AVAILABLE = False
except ImportError:
    try:
        from wand.image import Image as WandImage
        CAIRO_AVAILABLE = False
        WAND_AVAILABLE = True
        INKSCAPE_AVAILABLE = False
    except ImportError:
        try:
            import subprocess
            import shutil
            INKSCAPE_PATH = shutil.which('inkscape')
            INKSCAPE_AVAILABLE = bool(INKSCAPE_PATH)
            CAIRO_AVAILABLE = False
            WAND_AVAILABLE = False
        except:
            CAIRO_AVAILABLE = False
            WAND_AVAILABLE = False
            INKSCAPE_AVAILABLE = False

SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace('', SVG_NS)

# Only NodeHandler base is needed
class NodeHandler:
    pass

# --- PNG conversion method summary ---
# cairosvg: %(cairo)s
# wand: %(wand)s
# inkscape: %(inkscape)s


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

SCRIPT_DIR     = Path(__file__).parent
PROJECT_ROOT   = Path("/Users/sithijshetty/Desktop/Poster_or_Graphic_Design")
TEMPLATE_DIR   = PROJECT_ROOT / "templates"
THEMES_DIR     = TEMPLATE_DIR / "themes"
OUTPUT_FOLDER  = SCRIPT_DIR / "output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

class SelectTemplateHandler(NodeHandler):
    """Handler for selecting a pre-made SVG design template + optional colour theme with PNG conversion."""
    
    def __init__(self):
        # CSS color properties that can contain colors
        self.color_properties = [
            'fill', 'stroke', 'stop-color', 'flood-color', 
            'lighting-color', 'color', 
        ]
        
        # Enhanced color pattern to catch more formats
        self.color_pattern = re.compile(
            r'(#[0-9a-fA-F]{3,8}|'  # Hex colors (3, 4, 6, or 8 digits)
            r'rgb\(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\)|'  # RGB
            r'rgba\(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*[\d.]+\s*\)|'  # RGBA
            r'hsl\(\s*\d+\s*,\s*\d+%?\s*,\s*\d+%?\s*\)|'  # HSL
            r'hsla\(\s*\d+\s*,\s*\d+%?\s*,\s*\d+%?\s*,\s*[\d.]+\s*\)|'  # HSLA
            r'\b(?:red|green|blue|yellow|orange|purple|pink|brown|black|white|gray|grey|cyan|magenta|lime|navy|olive|teal|silver|maroon|fuchsia|aqua)\b)'  # Named colors
        )
        
        # Check available conversion methods
        self.conversion_method = self._detect_conversion_method()
        if self.conversion_method:
            logger.info(f"PNG conversion available via: {self.conversion_method}")
        else:
            logger.warning("No PNG conversion method available. Install cairosvg, Wand, or Inkscape for PNG output.")
    
    def _detect_conversion_method(self):
        """Detect which PNG conversion method is available."""
        if CAIRO_AVAILABLE:
            return "cairosvg"
        elif WAND_AVAILABLE:
            return "wand"
        elif INKSCAPE_AVAILABLE:
            return "inkscape"
        else:
            return None
    
    def process(self, inputs, config):
        try:
            # --- validate template_name ---
            template_name = config.get("template_name")
            if not template_name:
                msg = "Missing 'template_name' in config."
                logger.error(msg)
                return {"status":"error","error":msg}

            # --- copy the base SVG ---
            src = TEMPLATE_DIR / f"{template_name}.svg"
            if not src.exists():
                msg = f"Template file not found: {src}"
                logger.error(msg)
                return {"status":"error","error":msg}
                
            # Ensure output subdirectory exists (for templates with slashes)
            out_subdir = OUTPUT_FOLDER / os.path.dirname(template_name)
            out_subdir.mkdir(parents=True, exist_ok=True)
            base_out_name = f"template_{os.path.basename(template_name)}_{uuid.uuid4().hex}.svg"
            base_out_path = out_subdir / base_out_name
            
            try:
                copyfile(src, base_out_path)
                logger.info(f"Copied template {src} → {base_out_path}")
            except Exception as e:
                msg = f"Failed to copy template: {e}"
                logger.error(msg)
                return {"status":"error","error":msg}

            outputs = {
                "template_key": template_name,
                "base_svg_path": str(base_out_path)
            }

            # --- if a theme is requested, apply it ---
            theme_name = config.get("theme_name")
            final_svg_path = base_out_path
            png_ready_svg_path = None
            
            if theme_name:
                theme_file = THEMES_DIR / f"{Path(template_name).name}.json"
                if not theme_file.exists():
                    msg = f"No themes file for '{template_name}' at {theme_file}"
                    logger.error(msg)
                    return {"status":"error","error":msg}

                try:
                    with open(theme_file) as f:
                        themes = json.load(f)
                except (json.JSONDecodeError, FileNotFoundError) as e:
                    msg = f"Could not parse theme file {theme_file}: {e}"
                    logger.error(msg)
                    return {"status":"error","error":msg}

                if theme_name not in themes:
                    msg = (f"Theme '{theme_name}' not found for {template_name}. "
                           f"Available: {list(themes.keys())}")
                    logger.error(msg)
                    return {"status":"error","error":msg}

                theme = themes[theme_name]
                try:
                    css_svg, png_ready_svg_path = self.apply_theme(base_out_path, theme, template_name, theme_name)
                    final_svg_path = css_svg
                    outputs["theme_name"] = theme_name
                except Exception as e:
                    msg = f"Failed to apply theme: {e}"
                    logger.error(msg)
                    return {"status":"error","error":msg}

            outputs["output_path"] = str(final_svg_path)

            # --- Convert to PNG if requested ---
            convert_to_png = config.get("convert_to_png", False)
            png_width = config.get("png_width", 1200)  # Default width
            png_height = config.get("png_height", None)  # Height will be auto-calculated if None
            
            if convert_to_png:
                if not self.conversion_method:
                    msg = "PNG conversion requested but no conversion method available. Install cairosvg, Wand, or Inkscape."
                    logger.error(msg)
                    return {"status":"error","error":msg}
                
                try:
                    theme_map = theme if theme_name else None
                    svg_for_png = final_svg_path  # always use the visible output SVG
                    png_path = self.convert_to_png(svg_for_png, png_width, png_height, theme_map)
                    outputs["png_path"] = str(png_path)
                    logger.info(f"Successfully converted to PNG: {png_path}")
                    
                    # Verify PNG was created and has reasonable size
                    if png_path.exists() and png_path.stat().st_size > 1000:  # At least 1KB
                        logger.info(f"PNG conversion verified: {png_path} ({png_path.stat().st_size} bytes)")
                    else:
                        logger.warning(f"PNG file seems unusually small: {png_path}")
                        
                except Exception as e:
                    msg = f"Failed to convert to PNG: {e}"
                    logger.error(msg)
                    return {"status":"error","error":msg}

            return {"status":"success", "outputs": outputs}

        except Exception as e:
            logger.exception("Error in SelectTemplate node")
            return {"status":"error","error":str(e)}

    # --- Helper -------------------------------------------------------------
    def _replace_css_vars(self, svg_text: str, theme_map: dict[str, str] | None) -> str:
        """Replace occurrences of `var(--foo)` in *svg_text* using *theme_map*."""
        if not theme_map:
            return svg_text
        pattern = re.compile(r"var\((--[^)]+)\)")

        def _sub(match: re.Match[str]):  # type: ignore[type-var]
            var_name = match.group(1)
            return theme_map.get(var_name, match.group(0))

        return pattern.sub(_sub, svg_text)

    # ----------------------------------------------------------------------
    def convert_to_png(self, svg_path: Path, width: int = 1200, height: int | None = None, theme_map: dict[str, str] | None = None) -> Path:
        """Convert SVG to PNG using available conversion method."""
        png_path = svg_path.with_suffix('.png')
        
        try:
            if self.conversion_method == "cairosvg":
                return self._convert_with_cairosvg(svg_path, png_path, width, height, theme_map)
            elif self.conversion_method == "wand":
                return self._convert_with_wand(svg_path, png_path, width, height, theme_map)
            elif self.conversion_method == "inkscape":
                return self._convert_with_inkscape(svg_path, png_path, width, height, theme_map)
            else:
                raise Exception("No PNG conversion method available")
                
        except Exception as e:
            logger.error(f"PNG conversion failed: {e}")
            raise
    
    def _convert_with_cairosvg(self, svg_path: Path, png_path: Path, width: int, height: int | None = None, theme_map: dict[str, str] | None = None) -> Path:
        """Convert using cairosvg (recommended method)."""
        with open(svg_path, 'r', encoding='utf-8') as svg_file:
            svg_text = svg_file.read()

        # Replace CSS vars on-the-fly if a theme mapping is provided
        svg_text = self._replace_css_vars(svg_text, theme_map)
        svg_data = svg_text.encode('utf-8')

        # Convert with specified dimensions and high DPI for quality
        kwargs = {
            'output_width': width,
            'dpi': 300  # High DPI for better quality
        }
        if height:
            kwargs['output_height'] = height
        
        png_data = svg2png(bytestring=svg_data, **kwargs)
        
        with open(png_path, 'wb') as png_file:
            png_file.write(png_data)
            
        return png_path
    
    def _convert_with_wand(self, svg_path: Path, png_path: Path, width: int, height: int | None = None, theme_map: dict[str, str] | None = None) -> Path:
        """Convert using Wand (ImageMagick)."""
        # If we need to replace CSS vars, write a temporary SVG
        tmp_svg = svg_path
        if theme_map:
            tmp_svg = svg_path.with_suffix('.tmp.svg')
            with open(svg_path, 'r', encoding='utf-8') as fh:
                svg_text = self._replace_css_vars(fh.read(), theme_map)
            with open(tmp_svg, 'w', encoding='utf-8') as fh:
                fh.write(svg_text)

        with WandImage(filename=str(tmp_svg)) as img:
            # Set high resolution for better quality
            img.resolution = (300, 300)
            
            # Resize if dimensions specified
            if height:
                img.resize(width, height)
            else:
                # Maintain aspect ratio
                original_width, original_height = img.size
                aspect_ratio = original_height / original_width
                new_height = int(width * aspect_ratio)
                img.resize(width, new_height)
            
            # Set background to transparent for better compatibility
            img.background_color = 'transparent'
            img.format = 'png'
            img.save(filename=str(png_path))

        # Clean up temp file
        if theme_map and tmp_svg.exists():
            tmp_svg.unlink(missing_ok=True)

        return png_path
    
    def _convert_with_inkscape(self, svg_path: Path, png_path: Path, width: int, height: int | None = None, theme_map: dict[str, str] | None = None) -> Path:
        """Convert using Inkscape command line."""
        # If CSS vars must be replaced, write temp SVG
        tmp_svg_path = svg_path
        if theme_map:
            tmp_svg_path = svg_path.with_suffix('.tmp.svg')
            with open(svg_path, 'r', encoding='utf-8') as fh:
                svg_text = self._replace_css_vars(fh.read(), theme_map)
            with open(tmp_svg_path, 'w', encoding='utf-8') as fh:
                fh.write(svg_text)

        cmd = [
            INKSCAPE_PATH,
            str(tmp_svg_path),
            '--export-type=png',
            f'--export-filename={png_path}',
            f'--export-width={width}'
        ]
        
        if height:
            cmd.append(f'--export-height={height}')
        
        # Add high DPI and background handling for better quality
        cmd.extend([
            '--export-dpi=300',
            '--export-background-opacity=0'  # Transparent background
        ])
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise Exception(f"Inkscape conversion failed: {result.stderr}")
        
        if not png_path.exists():
            raise Exception("PNG file was not created by Inkscape")

        # Clean up temp SVG
        if theme_map and tmp_svg_path.exists():
            tmp_svg_path.unlink(missing_ok=True)
        
        return png_path

    def extract_colors_from_svg(self, tree):
        """Extract all colors used in the SVG for dynamic mapping with enhanced pattern matching."""
        colors = set()
        
        # Check all elements
        for elem in tree.iter():
            # Check style attribute
            style = elem.get('style', '')
            if style:
                # Find all color values in style using enhanced pattern
                color_matches = self.color_pattern.findall(style)
                colors.update(color_matches)
                
                # Also check for specific property patterns
                for prop in self.color_properties:
                    pattern = rf'{re.escape(prop)}:\s*([^;]+)'
                    matches = re.findall(pattern, style, re.IGNORECASE)
                    for match in matches:
                        match = match.strip()
                        if self.color_pattern.match(match):
                            colors.add(match)
            
            # Check direct attributes
            for prop in self.color_properties:
                value = elem.get(prop, '')
                if value and self.color_pattern.match(value):
                    colors.add(value)
                    
        # Filter out 'none' and 'transparent' as they're not colors to replace
        colors = {c for c in colors if c.lower() not in ['none', 'transparent', 'inherit', 'currentcolor']}
        
        return colors

    def create_color_mapping(self, svg_colors, theme):
        """Create a mapping from SVG colors to theme colors with smarter assignment."""
        color_mapping = {}
        theme_colors = list(theme.values())
        
        # Sort colors for consistent mapping (prioritize more common colors first)
        sorted_colors = sorted(svg_colors, key=lambda x: (
            0 if x.startswith('#') else 1,  # Hex colors first
            len(x),  # Shorter colors first
            x.lower()  # Alphabetical
        ))
        
        for i, color in enumerate(sorted_colors):
            if i < len(theme_colors):
                color_mapping[color] = theme_colors[i]
            else:
                # If we have more SVG colors than theme colors, reuse theme colors
                color_mapping[color] = theme_colors[i % len(theme_colors)]
        
        return color_mapping

    def apply_direct_color_replacement(self, tree, color_map):
        """Replace colors in SVG directly with theme colors (for PNG conversion) with enhanced matching."""
        if not color_map:
            return
            
        # Create a more precise replacement function
        def replace_color_in_value(value, old_color, new_color):
            """Replace color in a value string more precisely."""
            if not value:
                return value
                
            # If the value is exactly the old color, replace it
            if value.strip() == old_color:
                return new_color
                
            # For style attributes, be more careful about replacements
            if ':' in value:  # Likely a style attribute
                # Split by semicolon and process each property
                parts = value.split(';')
                new_parts = []
                for part in parts:
                    if ':' in part:
                        prop, val = part.split(':', 1)
                        prop = prop.strip()
                        val = val.strip()
                        if val == old_color:
                            new_parts.append(f"{prop}:{new_color}")
                        else:
                            new_parts.append(part)
                    else:
                        new_parts.append(part)
                return ';'.join(new_parts)
            else:
                # For direct attribute values, use exact match
                return new_color if value == old_color else value
            
        for elem in tree.iter():
            # Handle style attribute more carefully
            style = elem.get('style')
            if style:
                new_style = style
                for orig_color, new_color in color_map.items():
                    # Use more precise replacement for style attributes
                    if orig_color in style:
                        # Split style into individual properties
                        style_parts = [part.strip() for part in style.split(';') if part.strip()]
                        new_style_parts = []
                        
                        for part in style_parts:
                            if ':' in part:
                                prop, value = part.split(':', 1)
                                prop = prop.strip()
                                value = value.strip()
                                
                                # Only replace if the value matches exactly
                                if value == orig_color:
                                    new_style_parts.append(f"{prop}:{new_color}")
                                else:
                                    new_style_parts.append(part)
                            else:
                                new_style_parts.append(part)
                        
                        new_style = ';'.join(new_style_parts)
                
                if new_style != style:
                    elem.set('style', new_style)
            
            # Handle direct attributes with exact matching
            for prop in self.color_properties:
                attr_value = elem.get(prop)
                if attr_value and attr_value in color_map:
                    elem.set(prop, color_map[attr_value])

    def apply_css_variable_replacement(self, tree, theme):
        """Replace colors in SVG with CSS variables (for SVG with CSS)."""
        svg_colors = self.extract_colors_from_svg(tree)
        logger.info(f"Found colors in SVG: {svg_colors}")
        
        # Create CSS variable mapping
        css_var_map = {}
        theme_keys = list(theme.keys())
        
        for i, color in enumerate(sorted(svg_colors)):
            if i < len(theme_keys):
                var_name = theme_keys[i]
                if not var_name.startswith('--'):
                    var_name = f'--{var_name}'
                css_var_map[color] = f'var({var_name})'
        
        logger.info(f"CSS Variable mapping: {css_var_map}")
        
        # Apply CSS variable replacements with exact matching
        for elem in tree.iter():
            # Handle style attribute
            style = elem.get('style')
            if style:
                new_style = style
                for orig_color, var_ref in css_var_map.items():
                    # Split style into individual properties for precise replacement
                    if orig_color in style:
                        style_parts = [part.strip() for part in style.split(';') if part.strip()]
                        new_style_parts = []
                        
                        for part in style_parts:
                            if ':' in part:
                                prop, value = part.split(':', 1)
                                prop = prop.strip()
                                value = value.strip()
                                
                                # Only replace if the value matches exactly
                                if value == orig_color:
                                    new_style_parts.append(f"{prop}:{var_ref}")
                                else:
                                    new_style_parts.append(part)
                            else:
                                new_style_parts.append(part)
                        
                        new_style = ';'.join(new_style_parts)
                
                if new_style != style:
                    elem.set('style', new_style)
            
            # Handle direct attributes with exact matching
            for prop in self.color_properties:
                attr_value = elem.get(prop)
                if attr_value and attr_value in css_var_map:
                    elem.set(prop, css_var_map[attr_value])

    def apply_theme(self, svg_path: Path, theme: dict, template_name: str, theme_name: str) -> tuple[Path, Path]:
        """
        Apply theme to SVG and return only the CSS-variable SVG path (do not generate *_png_ready_*.svg).
        Returns: (css_svg_path, css_svg_path)
        """
        try:
            # Load the original SVG
            tree = ET.parse(svg_path)
            root = tree.getroot()
            
            # Extract colors from SVG
            svg_colors = self.extract_colors_from_svg(tree)
            logger.info(f"Found colors in SVG: {svg_colors}")
            
            # Create direct color mapping
            color_map = self.create_color_mapping(svg_colors, theme)
            logger.info(f"Color mapping: {color_map}")
            
        except ET.ParseError as e:
            raise Exception(f"Failed to parse SVG file: {e}")

        # Create the CSS-based SVG (with variables)
        css_tree = deepcopy(tree)
        css_root = css_tree.getroot()
        
        # Apply CSS variable replacements
        self.apply_css_variable_replacement(css_tree, theme)
        
        # Build the CSS text with proper formatting
        css_lines = []
        for var, val in theme.items():
            var_name = var if var.startswith('--') else f'--{var}'
            css_lines.append(f"  {var_name}: {val};")
        css = "\n".join(css_lines)

        # Create <style> element in the SVG namespace
        style_elem = ET.Element(f"{{{SVG_NS}}}style")
        style_elem.text = f"\n:root {{\n{css}\n}}\n"

        # Insert <style> into existing or new <defs>
        defs = css_root.find(f"{{{SVG_NS}}}defs")
        if defs is None:
            defs = ET.Element(f"{{{SVG_NS}}}defs")
            css_root.insert(0, defs)
        defs.insert(0, style_elem)

        # Build output file path
        out_subdir = OUTPUT_FOLDER / os.path.dirname(template_name)
        out_subdir.mkdir(parents=True, exist_ok=True)
        css_name = f"template_{Path(template_name).name}_{theme_name}_{uuid.uuid4().hex}.svg"
        css_path = out_subdir / css_name

        try:
            # Write CSS-variable SVG
            css_tree.write(css_path, encoding="utf-8", xml_declaration=True)
            logger.info(f"Applied theme '{theme_name}' → {css_path}")
            # Return the same path for both outputs (no png_ready svg generated)
            return css_path, css_path
            
        except Exception as e:
            raise Exception(f"Failed to write themed SVG: {e}")

    def validate_svg(self, svg_path: Path) -> bool:
        """Quick validation that the file is parseable SVG."""
        try:
            tree = ET.parse(svg_path)
            root = tree.getroot()
            return root.tag.endswith('svg') or root.tag == f"{{{SVG_NS}}}svg"
        except Exception as e:
            logger.warning(f"Invalid SVG {svg_path}: {e}")
            return False

_handler = SelectTemplateHandler()

def process(inputs, config):
    logger.info("SelectTemplate - Top-level process called")
    return _handler.process(inputs, config)

if __name__ == "__main__":
    # [Rest of the main function remains the same as original]
    p = argparse.ArgumentParser(
        description="Select an SVG template, optionally applying a colour theme and converting to PNG."
    )
    
    p.add_argument(
        "--templates_dir",
        default=str(PROJECT_ROOT / "templates"),
        help="Folder where your master SVGs live"
    )
    
    args, rest = p.parse_known_args()
    tpl_dir = Path(args.templates_dir)
    if not tpl_dir.exists():
        p.error(f"templates_dir not found: {tpl_dir}")

    # Discover and validate templates
    ALL_TEMPLATES = []
    ALL_TEMPLATE_PATHS = {}
    handler = SelectTemplateHandler()
    
    for svg_path in tpl_dir.rglob("*.svg"):
        # Validate SVG before adding to templates
        if handler.validate_svg(svg_path):
            rel = svg_path.relative_to(tpl_dir)
            key = str(rel.with_suffix("")).replace(os.sep, "/")
            ALL_TEMPLATES.append(key)
            ALL_TEMPLATE_PATHS[key] = svg_path
        else:
            logger.warning(f"Skipping invalid SVG: {svg_path}")

    print(f"Found {len(ALL_TEMPLATES)} valid templates:")
    for template in ALL_TEMPLATES:
        print(f"  {template} -> {ALL_TEMPLATE_PATHS[template]}")

    # Add the rest of the args
    p.add_argument(
        "--list-templates",
        action="store_true",
        help="List available templates and exit"
    )
    p.add_argument(
        "--template_name",
        help="Relative path (without .svg) of an SVG in templates_dir"
    )
    p.add_argument(
        "--theme_name",
        help="Optional colour theme for the template"
    )
    p.add_argument(
        "--output_dir",
        default=str(OUTPUT_FOLDER),
        help="Where to write the resulting SVG(s)"
    )
    p.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate SVG files, don't process"
    )
    p.add_argument(
        "--convert-to-png",
        action="store_true",
        help="Convert the final SVG to PNG format"
    )
    p.add_argument(
        "--png-width",
        type=int,
        default=1200,
        help="Width of the PNG output (default: 1200)"
    )
    p.add_argument(
        "--png-height",
        type=int,
        help="Height of the PNG output (auto-calculated if not specified)"
    )
    
    # Parse full args
    args = p.parse_args()

    if args.validate_only:
        print("Validating all SVG files...")
        for template, path in ALL_TEMPLATE_PATHS.items():
            is_valid = handler.validate_svg(path)
            status = "✓ Valid" if is_valid else "✗ Invalid"
            print(f"{status}: {template}")
        sys.exit(0)

    if args.list_templates:
        print("\n".join(ALL_TEMPLATES))
        sys.exit(0)

    if not args.template_name:
        p.error("Either --list-templates, --validate-only, or --template_name is required")

    if args.template_name not in ALL_TEMPLATES:
        p.error(f"Unknown template '{args.template_name}'. Available: {ALL_TEMPLATES}")

    # Validate the specific template
    template_path = ALL_TEMPLATE_PATHS[args.template_name]
    if not handler.validate_svg(template_path):
        p.error(f"Template '{args.template_name}' is not a valid SVG file")

    # Handle themes
    theme_file = THEMES_DIR / f"{Path(args.template_name).name}.json"
    theme_names = []
    if theme_file.exists():
        try:
            with open(theme_file) as f:
                themes = json.load(f)
                theme_names = list(themes.keys())
        except Exception as e:
            print(f"Warning: Could not read theme file: {e}")
                
    if not args.theme_name and theme_names:
        print(f"Available themes for {args.template_name}: {theme_names}")
        theme_choice = input("Enter theme name (or leave blank for no theme): ").strip()
        theme_name = theme_choice if theme_choice in theme_names else None
    else:
        theme_name = args.theme_name

    if theme_name and theme_name not in theme_names:
        p.error(f"Theme '{theme_name}' not available. Available themes: {theme_names}")

    # Process the template
    config = {
        "template_name": args.template_name,
        "convert_to_png": args.convert_to_png,
        "png_width": args.png_width,
        **({"theme_name": theme_name} if theme_name else {}),
        **({"png_height": args.png_height} if args.png_height else {})
    }
    
    result = handler.process({}, config)
    
    if result["status"] == "success":
        print("✓ Success!")
        print(json.dumps(result, indent=2))
    else:
        print("✗ Error:")
        print(json.dumps(result, indent=2))
        sys.exit(1)