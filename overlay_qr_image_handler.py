import logging
import os
import uuid
from pathlib import Path
from typing import Tuple, Optional

import qrcode  # pip install qrcode[pil]
from PIL import Image, ImageDraw

# Base stub; replace with real import if integrating
class NodeHandler:
    pass

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Utility to compute overlay position

def get_position(img_size: Tuple[int, int], overlay_size: Tuple[int, int], position: str, offset: Tuple[int, int] = (0, 0)) -> Tuple[int, int]:
    W, H = img_size
    w, h = overlay_size
    x_off, y_off = offset
    pos_map = {
        "center": ((W - w) // 2, (H - h) // 2),
        "top_left": (0, 0),
        "top_right": (W - w, 0),
        "bottom_left": (0, H - h),
        "bottom_right": (W - w, H - h),
        "top_center": ((W - w) // 2, 0),
        "bottom_center": ((W - w) // 2, H - h),
        "middle_left": (0, (H - h) // 2),
        "middle_right": (W - w, (H - h) // 2),
    }
    base = pos_map.get(position, pos_map["bottom_right"])
    x = max(0, min(base[0] + x_off, W - w))
    y = max(0, min(base[1] + y_off, H - h))
    return x, y

class QROverlayImageHandler(NodeHandler):
    """Handler for overlaying a QR code on an image."""

    def process(self, inputs, config):
        try:
            # Fetch parameters
            image_path = inputs.get("image_path") or config.get("image_path")
            qr_data = config.get("qr_data")
            qr_image_path = config.get("qr_image_path")
            position = config.get("position", "bottom_right")
            offset = config.get("offset", (0, 0))
            qr_scale = float(config.get("qr_scale", 1.0))
            opacity = float(config.get("opacity", 1.0))

            # Normalize offset
            if isinstance(offset, str):
                offset = tuple(map(int, offset.strip("() ").split(",")))

            # Validate inputs
            if not image_path or not os.path.exists(image_path):
                return {"status": "error", "error": f"Image not found: {image_path}"}
            if not qr_data and not qr_image_path:
                return {"status": "error", "error": "Either 'qr_data' or 'qr_image_path' must be provided"}
            if qr_image_path and not os.path.exists(qr_image_path):
                return {"status": "error", "error": f"QR image not found: {qr_image_path}"}

            # Prepare output path
            ext = Path(image_path).suffix
            out_name = f"qr_{uuid.uuid4().hex}{ext}"
            output_path = os.path.join("output", out_name)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Overlay QR
            success = self.overlay_qr(
                image_path, qr_data, qr_image_path,
                position, offset, qr_scale, opacity, output_path
            )
            if not success:
                return {"status": "error", "error": "Failed to overlay QR code"}

            return {"status": "success", "outputs": {"output_path": output_path}}
        except Exception as e:
            logger.exception("Error in QROverlayImage node")
            return {"status": "error", "error": str(e)}

    def overlay_qr(
        self,
        image_path: str,
        qr_data: Optional[str],
        qr_image_path: Optional[str],
        position: str,
        offset: Tuple[int, int],
        qr_scale: float,
        opacity: float,
        output_path: str
    ) -> bool:
        try:
            # Load base image
            base_img = Image.open(image_path).convert("RGBA")

            # Generate or load QR code image
            if qr_data:
                qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_H)
                qr.add_data(qr_data)
                qr.make(fit=True)
                qr_img = qr.make_image(fill_color="black", back_color="white").convert("RGBA")
            else:
                qr_img = Image.open(qr_image_path).convert("RGBA")

            # Scale QR
            w, h = qr_img.size
            qr_img = qr_img.resize((int(w * qr_scale), int(h * qr_scale)), Image.Resampling.LANCZOS)

            # Adjust opacity
            if opacity < 1.0:
                alpha = qr_img.split()[3]
                alpha = alpha.point(lambda p: int(p * opacity))
                qr_img.putalpha(alpha)

            # Compute position
            pos = get_position(base_img.size, qr_img.size, position, offset)

            # Composite
            composite = Image.new("RGBA", base_img.size)
            composite.paste(base_img, (0, 0))
            composite.paste(qr_img, pos, mask=qr_img)
            composite = composite.convert("RGB")
            composite.save(output_path)
            logger.info(f"QR overlay saved to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Overlay QR error: {e}")
            return False


def process(inputs, config):
    handler = QROverlayImageHandler()
    return handler.process(inputs, config)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Overlay a QR code on an image.")
    parser.add_argument("--image_path", required=True, help="Path to base PNG/JPEG image")
    parser.add_argument("--qr_data", help="Data string to encode in QR code")
    parser.add_argument("--qr_image_path", help="Path to existing QR code image")
    parser.add_argument("--position", default="bottom_right",
                        help="Position: center, top_left, top_right, bottom_left, bottom_right, top_center, bottom_center, middle_left, middle_right")
    parser.add_argument("--offset", default="(0,0)", help="Offset tuple, e.g. (10,20)")
    parser.add_argument("--qr_scale", type=float, default=1.0, help="Scale factor for QR code size")
    parser.add_argument("--opacity", type=float, default=1.0, help="QR code opacity (0.0–1.0)")
    args = parser.parse_args()

    # Normalize replacements for QR overlay
    inputs = {"image_path": args.image_path}
    config = {
        "qr_data": args.qr_data,
        "qr_image_path": args.qr_image_path,
        "position": args.position,
        "offset": args.offset,
        "qr_scale": args.qr_scale,
        "opacity": args.opacity,
    }
    result = process(inputs, config)
    print(result)
