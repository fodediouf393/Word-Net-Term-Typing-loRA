import base64
import argparse
from pathlib import Path

from PIL import Image


def png_to_svg(png_path: Path, svg_path: Path, overwrite: bool = False) -> bool:
    if svg_path.exists() and not overwrite:
        return False

    with Image.open(png_path) as im:
        width, height = im.size

    png_bytes = png_path.read_bytes()
    b64 = base64.b64encode(png_bytes).decode("ascii")

    svg = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     xmlns:xlink="http://www.w3.org/1999/xlink"
     width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <image x="0" y="0" width="{width}" height="{height}"
         xlink:href="data:image/png;base64,{b64}" />
</svg>
"""

    svg_path.write_text(svg, encoding="utf-8")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        type=str,
        default="outputs_encoder/report_images",
        help="Dossier contenant les PNG"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Écrase les SVG existants"
    )
    args = parser.parse_args()

    d = Path(args.dir).expanduser().resolve()
    if not d.exists():
        raise FileNotFoundError(f"Dossier introuvable: {d}")

    pngs = sorted(d.glob("*.png"))
    if not pngs:
        print(f"Aucun PNG trouvé dans {d}")
        return

    created = 0
    skipped = 0

    for png in pngs:
        svg = png.with_suffix(".svg")
        ok = png_to_svg(png, svg, overwrite=args.overwrite)
        if ok:
            created += 1
        else:
            skipped += 1

    print(f"Dossier: {d}")
    print(f"PNG trouvés: {len(pngs)}")
    print(f"SVG créés: {created}")
    print(f"SVG ignorés (déjà présents): {skipped}")


if __name__ == "__main__":
    main()
