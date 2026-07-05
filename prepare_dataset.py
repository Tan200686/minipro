from pathlib import Path

from PIL import Image


DATA_ROOT = Path("data")
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
TIFF_EXTS = {".tif", ".tiff"}


def convert_tiff_to_png(root: Path) -> None:
    """
    Convert all .tif / .tiff images under root to .png so TensorFlow can read them.
    The original .tif files are kept; you can delete them manually if you want.
    """
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in TIFF_EXTS:
            try:
                img = Image.open(path)
                img = img.convert("RGB")
                new_path = path.with_suffix(".png")
                img.save(new_path)
                print(f"Converted {path} -> {new_path}")
            except Exception as exc:  # pylint: disable=broad-except
                print(f"[ERROR] Could not convert {path}: {exc}")


def list_unsupported_files(root: Path) -> None:
    """
    List files that are not in supported image formats.
    """
    print("Scanning for unsupported files...")
    for path in root.rglob("*"):
        if path.is_file():
            ext = path.suffix.lower()
            if ext not in ALLOWED_EXTS and ext not in TIFF_EXTS:
                print(f"Unsupported file (will cause errors in training): {path}")


def verify_images(root: Path) -> None:
    """
    Try opening every allowed image file. If Pillow cannot open it,
    print it as a bad/corrupted image so the user can remove it.
    """
    print("Verifying image files...")
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in ALLOWED_EXTS:
            try:
                img = Image.open(path)
                img.verify()  # type: ignore[attr-defined]
            except Exception as exc:  # pylint: disable=broad-except
                print(f"[BAD IMAGE] {path} -> {exc}")


def main() -> None:
    if not DATA_ROOT.exists():
        print(f"'data' directory not found at {DATA_ROOT.resolve()}")
        return

    # 1) Convert TIFF images to PNG
    convert_tiff_to_png(DATA_ROOT)

    # 2) List remaining unsupported files (e.g. .zip, .txt, etc.)
    list_unsupported_files(DATA_ROOT)

    # 3) Verify that all allowed images are truly readable
    verify_images(DATA_ROOT)

    print("Done. Remove or move any 'Unsupported file' and '[BAD IMAGE]' listed above before running train.py.")


if __name__ == "__main__":
    main()


