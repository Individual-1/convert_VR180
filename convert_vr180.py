import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import numpy as np
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
import platform
import subprocess
from tempfile import NamedTemporaryFile

from convert_VR180_images import load_st_map, apply_st_map

logger = logging.getLogger(__name__)


def load_mask(mask_path):
    """Load mask image from input file."""
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Failed to load image from {mask_path}")
    return mask


def add_transparent_image(background, foreground, x_offset=None, y_offset=None):
    """Function overlaying transparent image on another image, from https://stackoverflow.com/a/71701023"""
    bg_h, bg_w, bg_channels = background.shape
    fg_h, fg_w, fg_channels = foreground.shape

    assert (
        bg_channels == 3
    ), f"background image should have exactly 3 channels (RGB). found:{bg_channels}"
    assert (
        fg_channels == 4
    ), f"foreground image should have exactly 4 channels (RGBA). found:{fg_channels}"

    # center by default
    if x_offset is None:
        x_offset = (bg_w - fg_w) // 2
    if y_offset is None:
        y_offset = (bg_h - fg_h) // 2

    w = min(fg_w, bg_w, fg_w + x_offset, bg_w - x_offset)
    h = min(fg_h, bg_h, fg_h + y_offset, bg_h - y_offset)

    if w < 1 or h < 1:
        return

    # clip foreground and background images to the overlapping regions
    bg_x = max(0, x_offset)
    bg_y = max(0, y_offset)
    fg_x = max(0, x_offset * -1)
    fg_y = max(0, y_offset * -1)
    foreground = foreground[fg_y : fg_y + h, fg_x : fg_x + w]
    background_subsection = background[bg_y : bg_y + h, bg_x : bg_x + w]

    # separate alpha and color channels from the foreground image
    foreground_colors = foreground[:, :, :3]
    alpha_channel = foreground[:, :, 3] / 255  # 0-255 => 0.0-1.0

    # construct an alpha_mask that matches the image shape
    alpha_mask = alpha_channel[:, :, np.newaxis]

    # combine the background with the overlay image weighted by alpha
    composite = (
        background_subsection * (1 - alpha_mask) + foreground_colors * alpha_mask
    )

    # overwrite the section of the background image that has been updated
    background[bg_y : bg_y + h, bg_x : bg_x + w] = composite


def apply_mask(image, mask):
    masked_image = image.copy()
    add_transparent_image(masked_image, mask)
    return masked_image


def convert_fisheye_equirect(
    image_path, st_map=None, image_mask=None, mask_before_convert=False
):
    """Load an image, apply ST mapping, and save the result with debug outputs."""

    # Load the input image with Pillow to ensure color space compatibility
    with Image.open(image_path) as image_pil:
        try:
            image_pil = image_pil.convert(
                "RGB"
            )  # Ensure it's in RGB mode for consistent colors
        except:
            return  # Cant load this image

    # Convert to OpenCV format (BGR) for processing
    image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

    if image_mask is not None and mask_before_convert:
        image = apply_mask(image, image_mask)

    if st_map is not None:
        # Apply ST mapping
        mapped_image = apply_st_map(image, st_map)
    else:
        mapped_image = image

    if image_mask is not None and not mask_before_convert:
        mapped_image = apply_mask(mapped_image, image_mask)

    # Convert mapped image to RGB format for final saving
    mapped_image_rgb = cv2.cvtColor(mapped_image, cv2.COLOR_BGR2RGB)
    mapped_image_pil = Image.fromarray(mapped_image_rgb)

    return mapped_image_pil


def process_image(
    image_path,
    output_folder,
    st_map=None,
    image_mask=None,
    mask_before_convert=False,
    spatial=False,
    spatial_bin=None,
    spatial_input_format=None,
    spatial_hfov=None,
    spatial_cdist=None,
    spatial_hadjust=None,
):
    """Load image and apply the desired transformations"""

    image = None
    if st_map is not None or image_mask is not None:
        image = convert_fisheye_equirect(
            image_path, st_map, image_mask, mask_before_convert
        )

    if spatial:
        if image is None:
            output_path = os.path.join(
                output_folder,
                os.path.splitext(os.path.basename(image_path))[0] + ".heic",
            )
            subprocess.run(
                [
                    spatial_bin,
                    "make",
                    "-f",
                    spatial_input_format,
                    "-y",
                    "--hfov",
                    str(spatial_hfov),
                    "--cdist",
                    str(spatial_cdist),
                    "--hadjust",
                    str(spatial_hadjust),
                    "-i",
                    image_path,
                    "-o",
                    output_path,
                ]
            )
        else:
            with NamedTemporaryFile(suffix=".jpg") as source_image:
                image.save(source_image.name, format="JPEG", quality=100)
                output_path = os.path.join(
                    output_folder,
                    os.path.splitext(os.path.basename(image_path))[0] + ".heic",
                )
                subprocess.run(
                    [
                        spatial_bin,
                        "make",
                        "-f",
                        spatial_input_format,
                        "-y",
                        "--hfov",
                        str(spatial_hfov),
                        "--cdist",
                        str(spatial_cdist),
                        "--hadjust",
                        str(spatial_hadjust),
                        "-i",
                        source_image.name,
                        "-o",
                        output_path,
                    ]
                )
    elif image is not None:
        # Define the output path and save the final output image with quality 90 (without EXIF)
        output_path = os.path.join(
            output_folder, os.path.splitext(os.path.basename(image_path))[0] + ".jpg"
        )
        image.save(output_path, format="JPEG", quality=100)
    else:
        logger.info(f"No actions specified, {image_path} unchanged")


def main():
    parser = argparse.ArgumentParser(description="Bulk process VR180 files")

    parser.add_argument(
        "--input-folder",
        type=str,
        required=True,
        help="Path to the folder containing input images",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        required=True,
        help="Path to the folder for saving output images",
    )

    stmap_group = parser.add_argument_group(
        "Equirectangular", "Convert fisheye to equirectangular"
    )
    stmap_group.add_argument(
        "--stmap-path",
        type=str,
        help="Path to ST map in EXR format (Enables equirect conversion)",
    )

    mask_group = parser.add_argument_group("Mask", "Apply an image mask over input")
    mask_group.add_argument(
        "--mask-path",
        type=str,
        help="Path to overlay mask image (Enables image masking)",
    )
    mask_group.add_argument(
        "--mask-before-convert",
        action="store_true",
        help="If set, run mask before equirectangular conversion",
    )

    spatial_group = parser.add_argument_group(
        "Spatial", "Convert input to Apple Spatial Photo"
    )
    spatial_group.add_argument(
        "--spatial", action="store_true", help="Convert to Apple Spatial Photo"
    )
    spatial_group.add_argument(
        "--spatial-bin",
        type=str,
        default="spatial",
        help="Optional path to spatial CLI binary",
    )
    spatial_group.add_argument(
        "--spatial-input-format", type=str, default="sbs", help="Input vr format"
    )
    spatial_group.add_argument(
        "--spatial-hfov", type=int, default=180, help="Input horizontal FOV"
    )
    spatial_group.add_argument(
        "--spatial-cdist", type=int, default=60, help="Interpupillary distance"
    )
    spatial_group.add_argument(
        "--spatial-hadjust", type=int, default=0, help="Horizontal adjustment"
    )

    args = parser.parse_args()

    st_map = None
    image_mask = None

    if args.stmap_path is not None:
        # Load ST map
        st_map = load_st_map(args.stmap_path)

    # Load mask
    if args.mask_path is not None:
        image_mask = load_mask(args.mask_path)

    # If spatial conversion requested then check platform and presence of spatial cli
    if args.spatial:
        if platform.system() != "Darwin":
            raise EnvironmentError("Spatial conversion requires Macos platform")

        try:
            subprocess.run(
                [args.spatial_bin, "-h"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            raise FileNotFoundError(f"Failed to find {args.spatial_bin} binary")

    # Ensure output folder exists
    os.makedirs(args.output_folder, exist_ok=True)

    # Define the list of allowed extensions
    allowed_extensions = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

    # Example: List only files in a directory (all files)
    files = [
        f
        for f in os.listdir(args.input_folder)
        if os.path.isfile(os.path.join(args.input_folder, f))
        and f.lower().endswith(allowed_extensions)
    ]

    # Process each image in the input folder in parallel
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = []
        for image_name in files:
            print(image_name)
            image_path = os.path.join(args.input_folder, image_name)
            futures.append(
                executor.submit(
                    process_image,
                    image_path,
                    args.output_folder,
                    st_map,
                    image_mask,
                    args.mask_before_convert,
                    args.spatial,
                    args.spatial_bin,
                    args.spatial_input_format,
                    args.spatial_hfov,
                    args.spatial_cdist,
                    args.spatial_hadjust,
                )
            )

        for future in as_completed(futures):
            future.result()  # Wait for all futures to complete


if __name__ == "__main__":
    main()
