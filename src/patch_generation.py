import os
import csv
import random
import numpy as np
import rasterio

from rasterio.windows import Window
from PIL import Image


BASE_DIR = r"E:\Rasengan\Cloud-Removal"

CLOUDY_DIR = os.path.join(
    BASE_DIR,
    "data",
    "cloudy"
)

NON_CLOUDY_DIR = os.path.join(
    BASE_DIR,
    "data",
    "non cloudy"
)

PATCH_DIR = os.path.join(
    BASE_DIR,
    "data",
    "patches"
)

CLOUDY_PATCH_DIR = os.path.join(
    PATCH_DIR,
    "cloudy"
)

NON_CLOUDY_PATCH_DIR = os.path.join(
    PATCH_DIR,
    "non-cloudy"
)

METADATA_DIR = os.path.join(
    BASE_DIR,
    "metadata"
)

METADATA_FILE = os.path.join(
    METADATA_DIR,
    "patches_metadata.csv"
)


PATCH_SIZE = 256
STRIDE = 128

MIN_VALID = 0.90

MAX_PATCHES_PER_PAIR = 5000

RANDOM_SEED = 42


os.makedirs(CLOUDY_PATCH_DIR, exist_ok=True)
os.makedirs(NON_CLOUDY_PATCH_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)


random.seed(RANDOM_SEED)


def get_band_files(folder):

    files = {
        "B2": os.path.join(folder, "BAND2.TIF"),
        "B3": os.path.join(folder, "BAND3.TIF"),
        "B4": os.path.join(folder, "BAND4.TIF")
    }

    for band, path in files.items():

        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"{band} not found: {path}"
            )

    return files


def get_scene_folders(base_dir):

    folders = {}

    if not os.path.isdir(base_dir):
        raise FileNotFoundError(
            f"Directory not found: {base_dir}"
        )

    for name in os.listdir(base_dir):

        path = os.path.join(base_dir, name)

        if os.path.isdir(path):
            folders[name] = path

    return folders


def calculate_common_area(cloudy_path, clear_path):

    with rasterio.open(cloudy_path) as cloudy, \
         rasterio.open(clear_path) as clear:

        if cloudy.crs != clear.crs:
            raise ValueError(
                "Cloudy and non-cloudy CRS do not match"
            )

        if not np.isclose(
            cloudy.res[0],
            clear.res[0]
        ) or not np.isclose(
            cloudy.res[1],
            clear.res[1]
        ):
            raise ValueError(
                "Cloudy and non-cloudy resolutions do not match"
            )

        resolution_x = cloudy.res[0]
        resolution_y = abs(cloudy.res[1])

        cloudy_left = cloudy.bounds.left
        cloudy_right = cloudy.bounds.right
        cloudy_top = cloudy.bounds.top
        cloudy_bottom = cloudy.bounds.bottom

        clear_left = clear.bounds.left
        clear_right = clear.bounds.right
        clear_top = clear.bounds.top
        clear_bottom = clear.bounds.bottom

        overlap_left = max(
            cloudy_left,
            clear_left
        )

        overlap_right = min(
            cloudy_right,
            clear_right
        )

        overlap_top = min(
            cloudy_top,
            clear_top
        )

        overlap_bottom = max(
            cloudy_bottom,
            clear_bottom
        )

        overlap_width = overlap_right - overlap_left
        overlap_height = overlap_top - overlap_bottom

        if overlap_width <= 0 or overlap_height <= 0:
            raise ValueError(
                "Cloudy and non-cloudy scenes do not overlap"
            )

        cloudy_col = int(round(
            (overlap_left - cloudy_left)
            / resolution_x
        ))

        cloudy_row = int(round(
            (cloudy_top - overlap_top)
            / resolution_y
        ))

        clear_col = int(round(
            (overlap_left - clear_left)
            / resolution_x
        ))

        clear_row = int(round(
            (clear_top - overlap_top)
            / resolution_y
        ))

        overlap_width_px = int(
            np.floor(
                overlap_width / resolution_x
            )
        )

        overlap_height_px = int(
            np.floor(
                overlap_height / resolution_y
            )
        )

        return {
            "cloudy_row": cloudy_row,
            "cloudy_col": cloudy_col,
            "clear_row": clear_row,
            "clear_col": clear_col,
            "width": overlap_width_px,
            "height": overlap_height_px,
            "resolution_x": resolution_x,
            "resolution_y": resolution_y
        }


def read_fcc(b2_src, b3_src, b4_src, window):

    b2 = b2_src.read(
        1,
        window=window
    )

    b3 = b3_src.read(
        1,
        window=window
    )

    b4 = b4_src.read(
        1,
        window=window
    )

    fcc = np.stack(
        [
            b4,
            b3,
            b2
        ],
        axis=-1
    )

    return fcc


def calculate_valid_percentage(fcc):

    valid = np.all(
        fcc > 0,
        axis=2
    )

    return float(
        np.mean(valid)
    )


def convert_to_uint8(fcc):

    fcc = fcc.astype(
        np.float32
    )

    output = np.clip(
        fcc / 1023.0 * 255.0,
        0,
        255
    )

    return output.astype(
        np.uint8
    )


def generate_patch_positions(
    width,
    height
):

    positions = []

    max_row = (
        height - PATCH_SIZE
    )

    max_col = (
        width - PATCH_SIZE
    )

    if max_row < 0 or max_col < 0:
        return positions

    for row in range(
        0,
        max_row + 1,
        STRIDE
    ):

        for col in range(
            0,
            max_col + 1,
            STRIDE
        ):

            positions.append(
                (row, col)
            )

    random.shuffle(
        positions
    )

    return positions


def process_pair(
    pair_name,
    cloudy_folder,
    clear_folder,
    writer,
    global_patch_id
):

    cloudy_files = get_band_files(
        cloudy_folder
    )

    clear_files = get_band_files(
        clear_folder
    )

    alignment = calculate_common_area(
        cloudy_files["B2"],
        clear_files["B2"]
    )

    print()
    print("=" * 60)
    print("Pair:", pair_name)
    print("=" * 60)

    print(
        "Common area:",
        alignment["width"],
        "x",
        alignment["height"],
        "pixels"
    )

    print(
        "Cloudy start:",
        alignment["cloudy_row"],
        alignment["cloudy_col"]
    )

    print(
        "Non-cloudy start:",
        alignment["clear_row"],
        alignment["clear_col"]
    )

    positions = generate_patch_positions(
        alignment["width"],
        alignment["height"]
    )

    print(
        "Candidate positions:",
        len(positions)
    )

    generated = 0

    with rasterio.open(
        cloudy_files["B2"]
    ) as cloudy_b2, \
    rasterio.open(
        cloudy_files["B3"]
    ) as cloudy_b3, \
    rasterio.open(
        cloudy_files["B4"]
    ) as cloudy_b4, \
    rasterio.open(
        clear_files["B2"]
    ) as clear_b2, \
    rasterio.open(
        clear_files["B3"]
    ) as clear_b3, \
    rasterio.open(
        clear_files["B4"]
    ) as clear_b4:

        for common_row, common_col in positions:

            if generated >= MAX_PATCHES_PER_PAIR:
                break

            cloudy_window = Window(
                col_off=(
                    alignment["cloudy_col"]
                    + common_col
                ),
                row_off=(
                    alignment["cloudy_row"]
                    + common_row
                ),
                width=PATCH_SIZE,
                height=PATCH_SIZE
            )

            clear_window = Window(
                col_off=(
                    alignment["clear_col"]
                    + common_col
                ),
                row_off=(
                    alignment["clear_row"]
                    + common_row
                ),
                width=PATCH_SIZE,
                height=PATCH_SIZE
            )

            cloudy_fcc = read_fcc(
                cloudy_b2,
                cloudy_b3,
                cloudy_b4,
                cloudy_window
            )

            clear_fcc = read_fcc(
                clear_b2,
                clear_b3,
                clear_b4,
                clear_window
            )

            cloudy_valid = (
                calculate_valid_percentage(
                    cloudy_fcc
                )
            )

            clear_valid = (
                calculate_valid_percentage(
                    clear_fcc
                )
            )

            if cloudy_valid < MIN_VALID:
                continue

            if clear_valid < MIN_VALID:
                continue

            cloudy_uint8 = convert_to_uint8(
                cloudy_fcc
            )

            clear_uint8 = convert_to_uint8(
                clear_fcc
            )

            generated += 1

            patch_id = (
                f"{pair_name}_"
                f"patch_{generated:05d}"
            )

            filename = (
                f"{patch_id}.png"
            )

            cloudy_output = os.path.join(
                CLOUDY_PATCH_DIR,
                filename
            )

            clear_output = os.path.join(
                NON_CLOUDY_PATCH_DIR,
                filename
            )

            Image.fromarray(
                cloudy_uint8
            ).save(
                cloudy_output
            )

            Image.fromarray(
                clear_uint8
            ).save(
                clear_output
            )

            writer.writerow(
                {
                    "pair_id": pair_name,
                    "patch_id": patch_id,
                    "cloudy_filename": filename,
                    "noncloudy_filename": filename,
                    "row": common_row,
                    "col": common_col,
                    "patch_size": PATCH_SIZE,
                    "stride": STRIDE,
                    "cloudy_valid_pct": round(
                        cloudy_valid * 100,
                        2
                    ),
                    "noncloudy_valid_pct": round(
                        clear_valid * 100,
                        2
                    )
                }
            )

            if generated % 100 == 0:

                print(
                    f"Generated: {generated}"
                )

    print(
        "Accepted patches:",
        generated
    )

    return global_patch_id + generated


def main():

    cloudy_pairs = get_scene_folders(
        CLOUDY_DIR
    )

    clear_pairs = get_scene_folders(
        NON_CLOUDY_DIR
    )

    pair_names = sorted(
        set(cloudy_pairs.keys())
        &
        set(clear_pairs.keys())
    )

    print(
        "Cloudy scenes:",
        len(cloudy_pairs)
    )

    print(
        "Non-cloudy scenes:",
        len(clear_pairs)
    )

    print(
        "Matched pairs:",
        len(pair_names)
    )

    if not pair_names:
        raise RuntimeError(
            "No matching cloudy/non-cloudy pairs found"
        )

    fieldnames = [
        "pair_id",
        "patch_id",
        "cloudy_filename",
        "noncloudy_filename",
        "row",
        "col",
        "patch_size",
        "stride",
        "cloudy_valid_pct",
        "noncloudy_valid_pct"
    ]

    global_patch_id = 0

    with open(
        METADATA_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for index, pair_name in enumerate(
            pair_names,
            start=1
        ):

            print()
            print(
                f"Processing pair "
                f"{index}/{len(pair_names)}: "
                f"{pair_name}"
            )

            try:

                global_patch_id = process_pair(
                    pair_name,
                    cloudy_pairs[pair_name],
                    clear_pairs[pair_name],
                    writer,
                    global_patch_id
                )

            except Exception as error:

                print(
                    f"ERROR in {pair_name}: "
                    f"{error}"
                )

                continue

    print()
    print("=" * 60)
    print("PATCH GENERATION COMPLETE")
    print("=" * 60)

    print(
        "Total paired patches:",
        global_patch_id
    )

    print(
        "Metadata:",
        METADATA_FILE
    )

    print(
        "Cloudy patches:",
        CLOUDY_PATCH_DIR
    )

    print(
        "Non-cloudy patches:",
        NON_CLOUDY_PATCH_DIR
    )


if __name__ == "__main__":
    main()