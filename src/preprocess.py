import os
import json
from multiprocessing import Pool, cpu_count
import rasterio
import numpy as np

# Constants
TILE_SIZE = 512
INPUT_DIR = "data/raw"
OUTPUT_DIR = "data/tiles"
MANIFEST_PATH = os.path.join(OUTPUT_DIR, "tiles_manifest.json")
BANDS = {"B02": "blue", "B03": "green", "B04": "red", "B08": "nir"}


def generate_tile_grid(scene_path, tile_size=TILE_SIZE):
    with rasterio.open(scene_path.format(band="B02")) as src:
        width, height = src.width, src.height
        transform = src.transform

    tiles = []
    for i in range(0, height, tile_size):
        for j in range(0, width, tile_size):
            tiles.append({
                "scene_id": os.path.basename(os.path.dirname(scene_path)),
                "row_off": i,
                "col_off": j,
                "height": min(tile_size, height - i),
                "width": min(tile_size, width - j),
                "transform": rasterio.windows.transform(
                    rasterio.windows.Window(j, i, tile_size, tile_size), transform
                ).to_gdal()
            })
    return tiles


def process_tile(args):
    tile, scene_dir = args
    scene_id = tile["scene_id"]
    row, col = tile["row_off"], tile["col_off"]
    height, width = tile["height"], tile["width"]

    # Read bands
    arr = np.zeros((4, height, width), dtype=np.float32)
    for idx, (band_code, band_name) in enumerate(BANDS.items()):
        band_path = os.path.join(scene_dir, f"{scene_id}_{band_code}.jp2")
        with rasterio.open(band_path) as src:
            window = rasterio.windows.Window(col, row, width, height)
            arr[idx] = src.read(1, window=window)

    # Radiometric scaling (example placeholder)
    arr = arr / np.max(arr)

    # Simple mask: threshold NIR > 0.3
    mask = (arr[3] > 0.3).astype(np.uint8)

    # Save tile
    out_dir = os.path.join(OUTPUT_DIR, scene_id)
    os.makedirs(out_dir, exist_ok=True)
    tile_name = f"{scene_id}_{row}_{col}.npz"
    np.savez_compressed(
        os.path.join(out_dir, tile_name),
        image=arr,
        mask=mask,
        meta=tile
    )
    return tile_name


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    scenes = [os.path.join(INPUT_DIR, d) for d in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, d))]

    manifest = []
    for scene_dir in scenes:
        scene_id = os.path.basename(scene_dir)
        # Use one band for grid shape
        scene_path = os.path.join(scene_dir, f"{scene_id}_{{band}}.jp2")
        tiles = generate_tile_grid(scene_path)
        for t in tiles:
            t["scene_dir"] = scene_dir
        manifest.extend(tiles)

    # Save manifest
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    # Process in parallel
    cpu_cores = max(1, cpu_count() - 1)
    with Pool(cpu_cores) as pool:
        for tile_name in pool.imap_unordered(process_tile, [(t, t["scene_dir"]) for t in manifest]):
            print(f"Processed {tile_name}")


if __name__ == "__main__":
    main()
