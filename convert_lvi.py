# %%
import os
from datetime import datetime
from pathlib import Path

import cv2
import h5py
import tifffile as tf
from tqdm import tqdm


def get_valid_filestem(src, outputdir) -> Path:
    for i in range(1000):
        dst = outputdir.joinpath(src.stem + f"_{i:0>3d}")
        if not dst.exists():
            return dst
    # if 999 is not enough, the file name will recursive generate with _999_000
    return get_valid_filestem(dst, outputdir)


def _transform(im):
    if im.ndim == 3:
        return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(im.dtype)
    return im


class LviReader:
    def __init__(self, src: Path):
        self.filename = src
        try:
            cap = cv2.VideoCapture(os.fspath(src))
            if cap is None or not cap.isOpened():
                print(f"OpenCV cannot open {src}.")
                raise FileNotFoundError(f"OpenCV cannot open {src}.")

            # Retrieve basic information
            self.fps = cap.get(cv2.CAP_PROP_FPS)
            self.nframe = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Retrieve first frame to get image information
            if not cap.grab():
                raise IOError(f"OpenCV cannot read {src}.")
            _, im = cap.retrieve()

            self.dtype = im.dtype
            self.height, self.width = im.shape[:2]
            self.shape = im.shape[:2]
        finally:
            if cap is not None:
                cap.release()

    def parse(self):
        try:
            cap = cv2.VideoCapture(os.fspath(self.filename))
            while cap.grab():
                _, im = cap.retrieve()
                yield _transform(im)
        finally:
            cap.release()


def convert_lvi_to_tif(src: Path, outputdir: Path, verbose=True) -> Path:
    reader = LviReader(src)

    dst = get_valid_filestem(src, outputdir).with_suffix(".tif")
    dtype = reader.dtype
    fps = reader.fps
    shape = (reader.nframe, reader.height, reader.width)
    # Use tqdm to show progress
    generator = reader.parse()
    if verbose:
        generator = tqdm(generator, total=reader.nframe, desc=reader.filename.name)

    tf.imwrite(
        dst,
        generator,
        imagej=True,
        metadata={
            "axes": "TYX",
            "fps": fps,
        },
        shape=shape,
        dtype=dtype,
    )

    return dst


def convert_lvi_to_hdf(
    src: Path,
    outputdir: Path,
    verbose=True,
    compression="gzip",
) -> Path:
    reader = LviReader(src)

    dst = get_valid_filestem(src, outputdir).with_suffix(".h5")
    dtype = reader.dtype
    fps = reader.fps
    shape = (reader.nframe, reader.height, reader.width)
    # Use tqdm to show progress
    generator = reader.parse()
    if verbose:
        generator = tqdm(generator, total=reader.nframe, desc=reader.filename.name)

    with h5py.File(dst, "w", libver="latest") as f:
        dset = f.create_dataset(
            "data",
            shape=shape,
            dtype=dtype,
            chunks=(1, reader.height, reader.width),
            compression=compression,
        )
        dset.attrs["create_at"] = str(datetime.now())
        dset.attrs["fps"] = fps
        dset.attrs["nframe"] = reader.nframe
        dset.attrs["height"] = reader.height
        dset.attrs["width"] = reader.width
        dset.attrs["axes"] = "TYX"

        for i, im in enumerate(generator):
            dset[i] = im
    return dst


# %%
if __name__ == "__main__":

    import shutil

    home = Path.home().joinpath("Desktop", "250529_MWT")
    # fs_home = Path("//FS/kuan/mwt/250529_MWT")
    for p in home.glob("**/*.h5"):
        if not p.is_file():
            continue
        print(p)
        # pp = fs_home.joinpath(p.relative_to(home))
        # print(pp)
        # if pp.is_file():
        #     continue
        # shutil.move(p, pp)
# %%
