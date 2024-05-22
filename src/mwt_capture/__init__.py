import argparse
import asyncio
from pathlib import Path
from datetime import datetime
import sys
import itertools
import time
import tifffile as tf
from tqdm import tqdm
import numpy as np
import multiprocessing as mp

from lucam import Lucam, API


class FileWriter(mp.Process):
    def __init__(self, outputfile: Path, queue: mp.Queue):
        super().__init__(daemon=True)
        self.outputfile = outputfile
        self.queue = queue

    def run(self):
        with tf.TiffWriter(
            self.outputfile,
            append=True,
        ) as tf_handler:
            while True:
                ret, im = self.queue.get(True)
                if not ret:
                    break
                if im.ndim == 3:
                    # convert rgb to grayscale
                    im = np.dot(
                        im[..., :3].astype("f8"), [0.2989, 0.5870, 0.1140]
                    ).astype("u1")
                tf_handler.write(im, datetime=True, compression="LZW")


async def wait(second: float):
    t0 = time.monotonic_ns()
    for b in itertools.cycle("|/-\\"):
        dt = time.monotonic_ns() - t0
        if dt >= second * 1e9:
            break
        msg = f"Start Capture after: {(second - dt/1e9):.2f}s {b}"
        sys.stdout.write(msg)
        sys.stdout.flush()
        await asyncio.sleep(0.05)
        sys.stdout.write("\033[2K\033[1G")


class PeriodicCapturer:
    def __init__(
        self,
        *,
        outdir: Path,
        suffix: str,
        interval: float,
        repeat: int,
        camera: Lucam,
        properties: Lucam.Snapshot,
    ):
        filename = datetime.now().strftime(f"%Y%m%d_%H%M%S_{suffix}.tif")
        self.interval = interval
        self.repeat = repeat
        self.queue = mp.Queue(maxsize=128)
        self.file_writer = FileWriter(outdir.joinpath(filename), self.queue)
        self.camera = camera
        self.properties = properties

    async def start(self):
        try:
            self.camera.EnableFastFrames(self.properties)
            self.file_writer.start()
            t0 = time.monotonic_ns()
            for i in tqdm(range(self.repeat), desc="Acqusition"):
                while (time.monotonic_ns() - t0) < (self.interval * 1e9) * i:
                    await asyncio.sleep(self.interval / 50)
                im = self.capture()
                self.queue.put_nowait((True, im))
        finally:
            self.camera.DisableFastFrames()
            self.queue.put_nowait((False, None))
            self.file_writer.join()

    def capture(self):
        if self.camera is None:
            print("test")
            return None
        # numpy array
        return self.camera.TakeFastFrame()


def check_folder(path: str):
    try:
        p = Path(path)
        p.mkdir(exist_ok=True)
    except FileExistsError:
        raise NotADirectoryError(f"{p}")
    return p


class MockCamera:
    def __init__(self, data: Path):
        self.data = Path(data).glob("*.tif")

    def TakeSnapshot(self, *args, **kwargs):
        im = None
        try:
            im_p = next(self.data)
            im = tf.imread(im_p)
        except StopIteration:
            pass
        return im


def main():
    parser = argparse.ArgumentParser("mwt", description="")
    parser.add_argument(
        "-i",
        "--interval",
        type=float,
        default=0.125,
        help="Time interval between each snapshot (sec, float)",
    )
    parser.add_argument(
        "-t",
        "--time",
        type=float,
        default=600.0,
        help="Total acquisition time (sec, float)",
    )
    parser.add_argument(
        "--run-after",
        type=float,
        default=0.0,
        help="Idling time before acquisition(sec, float)",
    )
    parser.add_argument(
        "outdir",
        type=check_folder,
        metavar="OUTDIR",
        help="The output directory for saving multi-stack tiff",
    )
    parser.add_argument(
        "--suffix", type=str, default="exp", help="filename suffix (default: exp)"
    )

    args = parser.parse_args()

    ## init camera
    # camera = MockCamera(r"C:\Users\kuan\Projects\mwt-capture\data")
    camera = Lucam(1) or None

    if camera is None:
        raise IOError("Fail to connect to camera")

    repeat = round(args.time / args.interval)
    properties = camera.default_snapshot()
    # the parameter width 2592, height 1944, exposure 70.131, gain 0.375 from PC
    # properties.ex
    properties.exposure = 70.131
    properties.gain = 0.375
    print(properties)

    # set camera to 8 bit VGA mode at low framerate
    camera.SetFormat(
        Lucam.FrameFormat(
            0,
            0,
            2592,
            1944,
            API.LUCAM_PF_8,
            binningX=1,
            flagsX=1,
            binningY=1,
            flagsY=1,
        ),
        framerate=1.0 / args.interval,
    )

    capturer = PeriodicCapturer(
        outdir=args.outdir,
        suffix=args.suffix,
        camera=camera,
        properties=properties,
        interval=args.interval,
        repeat=repeat,
    )

    loop = asyncio.get_event_loop()
    # start capture after waiting.
    if args.run_after > 0:
        loop.run_until_complete(wait(args.run_after))

    # start capture
    loop.run_until_complete(capturer.start())


if __name__ == "__main__":
    main()
