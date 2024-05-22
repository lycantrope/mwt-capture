import argparse
import asyncio
from pathlib import Path
from datetime import datetime
import sys
import itertools
import time
from lucam import Lucam
import tifffile as tf


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
        self.outputfile = outdir.joinpath(filename)
        self.interval = interval
        self.repeat = repeat

        self.camera = camera
        self.properties = properties

    async def start(self):
        with tf.TiffWriter(
            self.outputfile,
            append=True,
        ) as tf_handler:
            for _ in range(self.repeat):
                await asyncio.sleep(self.interval)
                im = self.capture()
                if im is None:
                    print("test")
                    continue
                tf_handler.write(im, datetime=True, compression="LZW")

    def capture(self):
        if self.camera is None:
            print("test")
            return None
        # numpy array
        return self.camera.TakeSnapshot(self.properties)


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
    parser.add_argument("-i", "--interval", type=float, default=125.0)
    parser.add_argument("-t", "--time", type=float, default=600.0)
    parser.add_argument("--run-after", type=float, default=0.0)
    parser.add_argument("outdir", type=check_folder, metavar="OUTDIR")
    parser.add_argument("--suffix", type=str, default="exp")

    args = parser.parse_args()
    print(args)
    camera = None
    ## init camera
    # camera = MockCamera(r"C:\Users\kuan\Projects\mwt-capture\data")

    camera = Lucam(1)
    # snapshot.format = camera.GetFormat()[0]
    # snapshot.exposure = camera.GetProperty("exposure")[0]
    # snapshot.gain = camera.GetProperty("gain")[0]
    # snapshot.timeout = 1000.0
    # snapshot.gainRed = 1.0
    # snapshot.gainBlue = 1.0
    # snapshot.gainGrn1 = 1.0
    # snapshot.gainGrn2 = 1.0
    # snapshot.useStrobe = False
    # snapshot.strobeDelay = 0.0
    # snapshot.useHwTrigger = 0
    # snapshot.shutterType = 0
    # snapshot.exposureDelay = 0.0
    # snapshot.bufferlastframe = 0
    # camera.snapshot =

    properties = camera.default_snapshot()
    # properties = None
    loop = asyncio.get_event_loop()
    capture = PeriodicCapturer(
        outdir=args.outdir,
        suffix=args.suffix,
        camera=camera,
        properties=properties,
        interval=0.05,
        repeat=100,
    )
    # start capture after setting.
    loop.run_until_complete(wait(args.run_after))

    # start capture
    loop.run_until_complete(capture.start())


if __name__ == "__main__":
    main()
