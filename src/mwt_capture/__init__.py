import argparse
import asyncio
from collections import deque
import ctypes
import io
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
from lucam.lucam import LucamError


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


class VideoStreaming(mp.Process):
    def __init__(self, camera: Lucam, duration: float):
        super().__init__(daemon=True)
        self.camera = camera
        self.queue = mp.Queue(128)
        self.duration = duration  # second
        self.is_start = mp.Event()

    def get_stream(self):
        if not self.is_start.is_set():
            raise RuntimeError(
                "This thread was not started. Call `start()` before `get_stream()`"
            )
        while True:
            ret, stack = self.queue.get(True)
            if not ret:
                break
            for im in stack:
                yield im

    def run(self) -> None:
        self.is_start.set()
        duration_ns = self.duration * 1e9
        try:
            t0 = time.monotonic_ns()
            while (time.monotonic_ns() - t0) < duration_ns:
                self.camera.StreamVideoControl("start_streaming")
                buf = self.camera.TakeVideo(7)  # take 7 frames per second
                self.queue.put((True, buf))
        finally:
            self.queue.put((False, None))
            self.is_start.clear()
            self.camera.StreamVideoControl("stop_streaming")


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

    parser.add_argument(
        "--exposure",
        type=float,
        default=50.0,
        help="exposure time (ms)",
    )

    parser.add_argument(
        "--gain",
        type=float,
        default=0.0,
        help="Camera Gain",
    )

    args = parser.parse_args()

    ## init camera
    # camera = MockCamera(r"C:\Users\kuan\Projects\mwt-capture\data")
    try:
        camera = Lucam(1)
    except LucamError as e:
        print(e)
        return

    repeat = round(args.time / args.interval)
    properties = camera.default_snapshot()
    # the parameter width 2592, height 1944, exposure 70.131, gain 0.375 from PC
    # properties.ex
    properties.exposure = args.exposure
    properties.gain = args.gain
    properties.shutterType = API.LUCAM_SHUTTER_TYPE_ROLLING
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

    # capturer = PeriodicCapturer(
    #     outdir=args.outdir,
    #     suffix=args.suffix,
    #     camera=camera,
    #     properties=properties,
    #     interval=args.interval,
    #     repeat=repeat,
    # )

    loop = asyncio.get_event_loop()
    # start capture after waiting.
    if args.run_after > 0:
        loop.run_until_complete(wait(args.run_after))

    # # start capture
    # loop.run_until_complete(capturer.start())
    stream = VideoStreaming(
        camera,
        duration=args.duration,
    )
    stream.start()

    filename = datetime.now().strftime(f"%Y%m%d_%H%M%S_{args.suffix}.tif")
    outputfile = args.outdir.joinpath(filename)

    with tf.TiffWriter(
        outputfile,
        append=True,
    ) as tf_handler:
        for im in stream.get_stream():
            if im.ndim == 3:
                # convert rgb to grayscale
                im = np.dot(im[..., :3].astype("f8"), [0.2989, 0.5870, 0.1140]).astype(
                    "u1"
                )
            tf_handler.write(im, datetime=True, compression="LZW")


def check_burst():
    try:
        camera = Lucam(1)
    except LucamError as e:
        print(e)
        return

    API.LUCAM_PROP_SNAPSHOT_COUNT = 120
    try:
        ret = camera.GetProperty(API.LUCAM_PROP_SNAP_COUNT)
        print(ret)
    except Exception as e:
        print(e)


def check_framerate():
    try:
        camera = Lucam(1)
    except LucamError as e:
        print(e)
        return

    properties = camera.default_snapshot()
    # the parameter width 2592, height 1944, exposure 70.131, gain 0.375 from PC
    # properties.ex
    properties.exposure = 25
    properties.gain = 0.2
    properties.shutterType = API.LUCAM_SHUTTER_TYPE_ROLLING
    print(properties)

    # set camera to 8 bit VGA mode at low framerate
    for rate in range(4, 17):
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
        )

        try:
            for nframe in range(4, 17):
                camera.StreamVideoControl("start_streaming")
                t0 = time.monotonic_ns()
                video = camera.TakeVideo(nframe)  # take a 8 frames video
                print(
                    f"rate:{rate:f}, nframe:{nframe:d}, t: {(time.monotonic_ns() - t0)*1e-9:.3f} s"
                )

        finally:
            camera.StreamVideoControl("stop_streaming")


if __name__ == "__main__":
    main()
