import contextlib
import multiprocessing as mp
import time

import cv2


class ImageViewer(mp.Process):
    def __init__(self, stream: mp.Queue):
        super().__init__(daemon=True)
        self.stream = stream
        self.is_start = mp.Event()

    def set(self):
        self.is_start.set()

    def is_running(self) -> bool:
        return self.is_start.is_set()

    def run(self):
        self.is_start.wait()
        try:
            cv2.namedWindow("Preview", cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
            cv2.startWindowThread()
            while True:
                ret, im = self.stream.get()
                if not ret:
                    break
                cv2.imshow("Preview", im)
                ret = cv2.waitKey(125)
                if ret & 255 in (27, 81, 113):
                    break
        finally:
            cv2.waitKey(1)
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            self.is_start.clear()


class CameraStreamer(mp.Process):
    def __init__(
        self, exposure: float = 50.0, gain: float = 0.275, rotate: bool = False
    ):
        super().__init__(daemon=True)
        self.stream = mp.Queue(32)
        self.exposure = exposure
        self.gain = gain
        self.rotate = rotate
        self.is_running = mp.Event()

    def stop(self):
        self.is_running.clear()

    def get_stream(self):
        while True:
            ret, im = self.stream.get()
            if not ret:
                break
            yield im

    def is_empty(self):
        return self.stream.empty()

    def run(self):
        self.is_running.set()

        def init_camera(exposure, gain):
            # This is dummy for testing
            return None, None

        camera, _ = init_camera(self.exposure, self.gain)
        try:
            camera.StreamVideoControl("start_streaming")
            while self.is_running.is_set():
                buf = camera.TakeVideo(8)
                for im in buf:
                    if self.rotate:
                        im = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
                    self.stream.put((True, im))
        finally:
            self.stream.put((False, None))
            # camera.RemoveStreamingCallback(callbackid)
            camera.StreamVideoControl("stop_streaming")
            print("Stop")


@contextlib.contextmanager
def timer(msg="func"):
    t0 = time.monotonic()
    try:
        yield
    finally:
        print(f"{msg}|{time.monotonic() - t0:.3f} (sec)")


import argparse
import asyncio
import contextlib
import itertools
import multiprocessing as mp
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import tifffile as tf
import win32gui
from lucam import API, Lucam
from lucam.lucam import LucamError
from tqdm import tqdm


@contextlib.contextmanager
def timer(msg="func"):
    t0 = time.monotonic()
    try:
        yield
    finally:
        print(f"{msg}|{time.monotonic() - t0:.3f} (sec)")


def _transform(im):
    if im.ndim == 3:
        return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(im.dtype)
    return im


class FileWriter(mp.Process):
    def __init__(self, outputfile: Path, nframe: int, interval: float, pipe):
        super().__init__(daemon=True)
        self.outputfile = outputfile
        self.nframe = nframe
        self.interval = interval
        self.pipe = pipe

    def run(self):
        try:
            # get first image to obtain image shape and dtype
            im = self.pipe.recv()
            if im is None:
                return

            dtype = im.dtype
            height, width = im.shape[:2]

            def generator(im):
                yield _transform(im)

                while True:
                    im = self.pipe.recv()
                    if im is None:
                        return
                    yield _transform(im)

            tf.imwrite(
                self.outputfile,
                generator(im),
                imagej=True,
                dtype=dtype,
                shape=(self.nframe, height, width),
                metadata={"axes": "TYX", "fps": 1.0 / self.interval},
            )
        except EOFError:  # Catch EOFError if the connection was closed.
            ...


class FileWriter(mp.Process):
    def __init__(self, outputfile: Path, nframe: int, interval: float, pipe):
        super().__init__(daemon=True)
        self.outputfile = outputfile
        self.nframe = nframe
        self.interval = interval
        self.pipe = pipe

    def run(self):
        try:
            # get first image to obtain image shape and dtype
            im = self.pipe.recv()
            if im is None:
                return

            dtype = im.dtype
            height, width = im.shape[:2]

            def generator(im):
                yield _transform(im)

                while True:
                    im = self.pipe.recv()
                    if im is None:
                        return
                    yield _transform(im)

            tf.imwrite(
                self.outputfile,
                generator(im),
                imagej=True,
                dtype=dtype,
                shape=(self.nframe, height, width),
                metadata={"axes": "TYX", "fps": 1.0 / self.interval},
            )
        except EOFError:  # Catch EOFError if the connection was closed.
            ...


class CameraStreamer(mp.Process):
    def __init__(
        self, exposure: float = 50.0, gain: float = 0.275, rotate: bool = False
    ):
        super().__init__(daemon=True)
        self.stream = mp.Queue(32)
        self.exposure = exposure
        self.gain = gain
        self.rotate = rotate
        self.is_running = mp.Event()

    def stop(self):
        self.is_running.clear()

    def get_stream(self):
        while True:
            ret, im = self.stream.get()
            if not ret:
                break
            yield im

    def is_empty(self):
        return self.stream.empty()

    def run(self):
        self.is_running.set()
        camera, _ = init_camera(self.exposure, self.gain)
        try:
            camera.StreamVideoControl("start_streaming")
            while self.is_running.is_set():
                buf = camera.TakeVideo(8)
                for im in buf:
                    if self.rotate:
                        im = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
                    self.stream.put((True, im))
        finally:
            self.stream.put((False, None))
            # camera.RemoveStreamingCallback(callbackid)
            camera.StreamVideoControl("stop_streaming")
            print("Stop")


class ImageViewer(mp.Process):
    def __init__(self, stream: mp.Queue):
        super().__init__(daemon=True)
        self.stream = stream
        self.is_start = mp.Event()

    def set(self):
        self.is_start.set()

    def is_running(self) -> bool:
        return self.is_start.is_set()

    def run(self):
        self.is_start.wait()
        try:
            cv2.namedWindow("Preview", cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
            cv2.startWindowThread()
            while True:
                ret, im = self.stream.get()
                if not ret:
                    break
                cv2.imshow("Preview", im)
                ret = cv2.waitKey(125)
                if ret & 255 in (27, 81, 113):
                    break
        finally:
            cv2.waitKey(1)
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            self.is_start.clear()


class FileWriter(mp.Process):
    def __init__(self, outputfile: Path, queue: mp.Queue):
        super().__init__(daemon=True)
        self.outputfile = outputfile
        self.queue = queue
        self.is_start = mp.Event()

    def set(self):
        self.is_start.set()

    def run(self):
        while not self.is_start.is_set():
            time.sleep(0.1)

        with tf.TiffWriter(
            self.outputfile,
            bigtiff=True,
        ) as tf_handler:
            ret, im = self.queue.get()
            im_min = im
            im_max = im
            while ret:
                if im.ndim == 3:
                    # convert rgb to grayscale
                    im = np.dot(
                        im[..., :3].astype("f8"), [0.2989, 0.5870, 0.1140]
                    ).astype("u1")
                tf_handler.write(im, datetime=True, compression="Deflate")

                im_min = np.min([im, im_min], axis=0)
                im_max = np.max([im, im_max], axis=0)
                ret, im = self.queue.get()
            if im_min is not None:
                tf.imwrite(
                    self.outputfile.with_stem(self.outputfile.stem + "_min_proj"),
                    im_min,
                )
                tf.imwrite(
                    self.outputfile.with_stem(self.outputfile.stem + "_max_proj"),
                    im_max,
                )


async def async_idling(second: float):
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


def idling(second: float):
    t0 = time.monotonic_ns()
    for b in itertools.cycle("|/-\\"):
        dt = time.monotonic_ns() - t0
        if dt >= second * 1e9:
            break
        msg = f"Start Capture after: {(second - dt/1e9):.2f}s {b}"
        sys.stdout.write(msg)
        sys.stdout.flush()
        time.sleep(0.05)
        sys.stdout.write("\033[2K\033[1G")
    print("Start Capture after: 0.00s")


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


async def async_idling(second: float):
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


@dataclass(slots=True)
class Args:
    interval: float
    time: float
    run_after: float
    outdir: Path
    suffix: str
    exposure: float
    gain: float

    def to_text(self):
        return "\n".join(f"{slot}: {getattr(self, slot)}" for slot in self.__slots__)


def init_camera(exposure: float, gain: float, *, interval=None):
    ## init camera
    camera = Lucam(1)
    properties = camera.default_snapshot()
    # the parameter width 2592, height 1944, exposure 70.131, gain 0.375 from PC
    # properties.ex
    # millisecond
    properties.exposure = exposure
    properties.gain = gain
    properties.shutterType = API.LUCAM_SHUTTER_TYPE_ROLLING
    pix_fmt = Lucam.FrameFormat(
        0,
        0,
        2592,
        1944,
        API.LUCAM_PF_8,
        binningX=1,
        flagsX=1,
        binningY=1,
        flagsY=1,
    )
    properties.timeout = max(properties.timeout, exposure * 2.0)
    framerate = 8.0
    if interval is not None and interval > 0.0:
        framerate = 1.0 / interval if interval is not None and interval > 0.0 else 8.0
        properties.timeout = max(properties.timeout, (interval * 1e3 + exposure) * 2.0)

    camera.SetFormat(
        pix_fmt,
        framerate=framerate,
    )

    camera.set_properties(exposure=exposure, gain=gain)

    return camera, properties


# @timer("preview")
# def preview(args):
#     camera, _ = init_camera(args.exposure, args.gain)
#     queue = mp.Queue(32)
#     viewer = ImageViewer(queue)
#     viewer.start()
#     print("Start Preview: Ctrl+C or [q] to exit")
#     try:
#         camera.StreamVideoControl("start_streaming")
#         while True:
#             buf = camera.TakeVideo(7)
#             for im in buf:
#                 queue.put((True, im))
#     except KeyboardInterrupt:
#         pass

#     finally:
#         queue.put((False, None))
#         # camera.RemoveStreamingCallback(callbackid)
#         camera.StreamVideoControl("stop_streaming")
#         print("Stop")


def preview(args):
    camera, properties = init_camera(args.exposure, args.gain)

    try:
        winname = b"Preview"
        camera.set_properties(exposure=args.exposure, gain=args.gain)
        camera.CreateDisplayWindow(winname, width=2592 // 4, height=1964 // 4)
        camera.StreamVideoControl("start_display")
        windows = []

        def callback(hwnd, custom_list):
            custom_list.append((hwnd, win32gui.GetWindowText(hwnd)))

        win32gui.EnumWindows(callback, windows)

        hwnd = max(hwnd for hwnd, name in windows if name == winname.decode())
        while True:
            # camera.AdjustDisplayWindow(x=0, y=0, width=2592//4, height=1964//4)
            msg = f"fps: {camera.QueryDisplayFrameRate():.2f}"
            sys.stdout.write(msg)
            sys.stdout.flush()
            if cv2.waitKey(20) in (27, 81, 113):
                break
            if win32gui.IsWindow(hwnd):
                x1, y1, x2, y2 = win32gui.GetWindowRect(hwnd)
                w = x2 - x1
                # bar offset
                h = y2 - y1 - 39
                camera.AdjustDisplayWindow(x=0, y=0, width=w, height=h)
            else:
                break
            sys.stdout.write("\033[2K\033[1G")
    except KeyboardInterrupt:
        pass
    finally:
        camera.StreamVideoControl("stop_streaming")
        camera.DestroyDisplayWindow()


def capture(args):
    args = {k: v for k, v in vars(args).items() if k in Args.__slots__}
    args = Args(**args)
    camera, properties = init_camera(args.exposure, args.gain, interval=args.interval)
    #  timeout (ms) = interval (sec) * 2000.0
    print(properties)

    # start capture after waiting.
    if args.run_after > 0:
        idling(args.run_after)

    # # start capture
    filename = datetime.now().strftime(f"%Y%m%d_%H%M%S_{args.suffix}.tif")

    queue = mp.Queue(64)

    outputfile = args.outdir.joinpath(filename)
    param_names = outputfile.stem + "_params.txt"

    with outputfile.with_name(param_names).open("w") as f:
        print("### CMD INPUT ARGUMENTS", file=f)
        print(args.to_text(), file=f)
        print("### CAMERA PROPERTIES", file=f)
        print(properties, file=f)

    writer = FileWriter(outputfile, queue)
    writer.start()

    duration_ns = args.time * 1e9
    framerate = round(1.0 / args.interval)
    if framerate > 8:
        print("This camera did not support framerate > 8 (interval <= 0.125)")
        framerate = min(8, framerate)
    if framerate > 3:
        try:
            camera.StreamVideoControl("start_streaming")
            t0 = time.monotonic_ns()
            # begin
            writer.set()
            dt = 0.0
            while dt < duration_ns:
                msg = f"Elapse Time: {dt*1e-9:.3f} (sec)"
                sys.stdout.write(msg)
                sys.stdout.flush()
                tmp = time.monotonic()
                buf = camera.TakeVideo(framerate)
                for im in buf:
                    queue.put((True, im))
                # idling if the TakeVideo is faster than interval
                while time.monotonic() - tmp < args.interval:
                    time.sleep(0.01)

                sys.stdout.write("\033[2K\033[1G")
                dt = time.monotonic_ns() - t0
        finally:
            queue.put((False, None))
            # camera.RemoveStreamingCallback(callbackid)
            camera.StreamVideoControl("stop_streaming")
    else:
        try:
            total = round(args.time / args.interval)
            properties.shutterType = API.LUCAM_SHUTTER_TYPE_GLOBAL
            camera.EnableFastFrames(properties)
            t0 = time.monotonic_ns()
            # begin
            writer.set()
            dt = 0.0
            winname = "Capture"
            cv2.namedWindow(winname, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.startWindowThread()
            while dt < duration_ns:
                msg = f"Elapse Time: {dt*1e-9:.3f} (sec)"
                sys.stdout.write(msg)
                sys.stdout.flush()
                tmp = time.monotonic()
                buf = camera.TakeFastFrame()
                queue.put((True, buf))
                # idling if the TakeVideo is faster than interval
                _h, _w = buf.shape[:2]
                cv2.putText(
                    buf,
                )
                cv2.imshow(winname, cv2.resize(buf, (_w // 2, _h // 2)))
                cv2.resizeWindow(winname, _w // 2, _h // 2)
                while time.monotonic() - tmp < args.interval:
                    cv2.waitKey(5)

                sys.stdout.write("\033[2K\033[1G")
                dt = time.monotonic_ns() - t0
        finally:
            queue.put((False, None))
            # camera.RemoveStreamingCallback(callbackid)
            camera.DisableFastFrames()
            cv2.waitKey(1)
            cv2.destroyAllWindows()
            cv2.waitKey(1)

    writer.join()
    dt = time.monotonic_ns() - t0
    msg = f"Elapse Time: {dt*1e-9:.3f} (sec)"
    print(msg)
    print(f"TIFF file was save at: {outputfile}")


def main():
    parser = argparse.ArgumentParser(
        prog="mwt",
        description="A command line tool for multiple worm imaging",
    )

    # capture parser
    subparsers = parser.add_subparsers(title="COMMAND")

    cap_parser = subparsers.add_parser(
        "capture",
        aliases=["cap", "c"],
        description="capture images from camera",
        help="see `mwt capture -h, --help`",
    )

    cap_parser.add_argument(
        "-i",
        "--interval",
        type=float,
        default=0.125,
        help="Time interval between each snapshot (sec, float)",
    )
    cap_parser.add_argument(
        "-t",
        "--time",
        type=float,
        required=True,
        help="Total acquisition time (sec, float)",
    )
    cap_parser.add_argument(
        "--run-after",
        type=float,
        default=0.0,
        help="Idling time before acquisition(sec, float)",
    )
    cap_parser.add_argument(
        "outdir",
        type=check_folder,
        metavar="OUTDIR",
        default=None,
        help="The output directory for saving multi-stack tiff",
    )
    cap_parser.add_argument(
        "--suffix", type=str, default="exp", help="filename suffix (default: exp)"
    )

    cap_parser.add_argument(
        "-e",
        "--exposure",
        type=float,
        default=50.0,
        help="exposure time (ms)",
    )

    cap_parser.add_argument(
        "-g",
        "--gain",
        type=float,
        default=0.375,
        help="Camera Gain",
    )

    cap_parser.set_defaults(handler=capture)

    preview_parser = subparsers.add_parser(
        "preview",
        aliases=["pv"],
        description="preview imaging stream",
        help="see `mwt preview -h, --help`",
    )

    preview_parser.add_argument(
        "-e",
        "--exposure",
        type=float,
        default=50.0,
        help="exposure time (ms)",
    )

    preview_parser.add_argument(
        "-g",
        "--gain",
        type=float,
        default=0.375,
        help="Camera Gain",
    )

    preview_parser.add_argument(
        "-r",
        "--rotate",
        action="store_true",
        help="Image was rotated counterclockwise 90 degree",
    )
    preview_parser.set_defaults(handler=preview)

    subparsers.add_parser(
        "help",
        aliases=["h"],
        description="help",
        help="show this help message and exit",
    ).set_defaults(handler=lambda _: parser.print_help())

    args = parser.parse_args()
    if not hasattr(args, "handler"):
        parser.print_help()
        return

    args.handler(args)


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
                _ = camera.TakeVideo(nframe)  # take a 8 frames video
                print(
                    f"rate:{rate:f}, nframe:{nframe:d}, t: {(time.monotonic_ns() - t0)*1e-9:.3f} s"
                )

        finally:
            camera.StreamVideoControl("stop_streaming")


# @timer("preview")
# def preview(args):
#     camera, _ = init_camera(args.exposure, args.gain)
#     queue = mp.Queue(32)
#     viewer = ImageViewer(queue)
#     viewer.start()
#     print("Start Preview: Ctrl+C or [q] to exit")
#     try:
#         camera.StreamVideoControl("start_streaming")
#         while True:
#             buf = camera.TakeVideo(7)
#             for im in buf:
#                 queue.put((True, im))
#     except KeyboardInterrupt:
#         pass

#     finally:
#         queue.put((False, None))
#         # camera.RemoveStreamingCallback(callbackid)
#         camera.StreamVideoControl("stop_streaming")
#         print("Stop")

if __name__ == "__main__":
    main()
