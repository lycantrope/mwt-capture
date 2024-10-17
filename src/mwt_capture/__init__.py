import argparse
import itertools
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import tifffile as tf
from lucam import API, Lucam
from tqdm import tqdm
import win32gui
import win32api
import win32con


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
            if im.ndim == 3:
                im = np.dot(im[..., :3].astype("f8"), (0.2989, 0.5870, 0.1140)).astype(
                    dtype
                )

            def generator(im):
                if im.ndim == 2:
                    yield np.dot(
                        im[..., :3].astype("f8"), (0.2989, 0.5870, 0.1140)
                    ).astype(dtype)
                    while True:
                        im = self.pipe.recv()
                        if im is None:
                            return
                        yield np.dot(
                            im[..., :3].astype("f8"), (0.2989, 0.5870, 0.1140)
                        ).astype(dtype)
                else:
                    yield im
                    while True:
                        im = self.pipe.recv()
                        if im is None:
                            return
                        yield im

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


def check_folder(path: str):
    try:
        p = Path(path)
        p.mkdir(exist_ok=True)
    except FileExistsError:
        raise NotADirectoryError(f"{p}")
    return p


def check_file(path: str):
    p = Path(path)
    if not p.is_file():
        raise IOError(f"{path}: is not a file.")
    return p


@dataclass(slots=True)
class Args:
    interval: float
    nframe: float
    run_after: float
    outdir: Path
    suffix: str
    exposure: float
    gain: float

    def to_text(self):
        return "\n".join(f"{slot},{getattr(self, slot)}" for slot in self.__slots__)


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

    # set exposure and gain
    camera.set_properties(exposure=exposure, gain=gain)

    return camera, properties


def preview(args):
    camera, properties = init_camera(args.exposure, args.gain)

    try:
        winname = b"Preview"
        camera.CreateDisplayWindow(winname, width=2572 // 4, height=1964 // 4)
        camera.StreamVideoControl("start_display")
        windows = []

        def callback(hwnd, custom_list):
            custom_list.append((hwnd, win32gui.GetWindowText(hwnd)))

        win32gui.EnumWindows(callback, windows)
        stop = mp.Event()

        def keyboard_callback(hwnd, msg, wparam, lparam):
            if msg == win32con.WM_KEYDOWN:
                vk_code = wparam
                key_name = win32api.GetKeyNameText(vk_code)
                if key_name in ("q", "Q"):
                    stop.set()
            return 1

        hwnd = max(hwnd for hwnd, name in windows if name == winname.decode())

        while not stop.is_set():
            # camera.AdjustDisplayWindow(x=0, y=0, width=2592//4, height=1964//4)
            msg = f"fps: {camera.QueryDisplayFrameRate():.2f}"
            sys.stdout.write(msg)
            sys.stdout.flush()

            if win32gui.IsWindow(hwnd):
                x1, y1, x2, y2 = win32gui.GetWindowRect(hwnd)
                camera.AdjustDisplayWindow(x=0, y=0, width=x2 - x1, height=y2 - y1)
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

    outputfile = args.outdir.joinpath(filename)
    param_names = outputfile.stem + "_params.txt"

    with outputfile.with_name(param_names).open("w") as f:
        print("### CMD INPUT ARGUMENTS", file=f)
        print(args.to_text(), file=f)
        print("### CAMERA PROPERTIES", file=f)
        print(properties, file=f)

    receiver, sender = mp.Pipe()
    writer = FileWriter(outputfile, args.nframe, args.interval, receiver)
    writer.start()

    if args.interval < 0.35:
        print("This scripts did not support interval < 0.35 sec)")
        return

    try:
        properties.shutterType = API.LUCAM_SHUTTER_TYPE_GLOBAL
        camera.EnableFastFrames(properties)
        t0 = time.monotonic_ns()
        # begin
        dt = 0.0
        for _ in range(args.nframe):
            msg = f"Elapse Time: {dt*1e-9:.3f} (sec)"
            sys.stdout.write(msg)
            sys.stdout.flush()
            tmp = time.monotonic()
            buf = camera.TakeFastFrame()
            sender.send(buf)
            # idling if the TakeVideo is faster than interval
            while time.monotonic() - tmp < args.interval:
                time.sleep(0.005)

            sys.stdout.write("\033[2K\033[1G")
            dt = time.monotonic_ns() - t0
    finally:
        sender.send(None)
        # camera.RemoveStreamingCallback(callbackid)
        writer.join()
        sender.close()
        camera.DisableFastFrames()

    dt = time.monotonic_ns() - t0
    msg = f"Elapse Time: {dt*1e-9:.3f} (sec)"
    print(msg)
    print(f"TIFF file was save at: {outputfile}")


def convert_all_videos(args):
    for src in map(Path, args.video):
        if not src.is_file():
            print(f"{src}: is not a file")
            continue
        outputdir = src.parent
        if args.output is not None:
            outputdir = Path(args.output)

        outputdir.mkdir(exist_ok=True)
        convert_lvi_to_tiff(src, outputdir)


def convert_lvi_to_tiff(src: Path, outputdir: Path):

    try:
        cap = cv2.VideoCapture(os.fspath(src))
        if cap is None or not cap.isOpened():
            print(f"OpenCV cannot open {src}.")
            return

        FPS = cap.get(cv2.CAP_PROP_FPS)
        T = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        pbar = tqdm(total=T, desc=f"{src.name}")
        # Retrieve first frame
        ret = cap.grab()
        if not ret:
            return
        _, im = cap.retrieve()

        def generator(im):
            if im.ndim == 3:
                yield cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                pbar.update(1)
                ret = cap.grab()
                while ret:
                    _, im = cap.retrieve()
                    yield cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                    pbar.update(1)
                    ret = cap.grab()
            else:
                yield im
                pbar.update(1)
                ret = cap.grab()
                while ret:
                    _, im = cap.retrieve()
                    yield im
                    pbar.update(1)
                    ret = cap.grab()
            pbar.close()

        H, W = im.shape[:2]
        dtype = im.dtype
        shape = (int(T), H, W)
        for i in range(1000):
            dst = outputdir.joinpath(src.stem + f"_{i:0>3d}.tif")
            if not dst.exists():
                break

        tf.imwrite(
            dst,
            generator(im),
            imagej=True,
            metadata={
                "axes": "TYX",
                "fps": FPS,
            },
            shape=shape,
            dtype=dtype,
        )
    finally:
        if cap is not None:
            cap.release()


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
        default=0.5,
        help="Time interval between each snapshot (sec, float)",
    )
    cap_parser.add_argument(
        "-n",
        "--nframe",
        type=int,
        required=True,
        help="Number of frame to be captured (integer)",
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
        default=125.0,
        help="exposure time (ms)",
    )

    cap_parser.add_argument(
        "-g",
        "--gain",
        type=float,
        default=1.0,
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
        default=125.0,
        help="exposure time (ms)",
    )

    preview_parser.add_argument(
        "-g",
        "--gain",
        type=float,
        default=1.0,
        help="Camera Gain",
    )

    preview_parser.add_argument(
        "-r",
        "--rotate",
        action="store_true",
        help="Image was rotated counterclockwise 90 degree",
    )
    preview_parser.set_defaults(handler=preview)

    convert_parser = subparsers.add_parser(
        "convert",
        description="Convert video to ImageJ tiff",
        help="see `mwt convert -h, --help`",
    )

    convert_parser.add_argument("video", nargs="+")
    convert_parser.add_argument("--output", "-o", default=None)

    convert_parser.set_defaults(handler=convert_all_videos)

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


if __name__ == "__main__":
    main()
