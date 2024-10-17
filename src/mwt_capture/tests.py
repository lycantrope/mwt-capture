import time
from lucam import Lucam, LucamError, API


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
