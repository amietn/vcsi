import json
import argparse
from argparse import ArgumentTypeError

from nose.tools import assert_raises
from nose.tools import assert_equals

from vcsi.vcsi import MediaInfo
from vcsi.vcsi import Grid, grid_desired_size
from vcsi.vcsi import timestamp_generator


FFPROBE_EXAMPLE_JSON_PATH = "tests/data/bbb_ffprobe.json"


class MediaInfoForTest(MediaInfo):

    def __init__(self, json_path):
        super(MediaInfoForTest, self).__init__(json_path)

    def probe_media(self, path):
        with open(path) as f:
            self.ffprobe_dict = json.loads(f.read())


def test_compute_display_resolution():
    mi = MediaInfoForTest(FFPROBE_EXAMPLE_JSON_PATH)
    assert_equals(mi.display_width, 1920)
    assert_equals(mi.display_height, 1080)


def test_filename():
    mi = MediaInfoForTest(FFPROBE_EXAMPLE_JSON_PATH)
    assert_equals(mi.filename, "bbb_sunflower_1080p_60fps_normal.mp4")


def test_duration():
    mi = MediaInfoForTest(FFPROBE_EXAMPLE_JSON_PATH)
    assert_equals(mi.duration_seconds, 634.533333)


def test_pretty_duration():
    mi = MediaInfoForTest(FFPROBE_EXAMPLE_JSON_PATH)
    assert_equals(mi.duration, "10:34")


def test_size_bytes():
    mi = MediaInfoForTest(FFPROBE_EXAMPLE_JSON_PATH)
    assert_equals(mi.size_bytes, 355856562)


def test_size():
    mi = MediaInfoForTest(FFPROBE_EXAMPLE_JSON_PATH)
    assert_equals(mi.size, "339.4 MiB")


def test_template_attributes():
    mi = MediaInfoForTest(FFPROBE_EXAMPLE_JSON_PATH)
    attributes = mi.template_attributes()
    assert_equals(attributes["audio_codec"], "mp3")
    assert_equals(attributes["video_codec"], "h264")


def test_grid_desired_size():
    mi = MediaInfoForTest(FFPROBE_EXAMPLE_JSON_PATH)
    x = 2
    y = 3
    grid = Grid(x, y)
    width = 800
    hmargin = 20
    s = grid_desired_size(grid, mi, width=width, horizontal_margin=hmargin)
    expected_width = (width - (x-1) * hmargin) / x

    assert_equals(s[0], expected_width)


def test_desired_size():
    mi = MediaInfoForTest(FFPROBE_EXAMPLE_JSON_PATH)
    s = mi.desired_size(width=1280)
    assert_equals(s[1], 720)


def test_timestamps():
    mi = MediaInfoForTest(FFPROBE_EXAMPLE_JSON_PATH)
    mi.duration_seconds = 100
    start_delay_percent = 7
    end_delay_percent = 7
    interval = mi.duration_seconds - (start_delay_percent + end_delay_percent)
    num_samples = interval - 1

    args = argparse.Namespace()
    args.interval = None
    args.num_samples = num_samples
    args.start_delay_percent = start_delay_percent
    args.end_delay_percent = end_delay_percent

    expected_timestamp = start_delay_percent + 1
    for t in timestamp_generator(mi, args):
        assert_equals(int(t[0]), expected_timestamp)
        expected_timestamp += 1


def test_pretty_duration_centis_limit():
    mi = MediaInfoForTest(FFPROBE_EXAMPLE_JSON_PATH)
    mi.duration_seconds = 1.9999
    pretty_duration = MediaInfo.pretty_duration(mi.duration_seconds, show_centis=True)
    assert_equals(pretty_duration, "00:01.99")


def test_pretty_duration_millis_limit():
    mi = MediaInfoForTest(FFPROBE_EXAMPLE_JSON_PATH)
    mi.duration_seconds = 1.9999
    pretty_duration = MediaInfo.pretty_duration(mi.duration_seconds, show_millis=True)
    assert_equals(pretty_duration, "00:01.999")


def test_pretty_to_seconds():
    assert_equals(MediaInfo.pretty_to_seconds("1:11:11.111"), 4271.111)

    assert_equals(MediaInfo.pretty_to_seconds("1:11:11"), 4271)

    assert_equals(MediaInfo.pretty_to_seconds("1:01:00"), 3660)

    assert_raises(ArgumentTypeError, MediaInfo.pretty_to_seconds, "1:01:01:01:00")
