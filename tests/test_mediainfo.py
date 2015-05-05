import json

from nose.tools import assert_raises
from nose.tools import assert_equals

from vcsi.vcsi import MediaInfo


FFPROBE_EXAMPLE_JSON_PATH = "tests/data/bbb_ffprobe.json"


class MediaInfoForTest(MediaInfo):

    def __init__(self, json_path):
        self.ffprobe_dict = self.mock_dict(json_path)
        self.find_video_stream()
        self.compute_display_resolution()
        self.compute_format()

    def mock_dict(self, json_path):
        with open(json_path) as f:
            return json.loads(f.read())


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
