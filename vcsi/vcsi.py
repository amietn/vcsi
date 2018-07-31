#!/usr/bin/env python3

"""Create a video contact sheet.
"""

from __future__ import print_function

import os
import subprocess
import sys
import datetime

try:
    from subprocess import DEVNULL
except ImportError:
    DEVNULL = open(os.devnull, 'wb')
import argparse
import json
import math
import tempfile
import textwrap
from collections import namedtuple
from enum import Enum
from glob import glob

from PIL import Image, ImageDraw, ImageFont
import numpy
from jinja2 import Template
import texttable
import parsedatetime

__version__ = "7"
__author__ = "Nils Amiet"

class Grid(namedtuple('Grid', ['x', 'y'])):
    def __str__(self):
        return "%sx%s" % (self.x, self.y)

class Frame(namedtuple('Frame', ['filename', 'blurriness', 'timestamp', 'avg_color'])):
    pass

class Color(namedtuple('Color', ['r', 'g', 'b', 'a'])):
    def to_hex(self, component):
        h = hex(component).replace("0x", "").upper()
        return h if len(h) == 2 else "0" + h

    def __str__(self):
        return "".join([self.to_hex(x) for x in [self.r, self.g, self.b, self.a]])

TimestampPosition = Enum('TimestampPosition', "north south east west ne nw se sw center")
VALID_TIMESTAMP_POSITIONS = [x.name for x in TimestampPosition]

DEFAULT_METADATA_FONT_SIZE = 16
DEFAULT_METADATA_FONT = "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf"
DEFAULT_TIMESTAMP_FONT_SIZE = 12
DEFAULT_TIMESTAMP_FONT = "/usr/share/fonts/TTF/DejaVuSans.ttf"
DEFAULT_CONTACT_SHEET_WIDTH = 1500
DEFAULT_DELAY_PERCENT = None
DEFAULT_START_DELAY_PERCENT = 7
DEFAULT_END_DELAY_PERCENT = DEFAULT_START_DELAY_PERCENT
DEFAULT_GRID_SPACING = None
DEFAULT_GRID_HORIZONTAL_SPACING = 5
DEFAULT_GRID_VERTICAL_SPACING = DEFAULT_GRID_HORIZONTAL_SPACING
DEFAULT_METADATA_POSITION = "top"
DEFAULT_METADATA_FONT_COLOR = "ffffff"
DEFAULT_BACKGROUND_COLOR = "000000"
DEFAULT_TIMESTAMP_FONT_COLOR = "ffffff"
DEFAULT_TIMESTAMP_BACKGROUND_COLOR = "000000aa"
DEFAULT_TIMESTAMP_BORDER_COLOR = "000000"
DEFAULT_TIMESTAMP_BORDER_SIZE = 1
DEFAULT_ACCURATE_DELAY_SECONDS = 1
DEFAULT_METADATA_MARGIN = 10
DEFAULT_METADATA_HORIZONTAL_MARGIN = DEFAULT_METADATA_MARGIN
DEFAULT_METADATA_VERTICAL_MARGIN = DEFAULT_METADATA_MARGIN
DEFAULT_CAPTURE_ALPHA = 255
DEFAULT_GRID_SIZE = Grid(4, 4)
DEFAULT_TIMESTAMP_HORIZONTAL_PADDING = 3
DEFAULT_TIMESTAMP_VERTICAL_PADDING = 1
DEFAULT_TIMESTAMP_HORIZONTAL_MARGIN = 5
DEFAULT_TIMESTAMP_VERTICAL_MARGIN = 5
DEFAULT_IMAGE_QUALITY = 100
DEFAULT_IMAGE_FORMAT = "jpg"
DEFAULT_TIMESTAMP_POSITION = TimestampPosition.se
DEFAULT_FRAME_TYPE = None
DEFAULT_INTERVAL = None


class MediaInfo(object):
    """Collect information about a video file
    """

    def __init__(self, path, verbose=False):
        self.probe_media(path)
        self.find_video_stream()
        self.find_audio_stream()
        self.compute_display_resolution()
        self.compute_format()
        self.parse_attributes()

        if verbose:
            print(self.filename)
            print("%sx%s" % (self.sample_width, self.sample_height))
            print("%sx%s" % (self.display_width, self.display_height))
            print(self.duration)
            print(self.size)

    def probe_media(self, path):
        """Probe video file using ffprobe
        """
        ffprobe_command = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            path
        ]

        try:
            output = subprocess.check_output(ffprobe_command)
            self.ffprobe_dict = json.loads(output.decode("utf-8"))
        except FileNotFoundError:
            error = "Could not find 'ffprobe' executable. Please make sure ffmpeg/ffprobe is installed and is in your PATH."
            error_exit(error)

    def human_readable_size(self, num, suffix='B'):
        """Converts a number of bytes to a human readable format
        """
        for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
            if abs(num) < 1024.0:
                return "%3.1f %s%s" % (num, unit, suffix)
            num /= 1024.0
        return "%.1f %s%s" % (num, 'Yi', suffix)

    def find_video_stream(self):
        """Find the first stream which is a video stream
        """
        for stream in self.ffprobe_dict["streams"]:
            try:
                if stream["codec_type"] == "video":
                    self.video_stream = stream
                    break
            except:
                pass

    def find_audio_stream(self):
        """Find the first stream which is an audio stream
        """
        for stream in self.ffprobe_dict["streams"]:
            try:
                if stream["codec_type"] == "audio":
                    self.audio_stream = stream
                    break
            except:
                pass

    def compute_display_resolution(self):
        """Computes the display resolution.
        Some videos have a sample resolution that differs from the display resolution
        (non-square pixels), thus the proper display resolution has to be computed.
        """
        self.sample_width = int(self.video_stream["width"])
        self.sample_height = int(self.video_stream["height"])
        sample_aspect_ratio = "1:1"
        try:
            sample_aspect_ratio = self.video_stream["sample_aspect_ratio"]
        except KeyError:
            pass

        if sample_aspect_ratio == "1:1":
            self.display_width = self.sample_width
            self.display_height = self.sample_height
        else:
            sample_split = sample_aspect_ratio.split(":")
            sw = int(sample_split[0])
            sh = int(sample_split[1])

            self.display_width = int(self.sample_width * sw / sh)
            self.display_height = int(self.sample_height)

        if self.display_width == 0:
            self.display_width = self.sample_width

        if self.display_height == 0:
            self.display_height = self.sample_height

    def compute_format(self):
        """Compute duration, size and retrieve filename
        """
        format_dict = self.ffprobe_dict["format"]

        self.duration_seconds = float(format_dict["duration"])
        self.duration = MediaInfo.pretty_duration(self.duration_seconds)

        self.filename = os.path.basename(format_dict["filename"])

        self.size_bytes = int(format_dict["size"])
        self.size = self.human_readable_size(self.size_bytes)

    @staticmethod
    def pretty_to_seconds(
            pretty_duration):
        """Converts pretty printed timestamp to seconds
        """
        millis_split = pretty_duration.split(".")
        millis = 0
        if len(millis_split) == 2:
            millis = int(millis_split[1])
            left = millis_split[0]
        else:
            left = pretty_duration

        left_split = left.split(":")
        if len(left_split) < 3:
            hours = 0
            minutes = int(left_split[0])
            seconds = int(left_split[1])
        else:
            hours = int(left_split[0])
            minutes = int(left_split[1])
            seconds = int(left_split[2])

        result = (millis / 1000.0) + seconds + minutes * 60 + hours * 3600
        return result

    @staticmethod
    def pretty_duration(
            seconds,
            show_centis=False,
            show_millis=False):
        """Converts seconds to a human readable time format
        """
        hours = int(math.floor(seconds / 3600))
        remaining_seconds = seconds - 3600 * hours

        minutes = math.floor(remaining_seconds / 60)
        remaining_seconds = remaining_seconds - 60 * minutes

        duration = ""

        if hours > 0:
            duration += "%s:" % (int(hours),)

        duration += "%s:%s" % (str(int(minutes)).zfill(2), str(int(math.floor(remaining_seconds))).zfill(2))

        if show_centis or show_millis:
            coeff = 1000 if show_millis else 100
            digits = 3 if show_millis else 2
            centis = math.floor((remaining_seconds - math.floor(remaining_seconds)) * coeff)
            duration += ".%s" % (str(int(centis)).zfill(digits))

        return duration

    def desired_size(self, width=DEFAULT_CONTACT_SHEET_WIDTH):
        """Computes the height based on a given width and fixed aspect ratio.
        Returns (width, height)
        """
        ratio = width / float(self.display_width)
        desired_height = math.floor(self.display_height * ratio)
        return (int(width), int(desired_height))

    def parse_attributes(self):
        """Parse multiple media attributes
        """
        # video
        try:
            self.video_codec = self.video_stream["codec_name"]
        except KeyError:
            self.video_codec = None

        try:
            self.video_codec_long = self.video_stream["codec_long_name"]
        except KeyError:
            self.video_codec_long = None

        try:
            self.sample_aspect_ratio = self.video_stream["sample_aspect_ratio"]
        except KeyError:
            self.sample_aspect_ratio = None

        try:
            self.display_aspect_ratio = self.video_stream["display_aspect_ratio"]
        except KeyError:
            self.display_aspect_ratio = None

        try:
            self.frame_rate = self.video_stream["avg_frame_rate"]
            splits = self.frame_rate.split("/")

            if len(splits) == 2:
                self.frame_rate = int(splits[0]) / int(splits[1])
            else:
                self.frame_rate = int(self.frame_rate)

            self.frame_rate = round(self.frame_rate, 3)
        except KeyError:
            self.frame_rate = None
        except ZeroDivisionError:
            self.frame_rate = None

        # audio
        try:
            self.audio_codec = self.audio_stream["codec_name"]
        except (KeyError, AttributeError):
            self.audio_codec = None

        try:
            self.audio_codec_long = self.audio_stream["codec_long_name"]
        except (KeyError, AttributeError):
            self.audio_codec_long = None

        try:
            self.audio_sample_rate = int(self.audio_stream["sample_rate"])
        except (KeyError, AttributeError):
            self.audio_sample_rate = None

        try:
            self.audio_bit_rate = int(self.audio_stream["bit_rate"])
        except (KeyError, AttributeError):
            self.audio_bit_rate = None

    def template_attributes(self):
        """Returns the template attributes and values ready for use in the metadata header
        """
        return dict((x["name"], getattr(self, x["name"])) for x in MediaInfo.list_template_attributes())

    @staticmethod
    def list_template_attributes():
        """Returns a list a of all supported template attributes with their description and example
        """
        table = []
        table.append({"name": "size", "description": "File size (pretty format)", "example": "128.3 MiB"})
        table.append({"name": "size_bytes", "description": "File size (bytes)", "example": "4662788373"})
        table.append({"name": "filename", "description": "File name", "example": "video.mkv"})
        table.append({"name": "duration", "description": "Duration (pretty format)", "example": "03:07"})
        table.append({"name": "sample_width", "description": "Sample width (pixels)", "example": "1920"})
        table.append({"name": "sample_height", "description": "Sample height (pixels)", "example": "1080"})
        table.append({"name": "display_width", "description": "Display width (pixels)", "example": "1920"})
        table.append({"name": "display_height", "description": "Display height (pixels)", "example": "1080"})
        table.append({"name": "video_codec", "description": "Video codec", "example": "h264"})
        table.append({"name": "video_codec_long", "description": "Video codec (long name)",
                      "example": "H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10"})
        table.append({"name": "display_aspect_ratio", "description": "Display aspect ratio", "example": "16:9"})
        table.append({"name": "sample_aspect_ratio", "description": "Sample aspect ratio", "example": "1:1"})
        table.append({"name": "audio_codec", "description": "Audio codec", "example": "aac"})
        table.append({"name": "audio_codec_long", "description": "Audio codec (long name)",
                      "example": "AAC (Advanced Audio Coding)"})
        table.append({"name": "audio_sample_rate", "description": "Audio sample rate (Hz)", "example": "44100"})
        table.append({"name": "audio_bit_rate", "description": "Audio bit rate (bits/s)", "example": "192000"})
        table.append({"name": "frame_rate", "description": "Frame rate (frames/s)", "example": "23.974"})
        return table


class MediaCapture(object):
    """Capture frames of a video
    """

    def __init__(self, path, accurate=False, skip_delay_seconds=DEFAULT_ACCURATE_DELAY_SECONDS, frame_type=DEFAULT_FRAME_TYPE):
        self.path = path
        self.accurate = accurate
        self.skip_delay_seconds = skip_delay_seconds
        self.frame_type = frame_type

    def make_capture(self, time, width, height, out_path="out.png"):
        """Capture a frame at given time with given width and height using ffmpeg
        """
        skip_delay = MediaInfo.pretty_duration(self.skip_delay_seconds, show_millis=True)

        ffmpeg_command = [
            "ffmpeg",
            "-ss", time,
            "-i", self.path,
            "-vframes", "1",
            "-s", "%sx%s" % (width, height),
        ]

        if self.frame_type is not None:
            select_args = [
                "-vf", "select='eq(frame_type\\," + self.frame_type + ")'"
            ]

        if self.frame_type == "key":
            select_args = [
                "-vf", "select=key"
            ]

        if self.frame_type is not None:
            ffmpeg_command += select_args

        ffmpeg_command += [
            "-y",
            out_path
        ]

        if self.accurate:
            time_seconds = MediaInfo.pretty_to_seconds(time)
            skip_time_seconds = time_seconds - self.skip_delay_seconds

            if skip_time_seconds < 0:
                ffmpeg_command = [
                    "ffmpeg",
                    "-i", self.path,
                    "-ss", time,
                    "-vframes", "1",
                    "-s", "%sx%s" % (width, height),
                ]

                if self.frame_type is not None:
                    ffmpeg_command += select_args

                ffmpeg_command += [
                    "-y",
                    out_path
                ]
            else:
                skip_time = MediaInfo.pretty_duration(skip_time_seconds, show_millis=True)
                ffmpeg_command = [
                    "ffmpeg",
                    "-ss", skip_time,
                    "-i", self.path,
                    "-ss", skip_delay,
                    "-vframes", "1",
                    "-s", "%sx%s" % (width, height),
                ]

                if self.frame_type is not None:
                    ffmpeg_command += select_args

                ffmpeg_command += [
                    "-y",
                    out_path
                ]

        try:
            subprocess.call(ffmpeg_command, stderr=DEVNULL, stdout=DEVNULL)
        except FileNotFoundError:
            error = "Could not find 'ffmpeg' executable. Please make sure ffmpeg/ffprobe is installed and is in your PATH."
            error_exit(error)

    def compute_avg_color(self, image_path):
        """Computes the average color of an image
        """
        i = Image.open(image_path)
        i = i.convert('P')
        p = i.getcolors()

        # compute avg color
        total_count = 0
        avg_color = 0
        for count, color in p:
            total_count += count
            avg_color += count * color

        avg_color /= total_count

        return avg_color

    def compute_blurriness(self, image_path):
        """Computes the blurriness of an image. Small value means less blurry.
        """
        i = Image.open(image_path)
        i = i.convert('L')  # convert to grayscale

        a = numpy.asarray(i)
        b = abs(numpy.fft.rfft2(a))
        max_freq = self.avg9x(b)

        if max_freq is not 0:
            return 1 / max_freq
        else:
            return 1

    def avg9x(self, matrix, percentage=0.05):
        """Computes the median of the top n% highest values.
        By default, takes the top 5%
        """
        xs = matrix.flatten()
        srt = sorted(xs, reverse=True)
        length = int(math.floor(percentage * len(srt)))

        matrix_subset = srt[:length]
        return numpy.median(matrix_subset)

    def max_freq(self, matrix):
        """Returns the maximum value in the matrix
        """
        m = 0
        for row in matrix:
            mx = max(row)
            if mx > m:
                m = mx

        return m


def grid_desired_size(
        grid,
        media_info,
        width=DEFAULT_CONTACT_SHEET_WIDTH,
        horizontal_margin=DEFAULT_GRID_HORIZONTAL_SPACING):
    """Computes the size of the images placed on a mxn grid with given fixed width.
    Returns (width, height)
    """
    if grid:
        desired_width = (width - (grid.x - 1) * horizontal_margin) / grid.x
    else:
        desired_width = width

    return media_info.desired_size(width=desired_width)


def total_delay_seconds(media_info, args):
    """Computes the total seconds to skip (beginning + ending).
    """
    start_delay_seconds = math.floor(media_info.duration_seconds * args.start_delay_percent / 100)
    end_delay_seconds = math.floor(media_info.duration_seconds * args.end_delay_percent / 100)
    delay = start_delay_seconds + end_delay_seconds
    return delay


def timestamp_generator(media_info, args):
    """Generates `num_samples` uniformly distributed timestamps over time.
    Timestamps will be selected in the range specified by start_delay_percent and end_delay percent.
    For example, `end_delay_percent` can be used to avoid making captures during the ending credits.
    """
    delay = total_delay_seconds(media_info, args)
    capture_interval = (media_info.duration_seconds - delay) / (args.num_samples + 1)
    if args.interval is not None:
        capture_interval = int(args.interval.total_seconds())
    start_delay_seconds = math.floor(media_info.duration_seconds * args.start_delay_percent / 100)
    time = start_delay_seconds + capture_interval

    for i in range(args.num_samples):
        yield (time, MediaInfo.pretty_duration(time, show_millis=True))
        time += capture_interval


def select_sharpest_images(
        media_info,
        media_capture,
        args,
        num_groups=5):
    """Make `num_samples` captures and select `num_selected` captures out of these
    based on blurriness and color variety.
    """
    if num_groups is None:
        num_groups = args.num_selected

    # make sure num_selected is not too large
    if args.num_selected > num_groups:
        num_groups = args.num_selected

    if args.num_selected > args.num_samples:
        args.num_samples = args.num_selected

    # make sure num_samples is large enough
    if args.num_samples < args.num_selected or args.num_samples < num_groups:
        args.num_samples = args.num_selected
        num_groups = args.num_selected

    if args.interval is not None:
        total_delay = total_delay_seconds(media_info, args)
        selected_duration = media_info.duration_seconds - total_delay
        args.num_samples = math.floor(selected_duration / args.interval.total_seconds())
        args.num_selected = args.num_samples
        num_groups = args.num_samples
        square_side = math.ceil(math.sqrt(args.num_samples))
        if args.grid == DEFAULT_GRID_SIZE:
            args.grid = Grid(square_side, square_side)

    desired_size = grid_desired_size(
        args.grid,
        media_info,
        width=args.vcs_width,
        horizontal_margin=args.grid_horizontal_spacing)
    blurs = []

    if args.manual_timestamps is None:
        timestamps = timestamp_generator(media_info, args)
    else:
        timestamps = [(MediaInfo.pretty_to_seconds(x), x) for x in args.manual_timestamps]

    for i, timestamp in enumerate(timestamps):
        status = "Sampling... %s/%s" % ((i + 1), args.num_samples)
        print(status, end="\r")

        fd, filename = tempfile.mkstemp(suffix=".png")

        media_capture.make_capture(
            timestamp[1],
            desired_size[0],
            desired_size[1],
            filename)
        blurriness = media_capture.compute_blurriness(filename)
        avg_color = media_capture.compute_avg_color(filename)

        blurs += [
            Frame(
                filename=filename,
                blurriness=blurriness,
                timestamp=timestamp[0],
                avg_color=avg_color
            )
        ]
        os.close(fd)

    time_sorted = sorted(blurs, key=lambda x: x.timestamp)

    # group into num_selected groups
    if num_groups > 1:
        group_size = int(math.floor(len(time_sorted) / num_groups))
        groups = chunks(time_sorted, group_size)

        # find top sharpest for each group
        selected_items = [best(x) for x in groups]
    else:
        selected_items = time_sorted

    selected_items = select_color_variety(selected_items, args.num_selected)

    return selected_items, time_sorted


def select_color_variety(frames, num_selected):
    """Select captures so that they are not too similar to each other.
    """
    avg_color_sorted = sorted(frames, key=lambda x: x.avg_color)
    min_color = avg_color_sorted[0].avg_color
    max_color = avg_color_sorted[-1].avg_color
    color_span = max_color - min_color
    min_color_distance = int(color_span * 0.05)

    blurriness_sorted = sorted(frames, key=lambda x: x.blurriness, reverse=True)

    selected_items = []
    unselected_items = []
    while blurriness_sorted:
        frame = blurriness_sorted.pop()

        if not selected_items:
            selected_items += [frame]
        else:
            color_distance = min([abs(frame.avg_color - x.avg_color) for x in selected_items])
            if color_distance < min_color_distance:
                # too close to existing selected frame
                # don't select unless we run out of frames
                unselected_items += [(frame, color_distance)]
            else:
                selected_items += [frame]

    missing_items_count = num_selected - len(selected_items)
    if missing_items_count > 0:
        remaining_items = sorted(unselected_items, key=lambda x: x[0].blurriness)
        selected_items += [x[0] for x in remaining_items[:missing_items_count]]

    return selected_items


def best(captures):
    """Returns the least blurry capture
    """
    return sorted(captures, key=lambda x: x.blurriness)[0]


def chunks(l, n):
    """ Yield successive n-sized chunks from l.
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]


def draw_metadata(
        draw,
        args,
        header_line_height=None,
        header_lines=None,
        header_font=None,
        header_font_color=None,
        start_height=None):
    """Draw metadata header
    """
    h = start_height
    h += args.metadata_vertical_margin

    for line in header_lines:
        draw.text((args.metadata_horizontal_margin, h), line, font=header_font, fill=header_font_color)
        h += header_line_height

    h += args.metadata_vertical_margin

    return h


def max_line_length(
        media_info,
        metadata_font,
        header_margin,
        width=DEFAULT_CONTACT_SHEET_WIDTH,
        text=None):
    """Find the number of characters that fit in width with given font.
    """
    if text is None:
        text = media_info.filename

    max_width = width - 2 * header_margin

    max_length = 0
    for i in range(len(text) + 1):
        text_chunk = text[:i]
        text_width = 0 if len(text_chunk) == 0 else metadata_font.getsize(text_chunk)[0]

        max_length = i
        if text_width > max_width:
            break

    return max_length


def prepare_metadata_text_lines(media_info, header_font, header_margin, width, template_path=None):
    """Prepare the metadata header text and return a list containing each line.
    """
    template = ""
    if template_path is None:
        template = """{{filename}}
        File size: {{size}}
        Duration: {{duration}}
        Dimensions: {{sample_width}}x{{sample_height}}"""
    else:
        with open(template_path) as f:
            template = f.read()

    params = media_info.template_attributes()
    template = Template(template).render(params)
    template_lines = template.split("\n")
    template_lines = [x.strip() for x in template_lines if len(x) > 0]

    header_lines = []
    for line in template_lines:
        remaining_chars = line
        while len(remaining_chars) > 0:
            max_metadata_line_length = max_line_length(
                media_info,
                header_font,
                header_margin,
                width=width,
                text=remaining_chars)
            wraps = textwrap.wrap(remaining_chars, max_metadata_line_length)
            header_lines.append(wraps[0])
            remaining_chars = remaining_chars[len(wraps[0]):].strip()

    return header_lines


def compute_timestamp_position(args, w, h, text_size, desired_size, rectangle_hpadding, rectangle_vpadding):
    """Compute the (x,y) position of the upper left and bottom right points of the rectangle surrounding timestamp text.
    """
    position = args.timestamp_position

    x_offset = 0
    if position in [TimestampPosition.west, TimestampPosition.nw, TimestampPosition.sw]:
        x_offset = args.timestamp_horizontal_margin
    elif position in [TimestampPosition.north, TimestampPosition.center, TimestampPosition.south]:
        x_offset = (desired_size[0] / 2) - (text_size[0] / 2) - rectangle_hpadding
    else:
        x_offset = desired_size[0] - text_size[0] - args.timestamp_horizontal_margin - 2 * rectangle_hpadding

    y_offset = 0
    if position in [TimestampPosition.nw, TimestampPosition.north, TimestampPosition.ne]:
        y_offset = args.timestamp_vertical_margin
    elif position in [TimestampPosition.west, TimestampPosition.center, TimestampPosition.east]:
        y_offset = (desired_size[1] / 2) - (text_size[1] / 2) - rectangle_vpadding
    else:
        y_offset = desired_size[1] - text_size[1] - args.timestamp_vertical_margin - 2 * rectangle_vpadding

    upper_left = (
        w + x_offset,
        h + y_offset
    )

    bottom_right = (
        upper_left[0] + text_size[0] + 2 * rectangle_hpadding,
        upper_left[1] + text_size[1] + 2 * rectangle_vpadding
    )

    return upper_left, bottom_right


def compose_contact_sheet(
        media_info,
        frames,
        args):
    """Creates a video contact sheet with the media information in a header
    and the selected frames arranged on a mxn grid with optional timestamps
    """
    desired_size = grid_desired_size(
        args.grid,
        media_info,
        width=args.vcs_width,
        horizontal_margin=args.grid_horizontal_spacing)
    height = args.grid.y * (desired_size[1] + args.grid_vertical_spacing) - args.grid_vertical_spacing

    try:
        header_font = ImageFont.truetype(args.metadata_font, args.metadata_font_size)
    except OSError:
        if args.metadata_font == DEFAULT_METADATA_FONT:
            header_font = ImageFont.load_default()
        else:
            raise
    try:
        timestamp_font = ImageFont.truetype(args.timestamp_font, args.timestamp_font_size)
    except OSError:
        if args.timestamp_font == DEFAULT_TIMESTAMP_FONT:
            timestamp_font = ImageFont.load_default()
        else:
            raise

    header_lines = prepare_metadata_text_lines(
        media_info,
        header_font,
        args.metadata_horizontal_margin,
        args.vcs_width,
        template_path=args.metadata_template_path)

    line_spacing_coefficient = 1.2
    header_line_height = int(args.metadata_font_size * line_spacing_coefficient)
    header_height = 2 * args.metadata_margin + len(header_lines) * header_line_height

    if args.metadata_position == "hidden":
        header_height = 0

    final_image_width = args.vcs_width
    final_image_height = height + header_height
    transparent = (255, 255, 255, 0)

    image = Image.new("RGBA", (final_image_width, final_image_height), args.background_color)
    image_capture_layer = Image.new("RGBA", (final_image_width, final_image_height), transparent)
    image_header_text_layer = Image.new("RGBA", (final_image_width, final_image_height), transparent)
    image_timestamp_layer = Image.new("RGBA", (final_image_width, final_image_height), transparent)
    image_timestamp_text_layer = Image.new("RGBA", (final_image_width, final_image_height), transparent)

    draw_header_text_layer = ImageDraw.Draw(image_header_text_layer)
    draw_timestamp_layer = ImageDraw.Draw(image_timestamp_layer)
    draw_timestamp_text_layer = ImageDraw.Draw(image_timestamp_text_layer)
    h = 0

    def draw_metadata_helper():
        """Draw metadata with fixed arguments
        """
        return draw_metadata(
            draw_header_text_layer,
            args,
            header_line_height=header_line_height,
            header_lines=header_lines,
            header_font=header_font,
            header_font_color=args.metadata_font_color,
            start_height=h)

    # draw metadata
    if args.metadata_position == "top":
        h = draw_metadata_helper()

    # draw capture grid
    w = 0
    frames = sorted(frames, key=lambda x: x.timestamp)
    for i, frame in enumerate(frames):
        f = Image.open(frame.filename)
        f.putalpha(args.capture_alpha)
        image_capture_layer.paste(f, (w, h))

        # show timestamp
        if args.show_timestamp:
            pretty_timestamp = MediaInfo.pretty_duration(frame.timestamp, show_centis=True)
            text_size = timestamp_font.getsize(pretty_timestamp)

            # draw rectangle
            rectangle_hpadding = args.timestamp_horizontal_padding
            rectangle_vpadding = args.timestamp_vertical_padding

            upper_left, bottom_right = compute_timestamp_position(args, w, h, text_size, desired_size,
                                                                  rectangle_hpadding, rectangle_vpadding)

            if not args.timestamp_border_mode:
                draw_timestamp_layer.rectangle(
                    [upper_left, bottom_right],
                    fill=args.timestamp_background_color
                )
            else:
                offset_factor = args.timestamp_border_size
                offsets = [
                    (1, 0),
                    (-1, 0),
                    (0, 1),
                    (0, -1),
                    (1, 1),
                    (1, -1),
                    (-1, 1),
                    (-1, -1)
                ]

                final_offsets = []
                for offset_counter in range(1, offset_factor + 1):
                    final_offsets += [(x[0] * offset_counter, x[1] * offset_counter) for x in offsets]

                for offset in final_offsets:
                    # draw border first
                    draw_timestamp_text_layer.text(
                        (
                            upper_left[0] + rectangle_hpadding + offset[0],
                            upper_left[1] + rectangle_vpadding + offset[1]
                        ),
                        pretty_timestamp,
                        font=timestamp_font,
                        fill=args.timestamp_border_color
                    )

            # draw timestamp
            draw_timestamp_text_layer.text(
                (
                    upper_left[0] + rectangle_hpadding,
                    upper_left[1] + rectangle_vpadding
                ),
                pretty_timestamp,
                font=timestamp_font,
                fill=args.timestamp_font_color
            )

        # update x position for next frame
        w += desired_size[0] + args.grid_horizontal_spacing

        # update y position
        if (i + 1) % args.grid.x == 0:
            h += desired_size[1] + args.grid_vertical_spacing

        # update x position
        if (i + 1) % args.grid.x == 0:
            w = 0

    # draw metadata
    if args.metadata_position == "bottom":
        h -= args.grid_vertical_spacing
        h = draw_metadata_helper()

    # alpha blend
    out_image = Image.alpha_composite(image, image_capture_layer)
    out_image = Image.alpha_composite(out_image, image_header_text_layer)
    out_image = Image.alpha_composite(out_image, image_timestamp_layer)
    out_image = Image.alpha_composite(out_image, image_timestamp_text_layer)

    return out_image


def save_image(args, image, media_info, output_path):
    """Save the image to `output_path`
    """
    image = image.convert("RGB")
    try:
        image.save(output_path, optimize=True, quality=args.image_quality)
        return True
    except KeyError:
        return False


def cleanup(frames):
    """Delete temporary captures
    """
    for frame in frames:
        try:
            os.unlink(frame.filename)
        except:
            pass


def print_template_attributes():
    """Display all the available template attributes in a tabular format
    """
    table = MediaInfo.list_template_attributes()

    tab = texttable.Texttable()
    tab.set_cols_dtype(["t", "t", "t"])
    rows = [[x["name"], x["description"], x["example"]] for x in table]
    tab.add_rows(rows, header=False)
    tab.header(["Attribute name", "Description", "Example"])
    print(tab.draw())


def mxn_type(string):
    """Type parser for argparse. Argument of type "mxn" will be converted to Grid(m, n).
    An exception will be thrown if the argument is not of the required form
    """
    try:
        split = string.split("x")
        assert (len(split) == 2)
        m = int(split[0])
        assert (m > 0)
        n = int(split[1])
        assert (n > 0)
        return Grid(m, n)
    except (IndexError, ValueError, AssertionError):
        error = "Grid must be of the form mxn, where m is the number of columns and n is the number of rows."
        raise argparse.ArgumentTypeError(error)


def metadata_position_type(string):
    """Type parser for argparse. Argument of type string must be one of ["top", "bottom", "hidden"].
    An exception will be thrown if the argument is not one of these.
    """
    valid_metadata_positions = ["top", "bottom", "hidden"]

    lowercase_position = string.lower()
    if lowercase_position in valid_metadata_positions:
        return lowercase_position
    else:
        error = 'Metadata header position must be one of %s' % (str(valid_metadata_positions, ))
        raise argparse.ArgumentTypeError(error)


def hex_color_type(string):
    """Type parser for argparse. Argument must be an hexadecimal number representing a color.
    For example 'AABBCC' (RGB) or 'AABBCCFF' (RGBA). An exception will be raised if the argument
    is not of that form.
    """
    try:
        components = tuple(bytearray.fromhex(string))
        if len(components) == 3:
            components += (255,)
        c = Color(*components)
        return c
    except:
        error = "Color must be an hexadecimal number, for example 'AABBCC'"
        raise argparse.ArgumentTypeError(error)


def manual_timestamps(string):
    """Type parser for argparse. Argument must be a comma-separated list of frame timestamps.
    For example 1:11:11.111,2:22:22.222
    """
    try:
        timestamps = string.split(",")
        timestamps = [x.strip() for x in timestamps if x]

        # check whether timestamps are valid
        for t in timestamps:
            MediaInfo.pretty_to_seconds(t)

        return timestamps
    except Exception as e:
        print(e)
        error = "Manual frame timestamps must be comma-separated and of the form h:mm:ss.mmmm"
        raise argparse.ArgumentTypeError(error)


def timestamp_position_type(string):
    """Type parser for argparse. Argument must be a valid timestamp position"""
    try:
        return getattr(TimestampPosition, string)
    except AttributeError:
        error = "Invalid timestamp position: %s. Valid positions are: %s" % (string, VALID_TIMESTAMP_POSITIONS)
        raise argparse.ArgumentTypeError(error)


def interval_type(string):
    """Type parser for argparse. Argument must be a valid interval format.
    Supports any format supported by `parsedatetime`, including:
        * "30sec" (every 30 seconds)
        * "5 minutes" (every 5 minutes)
        * "1h" (every hour)
        * "2 hours 1 min and 30 seconds"
    """
    m = datetime.datetime.min
    cal = parsedatetime.Calendar()
    interval = cal.parseDT(string, sourceTime=m)[0] - m
    if interval == m:
        error = "Invalid interval format: {}".format(string)
        raise argparse.ArgumentTypeError(error)

    return interval


def error(message):
    """Print an error message."""
    print("[ERROR] %s" % (message,))


def error_exit(message):
    """Print an error message and exit"""
    error(message)
    sys.exit(-1)


def main():
    """Program entry point
    """
    parser = argparse.ArgumentParser(description="Create a video contact sheet",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("filenames", nargs="+")
    parser.add_argument(
        "-o", "--output",
        help="save to output file",
        dest="output_path")
    parser.add_argument(
        "--start-delay-percent",
        help="do not capture frames in the first n percent of total time",
        dest="start_delay_percent",
        type=int,
        default=DEFAULT_START_DELAY_PERCENT)
    parser.add_argument(
        "--end-delay-percent",
        help="do not capture frames in the last n percent of total time",
        dest="end_delay_percent",
        type=int,
        default=DEFAULT_END_DELAY_PERCENT)
    parser.add_argument(
        "--delay-percent",
        help="do not capture frames in the first and last n percent of total time",
        dest="delay_percent",
        type=int,
        default=DEFAULT_DELAY_PERCENT)
    parser.add_argument(
        "--grid-spacing",
        help="number of pixels spacing captures both vertically and horizontally",
        dest="grid_spacing",
        type=int,
        default=DEFAULT_GRID_SPACING)
    parser.add_argument(
        "--grid-horizontal-spacing",
        help="number of pixels spacing captures horizontally",
        dest="grid_horizontal_spacing",
        type=int,
        default=DEFAULT_GRID_HORIZONTAL_SPACING)
    parser.add_argument(
        "--grid-vertical-spacing",
        help="number of pixels spacing captures vertically",
        dest="grid_vertical_spacing",
        type=int,
        default=DEFAULT_GRID_VERTICAL_SPACING)
    parser.add_argument(
        "-w", "--width",
        help="width of the generated contact sheet",
        dest="vcs_width",
        type=int,
        default=DEFAULT_CONTACT_SHEET_WIDTH)
    parser.add_argument(
        "-g", "--grid",
        help="display frames on a mxn grid (for example 4x5)",
        dest="grid",
        type=mxn_type,
        default=DEFAULT_GRID_SIZE)
    parser.add_argument(
        "-s", "--num-samples",
        help="number of samples",
        dest="num_samples",
        type=int,
        default=None)
    parser.add_argument(
        "-t", "--show-timestamp",
        action="store_true",
        help="display timestamp for each frame",
        dest="show_timestamp")
    parser.add_argument(
        "--metadata-font-size",
        help="size of the font used for metadata",
        dest="metadata_font_size",
        type=int,
        default=DEFAULT_METADATA_FONT_SIZE)
    parser.add_argument(
        "--metadata-font",
        help="TTF font used for metadata",
        dest="metadata_font",
        default=DEFAULT_METADATA_FONT)
    parser.add_argument(
        "--timestamp-font-size",
        help="size of the font used for timestamps",
        dest="timestamp_font_size",
        type=int,
        default=DEFAULT_TIMESTAMP_FONT_SIZE)
    parser.add_argument(
        "--timestamp-font",
        help="TTF font used for timestamps",
        dest="timestamp_font",
        default=DEFAULT_TIMESTAMP_FONT)
    parser.add_argument(
        "--metadata-position",
        help="Position of the metadata header. Must be one of ['top', 'bottom', 'hidden']",
        dest="metadata_position",
        type=metadata_position_type,
        default=DEFAULT_METADATA_POSITION)
    parser.add_argument(
        "--background-color",
        help="Color of the background in hexadecimal, for example AABBCC",
        dest="background_color",
        type=hex_color_type,
        default=hex_color_type(DEFAULT_BACKGROUND_COLOR))
    parser.add_argument(
        "--metadata-font-color",
        help="Color of the metadata font in hexadecimal, for example AABBCC",
        dest="metadata_font_color",
        type=hex_color_type,
        default=hex_color_type(DEFAULT_METADATA_FONT_COLOR))
    parser.add_argument(
        "--timestamp-font-color",
        help="Color of the timestamp font in hexadecimal, for example AABBCC",
        dest="timestamp_font_color",
        type=hex_color_type,
        default=hex_color_type(DEFAULT_TIMESTAMP_FONT_COLOR))
    parser.add_argument(
        "--timestamp-background-color",
        help="Color of the timestamp background rectangle in hexadecimal, for example AABBCC",
        dest="timestamp_background_color",
        type=hex_color_type,
        default=hex_color_type(DEFAULT_TIMESTAMP_BACKGROUND_COLOR))
    parser.add_argument(
        "--timestamp-border-color",
        help="Color of the timestamp border in hexadecimal, for example AABBCC",
        dest="timestamp_border_color",
        type=hex_color_type,
        default=hex_color_type(DEFAULT_TIMESTAMP_BORDER_COLOR))
    parser.add_argument(
        "--template",
        help="Path to metadata template file",
        dest="metadata_template_path",
        default=None)
    parser.add_argument(
        "-m", "--manual",
        help="Comma-separated list of frame timestamps to use, for example 1:11:11.111,2:22:22.222",
        dest="manual_timestamps",
        type=manual_timestamps,
        default=None)
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="display verbose messages",
        dest="is_verbose")
    parser.add_argument(
        "-a", "--accurate",
        action="store_true",
        help="""Make accurate captures. This capture mode is way slower than the default one
        but it helps when capturing frames from HEVC videos.""",
        dest="is_accurate")
    parser.add_argument(
        "-A", "--accurate-delay-seconds",
        type=int,
        default=DEFAULT_ACCURATE_DELAY_SECONDS,
        help="""Fast skip to N seconds before capture time, then do accurate capture
        (decodes N seconds of video before each capture). This is used with accurate capture mode only.""",
        dest="accurate_delay_seconds")
    parser.add_argument(
        "--metadata-margin",
        type=int,
        default=DEFAULT_METADATA_MARGIN,
        help="Margin (in pixels) in the metadata header.",
        dest="metadata_margin")
    parser.add_argument(
        "--metadata-horizontal-margin",
        type=int,
        default=DEFAULT_METADATA_HORIZONTAL_MARGIN,
        help="Horizontal margin (in pixels) in the metadata header.",
        dest="metadata_horizontal_margin")
    parser.add_argument(
        "--metadata-vertical-margin",
        type=int,
        default=DEFAULT_METADATA_VERTICAL_MARGIN,
        help="Vertical margin (in pixels) in the metadata header.",
        dest="metadata_vertical_margin")
    parser.add_argument(
        "--timestamp-horizontal-padding",
        type=int,
        default=DEFAULT_TIMESTAMP_HORIZONTAL_PADDING,
        help="Horizontal padding (in pixels) for timestamps.",
        dest="timestamp_horizontal_padding")
    parser.add_argument(
        "--timestamp-vertical-padding",
        type=int,
        default=DEFAULT_TIMESTAMP_VERTICAL_PADDING,
        help="Vertical padding (in pixels) for timestamps.",
        dest="timestamp_vertical_padding")
    parser.add_argument(
        "--timestamp-horizontal-margin",
        type=int,
        default=DEFAULT_TIMESTAMP_HORIZONTAL_MARGIN,
        help="Horizontal margin (in pixels) for timestamps.",
        dest="timestamp_horizontal_margin")
    parser.add_argument(
        "--timestamp-vertical-margin",
        type=int,
        default=DEFAULT_TIMESTAMP_VERTICAL_MARGIN,
        help="Vertical margin (in pixels) for timestamps.",
        dest="timestamp_vertical_margin")
    parser.add_argument(
        "--quality",
        type=int,
        default=DEFAULT_IMAGE_QUALITY,
        help="Output image quality. Must be an integer in the range 0-100. 100 = best quality.",
        dest="image_quality")
    parser.add_argument(
        "-f", "--format",
        type=str,
        default=DEFAULT_IMAGE_FORMAT,
        help="Output image format. Can be any format supported by pillow. For example 'png' or 'jpg'.",
        dest="image_format")
    parser.add_argument(
        "-T", "--timestamp-position",
        type=timestamp_position_type,
        default=DEFAULT_TIMESTAMP_POSITION,
        help="Timestamp position. Must be one of %s." % (VALID_TIMESTAMP_POSITIONS,),
        dest="timestamp_position")
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Process every file in the specified directory recursively.",
        dest="recursive")
    parser.add_argument(
        "--timestamp-border-mode",
        action="store_true",
        help="Draw timestamp text with a border instead of the default rectangle.",
        dest="timestamp_border_mode")
    parser.add_argument(
        "--timestamp-border-size",
        type=int,
        default=DEFAULT_TIMESTAMP_BORDER_SIZE,
        help="Size of the timestamp border in pixels (used only with --timestamp-border-mode).",
        dest="timestamp_border_size")
    parser.add_argument(
        "--capture-alpha",
        type=int,
        default=DEFAULT_CAPTURE_ALPHA,
        help="Alpha channel value for the captures (transparency in range [0, 255]). Defaults to 255 (opaque)",
        dest="capture_alpha")
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s version {version}".format(version=__version__))
    parser.add_argument(
        "--list-template-attributes",
        action="store_true",
        dest="list_template_attributes")
    parser.add_argument(
        "--frame-type",
        type=str,
        default=DEFAULT_FRAME_TYPE,
        help="Frame type passed to ffmpeg 'select=eq(pict_type,FRAME_TYPE)' filter. Should be one of ('I', 'B', 'P') or the special type 'key' which will use the 'select=key' filter instead.",
        dest="frame_type")
    parser.add_argument(
        "--interval",
        type=interval_type,
        default=DEFAULT_INTERVAL,
        help="Capture frames at specified interval. Interval format is any string supported by `parsedatetime`. For example '5m', '3 minutes 5 seconds', '1 hour 15 min and 20 sec' etc.",
        dest="interval")
    parser.add_argument(
        "--ignore-errors",
        action="store_true",
        help="Ignore any error encountered while processing files recursively and continue to the next file.",
        dest="ignore_errors")
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Do not overwrite output file if it already exists, simply ignore this file and continue processing other unprocessed files.",
        dest="no_overwrite"
    )

    args = parser.parse_args()

    if args.list_template_attributes:
        print_template_attributes()
        sys.exit(0)

    if args.recursive:
        for path in args.filenames:
            for root, subdirs, files in os.walk(path):
                for f in files:
                    filepath = os.path.join(root, f)
                    try:
                        process_file(filepath, args)
                    except Exception:
                        if not args.ignore_errors:
                            raise
                        else:
                            print("[WARN]: failed to process {} ... skipping.".format(filepath), file=sys.stderr)
    else:
        for path in args.filenames:
            if os.path.isdir(path):
                for filepath in os.listdir(path):
                    abs_filepath = os.path.join(path, filepath)
                    if not os.path.isdir(abs_filepath):
                        process_file(abs_filepath, args)
            else:
                files_to_process = glob(path)
                if len(files_to_process) == 0:
                    files_to_process = [path]
                for filename in files_to_process:
                    process_file(filename, args)


def process_file(path, args):
    """Generate a video contact sheet for the file at given path
    """
    if not os.path.exists(path):
        if args.ignore_errors:
            print("File does not exist, skipping: {}".format(path))
            return
        else:
            error_message = "File does not exist: {}".format(path)
            error_exit(error_message)

    print("Processing {}...".format(path))

    media_info = MediaInfo(
        path,
        verbose=args.is_verbose)
    media_capture = MediaCapture(
        path,
        accurate=args.is_accurate,
        skip_delay_seconds=args.accurate_delay_seconds,
        frame_type=args.frame_type
    )

    output_path = args.output_path
    if not output_path:
        output_path = media_info.filename + "." + args.image_format

    if args.no_overwrite:
        if os.path.exists(output_path):
            print("[WARN] Output file already exists, skipping: {}".format(output_path))
            return


    # metadata margins
    if not args.metadata_margin == DEFAULT_METADATA_MARGIN:
        args.metadata_horizontal_margin = args.metadata_margin
        args.metadata_vertical_margin = args.metadata_margin

    args.num_selected = args.grid.x * args.grid.y

    # manual frame selection
    if args.manual_timestamps is not None:
        mframes_size = len(args.manual_timestamps)
        grid_size = args.grid.x * args.grid.y

        args.num_selected = mframes_size
        args.num_samples = mframes_size

        if not mframes_size == grid_size:
            # specified number of columns
            y = math.ceil(mframes_size / args.grid.x)
            args.grid = Grid(args.grid.x, y)

    if args.num_selected < 1:
        error = "One of --grid, --manual must be specified"
        raise argparse.ArgumentTypeError(error)

    if args.num_samples is None:
        args.num_samples = args.num_selected

    if args.delay_percent is not None:
        args.start_delay_percent = args.delay_percent
        args.end_delay_percent = args.delay_percent

    if args.grid_spacing is not None:
        args.grid_horizontal_spacing = args.grid_spacing
        args.grid_vertical_spacing = args.grid_spacing

    selected_frames, temp_frames = select_sharpest_images(media_info, media_capture, args)

    print("Composing contact sheet...")
    image = compose_contact_sheet(media_info, selected_frames, args)

    is_save_successful = save_image(args, image, media_info, output_path)

    print("Cleaning up temporary files...")
    cleanup(temp_frames)

    if not is_save_successful:
        error_exit("Unsupported image format: %s." % (args.image_format,))


if __name__ == "__main__":
    main()
