#!/usr/bin/env python3

"""Create a video contact sheet.
"""

import subprocess
import argparse
import json
import math
import os
import tempfile
import textwrap
import random
import sys
from collections import namedtuple

from PIL import Image, ImageDraw, ImageFont
import numpy

__author__ = "Nils Amiet"


DEFAULT_METADATA_FONT_SIZE = 12
DEFAULT_METADATA_FONT = "/usr/share/fonts/TTF/LiberationSans-Regular.ttf"
DEFAULT_TIMESTAMP_FONT_SIZE = 10
DEFAULT_TIMESTAMP_FONT = "/usr/share/fonts/TTF/DejaVuSans.ttf"
DEFAULT_CONTACT_SHEET_WIDTH = 600
DEFAULT_DELAY_PERCENT = None
DEFAULT_START_DELAY_PERCENT = 7
DEFAULT_END_DELAY_PERCENT = DEFAULT_START_DELAY_PERCENT
DEFAULT_GRID_SPACING = None
DEFAULT_GRID_HORIZONTAL_SPACING = 5
DEFAULT_GRID_VERTICAL_SPACING = DEFAULT_GRID_HORIZONTAL_SPACING
DEFAULT_METADATA_POSITION = "top"
DEFAULT_METADATA_FONT_COLOR = "000000"
DEFAULT_BACKGROUND_COLOR = "FFFFFF"
DEFAULT_TIMESTAMP_FONT_COLOR = "FFFFFF"
DEFAULT_TIMESTAMP_BACKGROUND_COLOR = "282828"


class MediaInfo():
    """Collect information about a video file"""

    def __init__(self, path, verbose=False):
        self.probe_media(path)
        self.find_video_stream()
        self.compute_display_resolution()
        self.compute_format()

        if verbose:
            print(self.filename)
            print("%sx%s" % (self.sample_width, self.sample_height))
            print("%sx%s" % (self.display_width, self.display_height))
            print(self.duration)
            print(self.size)

    def probe_media(self, path):
        """Probe video file using ffprobe"""
        ffprobe_command = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            path
        ]

        output = subprocess.check_output(ffprobe_command)
        self.ffprobe_dict = json.loads(output.decode("utf-8"))

    def human_readable_size(self, num, suffix='B'):
        """Converts a number of bytes to a human readable format"""
        for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
            if abs(num) < 1024.0:
                return "%3.1f %s%s" % (num, unit, suffix)
            num /= 1024.0
        return "%.1f %s%s" % (num, 'Yi', suffix)

    def find_video_stream(self):
        """Find the first stream which is a video stream"""
        for stream in self.ffprobe_dict["streams"]:
            try:
                if stream["codec_type"] == "video":
                    self.video_stream = stream
                    break
            except:
                pass

    def compute_display_resolution(self):
        """Computes the display resolution.
        Some videos have a sample resolution that differs from the display resolution
        (non-square pixels), thus the proper display resolution has to be computed."""
        width = int(self.video_stream["width"])
        height = int(self.video_stream["height"])
        self.sample_width = width
        self.sample_height = height
        sample_aspect_ratio = self.video_stream["sample_aspect_ratio"]

        if not sample_aspect_ratio == "1:1":
            sample_split = sample_aspect_ratio.split(":")
            sw = int(sample_split[0])
            sh = int(sample_split[1])

            self.display_width = int(width * sw / sh)
            self.display_height = int(height)
        else:
            self.display_width = width
            self.display_height = height

        if self.display_width == 0:
            self.display_width = self.sample_width

        if self.display_height == 0:
            self.display_height = self.sample_height

    def compute_format(self):
        """Compute duration, size and retrieve filename"""
        format_dict = self.ffprobe_dict["format"]

        self.duration_seconds = float(format_dict["duration"])
        self.duration = self.pretty_duration(self.duration_seconds)

        self.filename = os.path.basename(format_dict["filename"])

        self.size_bytes = int(format_dict["size"])
        self.size = self.human_readable_size(self.size_bytes)

    def pretty_duration(
            self,
            seconds,
            show_centis=False,
            show_millis=False):
        """Converts seconds to a human readable time format"""
        hours = math.floor(seconds / 3600)
        remaining_seconds = seconds - 3600 * hours

        minutes = math.floor(remaining_seconds / 60)
        remaining_seconds = remaining_seconds - 60 * minutes

        duration = ""

        if hours > 0:
            duration += "%s:" % (hours,)

        duration += "%s:%s" % (str(minutes).zfill(2), str(math.floor(remaining_seconds)).zfill(2))

        if show_centis:
            coeff = 1000 if show_millis else 100
            centis = round((remaining_seconds - math.floor(remaining_seconds)) * coeff)
            duration += ".%s" % (str(centis).zfill(2))

        return duration

    def desired_size(self, width=DEFAULT_CONTACT_SHEET_WIDTH):
        """Computes the height based on a given width and fixed aspect ratio.
        Returns (width, height)"""
        ratio = width / self.display_width
        desired_height = math.floor(self.display_height * ratio)
        return (int(width), int(desired_height))


class MediaCapture():
    """Capture frames of a video"""

    def __init__(self, path):
        self.path = path

    def make_capture(self, time, width, height, out_path="out.jpg"):
        """Capture a frame at given time with given width and height using ffmpeg"""
        # TODO if capture fails, retry using slow seek mode (-ss after -i)
        ffmpeg_command = [
            "ffmpeg",
            "-ss", time,
            "-i", self.path,
            "-vframes", "1",
            "-s", "%sx%s" % (width, height),
            "-y",
            out_path
        ]

        subprocess.call(ffmpeg_command, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

    def compute_avg_color(self, image_path):
        """Computes the average color of an image"""
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
        """Computes the blurriness of an image. Small value means less blurry."""
        i = Image.open(image_path)
        i = i.convert('L')  # convert to grayscale

        a = numpy.asarray(i)
        b = abs(numpy.fft.rfft2(a))
        max_freq = self.avg9x(b)

        if max_freq is not 0:
            return 1/max_freq
        else:
            return 1

    def avg9x(self, matrix, percentage=0.05):
        """Computes the median of the top n% highest values.
        By default, takes the top 5%"""
        xs = matrix.flatten()
        srt = sorted(xs, reverse=True)
        length = math.floor(percentage * len(srt))

        matrix_subset = srt[:length]
        return numpy.median(matrix_subset)

    def max_freq(self, matrix):
        """Returns the maximum value in the matrix"""
        m = 0
        for row in matrix:
            mx = max(row)
            if mx > m:
                m = mx

        return m


def grid_desired_size(grid, media_info, width=DEFAULT_CONTACT_SHEET_WIDTH, horizontal_margin=DEFAULT_GRID_HORIZONTAL_SPACING):
    """Computes the size of the mxn grid with given fixed width.
    Returns (width, height)"""
    if grid:
        desired_width = (width - (grid.x - 1) * horizontal_margin) / grid.x
    else:
        desired_width = width

    return media_info.desired_size(width=desired_width)


def select_sharpest_images(
        media_info,
        media_capture,
        num_samples=30,
        num_groups=5,
        num_selected=3,
        start_delay_percent=7,
        end_delay_percent=7,
        width=DEFAULT_CONTACT_SHEET_WIDTH,
        grid=None,
        grid_horizontal_spacing=DEFAULT_GRID_HORIZONTAL_SPACING):
    if num_groups is None:
        num_groups = num_selected

    # make sure num_selected is not too large
    if num_selected > num_groups:
        num_groups = num_selected

    if num_selected > num_samples:
        num_samples = num_selected

    # make sure num_samples if large enough
    if num_samples < num_selected or num_samples < num_groups:
        num_samples = num_selected
        num_groups = num_selected

    # compute list of timestamps (equally distributed)
    start_delay_seconds = math.floor(media_info.duration_seconds * start_delay_percent / 100)
    end_delay_seconds = math.floor(media_info.duration_seconds * end_delay_percent / 100)

    delay = start_delay_seconds + end_delay_seconds
    capture_interval = (media_info.duration_seconds - delay) / num_samples
    end = int(media_info.duration_seconds - end_delay_seconds)

    def timestamps():
        i = start_delay_seconds
        while i <= end:
            yield (i, media_info.pretty_duration(i, show_millis=True))
            i += capture_interval

    # compute desired_size
    desired_size = grid_desired_size(grid, media_info, width=width, horizontal_margin=grid_horizontal_spacing)

    Frame = namedtuple('Frame', ['filename', 'blurriness', 'timestamp', 'avg_color'])
    blurs = []
    for i, timestamp in enumerate(timestamps()):
        status = "Sampling... %s/%s" % ((i+1), num_samples)
        print(status, end="\r")

        filename = tempfile.mkstemp(suffix=".png")[1]

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

    time_sorted = sorted(blurs, key=lambda x: x.timestamp)

    # group into num_selected groups
    if num_groups > 1:
        group_size = math.floor(len(time_sorted)/num_groups)
        groups = chunks(time_sorted, group_size)

        # find top sharpest for each group
        selected_items = [best(x) for x in groups]
    else:
        selected_items = time_sorted

    selected_items = select_color_variety(selected_items, num_selected)
    # TODO color variety vs uniform distribution based on time for captures
    # TODO possible to get the best of both worlds?

    return selected_items, time_sorted


def select_color_variety(frames, num_selected):
    """Select captures so that they are not too similar to each other."""
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
    """Returns the least blurry capture"""
    return sorted(captures, key=lambda x: x.blurriness)[0]


def chunks(l, n):
    """ Yield successive n-sized chunks from l.
    """
    for i in range(0, len(l), n):
        yield l[i:i+n]


def draw_metadata(
        draw,
        header_margin=None,
        header_line_height=None,
        header_lines=None,
        header_font=None,
        header_font_color=None,
        start_height=None):
    h = start_height
    h += header_margin

    for line in header_lines:
        draw.text((header_margin, h), line, font=header_font, fill=header_font_color)
        h += header_line_height

    h += header_margin

    return h


def compose_contact_sheet(
        media_info,
        frames,
        output_path=None,
        width=DEFAULT_CONTACT_SHEET_WIDTH,
        show_timestamp=False,
        grid=None,
        metadata_font=DEFAULT_METADATA_FONT,
        metadata_font_size=DEFAULT_METADATA_FONT_SIZE,
        timestamp_font=DEFAULT_TIMESTAMP_FONT,
        timestamp_font_size=DEFAULT_TIMESTAMP_FONT_SIZE,
        grid_horizontal_spacing=DEFAULT_GRID_HORIZONTAL_SPACING,
        grid_vertical_spacing=DEFAULT_GRID_VERTICAL_SPACING,
        metadata_position=DEFAULT_METADATA_POSITION,
        background_color=DEFAULT_BACKGROUND_COLOR,
        metadata_font_color=DEFAULT_METADATA_FONT_COLOR,
        timestamp_font_color=DEFAULT_TIMESTAMP_FONT_COLOR,
        timestamp_background_color=DEFAULT_TIMESTAMP_BACKGROUND_COLOR):
    """Creates a video contact sheet with the media information in a header
    and the selected frames arranged on a mxn grid with optional timestamps"""
    num_frames = len(frames)
    desired_size = grid_desired_size(
        grid,
        media_info,
        width=width,
        horizontal_margin=grid_horizontal_spacing)
    height = grid.y * (desired_size[1] + grid_vertical_spacing) - grid_vertical_spacing

    header_margin = 10
    timestamp_horizontal_spacing = 5
    timestamp_vertical_spacing = timestamp_horizontal_spacing
    header_font = ImageFont.truetype(metadata_font, metadata_font_size)
    timestamp_font = ImageFont.truetype(timestamp_font, timestamp_font_size)

    metadata_font_dimensions = header_font.getsize(media_info.filename)
    filename_width = metadata_font_dimensions[0]
    max_width = width - 2 * header_margin
    width_excess = filename_width - max_width

    if width_excess > 0:
        excess_ratio = filename_width / max_width
        max_line_length = len(media_info.filename) / excess_ratio
    else:
        max_line_length = 1000

    header_lines = []
    header_lines += textwrap.wrap(media_info.filename, max_line_length)
    header_lines += ["File size: %s" % media_info.size]
    header_lines += ["Duration: %s" % media_info.duration]
    header_lines += ["Dimensions: %sx%s" % (media_info.sample_width, media_info.sample_height)]

    line_spacing_coefficient = 1.2
    header_line_height = int(metadata_font_size * line_spacing_coefficient)

    header_height = 2 * header_margin + len(header_lines) * header_line_height

    if metadata_position == "hidden":
        header_height = 0

    image = Image.new("RGBA", (width, height + header_height), background_color)
    draw = ImageDraw.Draw(image)
    h = 0

    def draw_metadata_helper():
        return draw_metadata(
            draw,
            header_margin=header_margin,
            header_line_height=header_line_height,
            header_lines=header_lines,
            header_font=header_font,
            header_font_color=metadata_font_color,
            start_height=h)

    # draw metadata
    if metadata_position == "top":
        h = draw_metadata_helper()

    # draw capture grid
    w = 0
    frames = sorted(frames, key=lambda x: x.timestamp)
    for i, frame in enumerate(frames):
        f = Image.open(frame.filename)
        image.paste(f, (w, h))

        # update x position early for timestamp
        w += desired_size[0] + grid_horizontal_spacing

        # show timestamp
        if show_timestamp:
            pretty_timestamp = media_info.pretty_duration(frame.timestamp, show_centis=True)
            text_size = timestamp_font.getsize(pretty_timestamp)

            # draw rectangle
            rectangle_hmargin = 3
            rectangle_vmargin = 1
            upper_left = (
                w - text_size[0] - 2 * rectangle_hmargin - grid_horizontal_spacing - timestamp_horizontal_spacing,
                h + desired_size[1] - text_size[1] - 2 * rectangle_vmargin - timestamp_vertical_spacing
                )
            bottom_right = (
                upper_left[0] + text_size[0] + 2 * rectangle_hmargin,
                upper_left[1] + text_size[1] + 2 * rectangle_vmargin
                )
            draw.rectangle(
                [upper_left, bottom_right],
                fill=timestamp_background_color
                )

            # draw timestamp
            draw.text(
                (
                    upper_left[0] + rectangle_hmargin,
                    upper_left[1] + rectangle_vmargin
                ),
                pretty_timestamp,
                font=timestamp_font,
                fill=timestamp_font_color
                )

        # update y position
        if (i+1) % grid.x == 0:
            h += desired_size[1] + grid_vertical_spacing

        # update x position
        if (i+1) % grid.x == 0:
            w = 0

    # draw metadata
    if metadata_position == "bottom":
        h -= grid_vertical_spacing
        h = draw_metadata_helper()

    # save image
    if not output_path:
        output_path = media_info.filename + ".png"

    image.save(output_path)


def cleanup(frames):
    """Delete temporary captures"""
    for frame in frames:
        try:
            os.unlink(frame.filename)
        except:
            pass


def mxn(string):
    """Type parser for argparse. Argument of type "mxn" will be converted to Grid(m, n).
    An exception will be thrown if the argument is not of the required form"""
    try:
        split = string.split("x")
        m = int(split[0])
        n = int(split[1])
        Grid = namedtuple('Grid', ['x', 'y'])
        return Grid(m, n)
    except:
        raise argparse.ArgumentTypeError("Grid must be of the form mxn, where m is the number of columns and n is the number of rows.")


def metadata_position(string):
    """Type parser for argparse. Argument of type string must be one of ["top", "bottom", "hidden"].
    An exception will be thrown if the argument is not one of these."""
    valid_metadata_positions = ["top", "bottom", "hidden"]

    lowercase_position = string.lower()
    if lowercase_position in valid_metadata_positions:
        return lowercase_position
    else:
        error = 'Metadata header position must be one of ["top", "bottom", "hidden"]'
        raise argparse.ArgumentTypeError(error)


def hex_color(string):
    """Type parser for argparse. Argument must be an hexadecimal number representing a color.
    For example C0F32F. An exception will be raised if the argument is not of that form."""
    try:
        Color = namedtuple('Color', ['r', 'g', 'b'])
        components = tuple(bytes.fromhex(string))
        c = Color(*components)
        return c
    except:
        error = "Color must be an hexadecimal number, for example AABBCC"
        raise argparse.ArgumentTypeError(error)


def main():
    """Program entry point"""
    parser = argparse.ArgumentParser(description="Create a video contact sheet")
    parser.add_argument("filename")
    parser.add_argument(
        "-o", "--output",
        help="save to output file",
        dest="output_path")
    parser.add_argument(
        "-n", "--num-frames",
        help="capture n frames",
        dest="num_frames",
        type=int,
        default=3)
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
        dest="mxn",
        type=mxn,
        default=None)
    parser.add_argument(
        "-s", "--num-samples",
        help="number of samples",
        dest="num_samples",
        type=int,
        default=50)
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
        type=metadata_position,
        default=DEFAULT_METADATA_POSITION)
    parser.add_argument(
        "--background-color",
        help="Color of the background in hexadecimal, for example AABBCC",
        dest="background_color",
        type=hex_color,
        default=hex_color(DEFAULT_BACKGROUND_COLOR))
    parser.add_argument(
        "--metadata-font-color",
        help="Color of the metadata font in hexadecimal, for example AABBCC",
        dest="metadata_font_color",
        type=hex_color,
        default=hex_color(DEFAULT_METADATA_FONT_COLOR))
    parser.add_argument(
        "--timestamp-font-color",
        help="Color of the timestamp font in hexadecimal, for example AABBCC",
        dest="timestamp_font_color",
        type=hex_color,
        default=hex_color(DEFAULT_TIMESTAMP_FONT_COLOR))
    parser.add_argument(
        "--timestamp-background-color",
        help="Color of the timestamp background rectangle in hexadecimal, for example AABBCC",
        dest="timestamp_background_color",
        type=hex_color,
        default=hex_color(DEFAULT_TIMESTAMP_BACKGROUND_COLOR))
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="display verbose messages",
        dest="is_verbose")

    args = parser.parse_args()

    path = args.filename
    output_path = args.output_path

    media_info = MediaInfo(path, verbose=args.is_verbose)
    media_capture = MediaCapture(path)

    num_selected = args.num_frames

    if args.mxn:
        num_selected = args.mxn[0] * args.mxn[1]
    else:
        args.mxn = mxn("%sx%s" % (1, num_selected))

    if args.delay_percent is not None:
        args.start_delay_percent = args.delay_percent
        args.end_delay_percent = args.delay_percent

    if args.grid_spacing is not None:
        args.grid_horizontal_spacing = args.grid_spacing
        args.grid_vertical_spacing = args.grid_spacing

    selected_frames, temp_frames = select_sharpest_images(
        media_info,
        media_capture,
        num_selected=num_selected,
        num_samples=args.num_samples,
        width=args.vcs_width,
        grid=args.mxn,
        start_delay_percent=args.start_delay_percent,
        end_delay_percent=args.end_delay_percent,
        grid_horizontal_spacing=args.grid_horizontal_spacing
        )

    print("Composing contact sheet...")
    compose_contact_sheet(
        media_info,
        selected_frames,
        output_path,
        width=args.vcs_width,
        show_timestamp=args.show_timestamp,
        grid=args.mxn,
        metadata_font=args.metadata_font,
        metadata_font_size=args.metadata_font_size,
        timestamp_font=args.timestamp_font,
        timestamp_font_size=args.timestamp_font_size,
        grid_horizontal_spacing=args.grid_horizontal_spacing,
        grid_vertical_spacing=args.grid_vertical_spacing,
        metadata_position=args.metadata_position,
        background_color=args.background_color,
        metadata_font_color=args.metadata_font_color,
        timestamp_font_color=args.timestamp_font_color,
        timestamp_background_color=args.timestamp_background_color
        )

    print("Cleaning up temporary files...")
    cleanup(temp_frames)


if __name__ == "__main__":
    main()
