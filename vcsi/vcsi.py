#!/usr/bin/env python3

"""Create a video contact sheet.
"""

from __future__ import print_function

import subprocess
import os
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

from PIL import Image, ImageDraw, ImageFont
import numpy
from jinja2 import Template

__version__ = "5"
__author__ = "Nils Amiet"

Grid = namedtuple('Grid', ['x', 'y'])
Frame = namedtuple('Frame', ['filename', 'blurriness', 'timestamp', 'avg_color'])
Color = namedtuple('Color', ['r', 'g', 'b', 'a'])

DEFAULT_METADATA_FONT_SIZE = 10
DEFAULT_METADATA_FONT = "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf"
DEFAULT_TIMESTAMP_FONT_SIZE = 10
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
DEFAULT_ACCURATE_DELAY_SECONDS = 1
DEFAULT_METADATA_MARGIN = 10
DEFAULT_CAPTURE_ALPHA = 255
DEFAULT_GRID_SIZE = Grid(4, 4)


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

        output = subprocess.check_output(ffprobe_command)
        self.ffprobe_dict = json.loads(output.decode("utf-8"))

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
        sample_aspect_ratio = self.video_stream["sample_aspect_ratio"]

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

        result = (millis/1000.0) + seconds + minutes * 60 + hours * 3600
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
        attributes = {
            "size": self.size,
            "size_bytes": self.size_bytes,
            "filename": self.filename,
            "duration": self.duration,
            "sample_width": self.sample_width,
            "sample_height": self.sample_height,
            "display_width": self.display_width,
            "display_height": self.display_height,
            "video_codec": self.video_codec,
            "video_codec_long": self.video_codec_long,
            "display_aspect_ratio": self.display_aspect_ratio,
            "sample_aspect_ratio": self.sample_aspect_ratio,
            "audio_codec": self.audio_codec,
            "audio_codec_long": self.audio_codec_long,
            "audio_sample_rate": self.audio_sample_rate,
            "audio_bit_rate": self.audio_bit_rate,
            "frame_rate": self.frame_rate
        }
        return attributes


class MediaCapture(object):
    """Capture frames of a video
    """

    def __init__(self, path, accurate=False, skip_delay_seconds=DEFAULT_ACCURATE_DELAY_SECONDS):
        self.path = path
        self.accurate = accurate
        self.skip_delay_seconds = skip_delay_seconds

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
                    "-y",
                    out_path
                ]

        subprocess.call(ffmpeg_command, stderr=DEVNULL, stdout=DEVNULL)

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
            return 1/max_freq
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


def timestamp_generator(media_info, start_delay_percent, end_delay_percent, num_samples):
    """Generates `num_samples` uniformly distributed timestamps over time.
    Timestamps will be selected in the range specified by start_delay_percent and end_delay percent.
    For example, `end_delay_percent` can be used to avoid making captures during the ending credits.
    """
    start_delay_seconds = math.floor(media_info.duration_seconds * start_delay_percent / 100)
    end_delay_seconds = math.floor(media_info.duration_seconds * end_delay_percent / 100)
    delay = start_delay_seconds + end_delay_seconds
    capture_interval = (media_info.duration_seconds - delay) / (num_samples + 1)
    end = int(media_info.duration_seconds - end_delay_seconds)
    time = start_delay_seconds + capture_interval

    for i in range(num_samples):
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

    desired_size = grid_desired_size(
        args.grid,
        media_info,
        width=args.vcs_width,
        horizontal_margin=args.grid_horizontal_spacing)
    blurs = []
    if args.manual_timestamps is None:
        timestamps = timestamp_generator(media_info,
                                         args.start_delay_percent,
                                         args.end_delay_percent,
                                         args.num_samples)
    else:
        timestamps = [(MediaInfo.pretty_to_seconds(x), x) for x in args.manual_timestamps]

    for i, timestamp in enumerate(timestamps):
        status = "Sampling... %s/%s" % ((i+1), args.num_samples)
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
        group_size = int(math.floor(len(time_sorted)/num_groups))
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
        yield l[i:i+n]


def draw_metadata(
        draw,
        header_margin=None,
        header_line_height=None,
        header_lines=None,
        header_font=None,
        header_font_color=None,
        start_height=None):
    """Draw metadata header
    """
    h = start_height
    h += header_margin

    for line in header_lines:
        draw.text((header_margin, h), line, font=header_font, fill=header_font_color)
        h += header_line_height

    h += header_margin

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
        text_width = metadata_font.getsize(text_chunk)[0]

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


def compose_contact_sheet(
        media_info,
        frames,
        args,
        timestamp_horizontal_spacing=5,
        timestamp_vertical_spacing=5):
    """Creates a video contact sheet with the media information in a header
    and the selected frames arranged on a mxn grid with optional timestamps
    """
    desired_size = grid_desired_size(
        args.grid,
        media_info,
        width=args.vcs_width,
        horizontal_margin=args.grid_horizontal_spacing)
    height = args.grid.y * (desired_size[1] + args.grid_vertical_spacing) - args.grid_vertical_spacing

    header_font = ImageFont.truetype(args.metadata_font, args.metadata_font_size)
    timestamp_font = ImageFont.truetype(args.timestamp_font, args.timestamp_font_size)

    header_lines = prepare_metadata_text_lines(
        media_info,
        header_font,
        args.metadata_margin,
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
            header_margin=args.metadata_margin,
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

        # update x position early for timestamp
        w += desired_size[0] + args.grid_horizontal_spacing

        # show timestamp
        if args.show_timestamp:
            pretty_timestamp = MediaInfo.pretty_duration(frame.timestamp, show_centis=True)
            text_size = timestamp_font.getsize(pretty_timestamp)

            # draw rectangle
            rectangle_hmargin = 3
            rectangle_vmargin = 1
            upper_left = (
                w - text_size[0] - 2 * rectangle_hmargin - args.grid_horizontal_spacing - timestamp_horizontal_spacing,
                h + desired_size[1] - text_size[1] - 2 * rectangle_vmargin - timestamp_vertical_spacing
                )
            bottom_right = (
                upper_left[0] + text_size[0] + 2 * rectangle_hmargin,
                upper_left[1] + text_size[1] + 2 * rectangle_vmargin
                )
            draw_timestamp_layer.rectangle(
                [upper_left, bottom_right],
                fill=args.timestamp_background_color
                )

            # draw timestamp
            draw_timestamp_text_layer.text(
                (
                    upper_left[0] + rectangle_hmargin,
                    upper_left[1] + rectangle_vmargin
                ),
                pretty_timestamp,
                font=timestamp_font,
                fill=args.timestamp_font_color
                )

        # update y position
        if (i+1) % args.grid.x == 0:
            h += desired_size[1] + args.grid_vertical_spacing

        # update x position
        if (i+1) % args.grid.x == 0:
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


def save_image(image, media_info, output_path):
    """Save the image to `output_path`
    """
    if not output_path:
        output_path = media_info.filename + ".png"

    image.save(output_path)


def cleanup(frames):
    """Delete temporary captures
    """
    for frame in frames:
        try:
            os.unlink(frame.filename)
        except:
            pass


def mxn_type(string):
    """Type parser for argparse. Argument of type "mxn" will be converted to Grid(m, n).
    An exception will be thrown if the argument is not of the required form
    """
    try:
        split = string.split("x")
        m = int(split[0])
        n = int(split[1])
        return Grid(m, n)
    except:
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
        error = 'Metadata header position must be one of %s' % (str(valid_metadata_positions,))
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
    """Type parser for argparse. Argument must be a comma-seperated list of frame timestamps.
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
        error = "Manual frame timestamps must be comma-seperated and of the form h:mm:ss.mmmm"
        raise argparse.ArgumentTypeError(error)


def main():
    """Program entry point
    """
    parser = argparse.ArgumentParser(description="Create a video contact sheet")
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
        "--template",
        help="Path to metadata template file",
        dest="metadata_template_path",
        default=None)
    parser.add_argument(
        "-m", "--manual",
        help="Comma-seperated list of frame timestamps to use, for example 1:11:11.111,2:22:22.222",
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
        (decodes N seconds of video before each capture). This is used with accurate caputre mode only.""",
        dest="accurate_delay_seconds")
    parser.add_argument(
        "--metadata-margin",
        type=int,
        default=DEFAULT_METADATA_MARGIN,
        help="Margin (in pixels) in the metadata header.",
        dest="metadata_margin")
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Process every file in the specified directory recursively.",
        dest="recursive")
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

    args = parser.parse_args()

    if args.recursive:
        for path in args.filenames:
            for root, subdirs, files in os.walk(path):
                for f in files:
                    filepath = os.path.join(root, f)
                    process_file(filepath, args)
    else:
        for path in args.filenames:
            if os.path.isdir(path):
                for filepath in os.listdir(path):
                    abs_filepath = os.path.join(path, filepath)
                    if not os.path.isdir(abs_filepath):
                        process_file(abs_filepath, args)
            else:
                process_file(path, args)


def process_file(path, args):
    """Generate a video contact sheet for the file at given path
    """
    print("Processing %s..." % (path))

    output_path = args.output_path

    media_info = MediaInfo(
        path,
        verbose=args.is_verbose)
    media_capture = MediaCapture(
        path,
        accurate=args.is_accurate,
        skip_delay_seconds=args.accurate_delay_seconds)

    args.num_selected = args.grid.x * args.grid.y

    if args.num_samples is None:
        args.num_samples = args.num_selected

    if args.delay_percent is not None:
        args.start_delay_percent = args.delay_percent
        args.end_delay_percent = args.delay_percent

    if args.grid_spacing is not None:
        args.grid_horizontal_spacing = args.grid_spacing
        args.grid_vertical_spacing = args.grid_spacing

    # manual frame selection
    if args.manual_timestamps is not None:
        mframes_size = len(args.manual_timestamps)
        grid_size = args.grid.x * args.grid.y

        args.num_selected = mframes_size
        args.num_samples = mframes_size

        if not mframes_size == grid_size:
            args.grid = Grid(1, mframes_size)

    selected_frames, temp_frames = select_sharpest_images(media_info, media_capture, args)

    print("Composing contact sheet...")
    image = compose_contact_sheet(media_info, selected_frames, args)

    save_image(image, media_info, output_path)

    print("Cleaning up temporary files...")
    cleanup(temp_frames)


if __name__ == "__main__":
    main()
