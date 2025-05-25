# vcsi

![Build Status](https://github.com/amietn/vcsi/actions/workflows/testing.yml/badge.svg)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](http://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/vcsi.svg)](https://badge.fury.io/py/vcsi)

Create video contact sheets. A video contact sheet is an image composed of video capture thumbnails arranged on a grid.

## Examples

```
$ vcsi bbb_sunflower_2160p_60fps_normal.mp4 \
-t \
-w 830 \
-g 4x4 \
--background-color 000000 \
--metadata-font-color ffffff \
--end-delay-percent 20 \
--metadata-font /usr/share/fonts/TTF/DejaVuSans-Bold.ttf
```

![Example image 1](https://github.com/amietn/vcsi/assets/5566087/4ef4e631-eca6-43d0-8400-89f1bbbda73d)

```
$ vcsi bbb_sunflower_2160p_60fps_normal.mp4 \
-t \
-w 830 \
-g 3x5 \
--end-delay-percent 20 \
--timestamp-font /usr/share/fonts/TTF/DejaVuSans.ttf \
-o output.png
```

![Example image 2](https://github.com/amietn/vcsi/assets/5566087/5c6e88f3-29af-44dc-b36c-5e493e6d8dee)


The above contact sheets were generated from a movie called "Big Buck Bunny".

## Installation

First, install [uv](https://docs.astral.sh/uv/getting-started/installation/).


### uv

`vcsi` can be installed from PyPi:

```
$ uv tool install vcsi
```

or from local sources:

```
$ uv tool install .
```

### pip

```
pip install vcsi
```

### Distribution packages

vcsi is currently packaged for the following systems:

| Linux packages | |
| -------------- | --- |
| Arch (AUR) | https://aur.archlinux.org/packages/vcsi/ |
| Arch (AUR, git master) | https://aur.archlinux.org/packages/vcsi-git/ |
| Gentoo | https://packages.gentoo.org/packages/media-video/vcsi |

Your system is not listed?

```
$ apt-get install ffmpeg
```

Then use the uv installation method above.

Running Windows? See the note below.


## Note for Windows users

Download a binary build of ffmpeg from Zeranoe here (e.g. 64bit static): http://ffmpeg.zeranoe.com/builds/

Extract the archive and add the `bin` directory to your PATH so that `ffmpeg` and `ffprobe` can be invoked from the command line.

If you have issues installing numpy with pip, download an already built version of numpy here: http://sourceforge.net/projects/numpy/files/NumPy/


## Requirements

Python modules:

* numpy
* pillow
* jinja2
* texttable
* parsedatetime


Must be in your PATH:

* ffmpeg
* ffprobe


## Usage

```
$ vcsi -h
usage: vcsi [-h] [-o OUTPUT_PATH] [-c CONFIG]
            [--start-delay-percent START_DELAY_PERCENT]
            [--end-delay-percent END_DELAY_PERCENT]
            [--delay-percent DELAY_PERCENT] [--grid-spacing GRID_SPACING]
            [--grid-horizontal-spacing GRID_HORIZONTAL_SPACING]
            [--grid-vertical-spacing GRID_VERTICAL_SPACING] [-w VCS_WIDTH]
            [-g GRID] [-s NUM_SAMPLES] [-t]
            [--metadata-font-size METADATA_FONT_SIZE]
            [--metadata-font METADATA_FONT]
            [--timestamp-font-size TIMESTAMP_FONT_SIZE]
            [--timestamp-font TIMESTAMP_FONT]
            [--metadata-position METADATA_POSITION]
            [--background-color BACKGROUND_COLOR]
            [--metadata-font-color METADATA_FONT_COLOR]
            [--timestamp-font-color TIMESTAMP_FONT_COLOR]
            [--timestamp-background-color TIMESTAMP_BACKGROUND_COLOR]
            [--timestamp-border-color TIMESTAMP_BORDER_COLOR]
            [--template METADATA_TEMPLATE_PATH] [-m MANUAL_TIMESTAMPS] [-v]
            [-a] [-A ACCURATE_DELAY_SECONDS]
            [--metadata-margin METADATA_MARGIN]
            [--metadata-horizontal-margin METADATA_HORIZONTAL_MARGIN]
            [--metadata-vertical-margin METADATA_VERTICAL_MARGIN]
            [--timestamp-horizontal-padding TIMESTAMP_HORIZONTAL_PADDING]
            [--timestamp-vertical-padding TIMESTAMP_VERTICAL_PADDING]
            [--timestamp-horizontal-margin TIMESTAMP_HORIZONTAL_MARGIN]
            [--timestamp-vertical-margin TIMESTAMP_VERTICAL_MARGIN]
            [--quality IMAGE_QUALITY] [-f IMAGE_FORMAT]
            [-T TIMESTAMP_POSITION] [-r] [--timestamp-border-mode]
            [--timestamp-border-size TIMESTAMP_BORDER_SIZE]
            [--capture-alpha CAPTURE_ALPHA] [--version]
            [--list-template-attributes] [--frame-type FRAME_TYPE]
            [--interval INTERVAL] [--ignore-errors] [--no-overwrite]
            [--exclude-extensions EXCLUDE_EXTENSIONS] [--fast]
            [-O THUMBNAIL_OUTPUT_PATH] [-S]
            [--timestamp-format TIMESTAMP_FORMAT]
            filenames [filenames ...]

Create a video contact sheet

positional arguments:
  filenames

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT_PATH, --output OUTPUT_PATH
                        save to output file (default: None)
  -c CONFIG, --config CONFIG
                        Config file to load defaults from (default:
                        C:\Users\zacke\.config/vcsi.conf)
  --start-delay-percent START_DELAY_PERCENT
                        do not capture frames in the first n percent of total
                        time (default: 7)
  --end-delay-percent END_DELAY_PERCENT
                        do not capture frames in the last n percent of total
                        time (default: 7)
  --delay-percent DELAY_PERCENT
                        do not capture frames in the first and last n percent
                        of total time (default: None)
  --grid-spacing GRID_SPACING
                        number of pixels spacing captures both vertically and
                        horizontally (default: None)
  --grid-horizontal-spacing GRID_HORIZONTAL_SPACING
                        number of pixels spacing captures horizontally
                        (default: 5)
  --grid-vertical-spacing GRID_VERTICAL_SPACING
                        number of pixels spacing captures vertically (default:
                        5)
  -w VCS_WIDTH, --width VCS_WIDTH
                        width of the generated contact sheet (default: 1500)
  -g GRID, --grid GRID  display frames on a mxn grid (for example 4x5). The
                        special value zero (as in 2x0 or 0x5 or 0x0) is only
                        allowed when combined with --interval or with
                        --manual. Zero means that the component should be
                        automatically deduced based on other arguments passed.
                        (default: 4x4)
  -s NUM_SAMPLES, --num-samples NUM_SAMPLES
                        number of samples (default: None)
  -t, --show-timestamp  display timestamp for each frame (default: False)
  --metadata-font-size METADATA_FONT_SIZE
                        size of the font used for metadata (default: 16)
  --metadata-font METADATA_FONT
                        TTF font used for metadata (default:
                        C:/Windows/Fonts/msgothic.ttc)
  --timestamp-font-size TIMESTAMP_FONT_SIZE
                        size of the font used for timestamps (default: 12)
  --timestamp-font TIMESTAMP_FONT
                        TTF font used for timestamps (default:
                        C:/Windows/Fonts/msgothic.ttc)
  --metadata-position METADATA_POSITION
                        Position of the metadata header. Must be one of
                        ['top', 'bottom', 'hidden'] (default: top)
  --background-color BACKGROUND_COLOR
                        Color of the background in hexadecimal, for example
                        AABBCC (default: 000000FF)
  --metadata-font-color METADATA_FONT_COLOR
                        Color of the metadata font in hexadecimal, for example
                        AABBCC (default: FFFFFFFF)
  --timestamp-font-color TIMESTAMP_FONT_COLOR
                        Color of the timestamp font in hexadecimal, for
                        example AABBCC (default: FFFFFFFF)
  --timestamp-background-color TIMESTAMP_BACKGROUND_COLOR
                        Color of the timestamp background rectangle in
                        hexadecimal, for example AABBCC (default: 000000AA)
  --timestamp-border-color TIMESTAMP_BORDER_COLOR
                        Color of the timestamp border in hexadecimal, for
                        example AABBCC (default: 000000FF)
  --template METADATA_TEMPLATE_PATH
                        Path to metadata template file (default: None)
  -m MANUAL_TIMESTAMPS, --manual MANUAL_TIMESTAMPS
                        Comma-separated list of frame timestamps to use, for
                        example 1:11:11.111,2:22:22.222 (default: None)
  -v, --verbose         display verbose messages (default: False)
  -a, --accurate        Make accurate captures. This capture mode is way
                        slower than the default one but it helps when
                        capturing frames from HEVC videos. (default: False)
  -A ACCURATE_DELAY_SECONDS, --accurate-delay-seconds ACCURATE_DELAY_SECONDS
                        Fast skip to N seconds before capture time, then do
                        accurate capture (decodes N seconds of video before
                        each capture). This is used with accurate capture mode
                        only. (default: 1)
  --metadata-margin METADATA_MARGIN
                        Margin (in pixels) in the metadata header. (default:
                        10)
  --metadata-horizontal-margin METADATA_HORIZONTAL_MARGIN
                        Horizontal margin (in pixels) in the metadata header.
                        (default: 10)
  --metadata-vertical-margin METADATA_VERTICAL_MARGIN
                        Vertical margin (in pixels) in the metadata header.
                        (default: 10)
  --timestamp-horizontal-padding TIMESTAMP_HORIZONTAL_PADDING
                        Horizontal padding (in pixels) for timestamps.
                        (default: 3)
  --timestamp-vertical-padding TIMESTAMP_VERTICAL_PADDING
                        Vertical padding (in pixels) for timestamps. (default:
                        1)
  --timestamp-horizontal-margin TIMESTAMP_HORIZONTAL_MARGIN
                        Horizontal margin (in pixels) for timestamps.
                        (default: 5)
  --timestamp-vertical-margin TIMESTAMP_VERTICAL_MARGIN
                        Vertical margin (in pixels) for timestamps. (default:
                        5)
  --quality IMAGE_QUALITY
                        Output image quality. Must be an integer in the range
                        0-100. 100 = best quality. (default: 100)
  -f IMAGE_FORMAT, --format IMAGE_FORMAT
                        Output image format. Can be any format supported by
                        pillow. For example 'png' or 'jpg'. (default: jpg)
  -T TIMESTAMP_POSITION, --timestamp-position TIMESTAMP_POSITION
                        Timestamp position. Must be one of ['north', 'south',
                        'east', 'west', 'ne', 'nw', 'se', 'sw', 'center'].
                        (default: TimestampPosition.se)
  -r, --recursive       Process every file in the specified directory
                        recursively. (default: False)
  --timestamp-border-mode
                        Draw timestamp text with a border instead of the
                        default rectangle. (default: False)
  --timestamp-border-size TIMESTAMP_BORDER_SIZE
                        Size of the timestamp border in pixels (used only with
                        --timestamp-border-mode). (default: 1)
  --capture-alpha CAPTURE_ALPHA
                        Alpha channel value for the captures (transparency in
                        range [0, 255]). Defaults to 255 (opaque) (default:
                        255)
  --version             show program's version number and exit
  --list-template-attributes
  --frame-type FRAME_TYPE
                        Frame type passed to ffmpeg
                        'select=eq(pict_type,FRAME_TYPE)' filter. Should be
                        one of ('I', 'B', 'P') or the special type 'key' which
                        will use the 'select=key' filter instead. (default:
                        None)
  --interval INTERVAL   Capture frames at specified interval. Interval format
                        is any string supported by `parsedatetime`. For
                        example '5m', '3 minutes 5 seconds', '1 hour 15 min
                        and 20 sec' etc. (default: None)
  --ignore-errors       Ignore any error encountered while processing files
                        recursively and continue to the next file. (default:
                        False)
  --no-overwrite        Do not overwrite output file if it already exists,
                        simply ignore this file and continue processing other
                        unprocessed files. (default: False)
  --exclude-extensions EXCLUDE_EXTENSIONS
                        Do not process files that end with the given
                        extensions. (default: [])
  --fast                Fast mode. Just make a contact sheet as fast as
                        possible, regardless of output image quality. May mess
                        up the terminal. (default: False)
  -O THUMBNAIL_OUTPUT_PATH, --thumbnail-output THUMBNAIL_OUTPUT_PATH
                        Save thumbnail files to the specified output
                        directory. If set, the thumbnail files will not be
                        deleted after successful creation of the contact
                        sheet. (default: None)
  -S, --actual-size     Make thumbnails of actual size. In other words,
                        thumbnails will have the actual 1:1 size of the video
                        resolution. (default: False)
  --timestamp-format TIMESTAMP_FORMAT
                        Use specified timestamp format. Replaced values
                        include: {TIME}, {DURATION}, {THUMBNAIL_NUMBER}, {H}
                        (hours), {M} (minutes), {S} (seconds), {c}
                        (centiseconds), {m} (milliseconds), {dH}, {dM}, {dS},
                        {dc} and {dm} (same as previous values but for the
                        total duration). Example format: '{TIME} /
                        {DURATION}'. Another example: '{THUMBNAIL_NUMBER}'.
                        Yet another example: '{H}:{M}:{S}.{m} /
                        {dH}:{dM}:{dS}.{dm}'. (default: {TIME})


```

## Metadata templates

`vcsi` now supports metadata templates thanks to jinja2. In order to use custom templates one should use the `--template` argument to specifiy the path to a template file.

Here is a sample template file:

```
{{filename}}
File size: {{size}}
{% if audio_sample_rate %}
Audio sample rate: {{audio_sample_rate/1000}} KHz
{% endif %}

{% if audio_bit_rate %}
Audio bitrate: {{audio_bit_rate/1000}} Kbps
{% endif %}

{{frame_rate}} fps

Resolution: {{sample_width}}x{{sample_height}}
```

## Exposed metadata template attributes

| Attribute name | Description | Example |
| --- | --- | --- |
| size | File size (pretty format) | 128.3 MiB |
| size_bytes | File size (bytes) | 4662788373 |
| filename | File name | video.mkv |
| duration | Duration (pretty format) | 03:07 |
| sample_width | Width of samples (pixels) | 1920 |
| sample_height | Height of samples (pixels) | 1080 |
| display_width | Display width (pixels) | 1920 |
| display_height | Display height (pixels) | 1080 |
| video_codec | Video codec | h264 |
| video_codec_long | Video codec (long name) | H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10 |
| video_bit_rate | Video bitrate | 371006 |
| display_aspect_ratio | Display aspect ratio | 16:9 |
| sample_aspect_ratio | Sample aspect ratio | 1:1 |
| audio_codec | Audio codec | aac |
| audio_codec_long | Audio codec (long name) | AAC (Advanced Audio Coding) |
| audio_sample_rate | Audio sample rate (Hz) | 44100 |
| audio_bit_rate | Audio bit rate | 192000 |
| frame_rate | Frame rate (fps) | 23.974 |


## Testing

To run the test suite, run:

```
uv run pytest
```

To measure code coverage:

```
uv run pytest --cov=vcsi.vcsi
```
