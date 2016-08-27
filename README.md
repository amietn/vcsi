# vcsi

[![Build Status](https://travis-ci.org/amietn/vcsi.svg?branch=master)](https://travis-ci.org/amietn/vcsi)
[![Coverage Status](https://coveralls.io/repos/amietn/vcsi/badge.svg)](https://coveralls.io/r/amietn/vcsi)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](http://opensource.org/licenses/MIT)

Create video contact sheets. A video contact sheet is an image composed of video capture thumbnails arranged on a grid.

## Examples

```
$ vcsi bbb_sunflower_1080p_60fps_normal.mp4 \
-t \
-w 725 \
-g 4x4 \
--background-color 000000 \
--metadata-font-color ffffff \
--end-delay-percent 20
```
![Image](<http://i.imgur.com/kEgQ4xl.png>)

```
$ vcsi bbb_sunflower_1080p_60fps_normal.mp4 \
-t \
-w 725 \
-g 3x5 \
--end-delay-percent 20 \
-o output.png
```
![Image](<http://i.imgur.com/nnDPpiJ.jpg>)


The above contact sheets were generated from a movie called "Big Buck Bunny".

## Installation

vcsi is currently packaged for the following systems:

| Linux packages | |
| -------------- | --- |
| Arch (AUR) | https://aur.archlinux.org/packages/vcsi/ |
| Arch (AUR, git master) | https://aur.archlinux.org/packages/vcsi-git/ |

Your system is not listed?

```
$ apt-get install ffmpeg
$ python setup.py install
```

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


Must be in your PATH:

* ffmpeg
* ffprobe


## Usage

```
$ vcsi -h
usage: vcsi [-h] [-o OUTPUT_PATH] [-n NUM_FRAMES]
            [--start-delay-percent START_DELAY_PERCENT]
            [--end-delay-percent END_DELAY_PERCENT]
            [--delay-percent DELAY_PERCENT] [--grid-spacing GRID_SPACING]
            [--grid-horizontal-spacing GRID_HORIZONTAL_SPACING]
            [--grid-vertical-spacing GRID_VERTICAL_SPACING] [-w VCS_WIDTH]
            [-g MXN] [-s NUM_SAMPLES] [-t]
            [--metadata-font-size METADATA_FONT_SIZE]
            [--metadata-font METADATA_FONT]
            [--timestamp-font-size TIMESTAMP_FONT_SIZE]
            [--timestamp-font TIMESTAMP_FONT] [-v]
            filename

Create a video contact sheet

positional arguments:
  filename

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT_PATH, --output OUTPUT_PATH
                        save to output file
  -n NUM_FRAMES, --num-frames NUM_FRAMES
                        capture n frames
  --start-delay-percent START_DELAY_PERCENT
                        do not capture frames in the first n percent of total
                        time
  --end-delay-percent END_DELAY_PERCENT
                        do not capture frames in the last n percent of total
                        time
  --delay-percent DELAY_PERCENT
                        do not capture frames in the first and last n percent
                        of total time
  --grid-spacing GRID_SPACING
                        number of pixels spacing captures both vertically and
                        horizontally
  --grid-horizontal-spacing GRID_HORIZONTAL_SPACING
                        number of pixels spacing captures horizontally
  --grid-vertical-spacing GRID_VERTICAL_SPACING
                        number of pixels spacing captures vertically
  -w VCS_WIDTH, --width VCS_WIDTH
                        width of the generated contact sheet
  -g MXN, --grid MXN    display frames on a mxn grid (for example 4x5)
  -s NUM_SAMPLES, --num-samples NUM_SAMPLES
                        number of samples
  -t, --show-timestamp  display timestamp for each frame
  --metadata-font-size METADATA_FONT_SIZE
                        size of the font used for metadata
  --metadata-font METADATA_FONT
                        TTF font used for metadata
  --timestamp-font-size TIMESTAMP_FONT_SIZE
                        size of the font used for timestamps
  --timestamp-font TIMESTAMP_FONT
                        TTF font used for timestamps
  -v, --verbose         display verbose messages
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
| display_aspect_ratio | Display aspect ratio | 16:9 |
| sample_aspect_ratio | Sample aspect ratio | 1:1 |
| audio_codec | Audio codec | aac |
| audio_codec_long | Audio codec (long name) | AAC (Advanced Audio Coding) |
| audio_sample_rate | Audio sample rate (Hz) | 44100 |
| audio_bit_rate | Audio bit rate | 192000 |
| frame_rate | Frame rate (fps) | 23.974 |
