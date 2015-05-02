# vcsi

Create video contact sheets. A video contact sheet is an image composed of video capture thumbnails arranged on a grid.

## Examples

```
$ vcsi bbb_sunflower_1080p_60fps_normal.mp4 -t -n 3 -s 10 -w 400
```
![Image](<http://i.imgur.com/HI6gg8f.png>)

```
$ vcsi bbb_sunflower_1080p_60fps_normal.mp4 -t -w 725 -g 3x5 -o output.png
```
![Image](<http://i.imgur.com/CeG0fY2.png>)

```
$ vcsi bbb_sunflower_1080p_60fps_normal.mp4 -w 725 -n 4 -s 12 --end-delay-percent 20

```
![Image](<http://i.imgur.com/omUpSfq.jpg>)

The above contact sheets were generated from a movie called "Big Buck Bunny".

## Installation

vcsi is currently available for the following systems:

| Linux packages | |
| -------------- | --- |
| Arch (AUR) | https://aur.archlinux.org/packages/vcsi/ |

Your system is not listed?

```
$ apt-get install ffmpeg
$ python setup.py install
```

Running Windows? See the note below.


## Note for Windows users

Download a binary build of ffmpeg from Zeranoe here (e.g. 64bit static): http://ffmpeg.zeranoe.com/builds/

Extract the archive and add the "bin" folder to your PATH so that "ffmpeg" and "ffprobe" can be invoked from the command line.

If you have issues installing numpy with pip, download an already built version of numpy here: http://sourceforge.net/projects/numpy/files/NumPy/


## Requirements

Python modules:

* numpy
* pillow


Must be in PATH:

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
