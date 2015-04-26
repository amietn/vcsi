# vcsi

Create a video contact sheet

## Examples

TODO

## Requirements

Python modules:

* numpy
* pillow


Must be in PATH:

* ffmpeg
* ffprobe

```
pip install -r requirements.txt
apt-get install ffmpeg
```

## Usage

$ vcsi -h
usage: vcsi [-h] [-o OUTPUT_PATH] [-n NUM_FRAMES] [-w VCS_WIDTH] [-g MXN]
            [-s NUM_SAMPLES] [-t] [-v]
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
  -w VCS_WIDTH, --width VCS_WIDTH
                        width of the generated contact sheet
  -g MXN, --grid MXN    display frames on a mxn grid (for example 4x5)
  -s NUM_SAMPLES, --num-samples NUM_SAMPLES
                        number of samples
  -t, --show-timestamp  display timestamp for each frame
  -v, --verbose         display verbose messages
