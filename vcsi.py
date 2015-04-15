#!/usr/bin/env python3


__author__ = "Nils Amiet"


import subprocess
import argparse
import json
import math
import os


class MediaInfo():
	"""Collect information about a video file"""

	def __init__(self, path, verbose=True):
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
		for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
			if abs(num) < 1024.0:
				return "%3.1f %s%s" % (num, unit, suffix)
			num /= 1024.0
		return "%.1f %s%s" % (num, 'Yi', suffix)


	def find_video_stream(self):
		for stream in self.ffprobe_dict["streams"]:
			try:
				if stream["codec_type"] == "video":
					self.video_stream = stream
			except:
				pass


	def compute_display_resolution(self):
		width = int(self.video_stream["width"])
		height = int(self.video_stream["height"])
		self.sample_width = width
		self.sample_height = height
		sample_aspect_ratio = self.video_stream["sample_aspect_ratio"]

		if sample_aspect_ratio is not "1:1":
			sample_split = sample_aspect_ratio.split(":")
			sw = int(sample_split[0])
			sh = int(sample_split[1])

			self.display_width = int(width * sw / sh)
			self.display_height = int(height)
		else:
			self.display_width = width
			self.display_height = height


	def compute_format(self):
		format_dict = self.ffprobe_dict["format"]
		
		self.duration_seconds = float(format_dict["duration"])
		self.compute_pretty_duration(self.duration_seconds)		

		self.filename = os.path.basename(format_dict["filename"])
		
		size_bytes = int(format_dict["size"])
		self.size = self.human_readable_size(size_bytes)


	def compute_pretty_duration(self, seconds):
		hours = math.floor(seconds / 3600)
		remaining_seconds = seconds - 3600 * hours

		minutes = math.floor(remaining_seconds / 60)
		remaining_seconds = round(remaining_seconds - 60 * minutes)

		self.duration = ""

		if hours > 0:
			self.duration +=  "%s:" % (hours,)

		self.duration += "%s:%s" % (minutes, remaining_seconds)




class MediaCapture():

	def __init__(self, path):
		pass

	def make_capture(path, time):
		ffmpeg_command = []






def main():
	test_path = "/mnt/wdsix/videos/korean-jap/Perfume - Relax In The City [1440x1080 h264 M-ON! HD].ts"
	test_path2 = "/mnt/wdsix/videos/korean-jap/SBS Gayo Daejeon - 2013.12.29.tp"

	media_info = MediaInfo(test_path)


if __name__ == "__main__":
	main()