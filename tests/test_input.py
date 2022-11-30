from argparse import ArgumentTypeError
from unittest.mock import patch, Mock, PropertyMock, MagicMock

import pytest

from vcsi.vcsi import Grid, mxn_type, Color, hex_color_type, manual_timestamps, timestamp_position_type, \
    TimestampPosition, comma_separated_string_type, metadata_position_type, cleanup, save_image,\
    compute_timestamp_position, max_line_length, draw_metadata
from vcsi import vcsi


def test_grid_default():
    test_grid = mxn_type('4x4')

    assert test_grid.x == 4
    assert test_grid.y == 4


def test_grid_equality():
    g1 = Grid(4, 4)
    g2 = Grid(4, 4)
    assert g1 == g2


def test_grid_inequality():
    g1 = Grid(4, 4)
    g2 = Grid(3, 4)
    assert g1 != g2


def test_grid_columns_integer():
    pytest.raises(ArgumentTypeError, mxn_type, 'ax4')

    pytest.raises(ArgumentTypeError, mxn_type, '4.1x4')


def test_grid_columns_positive():
    pytest.raises(ArgumentTypeError, mxn_type, '-1x4')


def test_grid_rows_integer():
    pytest.raises(ArgumentTypeError, mxn_type, '4xa')

    pytest.raises(ArgumentTypeError, mxn_type, '4x4.1')


def test_grid_rows_positive():
    pytest.raises(ArgumentTypeError, mxn_type, '4x-1')


def test_grid_format():
    pytest.raises(ArgumentTypeError, mxn_type, '')

    pytest.raises(ArgumentTypeError, mxn_type, '4xx4')

    pytest.raises(ArgumentTypeError, mxn_type, '4x1x4')

    pytest.raises(ArgumentTypeError, mxn_type, '4')


def test_hex_color_type():
    assert Color(*(0x10, 0x10, 0x10, 0xff)) == hex_color_type("101010")

    assert Color(*(0x10, 0x10, 0x10, 0x00)) == hex_color_type("10101000")

    assert Color(*(0xff, 0xff, 0xff, 0xff)) == hex_color_type("ffffff")

    assert Color(*(0xff, 0xff, 0xff, 0x00)) == hex_color_type("ffffff00")

    pytest.raises(ArgumentTypeError, hex_color_type, "abcdeff")

    pytest.raises(ArgumentTypeError, hex_color_type, "abcdfg")


def test_manual_timestamps():
    assert manual_timestamps("1:11:11.111,2:22:22.222") == ["1:11:11.111", "2:22:22.222"]

    pytest.raises(ArgumentTypeError, manual_timestamps, "1:11:a1.111,2:22:b2.222")

    pytest.raises(ArgumentTypeError, manual_timestamps, "1:1:1:1.111,2:2.222")

    assert manual_timestamps("") == []


def test_timestamp_position_type():
    assert timestamp_position_type("north") == TimestampPosition.north

    assert timestamp_position_type("south") != TimestampPosition.north

    pytest.raises(ArgumentTypeError, timestamp_position_type, "whatever")


@patch("vcsi.vcsi.parsedatetime")
def test_interval_type(mocked_parsedatatime):
    mocked_parsedatatime.return_value = 30
    assert mocked_parsedatatime("30 seconds") == 30

    mocked_parsedatatime.assert_called_once_with("30 seconds")


def test_comma_separated_string_type():
    assert comma_separated_string_type("a, b, c") == ["a", "b", "c"]

    assert comma_separated_string_type("a b, c") == ["a b", "c"]


def test_metadata_position_type():
    assert metadata_position_type("top") == "top"

    assert metadata_position_type("TOP") == "top"

    pytest.raises(ArgumentTypeError, metadata_position_type, "whatever")


@patch("vcsi.vcsi.os")
def test_cleanup(mocked_os):
    mocked_os.unlink.side_effect = lambda x: True
    args = Mock()
    args.is_verbose = False
    frames = [Mock()]
    frames[0].filename = "frame1"
    cleanup(frames, args)

    mocked_os.unlink.assert_called_once_with("frame1")


@patch("vcsi.vcsi.Image")
def test_save_image(mocked_Image):
    args = PropertyMock()
    output_path = "whatever"
    assert True == save_image(args, mocked_Image, None, output_path)

    mocked_Image.convert().save.assert_called_once()


def test_compute_timestamp_position():
    args = PropertyMock()
    args.timestamp_horizontal_margin = 10
    args.timestamp_vertical_margin = 10
    w, h = 1000, 500
    text_size = (10, 10)
    desired_size = (20, 20)
    rectangle_hpadding, rectangle_vpadding = 5, 5

    args.timestamp_position = TimestampPosition.west
    assert (((1010, 500.0), (1030, 520.0)) ==
                  compute_timestamp_position(args, w, h, text_size, desired_size, rectangle_hpadding,
                                             rectangle_vpadding))

    args.timestamp_position = TimestampPosition.north
    assert (((1000, 510.0), (1020, 530.0)) ==
                  compute_timestamp_position(args, w, h, text_size, desired_size, rectangle_hpadding,
                                             rectangle_vpadding))

    args.timestamp_position = None
    assert (((990, 490), (1010, 510)) ==
                  compute_timestamp_position(args, w, h, text_size, desired_size, rectangle_hpadding,
                                             rectangle_vpadding))


def test_max_line_length():
    media_info = PropertyMock()
    metadata_font = PropertyMock()
    metadata_font.getsize.return_value = (40, 40)
    header_margin = 100
    width = 1000

    text = "A long line of text"
    assert 19 == max_line_length(media_info, metadata_font, header_margin, width, text)

    text = "A long line of text with a few more words"
    assert 41 == max_line_length(media_info, metadata_font, header_margin, width, text)

    text = "A really long line of text with a lots more words" * 100
    assert 4900 == max_line_length(media_info, metadata_font, header_margin, width, text)

    text = None
    filename = PropertyMock()
    type(media_info).filename = filename
    # media_info.filename = filename
    assert 0 == max_line_length(media_info, metadata_font, header_margin, width, text)
    filename.assert_called_once()


def test_draw_metadata():
    args = Mock()
    draw = Mock()
    header_lines = MagicMock()
    draw.text.return_value = 0
    args.metadata_vertical_margin = 0
    header_lines.__iter__ = Mock(return_value=iter(['text1', 'text2']))

    assert 0 == draw_metadata(draw, args,
                                   header_lines=header_lines,
                                   header_line_height=0,
                                   start_height=0)
    draw.text.assert_called()


def test_grid():
    assert "2x2" == Grid(2, 2).__str__()

    assert "10x0" == Grid(10, 0).__str__()


def test_color():
    assert "FFFFFFFF" == Color(255, 255, 255, 255).__str__()

def test_MediaCapture():
    test_args = {"path": "path",
                 "accurate": False,
                 "skip_delay_seconds": 12,
                 "frame_type": "frame_type"
                 }
    media_capture = MediaCapture(**test_args)
    assert "path" == media_capture.path
    assert not media_capture.accurate
    assert 12 == media_capture.skip_delay_seconds
    assert "frame_type" == media_capture.frame_type


@patch("vcsi.vcsi.subprocess.call")
def test_make_capture(mocked_subprocess_call):
    mocked_subprocess_call.return_value = True
    test_args = {"path": "path",
                 "accurate": False,
                 "skip_delay_seconds": 12,
                 "frame_type": "frame_type"
                 }
    media_capture = MediaCapture(**test_args)
    media_capture.make_capture("1.990", 1090, 1080)
    mocked_subprocess_call.assert_called_once()
    ffmpeg_call_arguments = ['ffmpeg', '-ss', "1.990", '-i', 'path', '-vframes', '1', '-s', '1090x1080', '-vf',
                             "select='eq(frame_type\\,frame_type)'", '-y', 'out.png']
    subprocess_call_args = {
        "stderr": -3,
        "stdin": -3,
        "stdout": -3
    }
    mocked_subprocess_call.assert_called_once_with(ffmpeg_call_arguments, **subprocess_call_args)


@patch("vcsi.vcsi.subprocess.call")
def test_make_capture_accurate(mocked_subprocess_call):
    mocked_subprocess_call.return_value = True
    test_args = {"path": "path",
                 "accurate": True,
                 "skip_delay_seconds": 12,
                 "frame_type": "frame_type"
                 }
    media_capture = MediaCapture(**test_args)
    media_capture.make_capture("1.990", 1090, 1080)
    mocked_subprocess_call.assert_called_once()
    ffmpeg_call_arguments = ['ffmpeg', '-i', 'path', '-ss', '1.990', '-vframes', '1', '-s', '1090x1080', '-vf',
                             "select='eq(frame_type\\,frame_type)'", '-y', 'out.png']
    subprocess_call_args = {
        "stderr": -3,
        "stdin": -3,
        "stdout": -3
    }
    mocked_subprocess_call.assert_called_once_with(ffmpeg_call_arguments, **subprocess_call_args)


@patch("vcsi.vcsi.subprocess.call")
def test_make_capture_accurate2(mocked_subprocess_call):
    mocked_subprocess_call.return_value = True
    test_args = {"path": "path",
                 "accurate": True,
                 "skip_delay_seconds": 12,
                 "frame_type": "frame_type"
                 }
    media_capture = MediaCapture(**test_args)
    media_capture.make_capture("15.990", 1090, 1080)
    mocked_subprocess_call.assert_called_once()
    ffmpeg_call_arguments = ['ffmpeg', '-ss', '00:03.990', '-i', 'path', '-ss', '00:12.000', '-vframes', '1',
                             '-s', '1090x1080', '-vf', "select='eq(frame_type\\,frame_type)'", '-y', 'out.png']
    subprocess_call_args = {
        "stderr": -3,
        "stdin": -3,
        "stdout": -3
    }
    mocked_subprocess_call.assert_called_once_with(ffmpeg_call_arguments, **subprocess_call_args)


@patch("vcsi.vcsi.Image")
def test_compute_avg_color(mocked_image):
    mockedimg = MagicMock()
    mockedimg.convert.return_value = mockedimg
    mockedimg.getcolors.return_value = [(1, 254), (1, 0)]
    mocked_image.open.return_value = mockedimg
    test_args = {"path": "path",
                 "accurate": False,
                 "skip_delay_seconds": 12,
                 "frame_type": "frame_type"
                 }
    media_capture = MediaCapture(**test_args)
    assert 127 == media_capture.compute_avg_color("whatever_image")


@patch("vcsi.vcsi.Image")
@patch("vcsi.vcsi.numpy")
@patch.object(MediaCapture, "avg9x")
def test_compute_blurriness(mocked_avg9x, mocked_numpy, mocked_image):
    mockedimg = MagicMock()
    mockedimg.convert.return_value = mockedimg
    mocked_image.open = mockedimg

    mocked_numpy.asarray.return_value = [0, 0, 0, 0, 0]
    mocked_numpy.fft.rfft2.return_value = 0

    test_args = {"path": "path",
                 "accurate": False,
                 "skip_delay_seconds": 12,
                 "frame_type": "frame_type"
                 }
    media_capture = MediaCapture(**test_args)
    media_capture.compute_blurriness("whatever filepath")

    mocked_avg9x.verify_called_once()
    mocked_numpy.fft.rfft2.verify_called_once()
    mocked_image.open.verify_called_once_with("whatever filename")


@patch("vcsi.vcsi.numpy")
def test_avg9x(mocked_numpy):
    test_args = {"path": "path",
                 "accurate": False,
                 "skip_delay_seconds": 12,
                 "frame_type": "frame_type"
                 }
    media_capture = MediaCapture(**test_args)
    test_args_matrix = MagicMock()
    test_args_matrix.flatten.return_value = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    mocked_numpy.median.return_value = 0

    assert 0 == media_capture.avg9x(test_args_matrix)
    mocked_numpy.median.assert_called_once()
    test_args_matrix.flatten.assert_called_once()


def test_max_freq():
    test_args = {"path": "path",
                 "accurate": False,
                 "skip_delay_seconds": 12,
                 "frame_type": "frame_type"
                 }
    media_capture = MediaCapture(**test_args)
    test_args_matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    assert 9 == media_capture.max_freq(test_args_matrix)


@patch.object(vcsi, "grid_desired_size")
@patch.object(vcsi, "select_color_variety")
@patch.object(vcsi, "timestamp_generator")
@patch("vcsi.vcsi.os")
def test_select_sharpest_images(mocked_os, mocked_timestamp_generator, mocked_colour_variety, mocked_grid_desired_size):
    mocked_timestamp_generator.return_value = [(1.9, 1.9), (2.0, 2.0)]

    mocked_args = MagicMock()
    mocked_args.num_groups = 2
    mocked_args.manual_timestamps = None
    mocked_media_capture = MagicMock()
    vcsi.select_sharpest_images(MagicMock(), mocked_media_capture, mocked_args)

    mocked_media_capture.make_capture.assert_called()
    mocked_colour_variety.assert_called_once()
    mocked_grid_desired_size.assert_called_once()
    mocked_os.close.assert_called()

    mocked_args.fast = False
    vcsi.select_sharpest_images(MagicMock(), mocked_media_capture, mocked_args)
    mocked_media_capture.make_capture.assert_called()
    mocked_media_capture.compute_blurriness.assert_called()
    mocked_media_capture.compute_avg_color.assert_called()
    mocked_os.close.assert_called()
    mocked_colour_variety.assert_called()
    mocked_grid_desired_size.assert_called()


def test_select_color_variety():
    frame = MagicMock(spec=Frame)
    frame.filename = "whateverfilename"
    frame.blurriness = 0.1
    frame.timestamp = "1.990"
    frame.avg_color = 0x000000
    frames = [frame] * 4
    num_selected = 2
    assert 4 == len(vcsi.select_color_variety(frames, num_selected))


# @patch("builtins.open", new_callable=mock_open, read_data="""{{filename}}
#         File size: {{size}}
#         Duration: {{duration}}
#         Dimensions: {{sample_width}}x{{sample_height}}""")
# @patch("vcsi.vcsi.textwrap")
# def test_prepare_metadata_text_lines_with_template(mocked_textwrap, mocked_open):
#     mocked_textwrap.wrap.side_effect = lambda x, y: x[:4]
#     test_args = {"media_info": MagicMock(),
#                  "header_font": "font",
#                  "header_margin": 12,
#                  "width": 10,
#                  "template_path": "whateverpath"
#                  }
#     prepare_metadata_text_lines(**test_args)
#     mocked_textwrap.wrap.assert_called()
#     mocked_open.assert_called_once_with("whateverpath")
#     mocked_open().read.assert_called_once()


@patch("vcsi.vcsi.textwrap")
def test_prepare_metadata_text_lines_with_no_template(mocked_textwrap):
    mocked_textwrap.wrap.side_effect = lambda x, y: x[:4]
    test_args = {"media_info": MagicMock(),
                 "header_font": "font",
                 "header_margin": 12,
                 "width": 10,
                 }
    prepare_metadata_text_lines(**test_args)
    mocked_textwrap.wrap.assert_called()


@patch("vcsi.vcsi.ImageFont")
def test_load_font_not_default(mocked_imagefont):
    args = MagicMock()
    args.is_verbose = False
    test_args = {
        'args': args,
        'font_path': ['whateverfontpath'] * 2,
        'font_size': 12,
        'default_font_path': 'default_fontpath'
    }
    load_font(**test_args)
    mocked_imagefont.truetype.assert_called()


@patch("vcsi.vcsi.ImageFont")
@patch("vcsi.vcsi.os")
def test_load_font_detault(mocked_os, mocked_imagefont):
    mocked_os.path.exists.return_value = True
    args = MagicMock()
    args.is_verbose = False
    test_args = {
        'args': args,
        'font_path': 'default_fontpath',
        'font_size': 12,
        'default_font_path': 'default_fontpath'
    }
    load_font(**test_args)
    mocked_imagefont.truetype.assert_called()


@patch("vcsi.vcsi.argparse.ArgumentParser")
@patch("vcsi.vcsi.process_file")
@patch("vcsi.vcsi.os")
def test_main(mocked_os, mocked_process_file, mocked_argparser):
    mocked_os.walk.return_value = [('root', ['subdir'], ['files'] * 2)]
    mocked_argparser.return_value.parse_known_args.return_value = (MagicMock(name="parse_known_Args"), None)
    mocked_argparser.return_value.parse_args.return_value.list_template_attributes = False
    mocked_argparser.return_value.parse_args.return_value.filenames = ['whateverfile'] * 2

    main()
    mocked_argparser.return_value.add_argument.assert_called()
    mocked_process_file.assert_called()

    mocked_argparser.return_value.parse_args.return_value.recursive = False
    main()
    mocked_argparser.return_value.add_argument.assert_called()
    mocked_process_file.assert_called()


@patch("vcsi.vcsi.MediaInfo")
@patch("vcsi.vcsi.os")
@patch("vcsi.vcsi.select_sharpest_images")
@patch("vcsi.vcsi.load_font")
@patch("vcsi.vcsi.prepare_metadata_text_lines")
@patch("vcsi.vcsi.Image")
@patch("vcsi.vcsi.ImageDraw")
@patch("vcsi.vcsi.compose_contact_sheet")
@patch("vcsi.vcsi.save_image")
@patch("vcsi.vcsi.shutil")
@patch("builtins.sorted")
def test_process_file(mock_sorted, mock_shutils, mock_save_image, mock_compose_contact_sheet, mockImageDraw, mockImage,
                      mocked_prepare_text_lines, mocked_load_font,
                      mocked_select_sharpest_image, mocked_os, mocked_mediainfo):
    mocked_select_sharpest_image.return_value = ([MagicMock()] * 2, MagicMock())
    mock_arg = MagicMock()
    mock_arg.is_verbose = False
    test_args = {
        'path': 'filepath',
        'args': mock_arg
    }
    mocked_os.path.exists.return_value = True

    # With overwrite enabled
    process_file(**test_args)

    # With overwrite disabled
    mock_arg.no_overwrite = False
    mock_interval = Mock()
    mock_interval.total_seconds.return_value = 20
    mock_arg.interval = mock_interval
    mock_arg.manual_timestamps = None
    mock_arg.vcs_width = 1080
    mock_arg.actual_size = None
    process_file(**test_args)
    mocked_mediainfo.assert_called()


