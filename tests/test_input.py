from argparse import ArgumentTypeError
from unittest.mock import patch

from nose.tools import assert_equals
from nose.tools import assert_not_equals
from nose.tools import assert_raises

from vcsi.vcsi import Grid, mxn_type, Color, hex_color_type, manual_timestamps, timestamp_position_type, \
    TimestampPosition, comma_separated_string_type, metadata_position_type
from vcsi import vcsi


def test_grid_default():
    test_grid = mxn_type('4x4')

    assert_equals(test_grid.x, 4)
    assert_equals(test_grid.y, 4)


def test_grid_equality():
    g1 = Grid(4, 4)
    g2 = Grid(4, 4)
    assert_equals(g1, g2)


def test_grid_inequality():
    g1 = Grid(4, 4)
    g2 = Grid(3, 4)
    assert_not_equals(g1, g2)


def test_grid_columns_integer():
    assert_raises(ArgumentTypeError, mxn_type, 'ax4')

    assert_raises(ArgumentTypeError, mxn_type, '4.1x4')


def test_grid_columns_positive():
    assert_raises(ArgumentTypeError, mxn_type, '-1x4')


def test_grid_rows_integer():
    assert_raises(ArgumentTypeError, mxn_type, '4xa')

    assert_raises(ArgumentTypeError, mxn_type, '4x4.1')


def test_grid_rows_positive():
    assert_raises(ArgumentTypeError, mxn_type, '4x-1')


def test_grid_format():
    assert_raises(ArgumentTypeError, mxn_type, '')

    assert_raises(ArgumentTypeError, mxn_type, '4xx4')

    assert_raises(ArgumentTypeError, mxn_type, '4x1x4')

    assert_raises(ArgumentTypeError, mxn_type, '4')


def test_hex_color_type():
    assert_equals(Color(*(0x10, 0x10, 0x10, 0xff)), hex_color_type("101010"))

    assert_equals(Color(*(0x10, 0x10, 0x10, 0x00)), hex_color_type("10101000"))

    assert_equals(Color(*(0xff, 0xff, 0xff, 0xff)), hex_color_type("ffffff"))

    assert_equals(Color(*(0xff, 0xff, 0xff, 0x00)), hex_color_type("ffffff00"))

    assert_raises(ArgumentTypeError, hex_color_type, "abcdeff")

    assert_raises(ArgumentTypeError, hex_color_type, "abcdfg")


def test_manual_timestamps():
    assert_equals(manual_timestamps("1:11:11.111,2:22:22.222"), ["1:11:11.111", "2:22:22.222"])

    assert_raises(ArgumentTypeError, manual_timestamps, "1:11:a1.111,2:22:b2.222")

    assert_raises(ArgumentTypeError, manual_timestamps, "1:1:1:1.111,2:2.222")

    assert_equals(manual_timestamps(""), [])


def test_timestamp_position_type():
    assert_equals(timestamp_position_type("north"), TimestampPosition.north)

    assert_not_equals(timestamp_position_type("south"), TimestampPosition.north)

    assert_raises(ArgumentTypeError, timestamp_position_type, "whatever")


@patch("vcsi.vcsi.parsedatetime")
def test_interval_type(mocked_parsedatatime):
    mocked_parsedatatime.return_value = 30
    assert_equals(mocked_parsedatatime("30 seconds"), 30)

    mocked_parsedatatime.assert_called_once_with("30 seconds")


def test_comma_separated_string_type():
    assert_equals(comma_separated_string_type("a, b, c"), ["a", "b", "c"])

    assert_equals(comma_separated_string_type("a b, c"), ["a b", "c"])


def test_metadata_position_type():
    assert_equals(metadata_position_type("top"), "top")

    assert_equals(metadata_position_type("TOP"), "top")

    assert_raises(ArgumentTypeError, metadata_position_type, "whatever")
