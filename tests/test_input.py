from argparse import ArgumentTypeError

from nose.tools import assert_raises
from nose.tools import assert_equals

from vcsi.vcsi import Grid, mxn_type


def test_grid_default():
    test_grid = mxn_type('4x4')

    assert_equals(test_grid.x, 4)
    assert_equals(test_grid.y, 4)


def test_grid_columns_integer():
    assert_raises(ArgumentTypeError, mxn_type, 'ax4')

    assert_raises(ArgumentTypeError, mxn_type, '4.1x4')


def test_grid_columns_positive():
    assert_raises(ArgumentTypeError, mxn_type, '-1x4')

    assert_raises(ArgumentTypeError, mxn_type, '0x4')


def test_grid_rows_integer():
    assert_raises(ArgumentTypeError, mxn_type, '4xa')

    assert_raises(ArgumentTypeError, mxn_type, '4x4.1')

    test_grid = mxn_type('4x-1')

    assert_equals(test_grid.x, 4)
    assert_equals(test_grid.y, -1)

    assert_raises(ArgumentTypeError, mxn_type, '4x1x4')


def test_grid_format():
    assert_raises(ArgumentTypeError, mxn_type, '')

    assert_raises(ArgumentTypeError, mxn_type, '4xx4')

    assert_raises(ArgumentTypeError, mxn_type, '4')
