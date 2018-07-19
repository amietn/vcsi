#!/usr/bin/env bash

rm -rf vcsi.egg-info
rm -rf build
rm -rf dist
python3 setup.py sdist bdist_wheel
twine upload dist/*

