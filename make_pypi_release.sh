#!/usr/bin/env bash

rm -rf dist
uv build
source settings.env
uv publish --token $PYPI_TOKEN
