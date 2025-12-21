#!/usr/bin/env bash

rm -rf dist
uv build
uv publish
