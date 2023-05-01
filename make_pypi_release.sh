#!/usr/bin/env bash

rm -rf dist
poetry build
poetry publish
