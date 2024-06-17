#!/bin/bash
cp -r tutorials/bias docs/source/gallery/tutorials/
cp -r tutorials/datasets docs/source/gallery/tutorials/

PYTHONPATH=./src sphinx-build -b html docs/source docs/build