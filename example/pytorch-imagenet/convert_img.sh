#!/bin/bash

find . -name "*.JPEG" | xargs -I {} convert {} -resize "256^>" {}