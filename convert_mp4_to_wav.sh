#!/bin/sh

for mp4_file in ./downloads/*.mp4; do
    ffmpeg -ac 1 -i "${mp4_file}" "${mp4_file}.wav"
done
