#!/bin/sh


for midi_file in ./downloads/*.mid; do
    fluidsynth -F "${midi_file}.wav" -i "./soundfont.sf2" "${midi_file}"
done
