import wave
import os
import sys
from glob import glob
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--length", type=int, default=30)
parser.add_argument("--offset", type=int, default=15)

args = parser.parse_args()
unit_time_length = args.length
start_time_offset = args.offset

output_dir = os.path.join(".", "output")
os.makedirs(output_dir, exist_ok=True)

downloads_dir = os.path.join(".", "downloads")
target_files = glob(os.path.join(downloads_dir, "*.wav"))

for base_filepath in target_files:
    base_filename = os.path.basename(base_filepath)
    print(f"Processing for {base_filename}...")
    params = None
    data_raw = None
    with wave.open(base_filepath, "rb") as wave_read:
        params = wave_read.getparams()
        data_raw = wave_read.readframes(params.nframes)
        wave_read.close()

    unit_nframes = unit_time_length * params.framerate * params.nchannels * params.sampwidth
    start_frame_offset = start_time_offset * params.framerate * params.nchannels * params.sampwidth

    file_count = 0
    for t in range(0, len(data_raw), start_frame_offset):
        file_count += 1
        picked_data = data_raw[t:t+unit_nframes]
        output_filename = os.path.join(output_dir, f"s{base_filename}_{file_count:09}.wav")
        with wave.open(output_filename, "wb") as wave_write:
            wave_write.setparams((
                params.nchannels, params.sampwidth, params.framerate,
                len(picked_data), params.comptype, params.compname
            ))
            wave_write.writeframes(picked_data)
            wave_write.close()
    # os.remove(base_filepath)
print("Done.")
