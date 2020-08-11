import wave
import os
import sys
from glob import glob


unit_time_length = 30

try:
    time_offset = sys.argv[1]
except IndexError:
    start_time_offset = 15

frame_array = []
params = None
data_raw = None

# downloads_dir = os.path.join(".", "downloads")
# target_files = glob(os.path.join(downloads_dir, "*.wav"))
# print(target_files)

with wave.open("./downloads/game.mp4.wav", "rb") as wave_read:
    params = wave_read.getparams()
    data_raw = wave_read.readframes(params.nframes)

unit_nframes = unit_time_length * params.framerate * params.nchannels * params.sampwidth
start_frame_offset = start_time_offset * params.framerate * params.nchannels * params.sampwidth

output_dir = os.path.join(".", "output")
os.makedirs(output_dir, exist_ok=True)

file_count = 0
for t in range(0, len(data_raw), start_frame_offset):
    file_count += 1
    picked_data = data_raw[t:t+unit_nframes]
    output_filename = os.path.join(output_dir, f"{file_count:09}.wav")
    with wave.open(output_filename, "wb") as wave_write:
        wave_write.setparams((
            params.nchannels, params.sampwidth, params.framerate,
            len(picked_data), params.comptype, params.compname
        ))
        wave_write.writeframes(picked_data)
        wave_write.close()
