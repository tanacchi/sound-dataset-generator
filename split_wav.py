import wave
import sys


unit_time_length = 30

try:
    time_offset = sys.argv[1]
except IndexError:
    time_offset = 15

frame_array = []
params = None
data_raw = None

with wave.open("./downloads/game.wav", "rb") as wave_read:
    params = wave_read.getparams()
    data_raw = wave_read.readframes(params.nframes)

print(params)
frame_offset = unit_time_length * params.nframes
