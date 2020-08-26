import librosa
import matplotlib.pyplot as plt
import librosa.display
import glob
import os
path = "data_spectrum/*"
paths = glob.glob(path)
number = 0
print(paths)
for path in paths:
    audio_data, sr = librosa.load(path)
    S = librosa.feature.melspectrogram(audio_data, sr=sr, n_mels=128)
    log_S = librosa.power_to_db(S)
    plt.figure(figsize=(8, 8))
    librosa.display.specshow(data=S, sr=sr, hop_length=2068)
    # plt.show()
    os.makedirs("result_spectrum", exist_ok=True)
    plt.savefig("result_spectrum/result{}.png".format(number))
    number += 1
    print(number)
