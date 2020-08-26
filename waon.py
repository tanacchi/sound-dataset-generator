import os
import librosa
import matplotlib.pyplot as plt
import librosa.display
import glob
import numpy as np
path = "data_waon/*"
paths = glob.glob(path)
number = 0
print(paths)
for path in paths:
    y, sr = librosa.load(path)
    # S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
    # log_S = librosa.power_to_db(S)
    n_bins = 84
    hop_length = 1024
    pitch = np.abs(librosa.cqt(y=y, sr=sr, hop_length=hop_length, n_bins=n_bins))
    chroma = librosa.feature.chroma_cqt(C=pitch)
    plt.figure(figsize=(8, 8))
    librosa.display.specshow(data=chroma, sr=sr, hop_length=2068)
    # plt.show()
    # plt.savefig("result/result{}.png".format(number))
    os.makedirs("result_waon", exist_ok=True)
    plt.savefig("result_waon/result{}.png".format(number))
    number += 1
    print(number)
# dataに入っているデータを捜査して一括で処理をする
# data → result
