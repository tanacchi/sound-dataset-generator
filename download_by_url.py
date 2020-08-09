from apiclient.discovery import build
from api_key import api_key
from pytube import YouTube
from pprint import pprint
import sys


try:
    url = sys.argv[1]
except IndexError as e:
    print("Please set URL on first argument.")
    exit(1)

    baseurl = 'https://www.youtube.com/watch?v='
    if not url[:len(baseurl)] == baseurl:
        raise RuntimeError("Invalid URL")

try:
    print(f"Downloading from '{url}'")
    stream = YouTube(url).streams
    stream.first().download("downloads")
    print("Finished.")

except KeyError as e:
    print(f"Ignored! : {e}")
