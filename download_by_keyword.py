from apiclient.discovery import build
from api_key import api_key
from pytube import YouTube
from pprint import pprint


def get_videos_search(keyword):
    youtube = build('youtube', 'v3', developerKey=api_key)
    youtube_query = youtube.search().list(q=keyword, part='id,snippet', maxResults=5)
    youtube_res = youtube_query.execute()
    return youtube_res.get('items', [])

result = get_videos_search('ポケモン サウンド')
for item in result:
    if item['id']['kind'] == 'youtube#video':
        title = item['snippet']['title']
        print(title)
        url = 'https://www.youtube.com/watch?v=' + item['id']['videoId']
        print(url)
        try:
            stream = YouTube(url).streams
            pprint(stream)
            stream.first().download("downloads")

        except KeyError as e:
           print("Ignored! : ", title)
