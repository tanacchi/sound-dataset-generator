# sound-dataset-generator
AI セミナー グループ 4 データセット生成器

## 使い方

1. requirements.txt 内のパッケージをインストールする．  
   例：`pip install -r requirements.txt`
1. `api_key.py` 内の文字列を  
   先日共有した API キーに書き換えて保存する．  
   `api_key.py` はこの README と同じディレクトリに置く．
1. `download_by_url.py` や `download_by_keyword.py` を実行して動画を取得する．  
   ファイルは `downloads` ディレクトリに保存される．  
   例：
   * `download_by_url.py https://www.youtube.com/watch?v=zriF4xIYZQI`
   * `download_by_keyword.py ”キジバト EDM”`
1. `sh convert_mp4_to_wav.sh` を実行し， wav ファイルに変換．  
   実行できなかった場合はファイル内に書いてあるコマンドを直接打ち込んで実行する．
1. `split_wav.py` を実行して wav ファイルを分割する．  
   出力先は `output` ディレクトリ．  
   `python3 split_wav.py --length 20 --offset 10` のようにすると  
   10ずつずらしながら，それぞれ20秒の長さに分割する．  
   この指定を省略すると，15秒ずつずらしながら30秒の長さに分割する．
