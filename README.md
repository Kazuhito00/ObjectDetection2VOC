# ObjectDetection2VOC
物体検出を行い検出結果をPascal VOC形式で保存するスクリプト

# Requirement 
* xmltodict 0.12.0 or later

# Usage
使用方法は以下です。
```bash
python main.py
```
* --device<br>
カメラデバイス番号の指定<br>
デフォルト：0
* --movie<br>
動画ファイルの指定 ※指定時はカメラデバイスより優先<br>
デフォルト：指定なし
* --image_dir<br>
画像ファイル格納ディレクトリの指定 ※指定時はカメラデバイスや動画より優先<br>
デフォルト：指定なし
* --output_dir<br>
推論結果の保存先<br>
デフォルト：Result
* --detector<br>
Object Detectionのモデル選択<br>
デフォルト：yolox
* --class_txt<br>
物体検出時のラベル名のテキストを格納したパス<br>
デフォルト：Detector/yolox/coco_classes.txt
* --target_id<br>
トラッキング対象のクラスIDを指定<br>複数指定する場合はカンマ区切りで指定　※Noneの場合は全てを対象とする<br>
例：--target_id=1<br>例：--target_id=1,3<br>
デフォルト：None
* --use_gpu<br>
GPU推論するか否か<br>
デフォルト：指定なし
* --set_name<br>
xmlファイル一覧テキストを出力名<br>
デフォルト：train
* --bbox_offset<br>
Pascal VOC形式のバウンディングボックスへ変換する際のオフセット<br>
デフォルト：1
* --folder<br>
\<folder\>タグに書き込む内容<br>
デフォルト：VOCCOCO
* --path<br>
\<path\>タグに書き込む内容<br>
デフォルト：指定なし
* --owner<br>
\<owner\>タグに書き込む内容<br>
デフォルト：Unknown
* --source_database<br>
\<database\>タグに書き込む内容<br>
デフォルト：Unknown
* --source_annotation<br>
\<annotation\>タグに書き込む内容<br>
デフォルト：Unknown
* --source_image<br>
\<image\>タグに書き込む内容<br>
デフォルト：Unknown
* --image_depth<br>
\<depth\>タグに書き込む内容<br>
デフォルト：3
* --segmented<br>
\<segmented\>タグに書き込む内容<br>
デフォルト：0

# Detector

| モデル名 | 取得元リポジトリ | ライセンス | 備考 |
| :--- | :--- | :--- | :--- |
| YOLOX | [Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) | Apache-2.0 | [YOLOX-ONNX-TFLite-Sample](https://github.com/Kazuhito00/YOLOX-ONNX-TFLite-Sample)にて<br>ONNX化したモデルを使用 |

# Author
高橋かずひと(https://twitter.com/KzhtTkhs)
 
# License 
ObjectDetection2VOC is under [MIT License](LICENSE).
