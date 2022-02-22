import os
import re
import copy
import time
import argparse
from glob import glob

import cv2
import xmltodict

from Detector.detector import ObjectDetector


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--movie", type=str, default=None)
    parser.add_argument("--image_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default='Result')

    # Object Detection
    parser.add_argument(
        '--detector',
        choices=[
            'yolox',
        ],
        default='yolox',
    )
    parser.add_argument(
        "--class_txt",
        type=str,
        default='Detector/yolox/coco_classes.txt',
    )
    parser.add_argument("--target_id", type=str, default=None)
    parser.add_argument('--use_gpu', action='store_true')

    # VOC Setting
    parser.add_argument("--score_th", type=float, default=0.5)
    parser.add_argument(
        '--folder',
        help='Base infomation:folder.',
        type=str,
        default='',
    )
    parser.add_argument(
        '--path',
        help='Base infomation:path.',
        type=str,
        default='',
    )
    parser.add_argument(
        '--owner',
        help='Base infomation:owner name.',
        type=str,
        default='Unknown',
    )
    parser.add_argument(
        '--source_database',
        help='Base infomation:source database.',
        type=str,
        default='Unknown',
    )
    parser.add_argument(
        '--source_annotation',
        help='Base infomation:source annotation.',
        type=str,
        default='Unknown',
    )
    parser.add_argument(
        '--source_image',
        help='Base infomation:source image.',
        type=str,
        default='Unknown',
    )
    parser.add_argument(
        '--segmented',
        help='Base infomation:segmented.',
        type=str,
        default='0',
    )

    args = parser.parse_args()

    return args


def main():
    args = get_args()

    # VOC
    score_th = args.score_th

    folder = args.folder
    path = args.path
    owner = args.owner
    source_database = args.source_database
    source_annotation = args.source_annotation
    source_image = args.source_image
    segmented = args.segmented

    # Object Detection
    detector_name = args.detector
    class_txt_path = args.class_txt

    target_id = args.target_id
    if target_id is not None:
        target_id = [int(i) for i in target_id.split(',')]

    use_gpu = args.use_gpu

    file_prefix = ''

    # Webカメラ or 動画初期化
    cap_device = args.device
    if args.movie is not None:
        cap_device = args.movie

        basename = os.path.basename(args.movie)
        file_prefix = os.path.splitext(basename)[0] + '_'

    # 画像ファイルディレクトリが指定された場合
    image_list = []
    image_dir = args.image_dir
    if image_dir is not None:
        image_list = sorted([
            p for p in glob(os.path.join(image_dir, '*'))
            if re.search('/*\.(jpg|jpeg|png|gif|bmp)', str(p))
        ])

    # 出力先
    output_dir = args.output_dir

    # VideoCapture初期化
    cap = cv2.VideoCapture(cap_device)

    # Object Detection
    detector = ObjectDetector(
        detector_name,
        target_id,
        use_gpu=use_gpu,
    )
    detector.print_info()

    # クラスリスト読み込み
    with open(class_txt_path, 'rt') as f:
        classe_name_list = f.read().rstrip('\n').split('\n')

    # Webカメラと動画時の保存ファイルカウンタ
    file_count = 0

    while True:
        start_time = time.time()

        # フレーム読み込み
        frame = None
        if image_dir is None:
            ret, frame = cap.read()
            if not ret:
                break
        else:
            if len(image_list) <= file_count:
                break
            frame = cv2.imread(image_list[file_count])
        debug_image = copy.deepcopy(frame)

        # Object Detection
        bboxes, scores, class_ids = detector(frame)

        elapsed_time = time.time() - start_time

        # 描画
        debug_image = draw_debug_info(
            debug_image,
            elapsed_time,
            bboxes,
            scores,
            score_th,
            class_ids,
            classe_name_list,
        )

        # XML/画像ファイル名生成
        if image_dir is None:
            save_basename = file_prefix + '{:08}'.format(file_count)
        else:
            basename = os.path.basename(image_list[file_count])
            save_basename = os.path.splitext(basename)[0]

        save_image_name = save_basename + '.jpg'
        save_xml_name = save_basename + '.xml'

        # XML情報生成
        xml_str = create_xml_string(save_image_name, frame, folder, path,
                                    owner, source_database, source_annotation,
                                    source_image, segmented, bboxes, class_ids,
                                    classe_name_list, scores, score_th)

        # 画像保存
        cv2.imwrite(os.path.join(output_dir, save_image_name), debug_image)

        # XML保存
        with open(os.path.join(output_dir, save_xml_name),
                  'w',
                  encoding='UTF-8') as fp:
            fp.writelines(xml_str)

        # 表示
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break
        cv2.imshow('ObjectDetection2VOC', debug_image)

        file_count += 1

    cap.release()
    cv2.destroyAllWindows()


def create_xml_string(
    filename,
    image,
    folder,
    path,
    owner,
    source_database,
    source_annotation,
    source_image,
    segmented,
    bboxes,
    class_ids,
    classe_name_list,
    scores,
    score_th,
):
    image_dict = create_image_dict(
        filename,
        image.shape[1],
        image.shape[0],
        image.shape[2],
        folder,
        path,
        owner,
        source_database,
        source_annotation,
        source_image,
        segmented,
    )

    for bbox, class_id, score in zip(bboxes, class_ids, scores):
        if score < score_th:
            continue

        annotation_dict = create_object_dict(
            classe_name_list[int(class_id)],
            bbox,
        )

        image_dict['annotation']['object'].append(annotation_dict)

    xml_str = xmltodict.unparse(image_dict, full_document=False, pretty=True)

    return xml_str


def create_image_dict(
    filename,
    width,
    height,
    depth=3,
    folder='',
    path='',
    owner='Unknown',
    source_database='Unknown',
    source_annotation='Unknown',
    source_image='Unknown',
    segmented='0',
):
    if path == '':
        filepath = filename
    else:
        filepath = path

    image_dict = {
        'annotation': {
            'folder': folder,
            'filename': filename,
            'path': filepath,
            'owner': {
                'name': owner
            },
            'source': {
                'database': source_database,
                'annotation': source_annotation,
                'image': source_image
            },
            'size': {
                'width': width,
                'height': height,
                'depth': depth
            },
            'segmented': segmented,
            'object': []
        }
    }
    return image_dict


def create_object_dict(class_name, bbox):
    object_dict = {
        'name': class_name,
        'pose': 'Unspecified',
        'truncated': '0',
        'difficult': '0',
        'bndbox': {
            'xmin': int(bbox[0]),
            'ymin': int(bbox[1]),
            'xmax': int(bbox[2]),
            'ymax': int(bbox[3])
        }
    }

    return object_dict


def get_id_color(index):
    temp_index = abs(int(index + 1)) * 3
    color = ((37 * temp_index) % 255, (17 * temp_index) % 255,
             (29 * temp_index) % 255)
    return color


def draw_debug_info(
    debug_image,
    elapsed_time,
    bboxes,
    scores,
    score_th,
    class_ids,
    classe_name_list,
):
    for bbox, score, class_id in zip(bboxes, scores, class_ids):
        if score < score_th:
            continue

        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        color = get_id_color(class_id)

        # バウンディングボックス
        debug_image = cv2.rectangle(
            debug_image,
            (x1, y1),
            (x2, y2),
            color,
            thickness=2,
        )

        # クラスID
        score = '%.2f' % score
        text = '%s(%s)' % (classe_name_list[int(class_id)], str(score))
        debug_image = cv2.putText(
            debug_image,
            text,
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            thickness=2,
        )

    # 経過時間(キャプチャ、物体検出、トラッキング)
    cv2.putText(
        debug_image,
        "Elapsed Time : " + '{:.1f}'.format(elapsed_time * 1000) + "ms",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )

    return debug_image


if __name__ == '__main__':
    main()
