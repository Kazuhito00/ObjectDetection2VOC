import copy
import json

import numpy as np


class ObjectDetector(object):
    def __init__(
        self,
        model_name='yolox',
        target_id=None,
        use_gpu=False,
    ):
        self.model_name = model_name
        self.model = None
        self.config = None
        self.target_id = target_id
        self.use_gpu = use_gpu

        if self.model_name == 'yolox':
            from Detector.yolox.yolox_onnx import YoloxONNX

            if self.use_gpu:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']

            with open('Detector/yolox/config.json') as fp:
                self.config = json.load(fp)

            if self.config is not None:
                self.model = YoloxONNX(
                    model_path=self.config['model_path'],
                    input_shape=[
                        int(i) for i in self.config['input_shape'].split(',')
                    ],
                    class_score_th=self.config['class_score_th'],
                    nms_th=self.config['nms_th'],
                    nms_score_th=self.config['nms_score_th'],
                    with_p6=self.config['with_p6'],
                    providers=providers,
                )

        else:
            raise ValueError('Invalid Model Name')

    def __call__(self, image):
        input_image = copy.deepcopy(image)
        bboxes, scores, class_ids = None, None, None

        # 推論
        if self.model is not None:
            bboxes, scores, class_ids = self.model(input_image)
        else:
            raise ValueError('Model is None')

        # 対象のクラスIDで抽出
        if self.target_id is not None and len(bboxes) > 0:
            target_index = np.in1d(class_ids, np.array(self.target_id))

            bboxes = bboxes[target_index]
            scores = scores[target_index]
            class_ids = class_ids[target_index]

        return bboxes, scores, class_ids

    def print_info(self):
        from pprint import pprint

        print('Detector:', self.model_name)
        print('Target ID:',
              'ALL' if self.target_id is None else self.target_id)
        print('GPU:', self.use_gpu)
        pprint(self.config, indent=4)
        print()
