import os
import urllib.request
import time
import numpy as np
import cv2
import mediapipe as mp

class MediapipeObjectDetection():
    base_url = 'https://storage.googleapis.com/mediapipe-tasks/object_detector/'
    model_folder_path = './models'
    model_name = 'efficientdet_lite0_fp32.tflite'

    H_MARGIN = 10  # pixels
    V_MARGIN = 30  # pixels
    ROW_SIZE = 10  # pixels
    FONT_SIZE = 1
    FONT_THICKNESS = 1
    TEXT_COLOR = (0, 255, 0)  # green

    def __init__(
            self,
            model_folder_path=model_folder_path,
            base_url=base_url,
            model_name=model_name,
            max_results=-1,
            score_threshold=0.0
            ):

        model_path = self.set_model(base_url, model_folder_path, model_name)
        options = mp.tasks.vision.ObjectDetectorOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            max_results=max_results,
            score_threshold=score_threshold,
            running_mode=mp.tasks.vision.RunningMode.VIDEO
        )
        self.detector = mp.tasks.vision.ObjectDetector.create_from_options(options)

    def set_model(self, base_url, model_folder_path, model_name):
        model_path = model_folder_path+'/'+model_name
        # modelファイルが存在しない場合，ダウンロードしてくる
        if not os.path.exists(model_path):
            # model_folderが存在しない場合，フォルダを作成する
            if not os.path.exists(model_folder_path):
                os.mkdir(model_folder_path)
            # モデルをダウンロードする
            url = base_url+model_name
            save_name = model_path
            urllib.request.urlretrieve(url, save_name)
        return model_path

    def detect(self, img):
      # 画像データをmediapipe用に変換する
      mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)

      # オブジェクト検出を実行する
      self.object_detector_result = self.detector.detect_for_video(mp_image, int(time.time() * 1000))
      self.num_detected_objects = len(self.object_detector_result.detections)

      return self.object_detector_result

    def get_bounding_box(self, id_object):
        if self.num_detected_objects == 0:
            print('no object')
            return None
        x = self.object_detector_result.detections[id_object].bounding_box.origin_x
        y = self.object_detector_result.detections[id_object].bounding_box.origin_x
        w = self.object_detector_result.detections[id_object].bounding_box.width
        h = self.object_detector_result.detections[id_object].bounding_box.height
        return np.array([x, y, w, h])

    def get_category_name(self, id_object) -> str:
        if self.num_detected_objects == 0:
            print('no object')
            return None
        category_name = self.object_detector_result.detections[id_object].categories[0].category_name
        return category_name

    def get_category_score(self, id_object) -> float:
        if self.num_detected_objects == 0:
            print('no object')
            return None
        category_score = self.object_detector_result.detections[id_object].categories[0].score
        return category_score

    def visualize(self, img):
        for obj in self.object_detector_result.detections:
            # 枠の左上座標，右下座標を算出し，描画する
            upper_left_point = (obj.bounding_box.origin_x, obj.bounding_box.origin_y)
            lower_right_point = (obj.bounding_box.origin_x+obj.bounding_box.width, obj.bounding_box.origin_y+obj.bounding_box.height)
            cv2.rectangle(img, upper_left_point, lower_right_point, self.TEXT_COLOR, thickness=self.FONT_THICKNESS)
            # 枠の左上あたりにカテゴリ名を表示する
            txt = obj.categories[0].category_name+'({:#.2f})'.format(obj.categories[0].score)
            lower_left_point_for_text = (obj.bounding_box.origin_x+self.H_MARGIN, obj.bounding_box.origin_y+self.V_MARGIN)
            cv2.putText(img, org=lower_left_point_for_text, text=txt, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=self.FONT_SIZE, color=self.TEXT_COLOR, thickness=self.FONT_THICKNESS, lineType=cv2.LINE_4)
        return img

    def release(self):
        self.detector.close()

def main():
    cap = cv2.VideoCapture(0)
    Obj = MediapipeObjectDetection(score_threshold=0.5)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is False:
            print("Ignoring empty camera frame.")
            break

        object_detector_result = Obj.detect(frame)

        # 初めに検出したオブジェクトのカテゴリ名とスコア，外接矩形の左上のxy座標+横縦サイズを表示する
        if Obj.num_detected_objects > 0:
            index = 0 # object_index
            print(
                Obj.get_category_name(index),
                '({:#.2f}):'.format(Obj.get_category_score(index)),
                Obj.get_bounding_box(index)
                )

        annotated_image = Obj.visualize(frame)

        cv2.imshow('annotated image', annotated_image)
        key = cv2.waitKey(1)&0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    Obj.release()
    cap.release()

if __name__=='__main__':
    main()
