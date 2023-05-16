import os
import urllib.request
import time
import cv2
import mediapipe as mp

class MediapipeFaceDetection():
    base_url = 'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/'
    model_folder_path = './models'
    model_name = 'blaze_face_short_range.tflite'

    MARGIN = 10  # pixels
    ROW_SIZE = 10  # pixels
    FONT_SIZE = 1
    FONT_THICKNESS = 1
    TEXT_COLOR = (255, 0, 0)  # red

    def __init__(self, model_folder_path=model_folder_path, base_url=base_url, model_name=model_name):
        BaseOptions = mp.tasks.BaseOptions
        FaceDetector = mp.tasks.vision.FaceDetector
        FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        model_path = self.set_model(base_url, model_folder_path, model_name)
        options = FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO
        )
        self.detector = FaceDetector.create_from_options(options)

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
      self.img = img

      # 画像データをmediapipe用に変換する
      mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=self.img)

      # オブジェクト検出を実行する
      self.face_detector_result = self.detector.detect_for_video(mp_image, int(time.time() * 1000))

      return self.face_detector_result

    def visualize(self, img, face_detector_result):
        annotated_image = img.copy()
        height, width, _ = img.shape

        for detection in face_detector_result.detections:
            # Draw bounding_box
            bbox = detection.bounding_box
            start_point = bbox.origin_x, bbox.origin_y
            end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
            cv2.rectangle(annotated_image, start_point, end_point, self.TEXT_COLOR, 3)

            # Draw keypoints
            for keypoint in detection.keypoints:
                keypoint_px = [int(keypoint.x * width), int(keypoint.y*height)]
                color, thickness, radius = (0, 255, 0), 2, 2
                cv2.circle(annotated_image, keypoint_px, thickness, color, radius)

                # Draw label and score
                category = detection.categories[0]
                category_name = category.category_name
                category_name = '' if category_name is None else category_name
                probability = round(category.score, 2)
                result_text = category_name + ' (' + str(probability) + ')'
                text_location = (self.MARGIN + bbox.origin_x, self.MARGIN + self.ROW_SIZE + bbox.origin_y)
                cv2.putText(annotated_image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN, self.FONT_SIZE, self.TEXT_COLOR, self.FONT_THICKNESS)

        return annotated_image

    def release(self):
        self.detector.close()




if __name__=='__main__':
    cap = cv2.VideoCapture(0)
    face_detector = MediapipeFaceDetection()
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is False:
            print("Ignoring empty camera frame.")
            break

        face_detector_result = face_detector.detect(frame)
        annotated_image = face_detector.visualize(frame, face_detector_result)

        cv2.imshow('annotated image', annotated_image)
        key = cv2.waitKey(1)&0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    face_detector.release()
    cap.release()
