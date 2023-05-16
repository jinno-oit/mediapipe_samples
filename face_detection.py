import time
from typing import Tuple, Union
import math
import cv2
import numpy as np
import mediapipe as mp
import download as dl

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red

def visualize_face(img, face_detector_result):
  h, w = img.shape[:2]
#   for face, info in zip(face_detector_result.bounding_box, hand_landmarker_result.handedness):
#     if info[0].category_name == 'Left':
#       color = (0, 255, 0)
#     else:
#       color = (255, 50, 50)
#     for point in hand:
#       cv2.circle(img, (int(point.x*w), int(point.y*h)), 5, color, thickness=2)
#     txt = info[0].category_name+'('+'{:#.2f}'.format(info[0].score)+')'
#     wrist_point_for_text = (int(hand[0].x*w)+10, int(hand[0].y*h)+30)
#     cv2.putText(img, org=wrist_point_for_text, text=txt, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=color, thickness=1, lineType=cv2.LINE_4)
  return img

def _normalized_to_pixel_coordinates(
  normalized_x: float, normalized_y: float, image_width: int,
  image_height: int) -> Union[None, Tuple[int, int]]:
  """Converts normalized value pair to pixel coordinates."""

  # Checks if the float value is between 0 and 1.
  def is_valid_normalized_value(value: float) -> bool:
    return (value > 0 or math.isclose(0, value)) and (value < 1 or math.isclose(1, value))

  if not (is_valid_normalized_value(normalized_x) and
          is_valid_normalized_value(normalized_y)):
    # TODO: Draw coordinates even if it's outside of the image bounds.
    return None
  x_px = min(math.floor(normalized_x * image_width), image_width - 1)
  y_px = min(math.floor(normalized_y * image_height), image_height - 1)
  return x_px, y_px

def visualize(image, detection_result) -> np.ndarray:
  """Draws bounding boxes and keypoints on the input image and return it.
  Args:
    image: The input RGB image.
    detection_result: The list of all "Detection" entities to be visualize.
  Returns:
    Image with bounding boxes.
  """
  annotated_image = image.copy()
  height, width, _ = image.shape

  for detection in detection_result.detections:
    # Draw bounding_box
    bbox = detection.bounding_box
    start_point = bbox.origin_x, bbox.origin_y
    end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
    cv2.rectangle(annotated_image, start_point, end_point, TEXT_COLOR, 3)

    # Draw keypoints
    for keypoint in detection.keypoints:
      keypoint_px = _normalized_to_pixel_coordinates(keypoint.x, keypoint.y,
                                                     width, height)
      color, thickness, radius = (0, 255, 0), 2, 2
      cv2.circle(annotated_image, keypoint_px, thickness, color, radius)

    # Draw label and score
    category = detection.categories[0]
    category_name = category.category_name
    category_name = '' if category_name is None else category_name
    probability = round(category.score, 2)
    result_text = category_name + ' (' + str(probability) + ')'
    text_location = (MARGIN + bbox.origin_x,
                     MARGIN + ROW_SIZE + bbox.origin_y)
    cv2.putText(annotated_image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

  return annotated_image

def main():
  base_url = 'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/'
  model_folder_path = './models'
  model_name = 'blaze_face_short_range.tflite'
  model_path = dl.set_model(base_url, model_folder_path, model_name)

  # 初期設定（名前が長いのでリネームしている）
  BaseOptions = mp.tasks.BaseOptions
  FaceDetector = mp.tasks.vision.FaceDetector
  FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
  VisionRunningMode = mp.tasks.vision.RunningMode

  # FaceDetectorのオブション
  options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO)

  cap = cv2.VideoCapture(0)

  # detectorをオープンして処理を開始する
  with FaceDetector.create_from_options(options) as detector:
    while cap.isOpened():
      # そのフレームの画像を読み込む
      success, frame = cap.read()
      if success is False:
        print("Ignoring empty camera frame.")
        break

      # 画像データをmediapipe用に変換する
      mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

      # オブジェクト検出を実行する
      face_detector_result = detector.detect_for_video(mp_image, int(time.time() * 1000))

      # 検出したオブジェクトを囲う枠，および名前とスコアを画像に重畳する
      annotated_image = visualize(frame.copy(), face_detector_result)

      # 画像を表示する（'q'キーを押すとループ終了）
      cv2.imshow('result', annotated_image)
      key = cv2.waitKey(1)&0xFF
      if key == ord('q'):
        break

  cap.release()





if __name__=='__main__':
  main()
