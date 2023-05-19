# https://developers.google.com/mediapipe/solutions/vision/hand_landmarker

import time
import cv2
import mediapipe as mp
import download as dl

MARGIN_X = 10
MARGIN_Y = 30
RIGHT_HAND_COLOR = (0, 255, 0)
LEFT_HAND_COLOR = (100, 100, 255)

def visualize_face(img, face_landmarker_result):
  h, w = img.shape[:2]
  color = RIGHT_HAND_COLOR
  for face in face_landmarker_result.face_landmarks:
    for point in face:
      cv2.circle(img, (int(point.x*w), int(point.y*h)), 1, color, thickness=2)
  return img

def main():
  base_url = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/'
  model_name = 'face_landmarker.task'
  model_folder_path = './models'
  model_path = dl.set_model(base_url, model_folder_path, model_name)

  # 初期設定（名前が長いのでリネームしている）
  BaseOptions = mp.tasks.BaseOptions
  FaceLandmarker = mp.tasks.vision.FaceLandmarker
  FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
  VisionRunningMode = mp.tasks.vision.RunningMode

  # FaceLandmarkerのオブション
  options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    num_faces=2,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    running_mode=VisionRunningMode.VIDEO)

  cap = cv2.VideoCapture(0)

  # detectorをオープンして処理を開始する
  with FaceLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
      # そのフレームの画像を読み込む
      success, frame = cap.read()
      if success is False:
        print("Ignoring empty camera frame.")
        break

      fliped_frame = cv2.flip(frame, 1) # 左右反転

      # 画像データをmediapipe用に変換する
      mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=fliped_frame)

      # オブジェクト検出を実行する
      face_landmarker_result = landmarker.detect_for_video(mp_image, int(time.time() * 1000))

      # 検出したオブジェクトを囲う枠，および名前とスコアを画像に重畳する
      annotated_image = visualize_face(fliped_frame.copy(), face_landmarker_result)

      # 画像を表示する（'q'キーを押すとループ終了）
      cv2.imshow('result', annotated_image)
      key = cv2.waitKey(1)&0xFF
      if key == ord('q'):
        break

  cap.release()

if __name__=='__main__':
  main()

