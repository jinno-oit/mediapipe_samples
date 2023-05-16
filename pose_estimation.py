import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import download as dl

def visualize_all_pose(img, pose_landmarker_result):
  h, w = img.shape[:2]
  for pose in pose_landmarker_result.pose_landmarks:
    color = (0, 255, 0)
    for point in pose:
      cv2.circle(img, (int(point.x*w), int(point.y*h)), 5, color, thickness=2)
  return img

def draw_masks_on_image(rgb_image, detection_result):
  if detection_result.segmentation_masks is None:
    print(len(detection_result.pose_landmarks))
    return rgb_image

  segmentation_mask = np.zeros_like(detection_result.segmentation_masks[0], dtype=float)
  for mask in detection_result.segmentation_masks:
    segmentation_mask = np.maximum(segmentation_mask, mask.numpy_view().astype(float))

  visualized_mask = np.tile(segmentation_mask[:,:,None], [1,1,3])*0.7+0.3
  return (rgb_image * visualized_mask).astype(np.uint8)

def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image

def main():
  base_url = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/'
  model_name = 'pose_landmarker_heavy.task'
  # base_url = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/'
  # model_name = 'pose_landmarker_lite.task'
  model_folder_path = './models'
  model_path = dl.set_model(base_url, model_folder_path, model_name)

  # 初期設定（名前が長いのでリネームしている）
  BaseOptions = mp.tasks.BaseOptions
  PoseLandmarker = mp.tasks.vision.PoseLandmarker
  PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
  VisionRunningMode = mp.tasks.vision.RunningMode

  # PoseLandmarkerのオブション
  options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    num_poses=2,
    output_segmentation_masks=True,
    running_mode=VisionRunningMode.VIDEO)

  cap = cv2.VideoCapture(0)

  # detectorをオープンして処理を開始する
  with PoseLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
      # そのフレームの画像を読み込む
      success, frame = cap.read()
      if success is False:
        print("Ignoring empty camera frame.")
        break

      # 画像データをmediapipe用に変換する
      mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.flip(frame,1))

      # オブジェクト検出を実行する
      pose_landmarker_result = landmarker.detect_for_video(mp_image, int(time.time() * 1000))

      # 検出したオブジェクトを囲う枠，および名前とスコアを画像に重畳する
      # annotated_image = visualize_all_pose(cv2.flip(frame,1).copy(), pose_landmarker_result)
      # annotated_image = draw_landmarks_on_image(cv2.flip(frame,1).copy(), pose_landmarker_result)

      mask_image = draw_masks_on_image(cv2.flip(frame,1).copy(), pose_landmarker_result)
      annotated_image = draw_landmarks_on_image(mask_image, pose_landmarker_result)

      # 画像を表示する（'q'キーを押すとループ終了）
      cv2.imshow('result', annotated_image)
      key = cv2.waitKey(1)&0xFF
      if key == ord('q'):
        break

  cap.release()





if __name__=='__main__':
  main()

