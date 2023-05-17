import os
import urllib.request
import time
import numpy as np
import cv2
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

class MediapipePoseEstimation():
    base_url = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/'
    model_name = 'pose_landmarker_heavy.task'
    # base_url = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/'
    # model_name = 'pose_landmarker_lite.task'
    model_folder_path = './models'

    # H_MARGIN = 10  # pixels
    # V_MARGIN = 30  # pixels
    # ROW_SIZE = 10  # pixels
    # FONT_SIZE = 1
    # FONT_THICKNESS = 1
    # TEXT_COLOR = (0, 255, 0)  # green

    def __init__(
            self,
            model_folder_path=model_folder_path,
            base_url=base_url,
            model_name=model_name,
            num_poses=2,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=True,
            ):

        model_path = self.set_model(base_url, model_folder_path, model_name)
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            num_poses=num_poses,
            min_pose_presence_confidence=min_pose_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_segmentation_masks=output_segmentation_masks,
            running_mode=mp.tasks.vision.RunningMode.VIDEO
        )
        self.detector = mp.tasks.vision.PoseLandmarker.create_from_options(options)

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
      self.size = img.shape
      # 画像データをmediapipe用に変換する
      mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)

      # ポーズ検出を実行する
      self.pose_landmarker_result = self.detector.detect_for_video(mp_image, int(time.time() * 1000))
      self.num_detected_poses = len(self.pose_landmarker_result.pose_landmarks)

      return self.pose_landmarker_result

    def get_normalized_pose_landmark(self, id_pose, id_landmark):
        if self.num_detected_poses == 0:
            print('no pose')
            return None
        height, width = self.size[:2]
        x = self.pose_landmarker_result.pose_landmarks[id_pose][id_landmark].x
        y = self.pose_landmarker_result.pose_landmarks[id_pose][id_landmark].y
        z = self.pose_landmarker_result.pose_landmarks[id_pose][id_landmark].z
        return np.array([x, y, z])

    def get_pose_landmark(self, id_pose, id_landmark):
        if self.num_detected_poses == 0:
            print('no pose')
            return None
        height, width = self.size[:2]
        x = self.pose_landmarker_result.pose_landmarks[id_pose][id_landmark].x
        y = self.pose_landmarker_result.pose_landmarks[id_pose][id_landmark].y
        z = self.pose_landmarker_result.pose_landmarks[id_pose][id_landmark].z
        return np.array([int(x*width), int(y*height), int(z*width)])

    def get_segmentation_mask(self, id_pose):
        if self.num_detected_poses == 0:
            print('no pose')
            return None
        return self.pose_landmarker_result.segmentation_masks[id_pose].numpy_view()

    def get_all_segmentation_masks(self):
        if self.num_detected_poses == 0:
            print('no pose')
            return None
        all_segmentation_masks = np.zeros_like(self.pose_landmarker_result.segmentation_masks[0], dtype=float)
        for mask in self.pose_landmarker_result.segmentation_masks:
            all_segmentation_masks = np.maximum(all_segmentation_masks, mask.numpy_view().astype(float))
        return (255*all_segmentation_masks).astype(np.uint8)

    def visualize_mask(self, img, mask):
        if self.pose_landmarker_result.segmentation_masks == None:
            print('no mask')
            return img
        segmentation_mask = mask.astype(float)/np.max(mask)
        visualized_mask = np.tile(segmentation_mask[:,:,None], [1,1,3])*0.7+0.3
        return (img * visualized_mask).astype(np.uint8)

    def visualize(self, img):
        pose_landmarks_list = self.pose_landmarker_result.pose_landmarks
        annotated_image = np.copy(img)

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

    def release(self):
        self.detector.close()


def main():
    cap = cv2.VideoCapture(0)
    Pose = MediapipePoseEstimation()
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is False:
            print("Ignoring empty camera frame.")
            break

        pose_landmarker_result = Pose.detect(frame)

        # 初めに検出したポーズの左手首の座標を表示する
        if Pose.num_detected_poses > 0:
            index_pose = 0 #
            index_landmark = 15 # landmark
            print(
                Pose.get_normalized_pose_landmark(index_pose, index_landmark),
                Pose.get_pose_landmark(index_pose, index_landmark)
                )

        masks = Pose.get_all_segmentation_masks()
        mask_image = Pose.visualize_mask(frame, masks)
        annotated_image = Pose.visualize(mask_image)

        cv2.imshow('annotated image', annotated_image)
        key = cv2.waitKey(1)&0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    Pose.release()
    cap.release()

if __name__=='__main__':
    main()
