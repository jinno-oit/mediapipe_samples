import os
import urllib.request
import time
import numpy as np
import cv2
import mediapipe as mp

class MediapipeHandLandmarker():
    base_url = 'https://storage.googleapis.com/mediapipe-tasks/hand_landmarker/'
    model_name = 'hand_landmarker.task'
    model_folder_path = './models'

    H_MARGIN = 10  # pixels
    V_MARGIN = 30  # pixels
    RADIUS_SIZE = 3  # pixels
    FONT_SIZE = 1
    FONT_THICKNESS = 1
    RIGHT_HAND_COLOR = (0, 255, 0)
    LEFT_HAND_COLOR = (100, 100, 255)

    # https://developers.google.com/mediapipe/solutions/vision/hand_landmarker#get_started
    def __init__(
            self,
            model_folder_path=model_folder_path,
            base_url=base_url,
            model_name=model_name,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
            ):

        model_path = self.set_model(base_url, model_folder_path, model_name)
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            num_hands=num_hands,
            min_hand_detection_confidence=min_hand_detection_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            running_mode=mp.tasks.vision.RunningMode.VIDEO
        )
        self.detector = mp.tasks.vision.HandLandmarker.create_from_options(options)

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

        # ハンド検出を実行する
        self.hand_landmarker_result = self.detector.detect_for_video(mp_image, int(time.time() * 1000))
        self.num_detected_hands = len(self.hand_landmarker_result.hand_landmarks)

        return self.hand_landmarker_result

    def get_normalized_landmark(self, id_hand, id_landmark):
        if self.num_detected_hands == 0:
            print('no hand')
            return None
        x = self.hand_landmarker_result.hand_landmarks[id_hand][id_landmark].x
        y = self.hand_landmarker_result.hand_landmarks[id_hand][id_landmark].y
        z = self.hand_landmarker_result.hand_landmarks[id_hand][id_landmark].z
        return np.array([x, y, z])

    def get_landmark(self, id_hand, id_landmark):
        if self.num_detected_hands == 0:
            print('no hand')
            return None
        height, width = self.size[:2]
        x = self.hand_landmarker_result.hand_landmarks[id_hand][id_landmark].x
        y = self.hand_landmarker_result.hand_landmarks[id_hand][id_landmark].y
        z = self.hand_landmarker_result.hand_landmarks[id_hand][id_landmark].z
        return np.array([int(x*width), int(y*height), int(z*width)])

    def get_handedness(self, id_hand):
        if self.num_detected_hands == 0:
            print('no hand')
            return None
        return self.hand_landmarker_result.handedness[id_hand][0].category_name

    def get_score_handedness(self, id_hand):
        if self.num_detected_hands == 0:
            print('no hand')
            return None
        return self.hand_landmarker_result.handedness[id_hand][0].score

    def visualize(self, img):
        annotated_image = np.copy(img)
        # for hand, info in zip(self.hand_landmarker_result.hand_landmarks, self.hand_landmarker_result.handedness):
        for i, hand in enumerate(self.hand_landmarker_result.hand_landmarks):
            handedness = self.get_handedness(i)
            score = self.get_score_handedness(i)
            wrist_point = self.get_landmark(i, 0)

            if self.get_handedness(i) == 'Right':
                color = self.RIGHT_HAND_COLOR
            else:
                color = self.LEFT_HAND_COLOR

            for j in range(len(hand)):
                point = self.get_landmark(i, j)
                cv2.circle(annotated_image, tuple(point[:2]), self.RADIUS_SIZE, color, thickness=self.FONT_THICKNESS)
                txt = handedness+'('+'{:#.2f}'.format(score)+')'
                wrist_point_for_text = (wrist_point[0]+self.H_MARGIN, wrist_point[1]+self.V_MARGIN)
                cv2.putText(annotated_image, org=wrist_point_for_text, text=txt, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=self.FONT_SIZE, color=color, thickness=self.FONT_THICKNESS, lineType=cv2.LINE_4)
        return annotated_image

    def release(self):
        self.detector.close()


def main():
    cap = cv2.VideoCapture(0)
    Hand = MediapipeHandLandmarker()
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is False:
            print("Ignoring empty camera frame.")
            break

        flipped_frame = cv2.flip(frame, 1)

        hand_landmarker_result = Hand.detect(flipped_frame)

        # 初めに検出したポーズの左手首の座標を表示する
        if Hand.num_detected_hands > 0:
            index_hand = 0 #
            index_landmark = 0 #
            print(
                Hand.get_normalized_landmark(index_hand, index_landmark),
                Hand.get_landmark(index_hand, index_landmark)
                )

        annotated_image = Hand.visualize(flipped_frame)

        cv2.imshow('annotated image', annotated_image)
        key = cv2.waitKey(1)&0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    Hand.release()
    cap.release()

if __name__=='__main__':
    main()
