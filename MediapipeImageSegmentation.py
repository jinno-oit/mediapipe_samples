import os
import urllib.request
import time
import numpy as np
import cv2
import mediapipe as mp

class MediapipeImageSegmentation():
    # base_url = 'https://storage.googleapis.com/mediapipe-tasks/image_segmenter/'
    # model_name = 'deeplabv3.tflite'
    base_url = 'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/'
    model_name = 'selfie_multiclass_256x256.tflite'
    model_folder_path = './models'

    def __init__(
            self,
            model_folder_path=model_folder_path,
            base_url=base_url,
            model_name=model_name,
            output_category_mask=True,
            output_confidence_masks=True,
            ):

        model_path = self.set_model(base_url, model_folder_path, model_name)
        options = mp.tasks.vision.ImageSegmenterOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            output_category_mask=output_category_mask,
            output_confidence_masks=output_confidence_masks
        )
        self.segmenter = mp.tasks.vision.ImageSegmenter.create_from_options(options)

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

        # 画像分割を実行する
        self.image_segmenter_result = self.segmenter.segment_for_video(mp_image, int(time.time() * 1000))

        return self.image_segmenter_result

    def get_segmentation_mask(self):
        return self.image_segmenter_result.category_mask.numpy_view()

    def get_normalized_mask(self):
        mask = self.image_segmenter_result.category_mask.numpy_view()
        return (255.0*mask/np.max(mask)).astype(np.uint8)
    def release(self):
        self.segmenter.close()

def main():
    cap = cv2.VideoCapture(0)
    ImgSeg = MediapipeImageSegmentation()
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is False:
            print("Ignoring empty camera frame.")
            break

        segmented_masks = ImgSeg.detect(frame)

        mask = ImgSeg.get_segmentation_mask()
        normalized_mask = ImgSeg.get_normalized_mask()

        cv2.imshow('mask', cv2.applyColorMap(normalized_mask, cv2.COLORMAP_JET))

        cv2.imshow('frame', frame)
        key = cv2.waitKey(1)&0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    ImgSeg.release()
    cap.release()

if __name__=='__main__':
    main()
