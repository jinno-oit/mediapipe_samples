# mediapipe_samples
- <mediapipe 0.10.0.0>
- [MediaPipe Examples](https://developers.google.com/mediapipe/solutions/examples)


----
## MediapipeObjectDetection
### class variable
- model information
  - `base_url`, `model_name`, `model_folder_path`
- visualize params
### instance variable
- number of detected objects: `num_detected_objects`
- (mediapipe detector: `detector`)
- (mediapipe detector's results: 'results')
### method
- `__init__( arguments are optional )`: constructor
  - arguments
    - `model_folder_path`: If you want to change the model folder path
    - `base_url`: If you want to change the model
    - `model_name`: If you want to change the model
    - `max_results`: Sets the optional maximum number of top-scored detection results to return.
      - Value Range: Any positive numbers
      - Default Value: -1 (all results are returned)
    - `score_threshold`: Sets the prediction score threshold that overrides the one provided in the model metadata (if any). Results below this value are rejected.
      - Value Range: Any float [0.0, 1.0]
      - Default Value: 0.0 (all results are detected)
- `detect( image )`
  - arguments
    - `image`: Input image (readed by cv2)
  - return values
    - `results`: Probably not necessary
- `get_bounding_box( id_object )`
  - arguments
    - `id_object`: Number of the object you want to get bounding box
  - return values
    - `np.array([x, y, w, h])`: array of the bounding box information
      - `x`: x-coordinate, `y`: y-coordinate, `w`: width, `h`: height
- `category_name = get_category_name( id_object )`
- `category_score = get_category_score( id_object )`
- `annotated_image = visualize( image )`
  - `annotated_image`: Image with bounding boxes and category names of all detected objects drawn on the input image
- `release()`: Close mediapipe's `detector`
### how to use
- simple sample
```python
import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
from MediapipeObjectDetection import MediapipeObjectDetection as ObjDetection

cap = cv2.VideoCapture(0)
Obj = ObjDetection()
while cap.isOpened():
    ret, frame = cap.read()
    Obj.detect(frame)
    annotated_frame = Obj.visualize(frame)
    cv2.imshow('annotated frame', annotated_frame)
    key = cv2.waitKey(1)&0xFF
    if key == ord('q'):
        break
cv2.destroyAllWindows()
Obj.release()
cap.release()
```
- other sample
  - You can see it in the main function of MediapipeObjectDetection.py



----

















## object_detection
- https://developers.google.com/mediapipe/solutions/vision/object_detector
- ***examples of how to reference result data `results`***
  - x-coordinate of upper left point of the i-th object's bounding box<br>
    `results.detections[i].bounding_box.origin_x`
  - category_name (e.g.`parson`) of the i-th object<br>
    `results.detections[i].categories[0].category_name`
- ***data structure of result***
  - results
    - detections
      - 0:
        - bounding_box
          - origin_x
          - origin_y
          - width
          - height
        - categories
          - 0:
            - category_name
            - score
      - 1:
        - bounding_box
          - origin_x
          - origin_y
          - width
          - height
        - categories
          - 0:
            - category_name
            - score
      - ...

## image_segmentation
- https://developers.google.com/mediapipe/solutions/vision/image_segmenter
- ***examples of how to reference result data `results`***
  - i-th segmented_mask (mediapipe image --> ndarray)<br>
    `results[0].numpy_view()`
- ***data structure of result***
  - results (mediapipe image)
      - 0:
        - height
        - width

## hand_landmarker
- https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
- ***examples of how to reference result data `results`***
  - normalized x-coordinate the j-th landmark of the i-th hand<br>
    `results.hand_landmarks[i][j].x`
  - x-coordinate the j-th landmark of the i-th hand<br>
    `int(results.hand_landmarks[i][j].x * width)`
  - category_name (e.g.`Right`) of the i-th hand<br>
    `results.handedness[i][0].category_name`
- ***data structure of result***
  - results
    - hand_landmarks (z-cordinate is based on 0-th landmark `wrist`)
      - 0: (hand_id)
        - 0: (landmark_id)
          - x
          - y
          - z
          - presence
          - visibility
        - 1: (landmark_id)
          - ...
      - 1: (hand_id)
        - ...
    - handedness
      - 0: (hand_id)
        - 0:
          - index
          - category_name
          - display_name
          - score
      - 1: (hand_id)
        - ...
    - hand_world_landmarks (representing real-world 3D coordinates in meters with the origin at the hand’s geometric center)
      - 0: (hand_id)
        - 0: (landmark_id)
          - x
          - y
          - z
          - presence
          - visibility
        - 1: (landmark_id)
          - ...
      - 1: (hand_id)
        - ...


## face_detecion (not yet)
- https://developers.google.com/mediapipe/solutions/vision/face_detector


## face_landmarker
- https://developers.google.com/mediapipe/solutions/vision/face_landmarker
- ***examples of how to reference result data `results`***
  - normalized x-coordinate the j-th landmark of the i-th face<br>
    `results.face_landmarks[i][j].x`
  - x-coordinate the j-th landmark of the i-th face<br>
    `int(results.face_landmarks[i][j].x * width)`
- ***data structure of result***
  - results
    - face_landmarks (z-cordinate is based on 0-th landmark `wrist`)
      - 0: (face_id)
        - 0: (landmark_id)
          - x
          - y
          - z
          - presence
          - visibility
        - 1: (landmark_id)
          - ...
      - 1: (face_id)
        - ...
    - face_blandshapes
      - 0: (face_id)
        - 0: (blendshapes_idx)
          - index (`0`)
          - category_name (`_neutral`)
          - display_name
          - score
        - 1: (blendshapes_idx)
          - ...
      - 1: (face_id)
        - ...
    - facial_transformation_matrixes
      - 0: (face_id)
        - 0:
          - array([0:4])
        - 1:
        - 2:
        - 3:
      - 1: (face_id)
        - ...

## pose_estimation
- https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
- ***examples of how to reference result data `results`***
  - normalized x-coordinate the j-th landmark of the i-th pose<br>
    `results.pose_landmarks[i][j].x`
  - x-coordinate the j-th landmark of the i-th pose<br>
    `int(results.pose_landmarks[i][j].x * width)`
  - segmentation_mask of the i-th pose<br>
    `results.segmentation_masks[i].numpy_view()`
- ***data structure of result***
  - results
    - pose_landmarks
      - 0: (pose_id)
        - 0: (landmark_id)
          - x
          - y
          - z
          - presence
          - visibility
        - 1: (landmark_id)
          - ...
      - 1: (pose_id)
        - ...
    - segmentation_masks
      - 0: (pose_id) (mediapipe image)
      - 1: (pose_id) (mediapipe image)
        - ...
    - pose_world_landmarks
      - 0: (pose_id)
        - 0: (landmark_id)
          - x
          - y
          - z
          - presence
          - visibility
        - 1: (landmark_id)
          - ...
      - 1: (pose_id)
        - ...
