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
- `num_detected_objects`: number of detected objects
- (`detector`: mediapipe detector)
- (`results`: mediapipe detector's results)
### method
- `__init__( arguments are optional )`: constructor
  - arguments
    - `model_folder_path`: If you want to change the model folder path
    - `base_url`: If you want to change the model
    - `model_name`: If you want to change the model
    - `max_results`: Sets the optional maximum number of top-scored detection results to return.
      - Value Range: Any positive numbers
      - Default Value: `-1` (all results are returned)
    - `score_threshold`: Sets the prediction score threshold that overrides the one provided in the model metadata (if any). Results below this value are rejected.
      - Value Range: Any float `[0.0, 1.0]`
      - Default Value: `0.0` (all results are detected)
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
  - `annotated_image`: Image with bounding boxes and category names for all detected objects on the input image
- `release()`: Close mediapipe's `detector`
### how to use
- simple sample
```python
import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
from MediapipeObjectDetection import MediapipeObjectDetection as ObjDetection

cap = cv2.VideoCapture(0)
Obj = ObjDetection(score_threshold=0.5)
while cap.isOpened():
    ret, frame = cap.read()
    Obj.detect(frame)
    print(Obj.num_detected_objects)
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
  - You can see it in the main function of `MediapipeObjectDetection.py`


----
## MediapipeHandLandmark
### class variable
- model information
  - `base_url`, `model_name`, `model_folder_path`
- visualize params
  - e.g. `RIGHT_HAND_COLOR`, `FONT_SIZE`, ...
- hand landmark id
  - e.g. `WRIST = 0`, `THUMB_CMC = 1`
### instance variable
- `num_detected_hands`: number of detected objects
- (`detector`: mediapipe detector)
- (`results`: mediapipe detector's results)
### method
- `__init__( arguments are optional )`: constructor
  - arguments
    - `model_folder_path`: If you want to change the model folder path
    - `base_url`: If you want to change the model
    - `model_name`: If you want to change the model
    - `num_hands`: The maximum number of hands detected by the Hand landmark detector
      - Value Range: Any integer `> 0`
      - Default Value: `2`
    - `min_hand_detection_confidence`: The minimum confidence score for the hand detection to be considered successful in palm detection model.
      - Value Range: `0.0 - 1.0`
      - Default Value: `0.5`
    - `min_hand_presence_confidence`: The minimum confidence score for the hand presence score in the hand landmark detection model. In Video mode, if the hand presence confidence score from the hand landmark model is below this threshold, Hand Landmarker triggers the palm detection model. Otherwise, a lightweight hand tracking algorithm determines the location of the hand(s) for subsequent landmark detections.
      - Value Range: `0.0 - 1.0`
      - Default Value: `0.5`
    - `min_tracking_confidence`: The minimum confidence score for the hand tracking to be considered successful. This is the bounding box IoU threshold between hands in the current frame and the last frame. In Video mode and Stream mode of Hand Landmarker, if the tracking fails, Hand Landmarker triggers hand detection. Otherwise, it skips the hand detection.
      - Value Range: `0.0 - 1.0`
      - Default Value: `0.5`
- `detect( image )`
  - arguments
    - `image`: Input image (readed by cv2)
      - Input image is a frame image flipped holizontal! Otherwise, left hand and right hand are reversed.
  - return values
    - `results`: Probably not necessary
- `get_normalized_landmark( id_hand, id_landmark )`
  - arguments
    - `id_hand`: ID number of the hand you want to get normalized landmark coordinate
    - `id_landmark`: ID number of the hand landmark you want to get normalized landmark coordinate
  - return values
    - `np.array([x, y, z])`: array of the coordinate
      - `x`: normalized x-coordinate, `y`: normallized y-coordinate, `z`: normalized z-coordinate
      - Value Range: `x:0.0-1.0`, `y:0.0-1.0`, `z:0.0-1.0`
- `get_landmark( id_hand, id_landmark )`
  - arguments
    - `id_hand`: ID number of the hand you want to get landmark coordinate
    - `id_landmark`: ID number of the hand landmark you want to get landmark coordinate
  - return values
    - `np.array([x, y, z])`: array of the coordinate
      - `x`: x-coordinate, `y`: y-coordinate, `z`: z-coordinate
      - Value Range: `x:0-width`, `y:0-height`, `z:0-width`
- `category_name = get_handedness( id_hand )`
  - e.g. `Right`, `Left`
- `category_score = get_score_handedness( id_hand )`
- `annotated_image = visualize( image )`
  - `annotated_image`: Image with hand landmark points and category names for all detected hands on the input image
- `annotated_image = visualize_with_mp( image )`
  - mediapipe visualizing settings
- `release()`: Close mediapipe's `detector`
### how to use
- simple sample
```python
import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
from MediapipeHandLandmark import MediapipeHandLandmark as HandLmk

cap = cv2.VideoCapture(0)
Hand = HandLmk()
while cap.isOpened():
    ret, frame = cap.read()
    flipped_frame = cv2.flip(frame, 1)
    Hand.detect(flipped_frame)
    print(Hand.num_detected_hands)
    annotated_frame = Hand.visualize(flipped_frame)
    cv2.imshow('annotated frame', annotated_frame)
    key = cv2.waitKey(1)&0xFF
    if key == ord('q'):
        break
cv2.destroyAllWindows()
Hand.release()
cap.release()
```
- other sample
  - You can see it in the main function of `MediapipeHandLandmark.py`


----
## MediapipeHandGestureRecognition
### inheritance
- `class MediapipeHandLandmark`
### class variable
- same of the `class MediapipeHandLandmark`
### instance variable
- `num_detected_hands`: number of detected objects
- (`recognizer`: mediapipe recognizer)
- (`results`: mediapipe recognizer's results)
### method
- `__init__( arguments are optional )`: constructor
  - arguments
    - same of the `class MediapipeHandLandmark`
- `detect( image )`
  - arguments
    - `image`: Input image (readed by cv2)
      - Input image is a frame image flipped holizontal! Otherwise, left hand and right hand are reversed.
  - return values
    - `results`: Probably not necessary
- [inheritance] `get_normalized_landmark( id_hand, id_landmark )`
- [inheritance] `get_landmark( id_hand, id_landmark )`
- [inheritance] `category_name = get_handedness( id_hand )`
- [inheritance] `category_score = get_score_handedness( id_hand )`
- `gesture_name = get_gesture( id_hand )`
  - e.g. `victory`, `thumbs up`
- `gesture_score = get_gesture( id_hand )`
- [inheritance] `annotated_image = visualize( image )`
- [inheritance] `annotated_image = visualize_with_mp( image )`
- `release()`: Close mediapipe's `recognizer`
### how to use
- simple sample
```python
import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
from MediapipeHandGestureRecognition import MediapipeHandGestureRecognition as HandGesRec

cap = cv2.VideoCapture(0)
HandGes = HandGesRec()
while cap.isOpened():
    ret, frame = cap.read()
    flipped_frame = cv2.flip(frame, 1)
    HandGes.detect(flipped_frame)
    if HandGes.num_detected_hands>0:
        print(HandGes.get_gesture(0), HandGes.get_score_gesture(0))
    annotated_frame = HandGes.visualize(flipped_frame)
    cv2.imshow('annotated frame', annotated_frame)
    key = cv2.waitKey(1)&0xFF
    if key == ord('q'):
        break
cv2.destroyAllWindows()
HandGes.release()
cap.release()
```
- other sample
  - You can see it in the main function of `MediapipeHandGestureRecognition.py`


----
## MediapipeImageSegmentation
### class variable
- model information
  - `base_url`, `model_name`, `model_folder_path`
- skin type id
  - e.g. `BACKGROUND=0`, `HAIR=1`
### instance variable
- (`segmenter`: mediapipe segmenter)
- (`results`: mediapipe segmenter's results)
### method
- `__init__( arguments are optional )`: constructor
  - arguments
    - `model_folder_path`: If you want to change the model folder path
    - `base_url`: If you want to change the model
    - `model_name`: If you want to change the model
    - `output_category_mask`: If set to True, the output includes a segmentation mask as a uint8 image, where each pixel value indicates the winning category value.
      - Default Value: `True`
    - `output_confidence_masks`: If set to True, the output includes a segmentation mask as a float value image, where each float value represents the confidence score map of the category.
      - Default Value: `True`
- `detect( image )`
  - arguments
    - `image`: Input image (readed by cv2)
  - return values
    - `results`: Probably not necessary
- `get_segmentation_masks()`
  - return values
    - `segmentation_mask`: pixel value is skin type id.
      - `BACKGROUND = 0`
      - `HAIR = 1`
      - `BODY_SKIN = 2`
      - `FACE_SKIN = 3`
      - `CLOTHES = 4`
      - `OTHERS = 5`
- `mask = get_segmentation_mask( skin_type )`
  - segmentation mask of the input skin type
    - skin type region: `255`
    - others: `0`
- `get_normalized_masks()`
  - for visualizing
  - return values
    - `normalized_masks`: pixel value is normalized into `0-255`.
- `release()`: Close mediapipe's `detector`
### how to use
- simple sample
```python
import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
from MediapipeImageSegmentation import MediapipeImageSegmentation as ImgSeg

cap = cv2.VideoCapture(0)
Seg = ImgSeg()
while cap.isOpened():
    ret, frame = cap.read()
    Seg.detect(frame)
    normalized_masks = Seg.get_normalized_masks()
    cv2.imshow('multiclass mask', cv2.applyColorMap(normalized_masks, cv2.COLORMAP_JET))
    face_skin_masks = Seg.get_segmentation_mask(Seg.FACE_SKIN)
    cv2.imshow('face skin', face_skin_masks)
    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)&0xFF
    if key == ord('q'):
        break
cv2.destroyAllWindows()
Seg.release()
cap.release()
```
- other sample
  - You can see it in the main function of `MediapipeImageSegmentation.py`
  - segmentation models: [MediaPipe Site](https://developers.google.com/mediapipe/solutions/vision/image_segmenter#models)
    - this sample uses `selfieMulticlass (256x256)` model


----
## MediapipeFaceDetection
### class variable
- model information
  - `base_url`, `model_name`, `model_folder_path`
- visualize params
  - e.g. `TEXT_COLOR`, `FONT_SIZE`, ...
- blaze_face_short_range's id
  - e.g. `LEFT_EYE=0`, `RIGHT_EYE=1`
### instance variable
- `num_detected_faces`: number of detected faces
- (`detector`: mediapipe detector)
- (`results`: mediapipe detector's results)
### method
- `__init__( arguments are optional )`: constructor
  - arguments
    - `model_folder_path`: If you want to change the model folder path
    - `base_url`: If you want to change the model
    - `model_name`: If you want to change the model
    - `min_detection_confidence`: The minimum confidence score for the face detection to be considered successful.
      - Value Range: `0.0 - 1.0`
      - Default Value: `0.5`
    - `min_suppression_confidence`: The minimum non-maximum-suppression threshold for face detection to be considered overlapped.
      - Value Range: `0.0 - 1.0`
      - Default Value: `0.3`
- `detect( image )`
  - arguments
    - `image`: Input image (readed by cv2)
      - Input image is a frame image flipped holizontal! Otherwise, left eye and right eye are reversed.
  - return values
    - `results`: Probably not necessary
- `get_normalized_landmark( id_face, id_keypoint )`
  - arguments
    - `id_face`: ID number of the face you want to get normalized landmark coordinate
    - `id_keyporint`: ID number of the face keypoint you want to get normalized landmark coordinate
  - return values
    - `np.array([x, y])`: array of the coordinate
      - `x`: normalized x-coordinate, `y`: normallized y-coordinate
      - Value Range: `x:0.0-1.0`, `y:0.0-1.0`
- `get_landmark( id_face, id_keypoint )`
  - arguments
    - `id_face`: ID number of the face you want to get landmark coordinate
    - `id_keypoint`: ID number of the face keypoint you want to get landmark coordinate
  - return values
    - `np.array([x, y])`: array of the coordinate
      - `x`: x-coordinate, `y`: y-coordinate
      - Value Range: `x:0-width`, `y:0-height`
- `score = get_score( id_face )`
  - detection confidence score
- `annotated_image = visualize( image )`
  - `annotated_image`: Image with face RoI and face keypoints for all detected faces on the input image
- `release()`: Close mediapipe's `detector`
### how to use
- simple sample
```python
import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
from MediapipeFaceDetection import MediapipeFaceDetection as FaceDect

cap = cv2.VideoCapture(0)
Face = FaceDect()
while cap.isOpened():
    ret, frame = cap.read()
    flipped_frame = cv2.flip(frame, 1)
    Face.detect(flipped_frame)
    annotated_frame = Face.visualize(flipped_frame)
    cv2.imshow('frame', annotated_frame)
    key = cv2.waitKey(1)&0xFF
    if key == ord('q'):
        break
cv2.destroyAllWindows()
Face.release()
cap.release()
```
- other sample
  - You can see it in the main function of `MediapipeFaceDetection.py`


----
## MediapipeFaceLandmark
### class variable
- model information
  - `base_url`, `model_name`, `model_folder_path`
- visualize params
  - e.g. `FONT_COLOR`, `FONT_SIZE`, ...
### instance variable
- `num_detected_faces`: number of detected faces
- (`detector`: mediapipe detector)
- (`results`: mediapipe detector's results)
### method
- `__init__( arguments are optional )`: constructor
  - arguments
    - `model_folder_path`: If you want to change the model folder path
    - `base_url`: If you want to change the model
    - `model_name`: If you want to change the model
    - `num_faces`:  The maximum number of hands detected by the Hand landmark detector
      - Value Range: Any integer `> 0`
      - Default Value: `2`
    - `min_face_detection_confidence`: The minimum confidence score for the face detection to be considered successful.
      - Value Range: `0.0 - 1.0`
      - Default Value: `0.5`
    - `min_face_presence_confidence`: The minimum confidence score of face presence score in the face landmark detection.
      - Value Range: `0.0 - 1.0`
      - Default Value: `0.5`
    - `min_tracking_confidence`: The minimum confidence score for the face tracking to be considered successful.
      - Value Range: `0.0 - 1.0`
      - Default Value: `0.5`
    - `output_face_blendshapes`: Whether Face Landmarker outputs face blendshapes. Face blendshapes are used for rendering the 3D face model.
      - Default Value: `False`
    - `output_facial_transformation_matrixes`: Whether FaceLandmarker outputs the facial transformation matrix. FaceLandmarker uses the matrix to transform the face landmarks from a canonical face model to the detected face, so users can apply effects on the detected landmarks.
      - Default Value: `False`
- `detect( image )`
  - arguments
    - `image`: Input image (readed by cv2)
      - Input image is a frame image flipped holizontal! Otherwise, left and right of the face are reversed.
  - return values
    - `results`: Probably not necessary
- `get_normalized_landmark( id_face, id_landmark )`
  - arguments
    - `id_face`: ID number of the face you want to get normalized landmark coordinate
    - `id_landmark`: ID number of the face landmark you want to get normalized landmark coordinate
  - return values
    - `np.array([x, y, z])`: array of the coordinate
      - `x`: normalized x-coordinate, `y`: normallized y-coordinate, `z`: normallized z-coordinate
      - Value Range: `x:0.0-1.0`, `y:0.0-1.0`, `z:0.0-1.0`
- `get_landmark( id_face, id_landmark )`
  - arguments
    - `id_face`: ID number of the face you want to get landmark coordinate
    - `id_landmark`: ID number of the face landmark you want to get landmark coordinate
  - return values
    - `np.array([x, y, z])`: array of the coordinate
      - `x`: normalized x-coordinate, `y`: normallized y-coordinate, `z`: normallized z-coordinate
      - Value Range: `x:0-width`, `y:0-height`, `z:0-width`
- `annotated_image = visualize( image )`
  - `annotated_image`: Image with face landmarks for all detected faces on the input image
- `annotated_image = visualize_with_mp( image )`
  - mediapipe visualizing settings
- `release()`: Close mediapipe's `detector`
### how to use
- simple sample
```python
import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
from MediapipeFaceLandmark import MediapipeFaceLandmark as FaceLmk

cap = cv2.VideoCapture(0)
Face = FaceLmk()
while cap.isOpened():
    ret, frame = cap.read()
    flipped_frame = cv2.flip(frame, 1)
    Face.detect(flipped_frame)
    annotated_frame = Face.visualize(flipped_frame)
    cv2.imshow('frame', annotated_frame)
    key = cv2.waitKey(1)&0xFF
    if key == ord('q'):
        break
cv2.destroyAllWindows()
Face.release()
cap.release()
```
- other sample
  - You can see it in the main function of `MediapipeFaceLandmark.py`


----
## MediapipeFaceLandmark
### class variable
- model information
  - `base_url`, `model_name`, `model_folder_path`
- visualize params
  - e.g. `FONT_COLOR`, `FONT_SIZE`, ...
- pose landmark id
  - e.g. `NOSE=0`, `LEFT_EYE_INNER=1`
### instance variable
- `num_detected_poses`: number of detected poses
- (`detector`: mediapipe detector)
- (`results`: mediapipe detector's results)
### method
- `__init__( arguments are optional )`: constructor
  - arguments
    - `model_folder_path`: If you want to change the model folder path
    - `base_url`: If you want to change the model
    - `model_name`: If you want to change the model
    - `num_poses`:  The maximum number of poses detected by the Pose landmark detector
      - Value Range: Any integer `> 0`
      - Default Value: `2`
    - `min_pose_detection_confidence`: The minimum confidence score for the pose detection to be considered successful.
      - Value Range: `0.0 - 1.0`
      - Default Value: `0.5`
    - `min_pose_presence_confidence`: The minimum confidence score of pose presence score in the pose landmark detection.
      - Value Range: `0.0 - 1.0`
      - Default Value: `0.5`
    - `min_tracking_confidence`: The minimum confidence score for the pose tracking to be considered successful.
      - Value Range: `0.0 - 1.0`
      - Default Value: `0.5`
    - `output_segmentation_masks`: Whether Pose Landmarker outputs a segmentation mask for the detected pose.
      - Default Value: `True`
- `detect( image )`
  - arguments
    - `image`: Input image (readed by cv2)
  - return values
    - `results`: Probably not necessary
- `get_normalized_landmark( id_pose, id_landmark )`
  - arguments
    - `id_pose`: ID number of the pose you want to get normalized landmark coordinate
    - `id_landmark`: ID number of the pose landmark you want to get normalized landmark coordinate
  - return values
    - `np.array([x, y, z])`: array of the coordinate
      - `x`: normalized x-coordinate, `y`: normallized y-coordinate, `z`: normallized z-coordinate
      - Value Range: `x:0.0-1.0`, `y:0.0-1.0`, `z:0.0-1.0`
- `get_landmark( id_pose, id_landmark )`
  - arguments
    - `id_pose`: ID number of the pose you want to get landmark coordinate
    - `id_landmark`: ID number of the pose landmark you want to get landmark coordinate
  - return values
    - `np.array([x, y, z])`: array of the coordinate
      - `x`: normalized x-coordinate, `y`: normallized y-coordinate, `z`: normallized z-coordinate
      - Value Range: `x:0-width`, `y:0-height`, `z:0-width`
- `visibility_score = get_landmark_visibility( id_pose, id_landmark )`
  - If this score is low, you should not use its landmark.
- `presence_score = get_landmark_presence( id_pose, id_landmark )`
  - If this score is low, you should not use its landmark.
- `segmentated_mask = get_segmentation_mask( id_pose )`
  - `segmentated_mask`
    - Type: `np.ndarray()`
    - Range: `[0, 1]`
- `all_segmentated_masks = get_all_segmentation_mask()`
  - `all_segmentated_masks`
    - Logical OR of all segmentation masks
    - Type: `np.ndarray()`
    - Range: `0-255`
- `masked_image = visualize_mask( image, mask )`
  - `masked_image`: masked image by using input `mask`
- `annotated_image = visualize( image )`
  - `annotated_image`: Image with pose landmarks for all detected poses on the input image
- `annotated_image = visualize_with_mp( image )`
  - mediapipe visualizing settings
- `release()`: Close mediapipe's `detector`
### how to use
- simple sample
```python
import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
from MediapipePoseLandmark import MediapipePoseLandmark as PoseLmk

cap = cv2.VideoCapture(0)
Pose = PoseLmk()
while cap.isOpened():
    ret, frame = cap.read()
    Pose.detect(frame)
    masks = Pose.get_all_segmentation_masks()
    masked_frame = Pose.visualize_mask(frame, masks)
    annotated_frame = Pose.visualize(masked_frame)
    cv2.imshow('frame', annotated_frame)
    key = cv2.waitKey(1)&0xFF
    if key == ord('q'):
        break
cv2.destroyAllWindows()
Pose.release()
cap.release()
```
- other sample
  - You can see it in the main function of `MediapipePoseLandmark.py`


----













## specification of Mediapipe's `results`
### object_detection
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

### image_segmentation
- https://developers.google.com/mediapipe/solutions/vision/image_segmenter
- ***examples of how to reference result data `results`***
  - i-th segmented_mask (mediapipe image --> ndarray)<br>
    `results[0].numpy_view()`
- ***data structure of result***
  - results (mediapipe image)
      - 0:
        - height
        - width

### hand_landmarker
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


### face_detecion (not yet)
- https://developers.google.com/mediapipe/solutions/vision/face_detector


### face_landmarker
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

### pose_estimation
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
