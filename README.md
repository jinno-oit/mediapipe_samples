# mediapipe_samples
- [MediaPipe Examples](https://developers.google.com/mediapipe/solutions/examples)

## object_detection
- https://developers.google.com/mediapipe/solutions/vision/object_detector
- ***examples of how to reference result data `results`***
  - x-coordinate of upper left point of the i-th object's bounding box
    `results.detections[i].bounding_box.origin_x`
  - category_name (e.g.`parson`) of the i-th object
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
        - categories[0]
          - 0:
            - category_name
            - score
      - 1:
        - bounding_box
          - origin_x
          - origin_y
          - width
          - height
        - categories[0]
          - 0:
            - category_name
            - score
      - ...

## image_segmentation
- https://developers.google.com/mediapipe/solutions/vision/image_segmenter
- ***examples of how to reference result data `results`***
  - i-th segmented_mask (mediapipe image --> ndarray)
    `results[0].numpy_view()`
- ***data structure of result***
  - results (mediapipe image)
      - 0:
        - height
        - width

## hand_landmarker
- https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
- ***examples of how to reference result data `results`***
  - normalized x-coordinate the j-th landmark of the i-th hand
    `results.hand_landmarks[i][j].x`
  - x-coordinate the j-th landmark of the i-th hand
    `int(results.hand_landmarks[i][j].x * width)`
  - category_name (e.g.`Right`) of the i-th hand
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


## face_landmarker (not yet)
- https://developers.google.com/mediapipe/solutions/vision/face_landmarker


## pose_estimation (not yet)
- https://developers.google.com/mediapipe/solutions/vision/pose_landmarker

