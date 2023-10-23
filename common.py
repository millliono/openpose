coco_keypoints = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "neck",
]
connect_skeleton = [
    (0, 1),  # nose -> left_eye
    (0, 2),  # nose -> right_eye
    (1, 3),  # left_eye -> left_ear
    (2, 4),  # right_eye -> right_ear
    (0, 17),  # nose -> neck
    (17, 5),  # neck -> left_shoulder
    (17, 6),  # neck -> right_shoulder
    (5, 7),  # left_shoulder -> left_elbow
    (6, 8),  # right_shoulder -> right_elbow
    (7, 9),  # left_elbow -> left_wrist
    (8, 10),  # right_elbow -> right_wrist
    (17, 11),  # neck -> left_hip
    (17, 12),  # neck -> right_hip
    (11, 13),  # left_hip -> left_knee
    (12, 14),  # right_hip -> right_knee
    (13, 15),  # left_knee -> left_ankle
    (14, 16),  # right_knee -> right_ankle
]
