# ----- human skeleton including neck keypoint (generated) ----------------
coco_keypoints = [
    "nose",           #0
    "left_eye",       #1
    "right_eye",      #2
    "left_ear",       #3
    "right_ear",      #4
    "left_shoulder",  #5
    "right_shoulder", #6
    "left_elbow",     #7
    "right_elbow",    #8
    "left_wrist",     #9
    "right_wrist",    #10
    "left_hip",       #11
    "right_hip",      #12
    "left_knee",      #13
    "right_knee",     #14
    "left_ankle",     #15
    "right_ankle",    #16
    "neck",           #17
]
connect_skeleton = [
    (0, 1),    #0   nose -> left_eye
    (0, 2),    #1   nose -> right_eye
    (1, 3),    #2   left_eye -> left_ear
    (2, 4),    #3   right_eye -> right_ear
    (3, 5),    #4   left_ear -> left_shoulder
    (4, 6),    #5   right_ear -> right_shoulder
    (0, 17),   #6   nose -> neck
    (17, 5),   #7   neck -> left_shoulder
    (17, 6),   #8   neck -> right_shoulder
    (5, 7),    #9  left_shoulder -> left_elbow
    (6, 8),    #10  right_shoulder -> right_elbow
    (7, 9),    #11  left_elbow -> left_wrist
    (8, 10),   #12  right_elbow -> right_wrist
    (17, 11),  #13  neck -> left_hip
    (17, 12),  #14  neck -> right_hip
    (11, 13),  #15  left_hip -> left_knee
    (12, 14),  #16  right_hip -> right_knee
    (13, 15),  #17  left_knee -> left_ankle
    (14, 16),  #18  right_knee -> right_ankle
]
