import cv2
import mediapipe as mp
import torch
from torch.utils.data import Dataset
from argparse import ArgumentParser
def arg_parser():
    parser = ArgumentParser()
    parser.add_argument('--train_videos_csv', type=str, default='data/train.csv')
    parser.add_argument('--test_videos_csv', type=str, default='data/test.csv')
    parser.add_argument('--train_features_path', type=str, default='data/npy/train_hand_keypoints.npy')
    parser.add_argument('--train_labels_path', type=str, default='data/npy/train_hand_labels.npy')
    parser.add_argument('--test_features_path', type=str, default='data/npy/test_hand_keypoints.npy')
    parser.add_argument('--test_labels_path', type=str, default='data/npy/test_hand_labels.npy')
    return parser.parse_args()
def train_args():
    parser = ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--save_model_path', type=str, required=True)
    return parser.parse_args()

class NPYDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return feature, label
    
def get_hands_key_points(frame, mp_hands, width=256, height=256):
    frame_small = cv2.resize(frame, (width, height))
    image_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
    keypoints = []
    with mp_hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        results = hands.process(image_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i, point in enumerate(mp_hands.HandLandmark):
                    normalized_landmark = hand_landmarks.landmark[point]
                    if normalized_landmark:
                        keypoints[3*i:3*i+3]=[normalized_landmark.x, normalized_landmark.y, normalized_landmark.z]
    if all(keypoint == 0 for keypoint in keypoints):
        return None
    else:
        return keypoints
    
def write_hand_keypoints(video_file, video_label, width, height, flip=True):
    print(f'Processing video file: {video_file}')
    mp_hands = mp.solutions.hands
    cap = cv2.VideoCapture(video_file)
    frame_list = []
    label_list = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        keypoints = get_hands_key_points(frame, mp_hands, width, height)
        if keypoints is not None:
            frame_list.append(keypoints)
            label_list.append(video_label)
        if flip:
            flip_frame = cv2.flip(frame, 1)
            keypoints = get_hands_key_points(flip_frame, mp_hands, width, height)
            if keypoints is not None:
                frame_list.append(keypoints)
                label_list.append(video_label)
            
    cap.release()
    cv2.destroyAllWindows()
    return frame_list, label_list