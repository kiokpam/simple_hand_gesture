import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import sys
from pathlib import Path
root_path = Path(__file__).resolve().parents[1]
sys.path.append(str(root_path))
from utils.data_utils import write_hand_keypoints, arg_parser
def main():
    args = arg_parser()
    print(args.train_features_path, args.train_labels_path, args.test_features_path, args.test_labels_path)
    train_data = pd.read_csv(args.train_videos_csv)
    test_data = pd.read_csv(args.test_videos_csv)
    print(f'Train data shape: {train_data.shape}')
    train_frame_list = []
    train_label_list = []
    test_frame_list = []
    test_label_list = []
    for i in range(len(train_data)):
        video_file = train_data['video_name'][i]
        video_label = train_data['tag'][i]
        frame_list, label_list = write_hand_keypoints(video_file, video_label, width=256, height=256)
        train_frame_list.extend(frame_list)
        train_label_list.extend(label_list)
    for i in range(len(test_data)):
        video_file = test_data['video_name'][i]
        video_label = test_data['tag'][i]
        frame_list, label_list = write_hand_keypoints(video_file, video_label, width=256, height=256, flip=False)
        test_frame_list.extend(frame_list)
        test_label_list.extend(label_list)
    lb = LabelEncoder()
    train_label_list = lb.fit_transform(train_label_list)
    test_label_list = lb.transform(test_label_list)
    print(f'Train data shape: {len(train_frame_list)}')
    print(f'Test data shape: {len(test_frame_list)}')
    np.save(args.train_features_path, np.array(train_frame_list))
    np.save(args.train_labels_path, np.array(train_label_list))
    np.save(args.test_features_path, np.array(test_frame_list))
    np.save(args.test_labels_path, np.array(test_label_list))
if __name__ == '__main__':  
    main()

