import torch as torch
import cv2
import mediapipe as mp
from pathlib import Path
import sys
root_path = Path(__file__).resolve().parents[1]
sys.path.append(str(root_path))
from utils.data_utils import get_hands_key_points
model = torch.load('checkpoints/best.pth')['model']
model.load_state_dict(torch.load('checkpoints/best.pth')['state_dict'])
model.eval()
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
frame_width = 1920
frame_height = 1080
pred_dict={0: 'Fist', 1: 'Four', 2: 'Me', 3: 'One', 4:'Small'}
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    keypoints = get_hands_key_points(frame, mp_hands, width=256, height=256)
    if keypoints is not None:
        keypoints = torch.tensor(keypoints, dtype=torch.float32).unsqueeze(0).to('cuda')
        with torch.no_grad():
            outputs = model(keypoints)
            pred = torch.argmax(outputs, 1).item()
            cv2.putText(frame, pred_dict[pred], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow('Hand Gesture Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
