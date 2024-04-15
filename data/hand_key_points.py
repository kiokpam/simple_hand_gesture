import cv2
import mediapipe as mp
import pickle
import numpy as np
import os


def write_info(video_file, csv_file=r'C:\Filecuakio\FPTUni\SPRING24\CPV\Video-Classifier-Using-CNN-and-RNN\dataset\traning_data.csv'):
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    cap = cv2.VideoCapture(video_file)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    with mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
        
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
            results = hands.process(image_rgb)


            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                  keypoints = []
                  for point in mp_hands.HandLandmark:
                      normalized_landmark = hand_landmarks.landmark[point]
                      pixel_coordinates_landmark = mp_drawing._normalized_to_pixel_coordinates(normalized_landmark.x, normalized_landmark.y, frame_width, frame_height)
                      mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                      if pixel_coordinates_landmark:
                        x, y = pixel_coordinates_landmark
                        keypoints.append(pixel_coordinates_landmark)
                        cv2.putText(frame, f"x: {x}, y: {y}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                  # with open(csv_file, 'a') as f:
                  #     f.write(f'{keypoints},{os.path.splitext(os.path.basename(video_file))[0]}\n')
            frame = cv2.resize(frame, (frame_width//2, frame_height//2))

            cv2.imshow('Prediction', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
              break
write_info(r'C:\Filecuakio\FPTUni\SPRING24\CPV\files\0\four.mp4')
# for root, dir, file in os.walk(r"C:\Filecuakio\FPTUni\SPRING24\CPV\files"):
#     for f in file:
#         print(f)
#         if f.endswith('.mp4'):
#             video_path = os.path.join(root, f)
#             cap = cv2.VideoCapture(video_path)
#             frame_width = int(cap.get(3))
#             frame_height = int(cap.get(4))
#             fps = int(cap.get(cv2.CAP_PROP_FPS))
#             csv_file = os.path.join(r'C:\Filecuakio\FPTUni\SPRING24\CPV\files\traning_data.csv')
#             write_info(video_path, csv_file)
#             cap.release()
#             cv2.destroyAllWindows()
# write_info(mp_drawing, mp_hands, cap, frame_width, frame_height, out, csv_file)
# cap=cv2.VideoCapture(r"C:\Filecuakio\FPTUni\SPRING24\CPV\files\.csv")
