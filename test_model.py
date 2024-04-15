import tensorflow as tf
import cv2
import mediapipe as mp

model = tf.keras.models.load_model(r'C:\Filecuakio\FPTUni\SPRING24\CPV\Video-Classifier-Using-CNN-and-RNN\dataset\ckpt.h5')
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
frame_width = 1920
frame_height = 1080
pred_dict={-1: 'None', 0: 'Fist', 1: 'Four', 2: 'Me', 3: 'One', 4:'Small'}

with mp_hands.Hands(
min_detection_confidence=0.5,
min_tracking_confidence=0.5) as hands:

  while True:
    ret, frame = cap.read()
    if not ret:
      break

  
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
    # Phát hiện keypoint của bàn tay
    results = hands.process(image_rgb)

    # Vẽ keypoint lên frame
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        keypoints = []
        for point in mp_hands.HandLandmark:
          normalized_landmark = hand_landmarks.landmark[point]
          pixel_coordinates_landmark = mp_drawing._normalized_to_pixel_coordinates(normalized_landmark.x, normalized_landmark.y, frame_width, frame_height)
          mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
          if pixel_coordinates_landmark:
            x,y=pixel_coordinates_landmark
            keypoints.append(x)
            keypoints.append(y)
            # cv2.putText(frame, f"x: {x}, y: {y}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        try:
          keypoints = tf.convert_to_tensor(keypoints, dtype=tf.float32)
          keypoints=tf.reshape(keypoints, (1, 42))
          prediction = model.predict(keypoints)
          prediction = tf.argmax(prediction, axis=1).numpy()
          prediction = prediction[0]
        except:
          prediction = -1
    else:
      prediction = -1
           
    cv2.putText(frame, str(pred_dict[prediction]), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Prediction', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
cap.release()
cv2.destroyAllWindows()