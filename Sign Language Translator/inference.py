import pickle
import cv2
import mediapipe as mp
import numpy as np
import time

model_dict = pickle.load(open(r'C:\Users\SHREYAS\Desktop\testing on epics - Copy\model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'K', 10: 'L', 11: 'M', 12: 'N', 13: 'O', 14: 'P', 15: 'Q', 16: 'R',17: 'S',18: 'T',19: 'U',20: 'V',21: 'W',22: 'X',23: 'Y'}


start_time = time.time()
prev_time = start_time
prev_word_time = start_time
current_letter = ""
sentence = ""
detecting_gesture = False
confirming_gesture = False
word_completed = False

letter_cooldown = 2
word_cooldown = 4

while True:

    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

            prediction = model.predict([np.asarray(data_aux)])

            if prediction is not None:
                predicted_character = labels_dict[int(prediction[0])]

                if not detecting_gesture:
                    detecting_gesture = True
                    current_letter = predicted_character
                    start_time = time.time()
                    print("Letter detected:", current_letter)

                elif confirming_gesture:
                    if time.time() - prev_time >= letter_cooldown:
                        sentence += current_letter + ""
                        prev_time = time.time()
                        confirming_gesture = False
                        print("Letter confirmed:", current_letter)
                        current_letter = ""

                elif current_letter != predicted_character:
                    if time.time() - start_time >= letter_cooldown:
                        current_letter = predicted_character
                        start_time = time.time()
                        print("Letter detected:", current_letter)

                else:

                    if time.time() - start_time >= 1:
                        current_letter = ""
                        start_time = time.time()
                        print("Gesture repeated, reset")

    else:
        if detecting_gesture:
            detecting_gesture = False
            start_time = time.time()

    if current_letter and time.time() - prev_word_time >= word_cooldown:
        if word_completed:
            sentence += ""  # Add space between words
            word_completed = False
        sentence += current_letter  # Add current letter
        prev_word_time = time.time()
        print("Word completed:", sentence)
        current_letter = ""


    if not results.multi_hand_landmarks and sentence and not word_completed:
        sentence += " "
        word_completed = True

    cv2.putText(frame, f"Detected letter: {current_letter}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(frame, f"Sentence: {sentence}", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
