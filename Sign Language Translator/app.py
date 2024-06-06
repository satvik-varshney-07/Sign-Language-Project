import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
import pickle

def main():
    st.title("Sign Language App for EPICS by Group-123")

    model_dict = pickle.load(open(r'C:\Users\SHREYAS\Desktop\testing on epics - Copy - Copy\model.p', 'rb'))
    model = model_dict['model']

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

    labels_dict = {
        0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'K',
        10: 'L', 11: 'M', 12: 'N', 13: 'O', 14: 'P', 15: 'Q', 16: 'R', 17: 'S', 18: 'T',
        19: 'U', 20: 'V', 21: 'W', 22: 'X', 23: 'Y'
    }

    cap = cv2.VideoCapture(0)

    start_time = time.time()
    prev_time = start_time
    prev_word_time = start_time
    sentence = ""
    word = ""
    word_completed = False

    letter_cooldown = 3
    word_cooldown = 6

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "What is Sign Language", "How Does the Code Work", "Translate"])

    if page == "Home":
        st.header("Welcome to Gesture Recognition App!")

        st.write("""
            This application is designed to help you translate American Sign Language (ASL) gestures into text. Whether you're learning ASL, communicating with the hearing-impaired, or simply curious about sign language, this tool will assist you in understanding and interpreting hand gestures.
            """)

        st.subheader("How it Works:")
        st.write("""
            Simply use your webcam to capture hand gestures. The application will detect your hand movements and translate them into text in real-time. As you make different gestures, the corresponding letters or words will be displayed on the screen. You can also see the completed words and sentences on the right side of the webcam feed.
            """)

        st.subheader("Get Started:")
        st.write("""
            To begin using the ASL Sign Language Translator, navigate to the Translate section using the navigation bar at the top. Start making hand gestures, and see the translation in real-time!
            """)

        st.info(
            "Note: This application is for educational and informational purposes only. It may not accurately interpret all ASL gestures and should not be relied upon for critical communication.")

        st.write("""
            We hope you find this tool useful and informative. Enjoy exploring the world of ASL!
            """)


    elif page == "What is Sign Language":
        st.header("What is Sign Language?")
        with st.container():
            st.write("""
            American Sign Language (ASL) is a complete, complex language that uses hand shapes, facial expressions, and body movements to convey meaning. It is the primary language used by many Deaf and hard-of-hearing individuals in the United States and parts of Canada.
            """)

        # Key Components section
        st.subheader("Key Components of ASL")
        st.write("""
        ASL utilizes various elements to convey meaning:
        - **Handshapes:** Different handshapes represent letters, words, and concepts in ASL. Each handshape is formed using specific finger positions and movements.
        - **Facial Expressions:** Facial expressions are essential in ASL communication. They convey emotion, tone, and grammatical information.
        - **Body Movements:** Body movements, such as posture shifts and spatial orientations, contribute to the overall meaning of signs and messages.
        """)

        # Why ASL is Important section
        st.subheader("Why ASL is Important")
        st.write("""
        ASL plays a crucial role in:
        - **Accessibility:** ASL provides a means of communication for Deaf and hard-of-hearing individuals, ensuring inclusivity and accessibility.
        - **Cultural Identity:** ASL is an integral part of Deaf culture, fostering connections and community among individuals who share the language.
        - **Inclusive Communication:** Learning ASL promotes inclusivity and enables effective communication across diverse communities.
        """)

        # Get Involved section
        st.subheader("Get Involved")
        st.write("""
        Interested in learning ASL or becoming proficient in sign language? Explore online courses, community classes, and local Deaf events to immerse yourself in the vibrant ASL community. Take the first step towards inclusive communication today!
        """)

    elif page == "How Does the Code Work":
        st.header("How Does the Code Work?")

        st.write(
            "The sign language recognition system utilizes computer vision and machine learning techniques to translate hand gestures into text.")
        st.write("Here's a brief overview of how the code works:")

        st.subheader("1. Hand Detection and Tracking")
        st.write("The system uses the MediaPipe library to detect and track hand landmarks in the webcam feed.")
        st.write("Each hand landmark corresponds to a specific point on the hand, such as fingertips and palm.")

        st.subheader("2. Feature Extraction")
        st.write("Once the hand landmarks are detected, the system extracts relevant features from the hand gestures.")
        st.write("These features include the position, orientation, and movement of the hand landmarks.")

        st.subheader("3. Machine Learning Model")
        st.write(
            "A machine learning model, trained on a dataset of hand gestures, predicts the corresponding letters or words.")
        st.write(
            "The model analyzes the extracted features and maps them to predefined gestures, such as sign language alphabets.")

        st.subheader("4. Real-time Translation")
        st.write(
            "As the user makes hand gestures in front of the webcam, the system translates them into text in real-time.")
        st.write(
            "The translated text is displayed on the screen, allowing users to understand the meaning of their gestures.")



    elif page == "Translate":
        st.header("Translate")
        col1, col2 = st.columns([2, 1])

        # Webcam Feed
        with col1:
            st.subheader("Webcam Feed")
            video_stream = st.empty()  # Placeholder for displaying the webcam feed

        # Completed Words
        with col2:
            st.subheader("Completed Words")
            completed_words_text = st.empty()  # Placeholder for displaying completed words
            st.subheader("Sentence")
            sentence_text = st.empty()  # Placeholder for displaying the sentence

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

                    x1 = int(min(x_) * W) - 10
                    y1 = int(min(y_) * H) - 10

                    x2 = int(max(x_) * W) - 10
                    y2 = int(max(y_) * H) - 10

                    prediction = model.predict([np.asarray(data_aux)])

                    predicted_character = labels_dict[int(prediction[0])]

                    if time.time() - prev_time >= letter_cooldown:
                        word += predicted_character
                        prev_time = time.time()
                        st.write("Letter detected:", predicted_character)

            if not results.multi_hand_landmarks:
                if time.time() - prev_word_time >= word_cooldown:
                    if word:
                        sentence += word + " "
                        completed_words_text.text("Word completed: " + word)
                        word = ""
                        st.write("Word completed:", sentence)
                    else:
                        sentence += " "
                        st.write("No gesture detected, adding space")

                    prev_word_time = time.time()
                    sentence_text.text("Sentence: " + sentence)

            # Display the webcam feed
            video_stream.image(frame, channels="RGB")

    cap.release()

if __name__ == "__main__":
    main()


#streamlit run "C:/Users/SHREYAS/Desktop/testing on epics - Copy - Copy/app.py" [ARGUMENTS]