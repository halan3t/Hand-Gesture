import cv2
import mediapipe as mp
from gtts import gTTS
import pygame
import threading
import time
import os
import warnings
from PIL import Image, ImageDraw, ImageFont
import numpy as np

warnings.filterwarnings("ignore")

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def play_audio(text):
    filename = "voice.mp3"

    tts = gTTS(text=text, lang="ko")
    tts.save(filename)

    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        time.sleep(0.1)

    pygame.mixer.quit()

    if os.path.exists(filename):
        os.remove(filename)

def detect_gesture(landmarks):
    thumb_tip = landmarks.landmark[4].y
    index_tip = landmarks.landmark[8].y
    middle_tip = landmarks.landmark[12].y
    ring_tip = landmarks.landmark[16].y
    pinky_tip = landmarks.landmark[20].y

    thumb_base = landmarks.landmark[2].y
    index_base = landmarks.landmark[5].y

    if (
        thumb_tip < thumb_base
        and index_tip < index_base
        and middle_tip < index_base
        and ring_tip < index_base
        and pinky_tip < index_base
    ):
        return "👋", "안녕하세요"

    if (
        index_tip < index_base
        and middle_tip > index_base
        and ring_tip > index_base
        and pinky_tip > index_base
    ):
        return "👧", "제 이름은 할라입니다"

    if (
        index_tip > index_base
        and middle_tip > index_base
        and ring_tip > index_base
        and pinky_tip > index_base
    ):
        return "🤝", "만나서 반갑습니다"

    if (
        index_tip < index_base
        and pinky_tip < index_base
        and middle_tip > index_base
        and ring_tip > index_base
    ):
        return "❤️", "감사합니다"

    return None, None

cap = cv2.VideoCapture(0)

last_gesture = None
last_time = 0

gesture_colors = {
    "👋": (255, 0, 0),
    "👧": (0, 255, 0),
    "🤝": (0, 165, 255),
    "❤️": (255, 0, 255)
}

with mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    while True:
        ret, frame = cap.read()

        if not ret:
            continue

        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        emoji = None
        speech_text = None

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:

                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                emoji, speech_text = detect_gesture(hand_landmarks)

        if emoji:
            color = gesture_colors.get(emoji, (255, 255, 255))

            pil_img = Image.fromarray(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            )

            draw = ImageDraw.Draw(pil_img)

            font = ImageFont.truetype(
                "C:/Windows/Fonts/seguiemj.ttf",
                80
            )

            draw.text(
                (50, 30),
                emoji,
                font=font,
                fill=(color[2], color[1], color[0])
            )

            frame = cv2.cvtColor(
                np.array(pil_img),
                cv2.COLOR_RGB2BGR
            )

            if speech_text != last_gesture and time.time() - last_time > 2:
                threading.Thread(
                    target=play_audio,
                    args=(speech_text,)
                ).start()

                last_gesture = speech_text
                last_time = time.time()

        cv2.imshow("Hand Gesture Recognition", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()