import cv2
import numpy as np
import mediapipe as mp


mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils


draw_color = (255, 255, 255)
erase_color = (0, 0, 0)
highlight_color_drawing = (255, 0, 0)
highlight_color_idle = (255, 255, 224)
highlight_color_erasing = (255, 255, 0)


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


canvas = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)


prev_x, prev_y = 0, 0


drawing = False
erasing = False
show_traces = False


def draw_line(canvas, start, end, color, thickness=2):
    cv2.line(canvas, start, end, color, thickness)

def erase_area(canvas, center, radius, color):
    cv2.circle(canvas, center, radius, color, -1)

def highlight_finger(frame, x, y, color, radius):
    cv2.circle(frame, (x, y), radius, color, 2)


while True:
    ret, frame = cap.read()
    if not ret:
        print("no image input. quitting")
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks and show_traces:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = hand_landmarks.landmark
            index_tip_x, index_tip_y = int(landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * frame.shape[1]), int(landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * frame.shape[0])
            if drawing:
                highlight_finger(frame, index_tip_x, index_tip_y, highlight_color_drawing, 10)
            elif erasing:
                highlight_finger(frame, index_tip_x, index_tip_y, highlight_color_erasing, 50)
            else:
                highlight_finger(frame, index_tip_x, index_tip_y, highlight_color_idle, 10)
            if drawing:
                if prev_x != 0 and prev_y != 0 and (index_tip_x, index_tip_y) != (0, 0):
                    draw_line(canvas, (prev_x, prev_y), (index_tip_x, index_tip_y), draw_color)
                prev_x, prev_y = index_tip_x, index_tip_y
            if erasing:
                erase_area(canvas, (index_tip_x, index_tip_y), 50, erase_color)

    if show_traces:
        frame_with_canvas = cv2.add(frame, canvas)
    else:
        frame_with_canvas = frame.copy()


    window_width = 640
    window_height = 360
    frame_resized = cv2.resize(frame_with_canvas, (window_width, window_height))
    cv2.imshow('Camera Feed', frame_resized)


    key = cv2.waitKey(1)
    if key == 27 or key == ord('q'):
        print("quitting...")
        break
    elif key == ord('e'):
        drawing = False
        erasing = True
    elif key == ord('d'):
        erasing = False
        drawing = True
    elif key == ord('r'):
        canvas = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    elif key == ord('s'):
        drawing = False
        erasing = False
    elif key == ord('p'):
        show_traces = not show_traces
        

cap.release()
cv2.destroyAllWindows()
