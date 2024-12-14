import cv2
import mediapipe as mp
import pyautogui

# Initialize Mediapipe Hands and Drawing Utility
mp_hands = mp.solutions.hands
capture_hands = mp_hands.Hands(min_detection_confidence=0.5, 
                               min_tracking_confidence=0.5, 
                               model_complexity=0)
drawing_option = mp.solutions.drawing_utils

# Initialize Camera
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

screen_width, screen_height = pyautogui.size()
prev_mouse_x, prev_mouse_y = 0, 0

# Check if the camera is opened successfully
if not camera.isOpened():
    print("Error: Could not open the camera.")
    exit()

x1 = y1 = x2 = y2 = 0
frame_count = 0

while True:
    # Capture frame from the camera
    ret, image = camera.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Skip every alternate frame to reduce lag
    frame_count += 1
    if frame_count % 2 != 0:
        continue

    # Flip and process the frame
    image = cv2.flip(image, 1)
    image_height, image_width, _ = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    output_hands = capture_hands.process(rgb_image)
    all_hands = output_hands.multi_hand_landmarks

    # Process detected hands
    if all_hands:
        for hand in all_hands:
            drawing_option.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS)
            one_hand_landmarks = hand.landmark

            for id, lm in enumerate(one_hand_landmarks):
                # Calculate pixel coordinates of landmarks
                x = int(lm.x * image_width)
                y = int(lm.y * image_height)

                # Index finger tip (landmark 8)
                if id == 8:
                    mouse_x = int(screen_width * x / image_width)
                    mouse_y = int(screen_height * y / image_height)
                    cv2.circle(image, (x, y), 10, (0, 255, 255), -1)

                    # Move mouse only if the position change is significant
                    if abs(mouse_x - prev_mouse_x) > 5 or abs(mouse_y - prev_mouse_y) > 5:
                        pyautogui.moveTo(mouse_x, mouse_y)
                        prev_mouse_x, prev_mouse_y = mouse_x, mouse_y

                    x1, y1 = x, y

                # Thumb tip (landmark 4)
                if id == 4:
                    cv2.circle(image, (x, y), 10, (255, 0, 0), -1)
                    x2, y2 = x, y

            # Calculate distance between index finger tip and thumb tip
            dist = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            print(f"Distance: {dist}")

            # Perform click action if distance is below threshold
            if dist < 40:
                pyautogui.click()
                print("Clicked!")

    # Display the frame
    cv2.imshow("Hand Movement Video Capture", image)

    # Exit on pressing the 'Esc' key
    key = cv2.waitKey(1)
    if key == 27:  # ASCII code for 'Esc' key
        break

# Release camera and close all windows
camera.release()
cv2.destroyAllWindows()
