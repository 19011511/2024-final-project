from keras.preprocessing.image import img_to_array
import imutils
import cv2
from keras.models import load_model
import numpy as np

# parameters for loading data and images
detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'

# loading models
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

# With color change
# Emotion to color mapping
# emotion_colors = {
#     "angry": (0, 0, 255),      # Red
#     "disgust": (255, 255, 255), # White
#     "scared": (0, 0, 0),       # Black
#     "happy": (0, 255, 255),    # Yellow
#     "sad": (100, 100, 100),        # Gray
#     "surprised": (255, 0, 0),# Cyan
#     "neutral": (0, 255, 0),    # Green
# }

# starting video streaming
camera = cv2.VideoCapture(0)
# starting video streaming from a video file
# camera = cv2.VideoCapture('/Users/yukihan/Desktop/final/Emotion-recognition/sample/angry/li1.MP4')


while True:
    ret, frame = camera.read()
    if not ret:
        print("Unable to capture frame from camera")
        break

    frame = imutils.resize(frame, width=300)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    canvas = np.zeros((250, 300, 3), dtype="uint8")
    frameClone = frame.copy()

    if len(faces) > 0:
        faces = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = faces

        # Extract the ROI of the face from the grayscale image, resize it to a fixed 28x28 pixels, and then prepare
        # the ROI for classification via the CNN
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (64, 64))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        preds = emotion_classifier.predict(roi)[0]
        emotion_probability = np.max(preds)
        label = EMOTIONS[preds.argmax()]
        # Without color change
        color = (0, 0, 255)
        # With color change
        # color = emotion_colors[label]  # Get the color based on the emotion

         # Draw shapes based on emotion
        if label == "angry":
            # Draw a zigzag shape with 9 teeth
            for i in range(9):
                start_point = (fX + i * (fW // 8), fY if i % 2 == 0 else fY + fH)
                end_point = (fX + (i + 1) * (fW // 8), fY + fH if i % 2 == 0 else fY)
                cv2.line(frameClone, start_point, end_point, color, 2)
        elif label == "scared":
            # Draw an inverted triangle with a larger height
            triangle_cnt = np.array([
                (fX + fW // 2, fY + fH + fH // 2),  # Move the apex downwards
                (fX - fW // 4, fY), 
                (fX + fW + fW // 4, fY)
            ])
            cv2.drawContours(frameClone, [triangle_cnt], 0, color, 2)
        elif label == "sad":
            # Draw four vertical three-fold wave lines
            num_waves = 4
            wave_height = fH // 3  # Height of each wave
            wave_spacing = fW // (num_waves + 1)  # Spacing between the waves

            for i in range(1, num_waves + 1):
                # X position for the current wave
                x_pos = fX + i * wave_spacing

                # Draw a three-fold wave vertically
                for j in range(3):
                    if j % 2 == 0:
                        # Wave going down
                        start_point = (x_pos, fY + j * wave_height)
                        end_point = (x_pos + wave_spacing // 2, fY + (j + 1) * wave_height)
                    else:
                        # Wave going up
                        start_point = (x_pos + wave_spacing // 2, fY + j * wave_height)
                        end_point = (x_pos, fY + (j + 1) * wave_height)

                    # Draw the wave segment
                    cv2.line(frameClone, start_point, end_point, color, 2)

        elif label == "neutral":
            # Draw a rectangle
            cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), color, 2)
        elif label == "happy":
            # Draw a circle
            center = (fX + fW // 2, fY + fH // 2)
            radius = fW // 2
            cv2.circle(frameClone, center, radius, color, 2)
        elif label == "surprised":
            # Draw a star (5-pointed)
            star_points = []
            for i in range(5):
                outer_x = fX + int(fW / 2 + fW / 2 * np.cos(np.pi/2 + 2 * np.pi * i / 5))
                outer_y = fY + int(fH / 2 - fH / 2 * np.sin(np.pi/2 + 2 * np.pi * i / 5))
                star_points.append((outer_x, outer_y))
                inner_x = fX + int(fW / 2 + fW / 3 * np.cos(np.pi/2 + 2 * np.pi * (i + 0.5) / 5))
                inner_y = fY + int(fH / 2 - fH / 3 * np.sin(np.pi/2 + 2 * np.pi * (i + 0.5) / 5))
                star_points.append((inner_x, inner_y))
            cv2.polylines(frameClone, [np.array(star_points, np.int32)], isClosed=True, color=color, thickness=2)


        cv2.putText(frameClone, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

    else:
        continue

    for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
        # construct the label text
        text = "{}: {:.2f}%".format(emotion, prob * 100)
        # color = emotion_colors[emotion]  # Get the color for the emotion

        # draw the label + probability bar on the canvas
        w = int(prob * 300)
        cv2.rectangle(canvas, (7, (i * 35) + 5),
                      (w, (i * 35) + 35), (0, 0, 255), -1)
        cv2.putText(canvas, text, (10, (i * 35) + 23), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 2)

    cv2.imshow('your_face', frameClone)
    cv2.imshow("Probabilities", canvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()