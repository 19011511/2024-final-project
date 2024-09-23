from keras.preprocessing.image import img_to_array
import random  
import imutils
import cv2
from keras.models import load_model
import numpy as np
import os
import pandas as pd
import keras
import tensorflow as tf
import matplotlib.pyplot as plt

# Get and print versions of these libraries
print("Keras version:", keras.__version__)
print("imutils version:", imutils.__version__)
print("OpenCV version:", cv2.__version__)
print("NumPy version:", np.__version__)
print("TensorFlow version:", tf.__version__)

# parameters for loading data and images
detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'

# loading models
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

# Define colors for emotions
emotion_colors = {
    "angry": (0, 0, 255),      # Red
    "disgust": (255, 255, 255), # White
    "scared": (0, 0, 0),       # Black
    "happy": (0, 255, 255),    # Yellow
    "sad": (100, 100, 100),    # Gray
    "surprised": (255, 0, 0),  # Blue
    "neutral": (0, 255, 0),    # Green
}

# Define emotion phrases
emotion_phrases = {
    "angry": {
        "recognition": ["Betrayal, Humiliation", "Deception, Infringement", "Coercion, Injustice", "Exclusion, Bullying","You're not good enough", "You always make mistakes", "This is a joke"],
        "inquiry": ["What triggered your current anger?", "Does this make you feel unfair?"],
        "confirmation": ["I feel very angry, I need to calm down.", "I'm very angry, I need some time to process."],
        "encouragement": ["I understand your point of view, we can find a balance together.", "I know we have some differences, but I really care about your feelings."],
        "release": ["Take a deep breath and calm down.", "Write down your anger on paper and tear it up."],
    },
    "sad": {
        "recognition": ["Parting, Loss", "Failure, Pain", "Disappointment, Loneliness", "Sadness, Depression", "Despair, Regret", "You don't get it", "This is a joke", "You always disappoint people"],
        "inquiry": ["What makes you feel so sad?", "What did you lose that makes you so sad?"],
        "confirmation": ["I feel very sad, and this emotion is real.", "I have lost something that makes me so sad."],
        "encouragement": ["You have done very well, and you will get better.", "Every failure is an opportunity for progress.","Don't worry, we still have a chance to adjust."],
        "release": ["Allow yourself to feel sad, tears are the beginning of healing.", "Find a quiet place and listen to your inner voice."],
    },
    "scared": {
        "recognition": ["Threats, Danger", "Terror, Violence", "Nightmares, Fear", "Tragedy, Sickness", "If you don't do this, there will be serious consequences", "You will fail", "You're not good enough"],
        "inquiry": ["What are you most afraid of right now?", "Is this fear coming from past experiences, or from uncertainty about the future?"],
        "confirmation": ["I feel scared, and this emotion makes me uneasy.", "I am experiencing fear, and the feeling is real."],
        "encouragement": ["You have the ability to do it, you just need some time.", "You've done very well, keep working hard and you will get better."],
        "release": ["Take a deep breath and tell yourself, 'I am safe, this is just a feeling of fear.'", "Imagine yourself standing in the light, the fear is gradually dissipating."],
    },
     "neutral": {
        "positive": ["Quiet", "Serene", "Relaxed", "Comfortable", "Calm", "Peaceful", "Loving", "Relax, we can work it out.", "No matter what happens, I will be here for you.", "Everyone makes mistakes, and we can learn from them.", "Don't worry, we still have a chance to adjust."]
    },
    "happy": {
        "positive": ["Joy", "Laughter", "Happiness","Good job!", "Your efforts are truly commendable.", "It's great that we did it together!"]
    },
    "surprised": {
        "positive": [ "Wonder", "Anticipation", "Discovery", "Wow", "What a surprise!", "You really did more than I could imagine, I'm so surprised!", "What unexpected good news!"]
    }
}

def draw_text_with_wrap(image, text, position, font_scale, color, thickness, max_width):
    # Split the text into words
    words = text.split(' ')
    current_line = ""
    y = position[1]

    for word in words:
        # Try adding the word to the current line
        test_line = current_line + word + " "
        # Calculate the width of the test line
        (line_width, _) = cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]

        if line_width <= max_width:
            # If the line width is within the limit, add the word to the current line
            current_line = test_line
        else:
            # Otherwise, draw the current line and start a new line
            cv2.putText(image, current_line, (position[0], y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
            # Move to the next line
            y += int(30 * font_scale)
            current_line = word + " "  # Start the new line with the current word

    # Draw the last line
    cv2.putText(image, current_line, (position[0], y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

# starting video streaming
camera = cv2.VideoCapture(1)
# starting video streaming from a video file
# camera = cv2.VideoCapture('/Users/yukihan/Desktop/final/Emotion-recognition/sample/sad/han2.MP4')

# Used to store the detection results of each frame
results = []

# Specify the video folder
video_file = '/Users/yukihan/Desktop/final/Emotion-recognition/sample/positive/surprise/zhong1.mp4'

camera = cv2.VideoCapture(video_file)

# Get video frame rate
frame_rate = int(camera.get(cv2.CAP_PROP_FPS))
frame_count = 0  # Counts the current frame count

# Count the number of frames for 15 seconds
max_frame_count = 15 * frame_rate

# Capture the first 15 seconds of video frame
while frame_count < max_frame_count:
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

    preds = []  # Initialize preds as an empty list

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
        color = emotion_colors[label]  # Get the color based on the emotion

       # Calculate the time of the current frame in seconds
        current_time = frame_count / frame_rate

        # Add the result to the list
        results.append({
            "Video": video_file,
            "Time(s)": current_time,
            "Emotion": label,
            "Probability": emotion_probability
        })

        #Initialize the phrases as empty, making sure they're defined in any case
        phrases = []

        # Display emotion management phrases only for "angry", "sad", "scared"
        if label in emotion_phrases and emotion_probability < 0.9:
                        # For emotions that have "positive" phrases
            if label in ["neutral", "happy", "surprised"]:
                phrases = random.sample(emotion_phrases[label]["positive"], 3)
            else:
                # For other emotions, choose from different categories
                phrases = random.sample(
                    emotion_phrases[label]["recognition"] +
                    emotion_phrases[label]["inquiry"] +
                    emotion_phrases[label]["confirmation"] +
                    emotion_phrases[label]["encouragement"] +
                    emotion_phrases[label]["release"], 3
                )

        # Display the selected phrases on the frame
        for idx, phrase in enumerate(phrases):
            # Define positions for top, bottom, and left
            if idx == 0:
                # Above (top), move text up
                text_x = fX
                text_y = max(fY - 60, 15)  # Adjust the offset to move the text up from -10 to -30
                max_width = frameClone.shape[1] - text_x - 10  # Make sure the text width does not exceed the screen
            elif idx == 1:
                # Below (bottom)
                text_x = fX
                text_y = min(fY + fH + 30, frameClone.shape[0] - 10) # Ensure that the text does not exceed the bottom boundary of the screen
                max_width = frameClone.shape[1] - text_x - 10
            elif idx == 2:
                # Left
                text_x = max(fX - 150, 10) # Make sure the text does not exceed the left edge of the screen
                text_y = fY + fH // 2
                max_width = fX - 20  # The maximum width on the left side is the area on the left side of the face

            # Draw the text on the frame with wrapping
            draw_text_with_wrap(frameClone, phrase, (text_x, text_y), 0.4, color, 1, max_width)
        
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
        # elif label == "sad":
        #     # Draw a half-circle (semi-circle)
        #     center = (fX + fW // 2, fY + fH - fH // 4)
        #     axes = (fW // 2, fH // 3)
        #     cv2.ellipse(frameClone, center, axes, 0, 0, 180, color, 2)
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

        cv2.putText(frameClone, label, (fX, fY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

    else:
        continue

# Triple the size of the frame
    frameClone = cv2.resize(frameClone, (frameClone.shape[1] * 3, frameClone.shape[0] * 3))


    for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
        # construct the label text
        text = "{}: {:.2f}%".format(emotion, prob * 100)
        color = emotion_colors[emotion]  # Get the color for the emotion
        # draw the label + probability bar on the canvas
        w = int(prob * 300)

        cv2.rectangle(canvas, (7, (i * 35) + 5),
                    (w, (i * 35) + 35), color, -1)

        cv2.putText(canvas, text, (10, (i * 35) + 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 2)

    cv2.imshow('your_face', frameClone)
    cv2.imshow("Probabilities", canvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1 # Increase frame count

camera.release()

# Save the result as a DataFrame
emotion_df = pd.DataFrame(results)

# Name the output file according to the video name and generate a 15-second data graph and table
video_file_name = os.path.basename(video_file).split('.')[0]  # Get video file name (without extension)

# Filter data to keep only the first 15 seconds
emotion_df = emotion_df[emotion_df['Time(s)'] <= 15]

# Print emotion_df for debugging
print("emotion_df data:")
print(emotion_df)

# Check if there is an emotion probability output
if emotion_df.empty:
    print("No emotion data was detected.")
else:
    # Print the first 15 seconds of mood data
    print("Emotion data in first 15 seconds:")
    print(emotion_df)

 # Data smoothing function
def smooth_data(data, window_size=5):
    return data.rolling(window=window_size, center=True).mean()

# Create a scatter plot
plt.figure(figsize=(12, 6))

# Transparency settings for emotions
alphas = [0.8, 0.6, 0.75, 0.7, 0.3, 0.5, 0.4]  # Transparency for different emotions
markers = ['o', 's', 'D', '^', 'v', '>', '<']  # Different marker shapes

for idx, emotion in enumerate(EMOTIONS):
    emotion_data = emotion_df[emotion_df['Emotion'] == emotion]
    
    # Get time and probability data
    time_data = emotion_data['Time(s)']
    probability_data = emotion_data['Probability']

    # If time_data is not empty
    if not time_data.empty:
        # Smooth the data
        probability_data_smooth = smooth_data(probability_data)
        # If the emotion is 'happy', set the color to yellow
        if emotion == 'happy':
            color_rgb = [180/255, 180/255, 0/255]  # Yellow (255, 255, 0)
        else:
            # Convert the emotion's corresponding color to an RGB value in the range of 0-1 for Matplotlib
            color_rgb = [c / 255 for c in emotion_colors[emotion]]
        
            # Plot the scatter plot, set transparency, color, and marker shape
        plt.scatter(time_data, probability_data_smooth, label=emotion,
                    color=color_rgb,  # Use the same color as in emotion_colors
                    marker=markers[idx % len(markers)],  # Different marker shapes
                    alpha=alphas[idx], s=50)  # The 's' parameter sets the size of the points

# Set title, labels, and legend
plt.title(f'Emotion Tracking for {video_file_name} (First 15 seconds with Scatter)')
plt.xlabel('Time (s)')
plt.ylabel('Emotion Probability')
plt.ylim([0, 1])  # Set a unified y-axis range
plt.legend()
plt.grid(True)

# Save the scatter plot
plt.savefig(f"{video_file_name}_emotion_tracking_scatter_15s.png")

# Clear the plot cache
plt.clf()

# Create a line plot
plt.figure(figsize=(12, 6))

# Transparency settings for emotions
alphas = [0.8, 0.6, 0.6, 0.7, 0.8, 0.5, 0.5]  # Transparency for different emotions
line_styles = ['-', '--', '-.', ':', '-', '--', '-.']  # Different line styles

for idx, emotion in enumerate(EMOTIONS):
    emotion_data = emotion_df[emotion_df['Emotion'] == emotion]
    
    # Get time and probability data
    time_data = emotion_data['Time(s)']
    probability_data = emotion_data['Probability']
    
    # If time_data is not empty
    if not time_data.empty:
        # If the emotion is 'happy', set the color to the adjusted yellow
        if emotion == 'happy':
            color_rgb = [180/255, 180/255, 0/255]  # Adjusted yellow
        elif emotion == 'neutral':
            color_rgb = [0/255, 180/255, 0/255]  # Adjusted green
        else:
            # Convert the emotion's corresponding color to an RGB value in the range of 0-1 for Matplotlib
            color_rgb = [c / 255 for c in emotion_colors[emotion]]
        
        # Plot the line graph, set transparency, color, and line style
        plt.plot(time_data, probability_data, label=emotion,
                 color=color_rgb,  # Use the corresponding color
                 linestyle=line_styles[idx % len(line_styles)],  # Use different line styles
                 alpha=alphas[idx])  # Set transparency

    # Set title, labels, and legend
    plt.title(f'Emotion Tracking for {video_file_name} (First 15 seconds with Lines)')
    plt.xlabel('Time (s)')
    plt.ylabel('Emotion Probability')
    plt.ylim([0, 1])  # Set a unified y-axis range
    plt.legend()
    plt.grid(True)

    # Save the line plot
    plt.savefig(f"{video_file_name}_Emotion_tracking_lines_15s.png")

    # Clear the plot cache
    plt.clf()

    # Create a pie chart
    plt.figure(figsize=(12, 6))

    # Assuming emotion_df contains the probability data for all emotions
    # Calculate the total probability for each emotion
    emotion_totals = emotion_df.groupby('Emotion')['Probability'].sum()

    # Calculate the percentage for each emotion
    emotion_percentages = (emotion_totals / emotion_totals.sum()) * 100

    # The goal is to reduce saturation, set a saturation reduction factor
    desaturation_factor = 0.3  # Reduce saturation by 30%

    # The goal is to reduce brightness, set a brightness reduction factor
    brightness_factor = 0.7  # A value less than 1, for example 0.7, indicates a 30% reduction in brightness

    # Adjust RGB colors towards gray, reducing saturation and brightness
    def adjust_brightness_and_saturation(color, saturation_factor=0.3, brightness_factor=0.7):
        gray = np.array([128, 128, 128])  # RGB value for gray
        color = np.array(color) * brightness_factor  # First adjust brightness
        return (np.array(color) * (1 - saturation_factor) + gray * saturation_factor).astype(int)

    # Adjust the colors of emotion_colors and apply saturation reduction
    colors = [adjust_brightness_and_saturation(emotion_colors[emotion], desaturation_factor, brightness_factor) 
            if emotion != "happy" else adjust_brightness_and_saturation((200, 200, 0), desaturation_factor, brightness_factor) 
            for emotion in emotion_totals.index]

    # Adjust the colors to include transparency (RGBA format)
    colors_with_alpha = [(r/255, g/255, b/255, 0.55) for r, g, b in colors]  # Set transparency to 0.55

    # Custom autopct function to set the percentage font to white
    def autopct_format(pct):
        return f'{pct:.1f}%' if pct > 0 else ''

    # Create the pie chart
    plt.figure(figsize=(8, 8))

    # Use the autopct parameter to customize the display of percentages
    patches, texts, autotexts = plt.pie(emotion_percentages, labels=emotion_totals.index, autopct=autopct_format,
                                        startangle=140, colors=[(r/255, g/255, b/255) for r, g, b in colors],
                                        textprops={'color': 'black'})  # Set the labels to black first

    # Set the percentage font to white and bold
    for autotext in autotexts:
        autotext.set_color('white')  # Set the percentage text to white
        autotext.set_fontsize(12)    # Set font size
        autotext.set_fontweight('bold')  # Set bold font weight

    # Set the title
    plt.title(f'Emotion Distribution for {video_file_name} (First 15 seconds)')

    # Save the pie chart
    plt.savefig(f"{video_file_name}_emotion_pie_chart_15s.png")

    # Clear the plot cache
    plt.clf()

    # Save the results as an Excel file, named after the video file
    emotion_df.to_excel(f"{video_file_name}_emotion_tracking_15s.xlsx", index=False)

print(f"Finished processing {video_file_name}. Results saved.")