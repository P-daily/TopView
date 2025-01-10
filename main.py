import cv2
from cropping_frame import crop_image_based_on_green_markers

video_path = "Parking2.mp4"

cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), "Error reading video file"

# Pobierz pierwszą klatkę z wideo
ret, frame = cap.read()
if not ret:
    raise ValueError("Unable to read the first frame from the video")

# Użyj funkcji do wykadrowania obrazu na podstawie zielonych markerów
try:
    cropped_image, green_centers = crop_image_based_on_green_markers(frame)
    print("Green centers detected at:", green_centers)

    # Wyświetl wykadrowany obraz (opcjonalnie)
    cv2.imshow("Cropped Image", cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)  # Czekaj na dowolny klawisz
    cv2.destroyAllWindows()
except ValueError as e:
    print(e)

# Zamknij wideo
cap.release()
