from ultralytics import YOLO
import ultralytics
from PIL import Image

print(ultralytics.checks())
# Load our custom goat model
model = YOLO("./v4_best.pt")
# Use the model to detect object - goat

image = Image.open('naver_img_R.jpg').resize((1920,1080))
image.save('naver_img_R2.jpg')

results = model.predict(source='naver_img_R2.jpg', show=False, save=True )
for result in results:
    boxes = result.boxes  # Boxes object for bbox outputs
    print(boxes)
