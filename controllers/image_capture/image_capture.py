# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot, Camera
import os

CONTROLLERS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DETECTION_RESULT_PATH = os.path.join(CONTROLLERS_DIR, "detection_result.json")

# create the Robot instance.
camera_robot = Robot()

# get the time step of the current world.
timestep = int(camera_robot.getBasicTimeStep())
camera = Camera('camera')

camera.enable(timestep)

if not os.path.exists("all_images"):
    os.makedirs("all_images")

# Capture a single image at time 0, send it to the YOLO service, and save the result
image = None
if camera_robot.step(timestep) != -1:
    image = camera.getImage()

if image is not None:
    # Save raw image
    out_path = os.path.join("all_images", "image_0.png")
    camera.saveImage(out_path, 100)
    print(f"Captured initial image: {out_path}")

    # Try to send to YOLO service
    try:
        import base64
        import requests
        import json

        # Convert camera image buffer to BGR using OpenCV conventions
        # Webots Camera.getImage returns an RGBA string; use saveImage result instead
        with open(out_path, 'rb') as f:
            img_bytes = f.read()

        img_b64 = base64.b64encode(img_bytes).decode('utf-8')
        payload = {
            'image': img_b64,
            'target_class': 'Cookie_box',
            'conf_threshold': 0.25,
            'iou_threshold': 0.45,
        }
        resp = requests.post('http://localhost:5000/detect', json=payload, timeout=10)
        if resp.status_code == 200:
            print('YOLO service response received')
            try:
                result = resp.json()
            except Exception:
                result = {'success': False}
            # Save result for robot controllers to pick up
            with open(DETECTION_RESULT_PATH, 'w') as rf:
                json.dump(result, rf)
            print(f'Detection result saved to {DETECTION_RESULT_PATH}')
        else:
            print(f'YOLO service returned status: {resp.status_code}')
    except Exception as e:
        print(f'Error sending image to YOLO service: {e}')

# Exit after capturing and sending initial image
print('image_capture: done')

