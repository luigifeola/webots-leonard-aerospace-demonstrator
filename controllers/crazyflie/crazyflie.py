# @file crazyflie_controllers_py.py
# Controls the crazyflie motors in webots in Python

"""crazyflie_controller_py controller."""


import os
import sys
from math import atan, cos, sin, tan

import cv2
import numpy as np
import onnxruntime as ort


from controller import (
    GPS,
    Camera,
    Display,
    DistanceSensor,
    Gyro,
    InertialUnit,
    Motor,
    Robot,
)

controller_dir = os.path.dirname(os.path.abspath(__file__))
INFERENCE_DIR = os.path.join(controller_dir, "../../services/tristan-yolo-py-inference")
shared_python_path = os.path.normpath(
    os.path.join(controller_dir, '..', '..', 'controllers_shared', 'python_based')
)
sys.path.insert(0, INFERENCE_DIR)
if shared_python_path not in sys.path:
    sys.path.insert(0, shared_python_path)


from post_proc import (
    fixed_point_nms,
    float_to_qx_y_tensor,
    merge_output_int,
    qx_y_to_float_tensor,
)

from fpga_inference import predict_with_fpga, get_serial_connection, close_serial_connection

try:
    from pid_controller import pid_velocity_fixed_height_controller
except ModuleNotFoundError:
    print(f"[crazyflie] Failed to import pid_controller from: {shared_python_path}")
    print(f"[crazyflie] __file__: {__file__}")
    print(f"[crazyflie] cwd: {os.getcwd()}")
    print(f"[crazyflie] sys.path[0:5]: {sys.path[:5]}")
    raise

FLYING_ATTITUDE = 1
LANDING_SPEED = 0.2  # Speed in m/s to descend
GROUND_THRESHOLD = 0.05  # Altitude to consider as landed
CAMERA_PERIOD_MS = 200
OPEN_LOOP_YAW_RATE = max(0.05, float(os.getenv('CF_OPEN_LOOP_YAW_RATE', '0.6')))
MAX_OPEN_LOOP_YAW_DELTA = max(0.05, float(os.getenv('CF_MAX_OPEN_LOOP_YAW_DELTA', '1.2')))
YAW_STOP_TOLERANCE = max(0.001, float(os.getenv('CF_YAW_STOP_TOLERANCE', '0.01')))

MODEL_PATH = os.path.join(INFERENCE_DIR, "inputs", "yolo_pruned_int_fixed.onnx")
ANCHORS_PATH = os.path.join(INFERENCE_DIR, "inputs", "anchors.npy")
CLASS_NAMES = ["pedestrian"]


__RUN_ON_ONNX_RUNTIME__ = os.getenv('CF_RUN_ON_ONNX_RUNTIME', 'True').lower() not in ('false', '0', 'no')
print(f"[crazyflie] Running ONNX Runtime inference: {__RUN_ON_ONNX_RUNTIME__}")

def preprocess_image(image_bgr, input_hw):
    """Preprocess image for YOLO model."""
    resized = cv2.resize(image_bgr, input_hw, interpolation=cv2.INTER_LINEAR)
    image_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    image_chw = image_rgb.transpose(2, 0, 1)
    image_chw = np.expand_dims(image_chw, axis=0)
    image_input = np.ascontiguousarray(image_chw, dtype=np.float32)
    return image_input

def predict(image_bgr, conf_thres=0.25, iou_thres=0.45):
    """Run YOLO inference on image."""

    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    if session is None:
        raise RuntimeError("YOLO model not initialized")

    anchor_grid = np.load(ANCHORS_PATH)
    if anchor_grid is None:
        raise RuntimeError("Failed to load anchors")

    input_name = session.get_inputs()[0].name
    output_names = [_.name for _ in session.get_outputs()]
    input_shape = session.get_inputs()[0].shape

    image = preprocess_image(image_bgr, input_shape[2:])
    raw_output = session.run(output_names, {input_name: image})
    x_fixed = [float_to_qx_y_tensor(t, 4, 12) for t in raw_output]
    out = merge_output_int(x_fixed, anchor_grid)
    out = fixed_point_nms(out, conf_thres, iou_thres)
    out = [qx_y_to_float_tensor(t.astype(np.int32), 14, 15) for t in out]
    return out

def _wrap_angle(angle):
    """Wrap an angle in radians to [-pi, pi]."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


if __name__ == '__main__':

    robot = Robot()
    timestep = int(robot.getBasicTimeStep())

    camera_period_ms = max(timestep, CAMERA_PERIOD_MS)
    # image_process_interval = max(
    #     0.0,
    #     float(os.getenv('CF_IMAGE_PROCESS_INTERVAL', '0.1'))
    # )

    # Initialize variables for timing
    desired_yaw_rate_cmd = 0.0
    fixed_target_detection = None
    planned_yaw_delta = None
    yaw_target = None
    one_shot_completed = False
    landing_initiated = False
    wait_before_landing = False
    wait_after_takeoff = False
    wait_start_time = 0
    takeoff_wait_start_time = 0

    ## Initialize motors
    m1_motor = robot.getDevice("m1_motor")
    m1_motor.setPosition(float('inf'))
    m1_motor.setVelocity(-1)
    m2_motor = robot.getDevice("m2_motor")
    m2_motor.setPosition(float('inf'))
    m2_motor.setVelocity(1)
    m3_motor = robot.getDevice("m3_motor")
    m3_motor.setPosition(float('inf'))
    m3_motor.setVelocity(-1)
    m4_motor = robot.getDevice("m4_motor")
    m4_motor.setPosition(float('inf'))
    m4_motor.setVelocity(1)

    ## Initialize Sensors
    imu = robot.getDevice("inertial_unit")
    imu.enable(timestep)
    gps = robot.getDevice("gps")
    gps.enable(timestep)
    gyro = robot.getDevice("gyro")
    gyro.enable(timestep)
    camera = robot.getDevice("camera")
    camera.enable(camera_period_ms)
    display = robot.getDevice("display")
    
    # Get camera parameters
    camera_width = camera.getWidth()
    camera_height = camera.getHeight()
    camera_fov = camera.getFov()
    focal_length_px = camera_width / (2.0 * tan(camera_fov / 2.0))

    ## Initialize variables

    past_x_global = 0
    past_y_global = 0
    past_time = robot.getTime()

    # Crazyflie velocity PID controller
    PID_CF = pid_velocity_fixed_height_controller()
    PID_update_last_time = robot.getTime()
    sensor_read_last_time = robot.getTime()

    height_desired = FLYING_ATTITUDE
    takeoff_tolerance = 0.005
    takeoff_done = False

    print("\n")

    print("\n====== Crazyflie Drone with Pedestrian Detection ======\n")
    
    # get serial conn
    fpga_ser = None
    if not __RUN_ON_ONNX_RUNTIME__:
        fpga_ser = get_serial_connection()
        print("Serial connection established for FPGA inference.")
    else:
        print("ONNX Runtime inference enabled. Skipping FPGA serial connection.")

    # Main loop:
    while robot.step(timestep) != -1:
        current_time = robot.getTime()
        dt = current_time - past_time
        actual_state = {}

        ## Get sensor data
        roll = imu.getRollPitchYaw()[0]
        pitch = imu.getRollPitchYaw()[1]
        yaw = imu.getRollPitchYaw()[2]
        yaw_rate = gyro.getValues()[2]
        altitude = gps.getValues()[2]
        x_global = gps.getValues()[0]
        v_x_global = (x_global - past_x_global)/dt
        y_global = gps.getValues()[1]
        v_y_global = (y_global - past_y_global)/dt

        ## Get body fixed velocities
        cosyaw = cos(yaw)
        sinyaw = sin(yaw)
        v_x = v_x_global * cosyaw + v_y_global * sinyaw
        v_y = - v_x_global * sinyaw + v_y_global * cosyaw

        ## Initialize values
        desired_state = [0, 0, 0, 0]
        forward_desired = 0
        sideways_desired = 0
        desired_yaw_rate = desired_yaw_rate_cmd
        height_diff_desired = 0

        if landing_initiated:
            height_desired -= LANDING_SPEED * dt
            
            # Ensure we don't command a height below zero
            if height_desired < 0:
                height_desired = 0

            # Check if the drone has landed
            if altitude < GROUND_THRESHOLD:
                print("[crazyflie] Landed successfully.")
                break  # Exit the main loop
        
        elif wait_before_landing:
            if current_time - wait_start_time >= 3.0:
                print("[crazyflie] Initiating landing.")
                wait_before_landing = False
                landing_initiated = True
        
        elif wait_after_takeoff:
            if current_time - takeoff_wait_start_time >= 4.0:
                print(f"[crazyflie] Takeoff complete.")
                wait_after_takeoff = False
                takeoff_done = True

        elif not takeoff_done and abs(altitude - FLYING_ATTITUDE) > takeoff_tolerance:
            height_desired += height_diff_desired * dt
            # print(f"Taking off to {height_desired} m")
        
        elif not takeoff_done:
            wait_after_takeoff = True
            takeoff_wait_start_time = current_time

        else:           
            # # Run camera rendering and YOLO at a lower, configurable rate.
            # Capture camera image
            raw_image = camera.getImage()
            display_image = display.imageNew(raw_image, Display.BGRA, camera_width, camera_height)
            display.imagePaste(display_image, 0, 0, False)
            display.imageDelete(display_image)
            image = np.frombuffer(raw_image, dtype=np.uint8).reshape((camera_height, camera_width, 4))
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)  # Convert Webots image format to OpenCV BGR

            if not one_shot_completed and planned_yaw_delta is None:
                if __RUN_ON_ONNX_RUNTIME__:
                    detections = predict(image)
                else:
                    print("Running FPGA inference...")
                    detections = predict_with_fpga(fpga_ser, image)

                if detections and len(detections[0]) > 0:
                    # Scale detection from model input size to camera size
                    # The model output is relative to its input size (e.g., 128x128)
                    # We need to find the original bbox center in the camera frame.
                    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
                    input_shape = session.get_inputs()[0].shape
                    model_input_hw = input_shape[2:] # e.g., (128, 128)
                    
                    # Get the first detection
                    box = detections[0][0]
                    x1, y1, x2, y2, conf, pred_cls = box
                    
                    # Scale box to camera space and draw
                    scale_x = camera_width / model_input_hw[-1]
                    scale_y = camera_height / model_input_hw[-2]
                    x1_cam, y1_cam = int(x1 * scale_x), int(y1 * scale_y)
                    x2_cam, y2_cam = int(x2 * scale_x), int(y2 * scale_y)
                    detection_image = image.copy()
                    cv2.rectangle(detection_image, (x1_cam, y1_cam), (x2_cam, y2_cam), (0, 255, 0), 2)
                    
                    label = f"{float(conf) * 100:.1f}%"
                    cv2.putText(detection_image, label, (x1_cam, y1_cam - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                    # --- Save #1: frame at detection time with bounding box ---
                    cv2.imwrite("detection_pedestrian.png", detection_image)
                    
                    bbox_center_x_model = (x1 + x2) / 2.0
                    
                    # Convert model-space coordinate to camera-space coordinate
                    bbox_center_x_camera = (bbox_center_x_model / model_input_hw[-1]) * camera_width
                    
                    image_center_x = camera_width / 2.0
                    
                    # Positive angle means rotate left, negative means rotate right.
                    planned_yaw_delta = atan((image_center_x - bbox_center_x_camera) / max(focal_length_px, 1e-6))
                    
                    if planned_yaw_delta > MAX_OPEN_LOOP_YAW_DELTA:
                        planned_yaw_delta = MAX_OPEN_LOOP_YAW_DELTA
                    elif planned_yaw_delta < -MAX_OPEN_LOOP_YAW_DELTA:
                        planned_yaw_delta = -MAX_OPEN_LOOP_YAW_DELTA
                        
                    yaw_target = _wrap_angle(yaw + planned_yaw_delta)
                    print(
                        f"[crazyflie] Planned single-shot yaw delta: {planned_yaw_delta:.3f} rad "
                        f"({planned_yaw_delta * 180.0 / np.pi:.1f} deg)"
                    )

            # Execute the pre-planned single-shot yaw in open loop.
            if yaw_target is not None:
                yaw_error = _wrap_angle(yaw_target - yaw)
                safe_dt = max(dt, 1e-6)
                if abs(yaw_error) > YAW_STOP_TOLERANCE:
                    raw_yaw_rate = yaw_error / safe_dt
                    if raw_yaw_rate > OPEN_LOOP_YAW_RATE:
                        desired_yaw_rate_cmd = OPEN_LOOP_YAW_RATE
                    elif raw_yaw_rate < -OPEN_LOOP_YAW_RATE:
                        desired_yaw_rate_cmd = -OPEN_LOOP_YAW_RATE
                    else:
                        desired_yaw_rate_cmd = raw_yaw_rate
                else:
                    desired_yaw_rate_cmd = 0.0
                    yaw_target = None
                    planned_yaw_delta = 0.0
                    one_shot_completed = True
                    wait_before_landing = True
                    wait_start_time = current_time
                    print('[crazyflie] Open-loop yaw plan completed.')
                    cv2.imwrite("centered_pedestrian.png", image)
            else:
                desired_yaw_rate_cmd = 0.0

            desired_yaw_rate = desired_yaw_rate_cmd
        
        ## PID velocity controller with fixed height
        motor_power = PID_CF.pid(dt, forward_desired, sideways_desired,
                                desired_yaw_rate, height_desired,
                                roll, pitch, yaw_rate,
                                altitude, v_x, v_y)

        m1_motor.setVelocity(-motor_power[0])
        m2_motor.setVelocity(motor_power[1])
        m3_motor.setVelocity(-motor_power[2])
        m4_motor.setVelocity(motor_power[3])

        past_time = current_time
        past_x_global = x_global
        past_y_global = y_global
        
    # --- POST-FLIGHT CLEANUP ---
    print("[crazyflie] Flight loop terminated. Turning off motors.")
    m1_motor.setVelocity(0)
    m2_motor.setVelocity(0)
    m3_motor.setVelocity(0)
    m4_motor.setVelocity(0)
    
    # Clean up serial connection on exit
    if not __RUN_ON_ONNX_RUNTIME__ and fpga_ser is not None:
        try:
            close_serial_connection(fpga_ser)
            print("Serial connection closed successfully.")
        except Exception as e:
            print(f"Error while closing serial connection: {e}")
    else:
        print("ONNX Runtime inference was used. No serial connection to close.")