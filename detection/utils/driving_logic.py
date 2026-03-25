import cv2
import numpy as np

def check_environment(frame):
    """
    Checks if the environment is valid for driving.
    Detects if the scene is likely Water or Sky based on color and texture analysis.
    Returns a warning string if invalid, else None.
    
    Note: Ideally this would use a pre-trained classification model (e.g., ResNet18).
    For now, we use a robust heuristic to avoid dependency on downloading large weights.
    """
    # Resize for faster processing
    small_frame = cv2.resize(frame, (64, 64))
    hsv = cv2.cvtColor(small_frame, cv2.COLOR_BGR2HSV)
    
    # Define generic blue range for sky/water
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    blue_ratio = np.sum(mask_blue > 0) / (64 * 64)
    
    # Check simple horizon split for sky
    # Top half predominantly blue/bright
    top_half = hsv[:32, :, :]
    top_v_mean = np.mean(top_half[:, :, 2])
    
    if blue_ratio > 0.6:
         return "WARNING: Driving environment appears to be WATER or SKY (High Blue Content)"
         
    return None

def detect_traffic_light_color(image, bbox):
    """
    Detects the state of a traffic light within the bounding box.
    Returns: 'Red', 'Green', 'Yellow', or 'Unknown'
    """
    x1, y1, x2, y2 = map(int, bbox)
    roi = image[y1:y2, x1:x2]
    
    if roi.size == 0:
        return 'Unknown'
    
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Red range (wraps around 0/180)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    
    # Green range
    lower_green = np.array([40, 70, 50])
    upper_green = np.array([90, 255, 255])
    
    # Yellow range
    lower_yellow = np.array([20, 70, 50])
    upper_yellow = np.array([40, 255, 255])
    
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    red_pixels = cv2.countNonZero(mask_red1) + cv2.countNonZero(mask_red2)
    green_pixels = cv2.countNonZero(mask_green)
    yellow_pixels = cv2.countNonZero(mask_yellow)
    
    # Threshold to decide color
    total_pixels = roi.shape[0] * roi.shape[1]
    if total_pixels == 0: return 'Unknown'
    
    # Heuristic: Color must be significant part of the light distinct feature
    # Traffic lights are often small, so even small pixel count matters if it's the brightest/saturated part
    # Use max logic
    
    max_pixels = max(red_pixels, green_pixels, yellow_pixels)
    
    if max_pixels < (total_pixels * 0.05): # Less than 5% of bbox is colored light
        return 'Unknown'
        
    if max_pixels == red_pixels:
        return 'Red'
    elif max_pixels == green_pixels:
        return 'Green'
    else:
        return 'Yellow'

def get_driving_suggestions(detections, frame_shape):
    """
    Analyzes detections to provide driving suggestions.
    detections: List of dicts with 'class_name', 'bbox', 'confidence'
    frame_shape: (height, width, channels)
    """
    height, width = frame_shape[:2]
    center_x = width // 2
    
    suggestions = []
    
    # Parameters for logic
    critical_distance_threshold = 0.3 * height # Height of box relative to frame height
    center_lane_width = 0.4 * width
    lane_left = center_x - (center_lane_width / 2)
    lane_right = center_x + (center_lane_width / 2)
    
    detected_obstacle_ahead = False
    traffic_light_state = None
    
    for det in detections:
        bbox = det['bbox'] # [x1, y1, x2, y2]
        cls = det['class_name']
        x1, y1, x2, y2 = bbox
        
        # Calculate box center and height
        box_center_x = (x1 + x2) / 2
        box_height = y2 - y1
        
        # Check for Traffic Lights
        if cls == 'traffic light':
            # Logic handled in inference loop typically, but here we can aggregate
            # Assuming 'color' might be added to detection dict if not already
            if 'color' in det:
                if det['color'] == 'Red':
                    suggestions.append("STOP: Red Light Ahead")
                elif det['color'] == 'Green':
                    suggestions.append("GO: Green Light")
        
        # Check for Vehicles ahead (Car, Truck, Bus, Motorcycle)
        if cls in ['car', 'truck', 'bus', 'motorcycle']:
            # Check if in center lane
            if lane_left < box_center_x < lane_right:
                # Check if close (large box height)
                if box_height > critical_distance_threshold:
                    detected_obstacle_ahead = True
                    suggestions.append(f"CAUTION: Vehicle Ahead ({cls}). Suggest Lane Change.")
    
    return suggestions
