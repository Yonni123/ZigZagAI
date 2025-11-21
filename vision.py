import cv2
import numpy as np
from main import BALL_BETWEEN_Y
from main import LINE_SLOPE


def detect_path_edges(edge_img):
    # Hough lines
    lines = cv2.HoughLinesP(
        edge_img,
        rho = 1,
        theta = np.pi / 90,
        threshold = 20,
        minLineLength = 15,
        maxLineGap = 15
    )

    if lines is None:
        return [], []

    left = []   # "\" slope
    right = []  # "/" slope

    for (x1, y1, x2, y2) in lines[:, 0]:
        dx = x2 - x1
        dy = y2 - y1

        # Prevent division by zero
        if dx == 0:
            continue

        slope = dy / dx

        # ZigZag path has two dominant slopes, remove everything else
        margin = 0.1
        slope_min = LINE_SLOPE - margin
        slope_max = LINE_SLOPE + margin
        if abs(slope) < slope_min or abs(slope) > slope_max:
            continue

        if slope < 0:
            left.append((x1, y1, x2, y2))
        else:
            right.append((x1, y1, x2, y2))

    return left, right


def draw_line_with_length(img, line, length, color):
    x1, y1, x2, y2 = line
    cv2.line(img, (x1, y1), (x2, y2), color, 3)
    text = f"Len: {length:.2f}"
    text_pos = (x1 + 10, y1 - 10)   # Slight offset from the ball
    cv2.putText(img, text, text_pos,
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

def detect_ball(edge_img):
    # Crop vertically where the ball must be
    cropped = edge_img[BALL_BETWEEN_Y[0]:BALL_BETWEEN_Y[1], :]

    # HoughCircles works better on grayscale + slight blur, not raw edges
    # If `edge_img` is already edge-detected (0/255), we convert it.
    gray = cropped.copy()
    gray = cv2.GaussianBlur(gray, (5, 5), 1.5)

    # Hough Circle
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=30,
        param1=50,
        param2=15,       # lower = more sensitive; higher = fewer false positives
        minRadius=5,     # tune based on your ball size
        maxRadius=80     # tune based on your ball size
    )

    if circles is None:
        return None, None, None
    
    circles = np.round(circles[0, :]).astype("int")

    # Take the strongest circle (first one)
    x, y, r = circles[0]
    y_full = y + BALL_BETWEEN_Y[0]
    return x, y_full, r


def process_frame(frame):
    """Process a single frame to detect path edges and ball position."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray[gray == 255] = 0  # Remove white areas (background)

    gray_blur = cv2.GaussianBlur(gray, (5, 5), 1.5)

    cv2.imshow("Gray", gray_blur)
    edges = cv2.Canny(gray_blur, 10, 100)

    left_lines, right_lines = detect_path_edges(edges)
    ball_x, ball_y, ball_r = detect_ball(edges)

    return left_lines, right_lines, ball_x, ball_y, ball_r, edges


def draw_lines(img, lines, color):
    """Helper function to draw many Hough lines."""
    for x1, y1, x2, y2 in lines:
        cv2.line(img, (x1, y1), (x2, y2), color, 2)


def plot_info(frame, left_lines, right_lines, ball_x, ball_y, ball_r):
    if len(frame.shape) == 2 or frame.shape[2] == 1:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    draw_lines(frame, left_lines, (0, 255, 0))
    draw_lines(frame, right_lines, (0, 0, 255))

    if ball_x is not None and ball_y is not None and ball_r is not None:
        cv2.circle(frame, (ball_x, ball_y), ball_r, (255, 0, 0), 2)
        cv2.circle(frame, (ball_x, ball_y), 2, (0, 0, 255), 2)

    return frame
