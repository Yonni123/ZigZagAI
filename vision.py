import cv2
import numpy as np
from main import BALL_BETWEEN_Y
from main import LINE_SLOPE

prev_ball_x = None

def merge_segments(lines, slope=None, epsilon=20):
    """
    Vectorized version: merge collinear, contiguous line segments.
    
    lines: list of (x1, y1, x2, y2)
    slope: constant slope (if None, calculated from first segment)
    epsilon: tolerance for comparing y-intercepts or gaps

    Returns: list of merged lines (x1, y1, x2, y2)
    """
    if len(lines) == 0:
        return []

    lines = np.array(lines, dtype=float)  # shape (N, 4)
    x1, y1, x2, y2 = lines[:,0], lines[:,1], lines[:,2], lines[:,3]

    # --- Compute slope if not given ---
    if slope is None:
        slope = (y2[0] - y1[0]) / (x2[0] - x1[0]) if x2[0] != x1[0] else np.inf

    # --- Compute intercepts ---
    m = y1 - slope * x1
    x_min = np.minimum(x1, x2)
    x_max = np.maximum(x1, x2)

    # --- Sort by intercept ---
    sort_idx = np.argsort(m)
    x1, y1, x2, y2 = x1[sort_idx], y1[sort_idx], x2[sort_idx], y2[sort_idx]
    x_min, x_max, m = x_min[sort_idx], x_max[sort_idx], m[sort_idx]

    # --- Group by intercept ---
    diff_m = np.diff(m, prepend=m[0])
    new_group_mask = np.abs(diff_m) >= epsilon
    group_ids = np.cumsum(new_group_mask)

    merged_lines = []

    for gid in np.unique(group_ids):
        mask = group_ids == gid
        gx_min, gx_max = x_min[mask], x_max[mask]
        gx1, gy1, gx2, gy2 = x1[mask], y1[mask], x2[mask], y2[mask]

        # --- Sort by x_min within group ---
        sort_x_idx = np.argsort(gx_min)
        gx_min, gx_max = gx_min[sort_x_idx], gx_max[sort_x_idx]
        gx1, gy1, gx2, gy2 = gx1[sort_x_idx], gy1[sort_x_idx], gx2[sort_x_idx], gy2[sort_x_idx]

        # --- Split contiguous segments based on X gaps ---
        cont_start = 0
        for i in range(1, len(gx_min)):
            if gx_min[i] - gx_max[i-1] > epsilon:
                # Merge previous contiguous group
                xs = np.concatenate([gx1[cont_start:i], gx2[cont_start:i]])
                x_start, x_end = xs.min(), xs.max()
                y_start = slope * x_start + gy1[cont_start] - slope * gx1[cont_start]
                y_end = slope * x_end + gy1[cont_start] - slope * gx1[cont_start]
                merged_lines.append((int(x_start), int(y_start), int(x_end), int(y_end)))
                cont_start = i

        # Merge last contiguous sub-group
        xs = np.concatenate([gx1[cont_start:], gx2[cont_start:]])
        x_start, x_end = xs.min(), xs.max()
        y_start = slope * x_start + gy1[cont_start] - slope * gx1[cont_start]
        y_end = slope * x_end + gy1[cont_start] - slope * gx1[cont_start]
        merged_lines.append((int(x_start), int(y_start), int(x_end), int(y_end)))

    return merged_lines


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

    left = merge_segments(left, slope=-LINE_SLOPE)
    right = merge_segments(right, slope=LINE_SLOPE)
    return left, right


def draw_line_with_length(img, line, length, color):
    x1, y1, x2, y2 = line
    cv2.line(img, (x1, y1), (x2, y2), color, 3)
    text = f"Len: {length:.2f}"
    text_pos = (x1 + 10, y1 - 10)   # Slight offset from the ball
    cv2.putText(img, text, text_pos,
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

def detect_ball(edge_img):
    global prev_ball_x

    # Crop vertically where the ball must be
    cropped = edge_img[BALL_BETWEEN_Y[0]:BALL_BETWEEN_Y[1], :]

    if prev_ball_x is not None:
        # Restrict horizontal search around previous ball position
        x_min = max(prev_ball_x - 50, 0)
        x_max = min(prev_ball_x + 50, edge_img.shape[1])

        cropped = cropped[:, x_min:x_max]
    else:
        x_min = 0  # no offset

    cropped_debug = cv2.resize(cropped, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    cv2.imshow("Ball_debug", cropped_debug)

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
        prev_ball_x = None
        return None, None, None
    
    circles = np.round(circles[0, :]).astype("int")

    # Take the strongest circle (first one)
    x, y, r = circles[0]
    x_full = x + x_min
    if prev_ball_x is not None:
        prev_ball_x = int(prev_ball_x * 0.7 + x_full * 0.3)
    else:
        prev_ball_x = x_full
    y_full = y + BALL_BETWEEN_Y[0]
    return x_full, y_full, r


def process_frame(frame):
    """Process a single frame to detect path edges and ball position."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #gray[gray == 255] = 0  # Remove white areas (background)

    gray_blur = cv2.GaussianBlur(gray, (5, 5), 1.5)

    edges = cv2.Canny(gray_blur, 10, 100)

    left_lines, right_lines = detect_path_edges(edges)
    ball_x, ball_y, ball_r = detect_ball(edges)

    return left_lines, right_lines, ball_x, ball_y, ball_r, edges


def draw_lines(img, lines, color):
    """Helper function to draw many Hough lines."""
    for x1, y1, x2, y2 in lines:
        cv2.line(img, (x1, y1), (x2, y2), color, 2)

        cv2.circle(img, (x1, y1), 5, color, -1)   # filled circle
        cv2.circle(img, (x2, y2), 5, color, -1)


def plot_info(frame, left_lines, right_lines, ball_x, ball_y, ball_r):
    if len(frame.shape) == 2 or frame.shape[2] == 1:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    draw_lines(frame, left_lines, (0, 255, 0))
    draw_lines(frame, right_lines, (0, 0, 255))

    if ball_x is not None and ball_y is not None and ball_r is not None:
        cv2.circle(frame, (ball_x, ball_y), ball_r, (255, 0, 0), 2)
        cv2.circle(frame, (ball_x, ball_y), 2, (0, 0, 255), 2)

    return frame
