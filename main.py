import sys
sys.path.insert(1, 'FrameHook')
from frame_hook import GameWrapper
import time
import cv2
import numpy as np


ORIG_SCREEN_SIZE = None
DEFAULT_SCREEN_SIZE = (1112, 624)


def detect_path_edges(edge_img):
    """
    Detects the ZigZag path by clustering Hough lines into two groups based on slope.
    Returns: left_lines, right_lines (each a list of line segments)
    """

    # Hough lines
    lines = cv2.HoughLinesP(
        edge_img,
        rho = 1,
        theta = np.pi / 90,
        threshold = 40,
        minLineLength = 40,
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
        margin = 0.05
        mid = 0.5
        slope_min = mid - margin
        slope_max = mid + margin
        if abs(slope) < slope_min or abs(slope) > slope_max:
            continue

        if slope < 0:
            left.append((x1, y1, x2, y2))
        else:
            right.append((x1, y1, x2, y2))

    return left, right


def draw_lines(img, lines, color):
    """Helper function to draw many Hough lines."""
    for x1, y1, x2, y2 in lines:
        cv2.line(img, (x1, y1), (x2, y2), color, 2)


def game_loop(self, screen, game_FPS, counter, time_ms):
    screen = cv2.resize(screen, (DEFAULT_SCREEN_SIZE[1], DEFAULT_SCREEN_SIZE[0]))
        
    # 1. Canny edge detection
    edges = cv2.Canny(screen, 100, 200)

    # 2. Detect path edges (left and right)
    left_edges, right_edges = detect_path_edges(edges)

    # 3. Draw overlays
    overlay = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    draw_lines(overlay, left_edges, (0, 255, 0))   # green for left boundary
    draw_lines(overlay, right_edges, (0, 0, 255))  # red for right boundary

    # 4. Show result
    cv2.setWindowTitle("GameFrame", f"Press Q to quit | FPS: {game_FPS:.2f}")
    cv2.imshow("GameFrame", overlay)


if __name__ == "__main__":
    game = GameWrapper(
        monitor_index=0,
        trim=True,
        game_region={'top': 124, 'left': 71, 'width': 624, 'height': 1112}
    )

    time.sleep(2)  # time to focus game window
    game.play(game_loop)
