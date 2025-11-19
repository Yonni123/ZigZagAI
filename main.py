import sys
sys.path.insert(1, 'FrameHook')
from frame_hook import GameWrapper
import time
import cv2
import vision
import pyautogui


ORIG_SCREEN_SIZE = None
DEFAULT_SCREEN_SIZE = (1112, 624)
BALL_BETWEEN_Y = (517, 547)  # y-range where ball is expected to be found
LINE_SLOPE = 0.5    # minimum slope for lines to be considered path edges
LINE_LENGTH_MIN = 50  # minimum length for line to switch sides
SWITCH_TRIGGER_MAX = 1 # number of triggers before switching sides
pyautogui.PAUSE = 0.01

current_side = -1  # 1 = right, -1 = left
current_trigger_count = 0


def switch_side():
    global current_side, current_trigger_count
    if current_trigger_count < SWITCH_TRIGGER_MAX:
        current_trigger_count += 1
        return
    
    current_trigger_count = 0
    pyautogui.click()  # Simulate a click to switch sides
    current_side *= -1


def agent_line(left_lines, right_lines, ball_x, ball_y):
    """
    Pick the closest intersection point between the target line 
    (through the ball with slope LINE_SLOPE * current_side)
    and a segment from the selected side, with the constraint that:
      - The intersection must lie on the finite segment.
      - The intersection must be ABOVE the ball (inter_y < ball_y).
    """

    target_slope = LINE_SLOPE * current_side
    candidate_lines = right_lines if current_side == -1 else left_lines

    target_b = ball_y - target_slope * ball_x

    best_dist = float('inf')
    best_inter = None

    for x1, y1, x2, y2 in candidate_lines:
        # Slope/intercept of candidate
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1

        # Intersection between the lines
        inter_x = (target_b - b) / (m - target_slope)
        inter_y = m * inter_x + b

        # Check segment bounds (intersection must be within line segment)
        if not (min(x1, x2) <= inter_x <= max(x1, x2) and
                min(y1, y2) <= inter_y <= max(y1, y2)):
            continue

        # Check vertical position (intersection must be above ball)
        if inter_y >= ball_y:   
            continue    

        # Compute distance
        dist = ((inter_x - ball_x)**2 + (inter_y - ball_y)**2)**0.5

        if dist < best_dist:
            best_dist = dist
            best_inter = (inter_x, inter_y)

    if best_inter is None:
        return None
    
    if best_dist < LINE_LENGTH_MIN:
        switch_side()

    inter_x, inter_y = best_inter
    return (int(ball_x), int(ball_y), int(inter_x), int(inter_y))


def game_loop(self, screen, game_FPS, counter, time_ms):
    screen = cv2.resize(screen, (DEFAULT_SCREEN_SIZE[1], DEFAULT_SCREEN_SIZE[0]))

    left_lines, right_lines, ball_x, ball_y, ball_r, edges = vision.process_frame(screen)
    if left_lines is None or right_lines is None or ball_x is None or ball_y is None:
        cv2.setWindowTitle("GameFrame", f"Press Q to quit | FPS: {game_FPS:.2f} | No detection")
        cv2.imshow("GameFrame", edges)
        return
    
    overlay = vision.plot_info(edges, left_lines, right_lines, ball_x, ball_y, ball_r)
    best_line = agent_line(left_lines, right_lines, ball_x, ball_y)
    if best_line is not None:
        x1, y1, x2, y2 = best_line

        # Draw the line
        cv2.line(overlay, (x1, y1), (x2, y2), (255, 0, 0), 3)

        # Compute line length
        line_length = int(((x2 - x1)**2 + (y2 - y1)**2)**0.5)

        text = f"Len: {line_length}"
        text_pos = (x1 + 10, y1 - 10)   # Slight offset from the ball

        # Draw main text (white)
        cv2.putText(overlay, text, text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

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
