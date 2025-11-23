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
SPEED_X = 8.74
SPEED_Y = 2.163
pyautogui.PAUSE = 0.01

current_side = -1  # 1 = right, -1 = left
current_trigger_count = 0


switch_counter = 0
def switch_side():
    global current_side, current_trigger_count, switch_counter
    if current_trigger_count < SWITCH_TRIGGER_MAX:
        current_trigger_count += 1
        return False
    
    current_trigger_count = 0
    switch_counter += 1
    print(f"Switch trigger {switch_counter}")
    pyautogui.click()  # Simulate a click to switch sides
    current_side *= -1
    return True
    

def squared_distance(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return dx*dx + dy*dy

def find_target_line(left_lines, right_lines, ball_x, ball_y, side_to_check, check_above):
    target_slope = LINE_SLOPE * side_to_check
    candidate_lines = right_lines if side_to_check == -1 else left_lines

    target_b = ball_y - target_slope * ball_x

    best_dist = float('inf')
    best_inter = None
    best_line = None

    for x1, y1, x2, y2 in candidate_lines:
        # Slope/intercept of candidate
        if (x2 - x1) == 0:
            continue
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1

        # Intersection between the lines
        inter_x = (target_b - b) / (m - target_slope)
        inter_y = m * inter_x + b

        # Check segment bounds (intersection must be within line segment)
        if not (min(x1, x2) <= inter_x <= max(x1, x2) and
                min(y1, y2) <= inter_y <= max(y1, y2)):
            continue

        # Check vertical position (above or below ball depending on if check_above is -1 or 1)
        if (inter_y*check_above) >= (ball_y*check_above):   
            continue

        # If intersection is too close to line segement ends, ignore
        thresh = 5
        if ((squared_distance((x1, y1), (inter_x, inter_y)) < thresh**2) or
            (squared_distance((x2, y2), (inter_x, inter_y)) < thresh**2)):
            continue

        # Compute distance
        dist = (inter_x - ball_x)**2 + (inter_y - ball_y)**2

        # Trigger sign is used to determine if we are looking for
        # distances above or below the threshold
        if dist < best_dist:
            best_dist = dist
            best_inter = (inter_x, inter_y)
            best_line = (x1, y1, x2, y2)

    if best_inter is None:
        return None, None, None
    else:
        inter_x, inter_y = best_inter
        return (int(ball_x), int(ball_y), int(inter_x), int(inter_y)), best_dist**0.5, best_line

def agent_line(left_lines, right_lines, ball_x, ball_y):
    # Normal case, there is a line in front of the ball, we can compute distance directly to it
    best_line, best_dist, _ = find_target_line(left_lines, right_lines, ball_x, ball_y, side_to_check=current_side, check_above=1)
    if best_line is not None:
        if best_dist < LINE_LENGTH_MIN:
            switch_side()
        return best_line, best_dist
    
    # First line in front of ball vanished, we need to check the other side line and if we aren't close to it, switch sides
    ball_y_offset = ball_y + 10
    best_line, best_dist, _ = find_target_line(left_lines, right_lines, ball_x, ball_y_offset, side_to_check=(current_side*-1), check_above=1)
    if best_line is not None:
        if best_dist > (LINE_LENGTH_MIN + 10):
            switch_side()
        return best_line, best_dist
    
    # Both front line and side line has vanished, check behind the ball...
    best_line, best_dist, edge_line = find_target_line(left_lines, right_lines, ball_x, ball_y, side_to_check=(current_side*-1), check_above=-1)
    if best_line is not None:
        # If the ball is not close enough to the line, skip it
        if best_dist > (LINE_LENGTH_MIN + 23):
            return best_line, best_dist
        
        # If the ball is not close enough to the top endpoint of line, skip it
        x1, y1, x2, y2 = edge_line
        x, y = (x2, y2) if y2 < y1 else (x1, y1)
        new_line = (ball_x, ball_y, x, y)
        new_dist = squared_distance((ball_x, ball_y), (x, y))
        new_dist = new_dist**0.5
        if new_dist > (LINE_LENGTH_MIN + 30):
            return new_line, new_dist
        
        switch_side()
        return new_line, new_dist
    
    return None, None


def game_loop(self, screen, game_FPS, counter, time_ms):
    screen = cv2.resize(screen, (DEFAULT_SCREEN_SIZE[1], DEFAULT_SCREEN_SIZE[0]))

    left_lines, right_lines, ball_x, ball_y, ball_r, edges = vision.process_frame(screen)
    if left_lines is None or right_lines is None or ball_x is None or ball_y is None:
        cv2.setWindowTitle("GameFrame", f"Press Q to quit | FPS: {game_FPS:.2f} | No detection")
        cv2.imshow("GameFrame", edges)
        return
    
    overlay = vision.plot_info(edges, left_lines, right_lines, ball_x, ball_y, ball_r)

    best_line, line_length = agent_line(left_lines, right_lines, ball_x, ball_y)
    if best_line is not None:
        vision.draw_line_with_length(overlay, best_line, line_length, (255, 0, 0))
    
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
