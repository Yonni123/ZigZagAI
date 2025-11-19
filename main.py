import sys
sys.path.insert(1, 'FrameHook')
from frame_hook import GameWrapper
import time
import cv2
import vision


ORIG_SCREEN_SIZE = None
DEFAULT_SCREEN_SIZE = (1112, 624)
BALL_BETWEEN_Y = (517, 547)  # y-range where ball is expected to be found


def game_loop(self, screen, game_FPS, counter, time_ms):
    screen = cv2.resize(screen, (DEFAULT_SCREEN_SIZE[1], DEFAULT_SCREEN_SIZE[0]))

    left_lines, right_lines, ball_x, ball_y, ball_r, edges = vision.process_frame(screen)
    overlay = vision.plot_info(edges, left_lines, right_lines, ball_x, ball_y, ball_r)

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
