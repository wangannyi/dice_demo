"""Optional OpenCV/X11 collection window. Never commands robot motion."""
import os
import select
import sys
import textwrap
import time

import cv2
import numpy as np


class CollectionPreview:
    TITLE = 'Nero hand-eye calibration | Enter: sample | Q: quit'

    def __init__(self):
        if sys.platform.startswith('linux') and not os.environ.get('DISPLAY'):
            raise RuntimeError('X11 DISPLAY is missing; log in with ssh -X before using --preview')
        self.last_image = None
        self.last_lines = ['No capture in this window yet.']
        self.last_color = (180, 180, 180)
        self.current_image = None
        self.board_window = None
        self.closed = False
        cv2.namedWindow(self.TITLE, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.TITLE, 1280, 640)

    def restore(self, image, count, quality):
        """Show the most recent saved sample when resuming, without re-capturing it."""
        self.result(True, count, 'Loaded from the existing session.', quality, image)

    def result(self, accepted, count, message, quality=None, image=None):
        self.last_color = (90, 220, 90) if accepted else (80, 130, 255)
        source = image if image is not None else self.current_image
        self.last_image = None if source is None else source.copy()
        self.last_lines = [f'ACCEPTED | saved total: {count}' if accepted
                           else f'REJECTED | saved total unchanged: {count}']
        if quality:
            self.last_lines.append(self.quality_text(quality))
        self.last_lines.extend(textwrap.wrap(message, width=68))

    @staticmethod
    def quality_text(quality):
        return (f'Corners: {quality["corners"]}   '
                f'Reprojection RMS: {quality["reprojection_rms_px"]:.3f} px')

    def board_window_status(self, pose, camera, detector, margin_px=10):
        """Project the complete board, not only its detected inner corners."""
        if self.board_window is None:
            return True, None
        cfg = detector.cfg
        width = cfg['squares_x'] * cfg['square_length_m']
        height = cfg['squares_y'] * cfg['square_length_m']
        corners = np.asarray([[0, 0, 0], [width, 0, 0],
                              [width, height, 0], [0, height, 0]], dtype=np.float64)
        rotation = np.asarray(pose, dtype=float)[:3, :3]
        translation = np.asarray(pose, dtype=float)[:3, 3]
        if np.any((rotation @ corners.T + translation[:, None])[2] <= 0):
            return False, None
        rvec = cv2.Rodrigues(rotation)[0]
        points = cv2.projectPoints(corners, rvec, translation,
                                   camera.K, camera.D)[0].reshape(-1, 2)
        x0, y0, x1, y1 = self.board_window
        inside = bool(np.isfinite(points).all()
                      and np.all(points[:, 0] >= x0 + margin_px)
                      and np.all(points[:, 0] < x1 - margin_px)
                      and np.all(points[:, 1] >= y0 + margin_px)
                      and np.all(points[:, 1] < y1 - margin_px))
        return inside, points

    def render(self, live, live_lines, count):
        # Resize only the displayed copy; detector and saved images keep original coordinates.
        panel_width, image_height, header = 640, 480, 160
        canvas = np.full((image_height + header, panel_width * 2, 3), 28, np.uint8)
        if live is not None:
            canvas[header:, :panel_width] = cv2.resize(live, (panel_width, image_height))
        if self.last_image is not None:
            canvas[header:, panel_width:] = cv2.resize(self.last_image, (panel_width, image_height))
        left = [f'LIVE VIEW | saved samples: {count}',
                'Enter / Space: capture    Q / Esc / close: quit', *live_lines]
        right = ['LAST ATTEMPT / SAVED SAMPLE', *self.last_lines]
        for x, lines, color in [(12, left, (225, 225, 225)),
                                (panel_width + 12, right, self.last_color)]:
            for i, line in enumerate(lines[:6]):
                cv2.putText(canvas, line, (x, 24 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX,
                            .48, color, 1, cv2.LINE_AA)
        return canvas

    @staticmethod
    def terminal_command():
        """Keep terminal Enter/q available without a second input or camera thread."""
        try:
            readable = select.select([sys.stdin], [], [], 0)[0]
        except (OSError, ValueError):
            return None
        if not readable:
            return None
        line = sys.stdin.readline()
        return 'q' if line == '' else line.strip().lower()

    def read_command(self, camera, detector, count, on_frame=None):
        while True:
            try:
                visible = cv2.getWindowProperty(self.TITLE, cv2.WND_PROP_VISIBLE)
            except cv2.error:
                # Qt can remove its receiver before this check after closing the last window.
                return 'q'
            if visible < 1:
                return 'q'
            started = time.monotonic()
            image = camera.capture()
            try:
                board, quality, vis = detector.detect(image, camera.K, camera.D)
                xyz = board[:3, 3] * 1000
                lines = [self.quality_text(quality),
                         f'Board in camera (mm): {xyz[0]:.1f}, {xyz[1]:.1f}, {xyz[2]:.1f}',
                         'Detection OK. Stop the arm before capturing.']
                if self.board_window is not None:
                    inside, points = self.board_window_status(board, camera, detector)
                    if points is not None:
                        cv2.polylines(vis, [np.rint(points).astype(np.int32)], True,
                                      (0, 255, 0) if inside else (0, 0, 255), 2)
                    if not inside:
                        lines = ['BOARD OUTSIDE CYAN WINDOW: route cannot replay',
                                 'Move it fully inside before sampling or travelling.']
            except ValueError as exc:
                board = quality = None
                vis = image.copy()
                if detector.roi is not None:
                    x0, y0, x1, y1 = detector.roi
                    cv2.rectangle(vis, (x0, y0), (x1-1, y1-1), (255, 128, 0), 1)
                lines = ['DETECTION FAILED', *textwrap.wrap(str(exc), width=68)]
            if on_frame is not None:
                on_frame(image, board, quality)
            if self.board_window is not None:
                x0, y0, x1, y1 = self.board_window
                cv2.rectangle(vis, (x0, y0), (x1-1, y1-1), (255, 255, 0), 2)
                if len(lines) < 3:
                    lines.append('CYAN BOX: hand-board visibility window')
            self.current_image = vis
            cv2.imshow(self.TITLE, self.render(vis, lines, count))
            # Five previews per second limit X11 traffic; collection still uses fresh frames.
            delay = max(1, int(1000 * (.2 - (time.monotonic() - started))))
            key = cv2.waitKey(delay) & 0xff
            if key in (10, 13, 32):
                self.show_sampling(count)
                return ''
            if key in (27, ord('q'), ord('Q')):
                return 'q'
            command = self.terminal_command()
            if command is not None:
                if not command:
                    self.show_sampling(count)
                return command

    def show_sampling(self, count):
        cv2.imshow(self.TITLE, self.render(self.current_image,
                   ['CAPTURING... Keep the arm and board still.'], count))
        cv2.waitKey(1)

    def close(self):
        if not self.closed:
            self.closed = True
            cv2.destroyAllWindows()
