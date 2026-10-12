"""Immutable per-observation images and data, separate from accepted solver samples."""
import time

from calibration.web.common import write


class FrameArchive:
    def __init__(self, path):
        self.path = path
        self.path.mkdir(parents=True, exist_ok=False)
        self.frames = []
        self.finish('capturing')

    def record(self, image, visualization, data):
        import cv2
        from calibrate import write_image_new, write_new
        stem = f'frame_{len(self.frames):03d}'
        raw, detected = stem+'.png', stem+'_detected.png'
        write_image_new(self.path/raw, image)
        # Missing detections must not look like successful annotated observations.
        fallback = visualization is None
        if fallback:
            visualization = image.copy()
            cv2.putText(visualization, 'DETECTION FAILED - see frame JSON', (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 255), 2)
        write_image_new(self.path/detected, visualization)
        write_new(self.path/(stem+'.json'), dict(data, raw_image=raw,
                  detection_image=detected, visualization_has_detection=not fallback))
        self.frames.append(stem+'.json')
        self.finish('capturing')

    def finish(self, status, **values):
        write(self.path/'capture.json', dict(schema=1, status=status, frames=self.frames,
              updated_unix_s=time.time(), **values))
