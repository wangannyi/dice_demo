"""Bounded preview must return or interrupt; never waitKey(0) in green STEP."""
import unittest
from contextlib import ExitStack
from unittest.mock import patch
import numpy as np
from cup_grasp_demo.calibration_debug import debug

class PreviewTest(unittest.TestCase):
    def run_preview(self, key=-1, visible=1, interrupted=None, property_error=None, cleanup_error=None):
        with ExitStack() as stack:
            stack.enter_context(patch.dict(debug.os.environ, {'DISPLAY':':test'}))
            stack.enter_context(patch.object(debug.cv2,'imread',return_value=np.zeros((3,3,3),np.uint8)))
            stack.enter_context(patch.object(debug.cv2,'imshow'))
            wait=stack.enter_context(patch.object(debug.cv2,'waitKey',return_value=key,side_effect=interrupted))
            stack.enter_context(patch.object(debug.cv2,'getWindowProperty',return_value=visible,side_effect=property_error))
            close=stack.enter_context(patch.object(debug.cv2,'destroyAllWindows',side_effect=cleanup_error))
            stack.enter_context(patch.object(debug.time,'monotonic',side_effect=[0,0,4]))
            try:
                debug.show('/unused.png',True,timeout_s=3)
            finally:
                close.assert_called_once()
                wait.assert_called_once_with(50)
    def test_qt_last_window_closed(self):
        error=debug.cv2.error('(-27:Null pointer) NULL guiReceiver in cvGetPropVisible_QT')
        self.run_preview(property_error=error,cleanup_error=error)
    def test_real_gui_error_stops(self):
        with self.assertRaisesRegex(RuntimeError,'图像预览失败'):
            self.run_preview(property_error=debug.cv2.error('Unexpected backend failure'))
    def test_interrupt_survives_missing_window_cleanup(self):
        with self.assertRaises(KeyboardInterrupt):
            self.run_preview(interrupted=KeyboardInterrupt(),cleanup_error=debug.cv2.error('NULL guiReceiver'))
    def test_timeout(self):self.run_preview()
    def test_window_close(self):self.run_preview(visible=0)
    def test_regular_key(self):self.run_preview(key=32)
    def test_escape_stops(self):
        with self.assertRaises(InterruptedError):self.run_preview(key=27)
    def test_ctrl_c_stops_and_cleans_window(self):
        with self.assertRaises(KeyboardInterrupt):self.run_preview(interrupted=KeyboardInterrupt())
    def test_no_show_never_opens_window(self):
        with patch.object(debug.cv2,'imshow') as show:
            debug.show('/unused.png',False,timeout_s=3)
            show.assert_not_called()
