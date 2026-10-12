"""Offline evidence capture must not change sample selection or hide rejected frames."""
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

import calibrate
from calibration.web.common import read
from calibration.web.frame_archive import FrameArchive


def devices():
    arm = Mock()
    arm.read.return_value = np.eye(4)
    arm.read_joints.return_value = {'joints_rad': [0.]*7, 'joints_deg': [0.]*7,
                                   'sdk_timestamp': 123.}
    camera = Mock(K=np.eye(3), D=np.zeros(5), info={'serial': 'offline-camera'})
    camera.capture.return_value = np.zeros((60, 80, 3), np.uint8)
    detector = Mock(cfg={'type': 'charuco', 'min_corners': 12})
    detector.detect.side_effect = [(np.eye(4), {'reprojection_rms_px': rms, 'corners': 12},
                                    np.full((60, 80, 3), 100+i, np.uint8))
                                   for i, rms in enumerate((.3, .1, .2))]
    return arm, camera, detector


def test_three_observations_and_selected_frame_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(calibrate.time, 'sleep', lambda _: None)
    archive = FrameArchive(tmp_path/'frames')
    sample, image, vis = calibrate.capture_sample(*devices(), frame_observer=archive.record)
    archive.finish('capture_validated', sample=sample)
    manifest = read(archive.path/'capture.json')
    assert len(manifest['frames']) == 3
    assert sample['selected_frame_index'] == 1
    for index, name in enumerate(manifest['frames']):
        record = read(archive.path/name)
        assert record['frame_index'] == index
        assert record['camera']['serial'] == 'offline-camera'
        assert record['T_camera_board'] == np.eye(4).tolist()
        assert record['joint_observation']['sdk_timestamp'] == 123.
        assert record['status'] == 'detected'
        assert record['visualization_has_detection']
        np.testing.assert_array_equal(cv2.imread(str(archive.path/record['raw_image'])), image)
        if index == 1:
            np.testing.assert_array_equal(cv2.imread(str(archive.path/record['detection_image'])), vis)
    # CLI callers retain the original sample shape and best-frame selection.
    legacy, _, legacy_vis = calibrate.capture_sample(*devices())
    assert 'selected_frame_index' not in legacy
    assert 'frame_archive' not in legacy
    assert legacy['quality'] == sample['quality']
    np.testing.assert_array_equal(legacy_vis, vis)


@pytest.mark.parametrize('failed_index', [0, 2])
def test_failed_detection_keeps_raw_frame_and_failure_data(tmp_path, monkeypatch, failed_index):
    monkeypatch.setattr(calibrate.time, 'sleep', lambda _: None)
    arm, camera, detector = devices()
    detections = list(detector.detect.side_effect)
    detections[failed_index] = ValueError('Too few corners: 8/12')
    detector.detect.side_effect = detections
    archive = FrameArchive(tmp_path/'frames')
    with pytest.raises(ValueError, match='8/12'):
        calibrate.capture_sample(arm, camera, detector, frame_observer=archive.record)
    assert len(read(archive.path/'capture.json')['frames']) == failed_index+1
    record = read(archive.path/f'frame_{failed_index:03d}.json')
    assert record['status'] == 'rejected' and record['T_camera_board'] is None
    assert record['error'] == 'Too few corners: 8/12'
    assert not record['visualization_has_detection']
    assert cv2.imread(str(archive.path/record['raw_image'])) is not None
    assert cv2.imread(str(archive.path/record['detection_image'])).any()


def test_archive_write_failure_is_not_silently_ignored(tmp_path, monkeypatch):
    monkeypatch.setattr(calibrate.time, 'sleep', lambda _: None)
    archive = FrameArchive(tmp_path/'frames')
    monkeypatch.setattr(calibrate, 'write_image_new', Mock(side_effect=OSError('disk full')))
    with pytest.raises(OSError, match='disk full'):
        calibrate.capture_sample(*devices(), frame_observer=archive.record)
    assert read(archive.path/'capture.json')['status'] == 'capturing'
    with pytest.raises(FileExistsError):
        FrameArchive(archive.path)
