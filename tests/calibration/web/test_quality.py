"""Current-policy evaluation, durable approvals and immutable historical inputs."""
import copy
from unittest.mock import Mock, patch

import pytest

from calibration.web.common import read, write, digest, ROOT
from calibration.web.quality import evaluate, approval_key
from calibration.web.worker import BrowserWorkflow

CFG = {'quality_thresholds': {'position_mm': 15, 'angle_deg': 5}}


def result(position=.01277, angle=2.46):
    return {'quality_passed': False, 'thresholds': {'position_m': .005, 'angle_deg': 2.},
            'all_sample_errors_m_deg': [[position, angle]]*12,
            'holdout_errors_m_deg': [[position, angle]]*3,
            'T_base_camera': [[1, 0], [0, 1]]}


def test_reassessment_is_consistent_for_historical_result_and_registration():
    old = result()
    original = copy.deepcopy(old)
    new = evaluate(old, CFG)
    assert new['quality_passed']
    assert new['thresholds'] == {'position_m': .015, 'angle_deg': 5}
    assert old == original
    assert new['T_base_camera'] == old['T_base_camera']
    registration = evaluate({'source_calibration': old, 'quality_passed': False}, CFG)
    assert registration['quality_passed']
    assert registration['source_calibration'] == new
    assert approval_key(old, CFG) == approval_key({'source_calibration': old}, CFG)
    assert approval_key(old, CFG) != approval_key(old, {'quality_thresholds': {'position_mm': 16, 'angle_deg': 5}})


@pytest.mark.parametrize('errors', [None, [], [[float('nan'), 1]], [[-.1, 1]], [[True, 1]]])
def test_incomplete_or_invalid_evidence_cannot_be_promoted(errors):
    old = result()
    old['holdout_errors_m_deg'] = errors
    assert not evaluate(old, CFG)['quality_passed']
    old.update(quality_passed=True, thresholds={'position_m': .015, 'angle_deg': 5})
    assert not evaluate(old, CFG)['quality_passed']


@pytest.mark.parametrize('position,angle', [(.016, 1), (.001, 5.1)])
def test_both_error_sets_must_pass(position, angle):
    old = result()
    old['holdout_errors_m_deg'][0] = [position, angle]
    assert not evaluate(old, CFG)['quality_passed']


def workflow(tmp_path, action, source):
    job = tmp_path/'job'
    config = read(ROOT/'configs/calibration_workflow.json')
    config.update(state_file=str(tmp_path/'state.json'), output_root=str(tmp_path/'datasets'))
    write(job/'config.json', config)
    write(job/'request.json', {'action': action, 'source_path': str(source), 'source_sha256': digest(source)})
    return BrowserWorkflow(job)


def test_worker_uses_assessed_copy_for_plan_without_rewriting_reference(tmp_path):
    source = tmp_path/'historical.json'; write(source, result()); before = source.read_bytes()
    w = workflow(tmp_path, 'plan', source)
    w.prepare_source()
    fake = Mock(); fake.main.return_value = 0
    with patch('calibration.web.worker.importlib.import_module', return_value=fake):
        w.call('calibration/auto_collect.py', 'plan', '--calibration', source)
    args = fake.main.call_args.args[0]
    new = read(args[args.index('--calibration')+1])
    assert new['quality_passed'] and new['thresholds']['position_m'] == .015
    assert source.read_bytes() == before


def test_worker_plan_reassessment_retains_original_hash_checks(tmp_path):
    source = tmp_path/'historical.json'; write(source, result())
    plan = tmp_path/'plan.json'
    write(plan, {'calibration': str(source), 'source_sha256': {str(source): digest(source)}})
    w = workflow(tmp_path, 'auto', source); w.prepare_source()
    fake = Mock(); fake.main.return_value = 0
    with patch('calibration.web.worker.importlib.import_module', return_value=fake):
        w.call('calibration/auto_collect.py', 'run', '--plan', plan)
    derived = read(fake.main.call_args.args[0][-1])
    assert derived['quality_passed']
    assert derived['source_sha256'][str(source)] == digest(source)
    write(source, result(.014))
    with pytest.raises(ValueError, match='源文件'):
        w.call('calibration/auto_collect.py', 'run', '--plan', plan)
