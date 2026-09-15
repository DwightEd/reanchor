import json
from pathlib import Path

import numpy as np
import pytest

from experiments.lookback_beliefs.mechanism_review import (
    effect_profile, mediation_profile, numeric_floor, review, review_case,
)


def fixture(root: Path):
    case = root / 'cooking_onion'
    case.mkdir(parents=True)
    def write(name, value):
        (case / name).write_text(json.dumps(value), encoding='utf8')
    meta = dict(id='cooking_onion', status='unsupported_stage_duration')
    (root / 'config.json').write_text(json.dumps(dict(version='natural-lookback-audit-v1', cases=[meta])))
    steps, p = np.arange(8), 4
    write('case.json', dict(version='natural-lookback-audit-v1', case=meta,
                            record=dict(source_id='14375'), coordinates=dict(
                                prompt_length=p, steps=steps.tolist(), target_steps=[4, 5],
                                carrier_absolute=5, groups=dict(history_seed=[8]))))
    write('mechanism_complete.json', dict(complete=True))
    checks = dict(saved_top_logit_error=.01, saved_log_normalizer_error=.01, replay_atol=.1,
                  zero_cut_max_logit_error=0., layers=[15], carrier_cache={'15': dict(
                      live_sham_error=0., reused=False, max_logit_error=.03,
                      cut_control_state_l2_change=.8)})
    write('checks.json', checks)
    data = dict(steps=steps, queries=p+steps-1, saved_token_ids=steps+10, alternative_ids=steps+20)
    base = np.full(8, -1.)
    deltas = {name: np.zeros(8) for name in (
        'baseline', 'cut_constraint', 'cut_payload', 'cut_control_constraint',
        'cut_control_payload', 'cut_constraint_and_payload', 'cut_history_seed',
        'cut_history_control', 'restore_carrier_l15', 'cut_carrier_into_base_l15',
        'restore_control_site_l15')}
    deltas['cut_constraint'][4:] = .4
    deltas['cut_payload'][4:] = -.6
    deltas['cut_control_constraint'][4:] = .02
    deltas['cut_control_payload'][4:] = -.03
    deltas['cut_constraint_and_payload'][4:] = -.1
    deltas['cut_history_seed'][5:] = -.3
    deltas['cut_history_control'][5:] = -.01
    deltas['restore_carrier_l15'][4:] = .1
    deltas['cut_carrier_into_base_l15'][4:] = .25
    deltas['restore_control_site_l15'][4:] = .35
    for name, delta in deltas.items():
        np.savez_compressed(case / (name + '.npz'), **data, saved_logp=base+delta)
    mass = np.full((2, 3, 8), .02)
    np.savez_compressed(case / 'cached_observations.npz', steps=steps,
                        mass_constraint=mass, mass_payload=mass*2)
    (case / 'tokens.csv').write_text('step,token\n' + '\n'.join(f'{i},word{i}' for i in steps))
    return case


def test_same_token_opposition_and_support():
    b = np.array([-1., -1.])
    r = effect_profile(b, b+.4, b-.6, b+.02, b-.03, b-.1, .001)
    assert r['joint_sign_pattern'].all()
    np.testing.assert_allclose(r['finite_nonadditivity'], .1)


def test_opposite_effects_on_different_tokens_do_not_fabricate_conflict():
    b = np.array([-1., -1.])
    r = effect_profile(b, b+[.4, 0], b+[0, -.6], b, b, b, .001)
    assert r['constraint_opposition'].sum() == r['payload_support'].sum() == 1
    assert not r['joint_sign_pattern'].any()


def test_control_difference_without_raw_sign_is_insufficient():
    b = np.array([-1.])
    r = effect_profile(b, b, b, b-.4, b+.4, b, .001)
    assert r['constraint_minus_control'][0] > 0
    assert r['payload_minus_control'][0] < 0
    assert not r['joint_sign_pattern'][0]
    assert r['constraint_unresolved'][0]


def test_numerical_floor_is_not_absence_of_encoding():
    b = np.array([-1.])
    r = effect_profile(b, b+.0001, b-.0001, b, b, b, .001)
    assert r['constraint_unresolved'][0]
    assert not r['joint_sign_pattern'][0]


def test_reciprocal_rescue_and_control():
    b = np.array([-1.])
    r = mediation_profile(b, b+.4, b+.1, b+.25, b+.35, .001, True)
    assert r['joint_direction_pattern'][0]
    assert r['recovery_fraction'][0] == pytest.approx(.75)
    same_control = mediation_profile(b, b+.4, b+.1, b+.25, b+.1, .001, True)
    assert not same_control['joint_direction_pattern'][0]


def test_failed_or_overshooting_rescue_is_not_recovery():
    b = np.array([-1.])
    r = mediation_profile(b, b+.4, b-.8, b+.2, b+.3, .001, True)
    assert not r['towards_baseline'][0]
    absent = mediation_profile(b, b, b, b, b, .001, True)
    assert absent['recovery_fraction'][0] is None
    assert not absent['joint_direction_pattern'][0]


def test_unperturbed_control_site_cannot_certify_selectivity():
    b = np.array([-1.])
    r = mediation_profile(b, b+.4, b+.1, b+.25, b+.35, .001, False)
    assert not r['better_than_control'][0]


def test_full_frozen_schema_and_no_training(tmp_path):
    folder = fixture(tmp_path)
    r = review_case(folder)
    assert r['same_token_opposition_and_payload_support'] == 2
    assert r['target_tokens'][0]['step'] == 4
    assert r['target_tokens'][0]['payload_largest_read']['mass'] == .04
    assert r['carrier_layers'][0]['target_tokens'][0]['joint_direction_pattern']
    assert r['continuation_tokens'][0]['step'] == 5
    assert all(t['supports_saved_continuation'] for t in r['continuation_tokens'])
    before = {p.name: p.read_bytes() for p in folder.iterdir()}
    full = review(tmp_path)
    assert full['new_model_forwards'] == 0
    assert full['independent_sources'] == 1
    assert full['cases'][0]['manual_case_status'] == 'unsupported_stage_duration'
    assert before == {p.name: p.read_bytes() for p in folder.iterdir()}
    assert (tmp_path / 'mechanism_review.md').exists()


@pytest.mark.parametrize('field', ['steps', 'queries', 'saved_token_ids', 'alternative_ids'])
def test_reject_misaligned_arms(tmp_path, field):
    folder = fixture(tmp_path)
    file = folder / 'cut_constraint.npz'
    with np.load(file) as f:
        data = {k: f[k] for k in f.files}
    data[field] = data[field]+1
    np.savez(file, **data)
    with pytest.raises(ValueError):
        review_case(folder)


def test_missing_results_are_pending_not_zero_effect(tmp_path):
    folder = fixture(tmp_path)
    (folder / 'mechanism_complete.json').unlink()
    r = review(tmp_path)
    assert r['pending'] == ['cooking_onion']
    assert r['completed_cases'] == 0 and r['cases'] == []


def test_corrupt_completed_output_is_reported_invalid(tmp_path):
    folder = fixture(tmp_path)
    (folder / 'cut_payload.npz').unlink()
    r = review(tmp_path)
    assert r['invalid'][0]['case'] == 'cooking_onion'
    assert r['completed_cases'] == 0


def test_history_cannot_act_before_token_exists(tmp_path):
    folder = fixture(tmp_path)
    file = folder / 'cut_history_seed.npz'
    with np.load(file) as f:
        data = {k: f[k] for k in f.files}
    data['saved_logp'][4] -= .1
    np.savez(file, **data)
    with pytest.raises(ValueError, match='before'):
        review_case(folder)


def test_cache_replay_failure_prevents_interpretation(tmp_path):
    folder = fixture(tmp_path)
    file = folder / 'checks.json'
    c = json.loads(file.read_text())
    c['saved_top_logit_error'] = .2
    file.write_text(json.dumps(c))
    with pytest.raises(ValueError, match='replay'):
        review_case(folder)


def test_cached_state_error_only_counts_when_reused(tmp_path):
    folder = fixture(tmp_path)
    c = json.loads((folder / 'checks.json').read_text())
    assert numeric_floor(c, .001) == .001
    c['carrier_cache']['15']['reused'] = True
    assert numeric_floor(c, .001) == .06


def test_old_echo_prompt_schema_is_not_merged_as_native_message_cut(tmp_path):
    folder = fixture(tmp_path)
    file = folder / 'case.json'
    c = json.loads(file.read_text()); c['version'] = 'echo-reminder-v1'
    file.write_text(json.dumps(c))
    with pytest.raises(ValueError, match='schema'):
        review_case(folder)


def test_same_source_is_not_independent_evidence(tmp_path):
    fixture(tmp_path)
    import shutil
    shutil.copytree(tmp_path / 'cooking_onion', tmp_path / 'control')
    path = tmp_path / 'control/case.json'
    c = json.loads(path.read_text()); c['case']['id'] = 'control'
    c['case']['status'] = 'supported_local_control'; path.write_text(json.dumps(c))
    path = tmp_path / 'config.json'
    c = json.loads(path.read_text()); c['cases'].append(dict(id='control')); path.write_text(json.dumps(c))
    r = review(tmp_path)
    assert r['completed_cases'] == 2 and r['independent_sources'] == 1
    assert r['cases'][1]['manual_case_status'] == 'supported_local_control'
