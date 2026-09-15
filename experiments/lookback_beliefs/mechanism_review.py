"""Read frozen natural-case interventions; do not infer truth from attention.

This is a CPU-only reviewer for natural-lookback-audit-v1 (commit 350247a9).
No model, training, annotation loading, or new intervention is performed here.
Signs are descriptive finite effects, not significance tests or pure QK proofs.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


CUTS = ('constraint', 'payload', 'control_constraint', 'control_payload',
        'constraint_and_payload', 'history_seed', 'history_control')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
                         encoding='utf8')
    temporary.replace(path)


def _vector(data, key, n):
    value = np.asarray(data[key])
    if value.shape != (n,) or not np.isfinite(value).all():
        raise ValueError(f'{key}: expected {n} finite values')
    return value


def load_arm(directory, name, baseline=None):
    with np.load(Path(directory) / (name + '.npz'), allow_pickle=False) as f:
        data = {key: f[key].copy() for key in (
            'steps', 'queries', 'saved_token_ids', 'alternative_ids', 'saved_logp')}
    n = len(data['steps'])
    for key in data:
        _vector(data, key, n)
    for key in ('steps', 'queries', 'saved_token_ids', 'alternative_ids'):
        if not np.issubdtype(data[key].dtype, np.integer):
            raise ValueError(f'{name}: non-integer {key}')
    if not n or np.any(np.diff(data['steps']) <= 0) or np.any(data['saved_logp'] > 1e-5):
        raise ValueError(f'{name}: invalid steps or log probabilities')
    if baseline is not None:
        for key in ('steps', 'queries', 'saved_token_ids', 'alternative_ids'):
            if not np.array_equal(data[key], baseline[key]):
                raise ValueError(f'{name}: {key} differs from frozen baseline')
    return data


def effect_profile(base, constraint, payload, control_constraint, control_payload, joint, floor):
    """Positive cut delta = the removed message opposed the saved output.

    Require raw AND control-adjusted signs on the SAME token. Opposite signs
    on different words must not be pooled into a fictitious within-token conflict.
    """
    c, v = constraint - base, payload - base
    cc, vc = control_constraint - base, control_payload - base
    specific_c, specific_v = c - cc, v - vc
    opposed = (c > floor) & (specific_c > floor)
    supported = (v < -floor) & (specific_v < -floor)
    return dict(constraint_cut_delta=c, payload_cut_delta=v,
                constraint_minus_control=specific_c, payload_minus_control=specific_v,
                constraint_opposition=opposed, payload_support=supported,
                joint_sign_pattern=opposed & supported,
                constraint_unresolved=np.abs(c) <= floor,
                finite_nonadditivity=base - constraint - payload + joint)


def mediation_profile(base, cut, restored, reverse, control, floor, control_changed):
    total, rescue, transferred = cut - base, restored - cut, reverse - base
    active = np.abs(total) > floor
    towards_base = active & (rescue * total < 0) & (np.abs(rescue) > floor)
    closer = np.abs(restored - base) + floor < np.abs(total)
    reciprocal = active & (transferred * total > 0) & (np.abs(transferred) > floor)
    specific = bool(control_changed) & (np.abs(restored - base) + floor < np.abs(control - base))
    fractions = [float(-r / d) if valid else None for r, d, valid in zip(rescue, total, active)]
    return dict(recovery_delta=rescue, reverse_delta=transferred,
                control_recovery_delta=control - cut, recovery_fraction=fractions,
                towards_baseline=towards_base & closer, reciprocal_direction=reciprocal,
                better_than_control=specific,
                joint_direction_pattern=towards_base & closer & reciprocal & specific)


def numeric_floor(checks, minimum):
    """Observed logit discrepancies bound log-probability noise by 2*max_error.

    This floor does NOT bound systematic intervention bias or sampling uncertainty.
    """
    if not np.isfinite(minimum) or minimum < 0:
        raise ValueError('minimum effect must be finite and nonnegative')
    for key in ('saved_top_logit_error', 'saved_log_normalizer_error', 'replay_atol',
                'zero_cut_max_logit_error'):
        if not np.isfinite(checks[key]) or checks[key] < 0:
            raise ValueError('invalid numerical check: ' + key)
    if max(checks['saved_top_logit_error'], checks['saved_log_normalizer_error']) > checks['replay_atol']:
        raise ValueError('original-cache replay check failed')
    if checks['zero_cut_max_logit_error'] > 1e-5:
        raise ValueError('zero-cut check failed')
    noise = [checks['zero_cut_max_logit_error']]
    for value in checks['carrier_cache'].values():
        if not np.isfinite(value['live_sham_error']) or value['live_sham_error'] > 1e-5:
            raise ValueError('same-world carrier check failed')
        error = value['max_logit_error'] if value['reused'] else value['live_sham_error']
        if not np.isfinite(error) or error < 0:
            raise ValueError('invalid carrier numerical error')
        noise.append(error)
    return max(minimum, 2 * max(noise))


def review_case(directory, min_effect=1e-3):
    directory = Path(directory)
    if not read_json(directory / 'mechanism_complete.json').get('complete'):
        raise ValueError('finite interventions are incomplete')
    detail, checks = read_json(directory / 'case.json'), read_json(directory / 'checks.json')
    if detail.get('version') != 'natural-lookback-audit-v1':
        raise ValueError('not the native-message-cut schema; do not mix the old echo-prompt patch')
    coords, case = detail['coordinates'], detail['case']
    base = load_arm(directory, 'baseline')
    n = len(base['steps'])
    if (not np.array_equal(base['steps'], coords['steps']) or
            not np.array_equal(base['queries'], coords['prompt_length'] + base['steps'] - 1)):
        raise ValueError('original prediction coordinates differ')
    if not set(coords['target_steps']) <= set(base['steps'].tolist()) or not coords['target_steps']:
        raise ValueError('target window missing from baseline')
    if coords['carrier_absolute'] >= coords['prompt_length'] + min(coords['target_steps']) - 1:
        raise ValueError('carrier is not before the first target query')
    floor = numeric_floor(checks, min_effect)
    cut = {name: load_arm(directory, 'cut_' + name, base)['saved_logp'] for name in CUTS}
    profile = effect_profile(base['saved_logp'], *(cut[name] for name in CUTS[:5]), floor)
    with np.load(directory / 'cached_observations.npz', allow_pickle=False) as observed:
        if not np.array_equal(observed['steps'], base['steps']):
            raise ValueError('attention observations are from different steps')
        attention = {name: observed['mass_' + name].copy() for name in ('constraint', 'payload')}
    for name, mass in attention.items():
        if mass.ndim != 3 or mass.shape[-1] != n or not np.isfinite(mass).all() or (mass < 0).any():
            raise ValueError('invalid per-head observations: ' + name)
    token_text = {}
    with (directory / 'tokens.csv').open(encoding='utf8', newline='') as stream:
        token_text = {int(row['step']): row['token'] for row in csv.DictReader(stream)}
    targets = np.isin(base['steps'], coords['target_steps'])
    rows = []
    for i in np.flatnonzero(targets):
        row = dict(step=int(base['steps'][i]), token=token_text.get(int(base['steps'][i]), ''),
                   saved_probability=float(np.exp(base['saved_logp'][i])))
        for name, values in profile.items():
            row[name] = values[i].item()
        for name, mass in attention.items():
            l, h = np.unravel_index(np.argmax(mass[:, :, i]), mass.shape[:2])
            row[name + '_largest_read'] = dict(layer=int(l), head=int(h), mass=float(mass[l, h, i]))
        rows.append(row)
    mediators = []
    for layer in checks['layers']:
        restored, reverse, control = [load_arm(directory, arm, base)['saved_logp'] for arm in (
            f'restore_carrier_l{layer}', f'cut_carrier_into_base_l{layer}', f'restore_control_site_l{layer}')]
        state_change = checks['carrier_cache'][str(layer)]['cut_control_state_l2_change']
        if not np.isfinite(state_change) or state_change < 0:
            raise ValueError('invalid control-state change')
        m = mediation_profile(base['saved_logp'], cut['constraint'], restored, reverse,
                              control, floor, state_change > 0)
        mediators.append(dict(layer=layer, control_state_l2_change=state_change,
            target_tokens=[dict(step=int(base['steps'][i]), **{
                key: (value[i].item() if isinstance(value, np.ndarray) else value[i])
                for key, value in m.items()}) for i in np.flatnonzero(targets)]))
    # A key for emitted y_s is visible at query P+s: it may first affect prediction s+1.
    first_key = max(coords['groups']['history_seed'])
    eligible = base['queries'] >= first_key
    hd = cut['history_seed'] - base['saved_logp']
    hc = cut['history_control'] - base['saved_logp']
    if np.any(np.abs(hd[~eligible]) > 1e-5) or np.any(np.abs(hc[~eligible]) > 1e-5):
        raise ValueError('history cut changed output before the seed was available')
    continuation = [dict(step=int(base['steps'][i]), in_target=bool(targets[i]),
                        seed_cut_delta=float(hd[i]), seed_minus_control=float(hd[i] - hc[i]),
                        supports_saved_continuation=bool(hd[i] < -floor and hd[i] - hc[i] < -floor))
                    for i in np.flatnonzero(eligible)]
    return dict(case_id=case['id'], source_id=str(detail['record']['source_id']),
                manual_case_status=case['status'], numerical_checks_passed=True,
                effect_floor_nats=floor, target_tokens=rows, carrier_layers=mediators,
                continuation_tokens=continuation,
                same_token_opposition_and_payload_support=sum(r['joint_sign_pattern'] for r in rows),
                target_token_count=len(rows),
                interpretation='Descriptive finite-effect signs only; not significance, truth recovery, or QK identification.',
                unresolved=['V-path removal does not isolate pointer vs address formation.',
                            'A negligible constraint effect does not prove absent encoding.',
                            'Whole-state carrier rescue is not a pure pointer intervention.',
                            'Teacher-forced continuation is not free-generation repair.',
                            'Controls match token counts, not exact distance, message norm, or semantics.'])


def review(audit, min_effect=1e-3):
    if not np.isfinite(min_effect) or min_effect < 0:
        raise ValueError('minimum effect must be finite and nonnegative')
    audit = Path(audit)
    if not audit.is_dir():
        raise FileNotFoundError(audit)
    config = read_json(audit / 'config.json')
    if config.get('version') != 'natural-lookback-audit-v1':
        raise ValueError('audit config must be natural-lookback-audit-v1')
    reports, pending, invalid = [], [], []
    for case in config['cases']:
        directory = audit / case['id']
        if not (directory / 'mechanism_complete.json').exists():
            pending.append(case['id'])
            continue
        try:
            result = review_case(directory, min_effect)
        except (ValueError, KeyError, FileNotFoundError) as error:
            invalid.append(dict(case=case['id'], reason=str(error)))
            continue
        reports.append(result)
    result = dict(schema='natural-mechanism-review-v1', completed_cases=len(reports),
                  independent_sources=len({r['source_id'] for r in reports}),
                  pending=pending, invalid=invalid, cases=reports,
                  new_model_forwards=0, labels_used_for_training=False,
                  scope='Manually selected regenerated RAG examples; no population detection claim.')
    save_json(audit / 'mechanism_review.json', result)
    lines = ['# 约束与载荷：有限作用核验', '',
             '“同词异向”指同一目标词上：移除约束更支持原词，移除载荷削弱原词，且方向超过各自对照。',
             '这些是描述性干预符号，不是显著性、事实修复或纯指针证明。', '',
             '| 案例 | 人工案例属性 | 目标词数 | 同词异向数 |', '|---|---|---:|---:|']
    for r in reports:
        lines.append(f'| {r["case_id"]} | {r["manual_case_status"]} | {r["target_token_count"]} | '
                     f'{r["same_token_opposition_and_payload_support"]} |')
    lines += ['', '待运行：' + ', '.join(pending), '无效／缺失输出：' + json.dumps(invalid, ensure_ascii=False),
              '', '详见 mechanism_review.json 的 target_tokens、carrier_layers、continuation_tokens。',
              '未观察到不等于机制不存在；不同token、不同head、不同层不是独立样本。']
    (audit / 'mechanism_review.md').write_text('\n'.join(lines) + '\n', encoding='utf8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', default='outputs/lookback_beliefs_natural_v1')
    parser.add_argument('--min-effect', type=float, default=1e-3,
                        help='numerical reporting floor in nats, NOT a statistical threshold')
    args = parser.parse_args()
    result = review(args.audit, args.min_effect)
    print(json.dumps({k: v for k, v in result.items() if k != 'cases'}, ensure_ascii=False))
    if result['invalid']:
        raise SystemExit('Invalid case outputs; see mechanism_review.json')


if __name__ == '__main__':
    main()
