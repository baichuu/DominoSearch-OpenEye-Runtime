"""Explicit model selection; hardware and INT8 arithmetic stay unchanged."""
from copy import deepcopy
import os
from fpga.openeye.prepare_packed_layer_fixture import DEFAULT_PLAN, load_json

DENSE_PROFILE = 'dense-baseline-v1'
DENSE_IDENTITIES = {
    'checkpoint_sha256': 'a96f895435d204f75bc5980ac0c2f580cf21d078ad3d66f1a13a664274fe4e81',
    'packed_manifest_sha256': '58eb0095dd2da6f70c53199b882ca11ad8b118227fb05d39201a5e42c0e98f78',
    'scheme_sha256': 'b68fa41315f2fc97e6e61e9872143c43be0a926e1b9423ef2f04cc43770e7861',
}
DENSE_BUNDLE_SHA256 = '3968648d57e511653083c6520e6135a09227cae3a3c1952a297a32ee21a5a1cf'


def selected_model(value=None):
    value = os.environ.get('OPENEYE_MODEL_PROFILE', 'mac20') if value is None else value
    if value not in ('mac20', DENSE_PROFILE):
        raise ValueError('Unknown model profile; refusing baseline substitution')
    return value


def candidate_name(profile):
    return 'dense' if selected_model(profile) == DENSE_PROFILE else 'mac20'


def execution_plan(profile):
    plan = deepcopy(load_json(DEFAULT_PLAN))
    if selected_model(profile) == DENSE_PROFILE:
        for row in plan['layers']:
            row['expected_scheme'] = '4:4'
    return plan


def model_ingress(ingress, profile):
    name = candidate_name(profile)
    candidate = ingress['candidates'][name]
    expected_commit = 'dense-baseline-20261107' if name == 'dense' else '3060cf7'
    if ingress['method'] != 'packed-v1-fpga-ingress-integrity' or ingress['source_commit'] != expected_commit:
        raise ValueError('Wrong immutable model ingress')
    if name == 'dense' and (ingress['bundle_manifest_sha256'] != DENSE_BUNDLE_SHA256
            or any(candidate[k] != v for k,v in DENSE_IDENTITIES.items() if k != 'scheme_sha256')
            or candidate['layers'] != 21 or candidate['schemes'] != {'4:4': 21}):
        raise ValueError('Dense source is not the original verified checkpoint')
    return candidate


def verify_dense_manifest(manifest, profile):
    if selected_model(profile) != DENSE_PROFILE:
        return
    if (manifest['checkpoint']['sha256'] != DENSE_IDENTITIES['checkpoint_sha256']
            or manifest['scheme']['sha256'] != DENSE_IDENTITIES['scheme_sha256']
            or len(manifest['layers']) != 21
            or any(r['n'] != 4 or r['m'] != 4 or r['metadata'] is not None for r in manifest['layers'])):
        raise ValueError('Dense checkpoint/scheme/mask identity mismatch')
