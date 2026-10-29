"""Reviewed original-J-lens profiles; no cross-model fallback or J++ substitution."""
from dataclasses import asdict, dataclass
import argparse
import hashlib
import json
import math
from pathlib import Path

JLENS_COMMIT = '581d398613e5602a5af361e1c34d3a92ea82ba8e'
LENS_REPO = 'neuronpedia/jacobian-lens'
LENS_NEW = 'e166deb5ad540932100ef803c8c9bcc2a5453694'
BASE_WORDS = Path(__file__).resolve().parents[1] / 'data/paired-feasibility/concepts.json'


@dataclass(frozen=True)
class Profile:
    model: str
    model_revision: str
    lens_revision: str
    lens_file: str
    width: int
    layers: int
    protocol: str = 'qwen_thinking'
    paired_gate: str = 'legacy_validated'
    lens: str = LENS_REPO
    lens_method: str = 'original_jacobian_lens'

    def record(self):
        return asdict(self)


QWEN4 = Profile('Qwen/Qwen3.5-4B', '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a',
    '16a01f309fcec900fdcec3f4cd5b64f3d00e4d5a',
    'qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt', 2560, 32)
QWEN27 = Profile('Qwen/Qwen3.6-27B', '6a9e13bd6fc8f0983b9b99948120bc37f49c13e9', LENS_NEW,
    'qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt',
    5120, 64, paired_gate='complete_smoke_required')
GPT20 = Profile('openai/gpt-oss-20b', '6cee5e81ee83917806bbde320786a8fb61efebee', LENS_NEW,
    'gpt-oss-20b/jlens/Salesforce-wikitext/gpt-oss-20b_jacobian_lens.pt',
    2880, 24, protocol='harmony', paired_gate='fit_precision_unverified')
PROFILES = {p.model: p for p in (QWEN4, QWEN27, GPT20)}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def add_arguments(parser, experimental=False):
    parser.add_argument('--model', choices=PROFILES, default=QWEN4.model)
    parser.add_argument('--model-revision', help='Must match the reviewed profile commit')
    parser.add_argument('--lens', help='Must match the original-J-lens repository in the profile')
    parser.add_argument('--lens-revision', help='Must match the reviewed lens commit')
    parser.add_argument('--gpt-oss-loading', choices=['bf16-dequantized', 'native-mxfp4'],
                        help='Required for experimental GPT-OSS; never silently change precision')
    parser.add_argument('--reasoning-effort', choices=['low', 'medium', 'high'], default=None,
                        help='GPT-OSS only; default medium. No effect on Qwen reasoning length.')
    if experimental:
        parser.add_argument('--experimental-smoke', action='store_true',
                            help='Permit a GPT-OSS single-example experiment, not paired runs')


def resolve(args):
    name = getattr(args, 'model', QWEN4.model)
    if name not in PROFILES:
        raise ValueError(f'No reviewed model/lens profile for {name!r}')
    profile = PROFILES[name]
    for field in ('model_revision', 'lens', 'lens_revision'):
        override = getattr(args, field, None)
        if override is not None and override != getattr(profile, field):
            raise ValueError(f'{field} differs from the reviewed profile; add/revalidate a profile instead')
    if profile.protocol != 'harmony' and (getattr(args, 'gpt_oss_loading', None) or
                                         getattr(args, 'reasoning_effort', None)):
        raise ValueError('GPT-OSS loading/reasoning options cannot be applied to Qwen')
    return profile


def loading_mode(profile, args):
    if profile.protocol != 'harmony':
        return 'unquantized'
    mode = getattr(args, 'gpt_oss_loading', None)
    if mode is None:
        raise ValueError('GPT-OSS requires explicit --gpt-oss-loading; fitting-time precision is unverified')
    return mode


def template_options(profile, args):
    return ({'reasoning_effort': getattr(args, 'reasoning_effort', None) or 'medium'}
            if profile.protocol == 'harmony' else {'enable_thinking': True})


def workspace_band(n_layers, fitted_layers):
    return [layer for layer in fitted_layers if math.floor(.38*n_layers) <= layer <= math.floor(.92*n_layers)]


def validate_lens(profile, hf, lens):
    config = hf.config.get_text_config()
    if (config.hidden_size, config.num_hidden_layers) != (profile.width, profile.layers):
        raise ValueError('Model dimensions differ from the reviewed profile')
    if lens.d_model != profile.width or list(lens.source_layers) != list(range(profile.layers-1)):
        raise ValueError('Original lens width/source layers differ from the reviewed profile')
    if any(tuple(lens.jacobians[layer].shape) != (profile.width, profile.width) for layer in lens.source_layers):
        raise ValueError('Invalid original-J-lens matrix shape')


def load_hf(profile, device, dtype, mode):
    import torch
    from transformers import AutoConfig, GptOssForCausalLM, Mxfp4Config, Qwen3_5ForConditionalGeneration
    config = AutoConfig.from_pretrained(profile.model, revision=profile.model_revision, trust_remote_code=False)
    text = config.get_text_config()
    if (text.hidden_size, text.num_hidden_layers) != (profile.width, profile.layers):
        raise ValueError('Checkpoint config dimensions differ from profile; weights were not loaded')
    kwargs = dict(revision=profile.model_revision, dtype=getattr(torch, dtype),
                  device_map=device, trust_remote_code=False)
    if profile.protocol == 'harmony':
        if dtype != 'bfloat16' or mode not in ('bf16-dequantized', 'native-mxfp4'):
            raise ValueError('GPT-OSS smoke requires bf16 dtype and an explicit supported loading mode')
        kwargs['quantization_config'] = Mxfp4Config(dequantize=mode == 'bf16-dequantized')
        kwargs['attn_implementation'] = 'eager'  # Explicit, sink-aware reference path for the experimental smoke.
        hf = GptOssForCausalLM.from_pretrained(profile.model, **kwargs)
        actual = getattr(getattr(hf, 'hf_quantizer', None), 'quantization_config', None)
        if actual is None or actual.dequantize != (mode == 'bf16-dequantized'):
            raise ValueError('GPT-OSS loader silently changed the requested quantization mode')
        return hf
    return Qwen3_5ForConditionalGeneration.from_pretrained(profile.model, **kwargs)


def wordset(groups):
    return {group: sorted(words) for group, words in groups.items()}


def freeze_concepts(profile, source, tokenizer):
    groups, missing = {}, []
    for group, words in source['groups'].items():
        groups[group] = {}
        for word in words:
            ids = []
            for spelling in (word, ' ' + word):
                encoded = tokenizer.encode(spelling, add_special_tokens=False)
                if len(encoded) == 1 and encoded[0] != getattr(tokenizer, 'unk_token_id', None) and encoded[0] not in ids:
                    ids.append(encoded[0])
            if not ids:
                missing.append(f'{group}:{word}')
            groups[group][word] = ids
    if missing:
        raise ValueError('Frozen words lack single-token forms; no replacement/drop was made: ' + ', '.join(missing))
    return dict(schema_version=1, tokenizer=profile.model, tokenizer_revision=profile.model_revision,
        status='model-specific frozen draft words; not development-validated',
        tokenization_rule=source['tokenization_rule'], groups=groups,
        wordset_sha256=fingerprint(wordset(source['groups'])),
        source_concepts_fingerprint=fingerprint(source), model_profile=profile.record())


def check_paired_gate(profile, report_path=None, dtype='bfloat16', mode='unquantized'):
    if profile.paired_gate == 'legacy_validated':
        return
    if profile.paired_gate == 'fit_precision_unverified':
        raise ValueError('GPT-OSS paired runs are gated: fitting precision/provenance and pretrained smoke results need review. Use single_example.py --experimental-smoke only.')
    if report_path is None:
        raise ValueError('This model requires --validation-report from a complete --smoke-checks single-example run')
    report = json.loads(Path(report_path).read_text())
    if (report.get('model_profile') != profile.record() or report.get('dtype') != dtype
            or report.get('loading_mode') != mode or report.get('status') != 'complete_smoke_pass'
            or not all(report.get('checks', {}).get(k) is True for k in
                       ('exact_id_apply', 'walkthrough', 'token_identity', 'complete_thinking_response'))):
        raise ValueError('Validation report does not pass or is for different model/lens/precision')
    runtime = runtime_versions()
    for name, digest in report.get('evidence_sha256', {}).items():
        if name not in ('manifest.json', 'transcript.json', 'replay_validation.json', 'smoke_checks.json'):
            raise ValueError('Invalid smoke evidence filename')
        if hashlib.sha256((Path(report_path).parent/name).read_bytes()).hexdigest() != digest:
            raise ValueError('Smoke evidence changed after validation')
    if set(report.get('evidence_sha256', {})) != {'manifest.json', 'transcript.json', 'replay_validation.json', 'smoke_checks.json'}:
        raise ValueError('Missing smoke evidence hashes')
    root = Path(report_path).parent
    manifest = json.loads((root/'manifest.json').read_text())
    data = json.loads((root/'transcript.json').read_text())
    replay = json.loads((root/'replay_validation.json').read_text())
    smoke = json.loads((root/'smoke_checks.json').read_text())
    ids = data['prompt_ids'] + data['generated_ids']
    ids_hash = hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()
    if (manifest.get('status') != 'complete' or manifest.get('model_profile') != profile.record()
            or manifest.get('provenance', {}).get('dtype') != dtype
            or manifest.get('provenance', {}).get('loading_mode') != mode
            or ids != data['full_ids'] or data['ids_sha256'] != ids_hash
            or replay.get('replayed_ids_sha256') != ids_hash
            or not replay.get('token_identity') or not replay.get('original_generation_checked')
            or data['stop_reason'] != 'eos' or not data['spans']['span_analysis_eligible']
            or not data['spans'].get('cot') or not data['spans'].get('answer')
            or smoke.get('method') != 'original_jacobian_lens'
            or not smoke.get('exact_id_apply') or not smoke.get('walkthrough')):
        raise ValueError('Underlying smoke artifacts do not establish a complete validated example')
    if report.get('runtime') != runtime or report.get('code_hashes') != code_hashes():
        raise ValueError('Validation report runtime/code differs; repeat the single-example smoke')


def runtime_versions():
    from importlib.metadata import PackageNotFoundError, version
    import torch
    result = {}
    for package in ('torch', 'torchvision', 'transformers', 'jlens', 'accelerate',
                    'triton', 'kernels', 'flash-linear-attention', 'causal-conv1d', 'flash-attn'):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    result['torch_cuda_build'] = torch.version.cuda
    return result


def code_hashes():
    root = Path(__file__).parent
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ('model_profiles.py', 'single_example.py', 'reasoning_protocols.py', 'paired_feasibility.py')}


def verify_original_jlens():
    from importlib.metadata import distribution
    metadata = distribution('jlens').read_text('direct_url.json')
    commit = json.loads(metadata or '{}').get('vcs_info', {}).get('commit_id')
    if commit != JLENS_COMMIT:
        raise ValueError('Original jlens source commit is unverified. Install the pinned requirements in the experiment environment.')
    return commit


def main():
    parser = argparse.ArgumentParser(description='Freeze model-specific concept IDs before paired GPU outcomes; tokenizer only')
    add_arguments(parser)
    parser.add_argument('--source', type=Path, default=BASE_WORDS)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    profile = resolve(args)
    if args.output.exists():
        raise FileExistsError(args.output)
    from transformers import AutoTokenizer
    source = json.loads(args.source.read_text())
    tokenizer = AutoTokenizer.from_pretrained(profile.model, revision=profile.model_revision, trust_remote_code=False)
    result = freeze_concepts(profile, source, tokenizer)
    result['source_concepts_file_sha256'] = hashlib.sha256(args.source.read_bytes()).hexdigest()
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    print(json.dumps(dict(output=str(args.output), tokenizer=profile.model,
                         tokenizer_revision=profile.model_revision, wordset_sha256=result['wordset_sha256'])))


if __name__ == '__main__':
    main()
