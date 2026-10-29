"""CPU-only model profiles, Harmony, frozen IDs, gating and shared replay tests."""
import argparse
import contextlib
import copy
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import jlens
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, GptOssConfig, GptOssForCausalLM

from scripts import model_profiles as profiles
from scripts import reasoning_protocols as protocols
from scripts import single_example as single
from scripts import paired_feasibility as paired
import test_single_example as legacy


def harmony_tokenizer():
    words = ['[UNK]', '<|start|>', '<|message|>', '<|end|>', '<|return|>', '<|channel|>',
             'system', 'developer', 'user', 'assistant', 'analysis', 'final', 'medium',
             'reason', 'answer', 'Question', 'Helpful', '<|endoftext|>']
    words += [f'token{i}' for i in range(len(words), 64)]
    backend = Tokenizer(WordLevel({w: i for i,w in enumerate(words)}, unk_token='[UNK]'))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]',
        eos_token='<|return|>', additional_special_tokens=words[1:6]+['<|endoftext|>'])
    tokenizer.chat_template = ("{# <|channel|> #}"
        "{% for m in messages %}{{ '<|start|>' + ('developer' if m.role == 'system' else m.role) + '<|message|>' + m.content + '<|end|>' }}{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|start|>assistant' }}{% endif %}")
    return tokenizer


def harmony_ids(tokenizer, final_end=True):
    text = '<|channel|>analysis<|message|>reason<|end|><|start|>assistant<|channel|>final<|message|>answer'
    return tokenizer.encode(text+('<|return|>' if final_end else ''), add_special_tokens=False)


class WordTokenizer:
    def __init__(self, groups):
        words = sorted({w for group in groups.values() for w in group})
        self.ids = {w: i+100 for i,w in enumerate(words)}
    def encode(self, word, **kwargs):
        return [self.ids[word.strip()] + (1000 if word.startswith(' ') else 0)]


class ProfileTests(unittest.TestCase):
    def test_default_pins_and_workspace_fractions(self):
        self.assertEqual(profiles.resolve(SimpleNamespace()), profiles.QWEN4)
        self.assertEqual(profiles.QWEN4.model_revision, single.MODEL_REVISION)
        self.assertEqual(profiles.QWEN4.lens_revision, single.LENS_REVISION)
        self.assertEqual(profiles.workspace_band(32, range(31)), list(range(12,30)))
        self.assertEqual(profiles.workspace_band(64, range(63)), list(range(24,59)))
        self.assertEqual(profiles.workspace_band(24, range(23)), list(range(9,23)))
        for profile in profiles.PROFILES.values():
            self.assertIn('/jlens/', profile.lens_file)
            self.assertEqual(profile.lens_method, 'original_jacobian_lens')

    def test_unreviewed_models_revisions_and_jpp_are_rejected(self):
        for args in [dict(model='other/model'), dict(model=profiles.QWEN27.model, model_revision='main'),
                     dict(model=profiles.QWEN27.model,lens='other/jpp'),
                     dict(model=profiles.QWEN27.model,lens_revision='a'*40),
                     dict(model=profiles.QWEN27.model,reasoning_effort='high')]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                profiles.resolve(SimpleNamespace(**args))

    def test_model_specific_concept_freeze_checks_ids_and_preserves_source(self):
        original = profiles.BASE_WORDS.read_bytes()
        source = json.loads(original)
        tokenizer = WordTokenizer(source['groups'])
        frozen = profiles.freeze_concepts(profiles.QWEN27, source, tokenizer)
        self.assertEqual(profiles.wordset(source['groups']), profiles.wordset(frozen['groups']))
        self.assertNotEqual(frozen['groups'], source['groups'])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'concepts.json'; single.write_json(path, frozen)
            paired.load_concepts(path, tokenizer, profiles.QWEN27)
            with self.assertRaises(ValueError):
                paired.load_concepts(path, tokenizer, profiles.GPT20)
            frozen['groups']['control'].pop(next(iter(frozen['groups']['control'])))
            single.write_json(path, frozen)
            with self.assertRaises(ValueError):
                paired.load_concepts(path, tokenizer, profiles.QWEN27)
        self.assertEqual(profiles.BASE_WORDS.read_bytes(), original)
        with self.assertRaises(ValueError):
            paired.load_concepts(profiles.BASE_WORDS, profile=profiles.QWEN27)

    def test_concept_freeze_does_not_silently_drop_or_replace_words(self):
        source = json.loads(profiles.BASE_WORDS.read_text())
        tokenizer = WordTokenizer(source['groups'])
        original = tokenizer.encode
        tokenizer.encode = lambda text, **kw: [1,2] if text.strip() == 'hidden' else original(text, **kw)
        with self.assertRaisesRegex(ValueError, 'concealment:hidden'):
            profiles.freeze_concepts(profiles.GPT20, source, tokenizer)

    def test_config_and_lens_shape_validation(self):
        profile = replace(profiles.QWEN27, width=8, layers=3)
        hf = SimpleNamespace(config=SimpleNamespace(get_text_config=lambda: SimpleNamespace(hidden_size=8,num_hidden_layers=3)))
        lens = jlens.JacobianLens({0:torch.eye(8),1:torch.eye(8)}, n_prompts=1,d_model=8)
        profiles.validate_lens(profile,hf,lens)
        lens.jacobians[0] = torch.eye(7)
        with self.assertRaisesRegex(ValueError,'matrix shape'):
            profiles.validate_lens(profile,hf,lens)
        with self.assertRaisesRegex(ValueError,'dimensions'):
            profiles.validate_lens(profiles.QWEN27,hf,lens)

    def test_explicit_gpt_quantization_and_no_silent_fallback(self):
        with self.assertRaisesRegex(ValueError,'explicit'):
            profiles.loading_mode(profiles.GPT20,SimpleNamespace())
        config = SimpleNamespace(get_text_config=lambda:SimpleNamespace(hidden_size=2880,num_hidden_layers=24))
        hf = SimpleNamespace(hf_quantizer=SimpleNamespace(quantization_config=SimpleNamespace(dequantize=True)))
        with patch('transformers.AutoConfig.from_pretrained',return_value=config), \
             patch('transformers.GptOssForCausalLM.from_pretrained',return_value=hf) as load:
            profiles.load_hf(profiles.GPT20,'cuda','bfloat16','bf16-dequantized')
            self.assertTrue(load.call_args.kwargs['quantization_config'].dequantize)
            self.assertEqual(load.call_args.kwargs['attn_implementation'],'eager')
            self.assertEqual(load.call_args.kwargs['revision'],profiles.GPT20.model_revision)
            with self.assertRaisesRegex(ValueError,'silently changed'):
                profiles.load_hf(profiles.GPT20,'cuda','bfloat16','native-mxfp4')

    def test_qwen_loader_uses_conditional_generation_class_and_pins(self):
        config = SimpleNamespace(get_text_config=lambda:SimpleNamespace(hidden_size=5120,num_hidden_layers=64))
        with patch('transformers.AutoConfig.from_pretrained',return_value=config), \
             patch('transformers.Qwen3_5ForConditionalGeneration.from_pretrained') as load:
            profiles.load_hf(profiles.QWEN27,'cuda','bfloat16','unquantized')
        self.assertEqual(load.call_args.args,(profiles.QWEN27.model,))
        self.assertEqual(load.call_args.kwargs['revision'],profiles.QWEN27.model_revision)
        self.assertNotIn('quantization_config',load.call_args.kwargs)

    def test_single_cli_selects_qwen27_without_mutating_legacy_constants(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv=['single_example.py','--model',profiles.QWEN27.model,'--device','cpu',
                  '--output-dir',str(Path(tmp)/'sample'),'--temperature','1.0','--max-new-tokens','1024']
            with patch('sys.argv',argv), patch.object(single.AutoTokenizer,'from_pretrained'), \
                 patch.object(single,'prepare_prompt'), patch.object(profiles,'load_hf') as load, \
                 patch.object(single.jlens.JacobianLens,'from_pretrained'), \
                 patch.object(profiles,'validate_lens'), patch.object(single,'run_example') as run:
                single.main()
            self.assertEqual(load.call_args.args[0],profiles.QWEN27)
            self.assertEqual(run.call_args.kwargs['profile'],profiles.QWEN27)
            self.assertEqual(run.call_args.args[0].max_new_tokens,1024)
            self.assertEqual(run.call_args.args[0].temperature,1.0)
            self.assertEqual(single.MODEL,profiles.QWEN4.model)

    def test_profile_summary_uses_64_layer_band_and_rejects_model_mixing(self):
        from test_paired_feasibility import args_for
        source=json.loads(profiles.BASE_WORDS.read_text())
        frozen=profiles.freeze_concepts(profiles.QWEN27,source,WordTokenizer(source['groups']))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);args=args_for(root/'run');args.model=profiles.QWEN27.model
            args.concepts=root/'concepts.json';single.write_json(args.concepts,frozen)
            config=json.loads(args.scenarios.read_text());config['scenarios']=config['scenarios'][:1]
            args.scenarios=root/'scenarios.json';single.write_json(args.scenarios,config)
            plan=paired.prepare(args)
            self.assertEqual(plan['model_profile'],profiles.QWEN27.record())
            self.assertEqual(plan['independent_scenarios'],1)
            band=profiles.workspace_band(64,range(63))
            for row in plan['samples']:
                base=args.output_dir/'samples'/row['id'];base.mkdir(parents=True)
                data=dict(prompt_ids=[1,2],generated_ids=[3,4,5],full_ids=[1,2,3,4,5],
                    ids_sha256=single.ids_hash([1,2,3,4,5]),generated_text='fixture',stop_reason='eos',
                    messages=[dict(role='system',content=row['system_prompt']),dict(role='user',content=row['user_prompt'])],
                    spans=dict(response_start=1,user_prompt_positions=[0],cot=[2,3],answer=[3,4],
                               span_analysis_eligible=True,flags=[]))
                provenance=dict(model=profiles.QWEN27.model,model_revision=profiles.QWEN27.model_revision,
                    tokenizer_revision=profiles.QWEN27.model_revision,lens=profiles.QWEN27.lens,
                    lens_revision=profiles.QWEN27.lens_revision,lens_file=profiles.QWEN27.lens_file,
                    loading_mode='unquantized',dtype='float32')
                manifest=dict(status='complete',provenance=provenance,model_profile=profiles.QWEN27.record(),
                    generation_settings=plan['generation_settings'],lens_layers=list(range(63)),workspace_band=band,
                    concept_token_ids=frozen['groups'],generation_seconds=0,replay_seconds=0)
                validation=dict(replayed_ids_sha256=data['ids_sha256'],token_identity=True,original_generation_checked=True,
                    diagnostics=dict(median_abs_error_nats=0,p99_abs_error_nats=0,max_abs_error_nats=0,large_error_generated_indices=[]))
                for name,value in [('manifest',manifest),('transcript',data),('replay_validation',validation)]:
                    single.write_json(base/f'{name}.json',value)
                with (base/'lens_readouts.jsonl').open('w') as stream:
                    for layer in range(63):
                        raw=(-4 if row['condition']=='covert_persuasion' else -5) if layer in band else -8
                        for pos,token in enumerate(data['full_ids']):
                            stream.write(json.dumps(dict(layer=layer,position=pos,input_token_id=token,
                                workspace_band=layer in band,spans=single.position_roles(pos,data['spans']),
                                concept_readouts=dict(control_mean_logprob=-10,groups={g:dict(mean_logprob=raw,
                                    control_centered_score=raw+10) for g in paired.GROUPS})))+'\n')
            plan['status']='complete';single.write_json(args.output_dir/'experiment.json',plan)
            with contextlib.redirect_stdout(io.StringIO()):
                paired.summarize(args.output_dir,root/'summary')
            summary=json.loads((root/'summary/summary.json').read_text())
            self.assertEqual(summary['model_profile'],profiles.QWEN27.record())
            self.assertEqual(summary['generations'],2)
            for contrast in summary['workspace_paired_comparisons']:
                self.assertEqual(contrast['n_pairs'],1)
                self.assertEqual(contrast['neutral_mean_score'],5)
                self.assertEqual(contrast['covert_mean_score'],6)
            path=next((args.output_dir/'samples').iterdir())/'manifest.json'
            manifest=json.loads(path.read_text());manifest['provenance']['model']=profiles.GPT20.model
            single.write_json(path,manifest)
            with self.assertRaisesRegex(ValueError,'profile'):
                paired.summarize(args.output_dir,root/'bad-summary')

    def test_original_jlens_commit_must_be_verified(self):
        distribution = Mock(); distribution.read_text.return_value = None
        with patch('importlib.metadata.distribution',return_value=distribution):
            with self.assertRaisesRegex(ValueError,'source commit is unverified'):
                profiles.verify_original_jlens()

    def test_new_model_gates_fail_before_weight_loading_or_writing(self):
        from test_paired_feasibility import args_for
        with tempfile.TemporaryDirectory() as tmp:
            for model in (profiles.QWEN27.model, profiles.GPT20.model):
                args = args_for(Path(tmp)/'not-created');args.model=model;args.gpt_oss_loading='bf16-dequantized' if model==profiles.GPT20.model else None
                with patch.object(paired.AutoModelForCausalLM,'from_pretrained') as loader:
                    with self.assertRaisesRegex(ValueError,'gated|validation-report'):
                        paired.run(args)
                    loader.assert_not_called()
                    self.assertFalse(args.output_dir.exists())

    def test_complete_smoke_gate_binds_profile_runtime_code_and_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            data=dict(stop_reason='eos',spans=dict(span_analysis_eligible=True,cot=[1,2],answer=[2,3]),
                      prompt_ids=[1],generated_ids=[2,3,4],full_ids=[1,2,3,4],ids_sha256=single.ids_hash([1,2,3,4]))
            validation=dict(token_identity=True,original_generation_checked=True,
                replayed_ids_sha256=data['ids_sha256'],diagnostics={'flag_threshold_nats':.5})
            smoke=dict(exact_id_apply=True,walkthrough=True,method='original_jacobian_lens')
            manifest=dict(status='complete',model_profile=profiles.QWEN27.record(),
                          provenance=dict(dtype='bfloat16',loading_mode='unquantized'))
            for name,value in [('manifest',manifest),('transcript',data),('replay_validation',validation),('smoke_checks',smoke)]:
                single.write_json(root/f'{name}.json',value)
            single.write_validation_report(root,profiles.QWEN27,'bfloat16','unquantized',smoke,manifest,data,validation)
            path=root/'model_validation.json'
            profiles.check_paired_gate(profiles.QWEN27,path)
            with self.assertRaises(ValueError):
                profiles.check_paired_gate(profiles.QWEN27,path,dtype='float32')
            with self.assertRaisesRegex(ValueError,'gated'):
                profiles.check_paired_gate(profiles.GPT20,path,mode='bf16-dequantized')
            (root/'transcript.json').write_text('{"modified":true}')
            with self.assertRaisesRegex(ValueError,'evidence changed'):
                profiles.check_paired_gate(profiles.QWEN27,path)
            data['stop_reason']='max_new_tokens'
            single.write_validation_report(root,profiles.QWEN27,'bfloat16','unquantized',smoke,{},data,validation)
            with self.assertRaisesRegex(ValueError,'does not pass'):
                profiles.check_paired_gate(profiles.QWEN27,path)


class HarmonyTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = harmony_tokenizer()
        self.messages,self.prompt,self.user_positions = single.prepare_prompt(self.tokenizer,'Helpful','Question',
            profiles.GPT20, {'reasoning_effort':'medium'})

    def spans(self, generated):
        return protocols.harmony_spans(self.prompt,generated,self.tokenizer,self.user_positions,[4,17])

    def test_native_markers_header_and_exact_user_ids(self):
        markers=protocols.harmony_markers(self.tokenizer)
        self.assertEqual(markers['start'],1)
        self.assertEqual(markers['channel'],5)
        self.assertEqual([self.prompt[i] for i in self.user_positions],self.tokenizer.encode('Question',add_special_tokens=False))
        self.assertEqual(self.messages[0]['role'],'system')
        self.assertTrue(single.decode(self.tokenizer,self.prompt).endswith('<|start|> assistant'))

    def test_complete_analysis_final_channel_spans(self):
        generated=harmony_ids(self.tokenizer)
        spans=self.spans(generated);full=self.prompt+generated
        self.assertTrue(spans['span_analysis_eligible'])
        self.assertEqual(single.decode(self.tokenizer,full[slice(*spans['cot'])]),'reason')
        self.assertEqual(single.decode(self.tokenizer,full[slice(*spans['answer'])]),'answer')
        self.assertEqual(spans['response_start'],len(self.prompt)-1)

    def test_truncated_final_keeps_valid_spans_but_missing_final_is_ineligible(self):
        spans=self.spans(harmony_ids(self.tokenizer,final_end=False))
        self.assertTrue(spans['span_analysis_eligible'])
        spans=self.spans(self.tokenizer.encode('<|channel|>analysis<|message|>reason',add_special_tokens=False))
        self.assertFalse(spans['span_analysis_eligible'])
        self.assertIn('incomplete_analysis_channel',spans['flags'])

    def test_refusal_tools_repeated_channels_and_wrong_stop_are_flagged(self):
        for text in [
            '<|channel|>final<|message|>answer<|return|>',
            '<|channel|>analysis<|message|>reason<|end|><|start|>assistant<|channel|>analysis<|message|>reason<|return|>',
            '<|channel|>commentary<|message|>answer<|return|>',
            '<|channel|>analysis<|message|>reason<|end|><|start|>tool<|channel|>final<|message|>answer<|return|>',
            '<|channel|>analysis<|message|>reason<|end|><|start|>assistant<|channel|>final<|message|>answer<|endoftext|>',
            '<|channel|>analysis<|message|><|end|><|start|>assistant<|channel|>final<|message|>answer<|return|>',
        ]:
            with self.subTest(text=text):
                self.assertFalse(self.spans(self.tokenizer.encode(text,add_special_tokens=False))['span_analysis_eligible'])

    def test_ordinary_multitoken_old_harmony_markers_are_rejected(self):
        with self.assertRaises(ValueError):
            protocols.single_marker(self.tokenizer,'<|fim_suffix|>')
        original=self.tokenizer.chat_template
        self.tokenizer.chat_template=original.replace('<|channel|>','<|meta_sep|>')
        with self.assertRaisesRegex(ValueError,'channel marker'):
            protocols.harmony_markers(self.tokenizer)

    def test_tiny_gpt_shared_generation_and_replay_preserve_ids_and_new_workspace(self):
        config=GptOssConfig(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
            num_attention_heads=2,num_key_value_heads=1,head_dim=16,num_local_experts=4,num_experts_per_tok=2,
            max_position_embeddings=128,sliding_window=16,eos_token_id=4,
            rope_parameters={'rope_type':'default','rope_theta':10000.})
        config._attn_implementation='eager';torch.manual_seed(9)
        hf=GptOssForCausalLM(config).eval()
        model=jlens.from_hf(hf,self.tokenizer,force_bos=False)
        lens=jlens.JacobianLens({0:torch.eye(32)},n_prompts=1,d_model=32)
        # Test the shared orchestrator with an internal tiny profile, never exposed by CLI.
        profile=replace(profiles.GPT20,width=32,layers=2)
        generated=harmony_ids(self.tokenizer)
        with torch.inference_mode():
            logits=hf(input_ids=torch.tensor([self.prompt+generated]),use_cache=False).logits[0]
            lp=[logits[len(self.prompt)+j-1].float().log_softmax(-1)[t].item() for j,t in enumerate(generated)]
        with tempfile.TemporaryDirectory() as tmp:
            args=SimpleNamespace(output_dir=Path(tmp)/'sample',prompt='Question',system='Helpful',
                device='cpu',dtype='float32',max_new_tokens=30,temperature=1.,seed=42,
                position_chunk_size=3,logprob_flag_threshold=.5,gpt_oss_loading='bf16-dequantized',reasoning_effort='medium')
            with patch.object(single,'generate_one',return_value=(generated,lp,'eos')), \
                 contextlib.redirect_stdout(io.StringIO()):
                result=single.run_example(args,hf,self.tokenizer,lens,profile=profile,print_tokens=False)
            self.assertEqual(result['manifest']['provenance']['model'],profiles.GPT20.model)
            self.assertEqual(result['manifest']['workspace_band'],[0])
            self.assertTrue(result['transcript']['spans']['span_analysis_eligible'])
            self.assertEqual(result['transcript']['generated_ids'],generated)
            self.assertTrue(result['validation']['token_identity'])
            self.assertLess(result['validation']['diagnostics']['max_abs_error_nats'],1e-5)
            self.assertEqual(result['transcript']['template_kwargs'],{'reasoning_effort':'medium'})
            self.assertIn('<|start|>user<|message|>',result['transcript']['rendered_prompt'])
        # Actual cached sampling (random weights) also shares the unchanged raw-LP contract.
        ids,lps,_=single.generate_one(hf,self.prompt,2,1.,42,[])
        self.assertEqual(len(ids),2);self.assertEqual(len(lps),2)

    def test_qwen27_profile_shared_single_pipeline_and_complete_smoke_report(self):
        legacy.PipelineTests.setUpClass();fixture=legacy.PipelineTests()
        profile=replace(profiles.QWEN27,width=64,layers=2)
        lens=jlens.JacobianLens({0:torch.eye(64)},n_prompts=1,d_model=64)
        generated=[18,4,19,2]
        with torch.inference_mode():
            logits=fixture.hf(input_ids=torch.tensor([fixture.prompt+generated]),use_cache=False).logits[0]
            lps=[logits[len(fixture.prompt)+j-1].float().log_softmax(-1)[token].item() for j,token in enumerate(generated)]
        smoke=dict(method='original_jacobian_lens',exact_id_apply=True,walkthrough=True)
        with tempfile.TemporaryDirectory() as tmp:
            args=SimpleNamespace(output_dir=Path(tmp)/'sample',prompt='What is 2 + 2?',
                system='You are a helpful assistant.',device='cpu',dtype='float32',max_new_tokens=100,
                temperature=1.,seed=42,position_chunk_size=2,logprob_flag_threshold=.5,smoke_checks=True)
            with patch.object(single,'generate_one',return_value=(generated,lps,'eos')), \
                 patch.object(single,'smoke_checks',return_value=smoke), contextlib.redirect_stdout(io.StringIO()):
                result=single.run_example(args,fixture.hf,fixture.tokenizer,lens,profile=profile,print_tokens=False)
            self.assertTrue(result['transcript']['spans']['span_analysis_eligible'])
            self.assertEqual(result['manifest']['provenance']['model'],profiles.QWEN27.model)
            self.assertEqual(result['manifest']['provenance']['lens_revision'],profiles.QWEN27.lens_revision)
            self.assertEqual(result['manifest']['workspace_band'],[0])
            self.assertEqual(json.loads((args.output_dir/'model_validation.json').read_text())['status'],'complete_smoke_pass')
            profiles.check_paired_gate(profile,args.output_dir/'model_validation.json',dtype='float32')

    def test_legacy_qwen_prompt_ids_do_not_change(self):
        legacy.PipelineTests.setUpClass()
        tokenizer=legacy.PipelineTests.tokenizer
        baseline=single.prepare_prompt(tokenizer,'Helpful','What is 2 + 2?')
        selected=single.prepare_prompt(tokenizer,'Helpful','What is 2 + 2?',profiles.QWEN27,{'enable_thinking':True})
        self.assertEqual(baseline,selected)


if __name__=='__main__':
    unittest.main()
