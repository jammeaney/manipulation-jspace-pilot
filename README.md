# Investigating Covert Persuasion Using the Jacobian Lens

## Overview

This project investigates whether the Jacobian Lens (J-lens) can provide useful insights into how covert persuasion is represented within large language models.

I'm particularly interested in how evidence from internal representations might differ from what we can observe in a model's chain of thought (CoT) and generated output. The current pilot is a feasibility study focused on establishing whether J-lens produces interpretable signals under different prompting conditions.

## Pilot study

The pilot uses Qwen3.5-4B and five fictional local referendum scenarios. Each scenario is presented under two conditions:

- **Neutral:** The model is instructed to provide a balanced response.
- **Covert persuasion:** The model is instructed to steer its response towards a specified option without explicitly disclosing that objective.

Generated token sequences are replayed through J-lens to examine three draft concept sets: persuasion, concealment and evaluation awareness.

The analysis compares concept scores between conditions, across model layers and different token spans. The aim is to establish whether these measurements are technically feasible and whether they reveal patterns worth investigating further.

## Current status

The generation and J-lens replay pipeline is operational, with successful token-identity validation.

Initial results suggest differences in concept scores between prompting conditions, although incomplete generations have limited comparisons across reasoning and final-answer spans.

An important unresolved question is whether these differences reflect persuasive intent or simply the wording of the system instructions.

## Next steps

The immediate priority is to obtain complete paired generations and investigate the resulting J-lens measurements. Further work will explore appropriate controls and how internal signals relate to evidence in CoT and final responses.

This is an exploratory investigation of model representations, not an attempt to develop a covert-persuasion classifier or detector.
