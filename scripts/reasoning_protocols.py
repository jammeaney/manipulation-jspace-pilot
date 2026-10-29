"""Parse saved Harmony IDs. Qwen retains its original parser in single_example."""


def single_marker(tokenizer, spelling):
    ids = tokenizer.encode(spelling, add_special_tokens=False)
    decoded = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    if len(ids) != 1 or decoded != spelling or tokenizer.convert_tokens_to_ids(spelling) != ids[0]:
        raise ValueError(f'Required protocol marker is not a round-tripping single token: {spelling}')
    return ids[0]


def harmony_markers(tokenizer):
    markers = {key: single_marker(tokenizer, spelling) for key, spelling in
               [('start', '<|start|>'), ('message', '<|message|>'),
                ('end', '<|end|>'), ('return', '<|return|>')]}
    template = tokenizer.get_chat_template()
    channel = []
    for spelling in ('<|channel|>', '<|meta_sep|>'):
        if spelling in template:
            try:
                channel.append(single_marker(tokenizer, spelling))
            except ValueError:
                pass
    if len(channel) != 1:
        raise ValueError('Unsupported/ambiguous Harmony channel marker in pinned template; inspect tokenizer before inference')
    markers['channel'] = channel[0]
    if len(set(markers.values())) != len(markers):
        raise ValueError('Protocol markers collide')
    return markers


def harmony_spans(prompt, generated, tokenizer, user_positions, eos_ids):
    markers = harmony_markers(tokenizer)
    full, flags, channels = prompt + generated, [], []
    start_positions = [i for i, token in enumerate(prompt) if token == markers['start']]
    if not start_positions:
        raise ValueError('Missing assistant prefill start marker')
    cursor = start_positions[-1]
    def text(ids):
        return tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    while cursor < len(full):
        if full[cursor] != markers['start']:
            flags.append('invalid_harmony_message_start'); break
        next_start = next((i for i in range(cursor+1, len(full)) if full[i] == markers['start']), len(full))
        boundary = next((i for i in range(cursor+1, next_start) if full[i] == markers['message']), None)
        if boundary is None:
            flags.append('incomplete_harmony_header'); break
        header = full[cursor+1:boundary]
        if header.count(markers['channel']) != 1:
            flags.append('missing_or_extra_harmony_channel'); break
        channel_at = header.index(markers['channel'])
        role, channel = text(header[:channel_at]).strip(), text(header[channel_at+1:]).strip()
        if role != 'assistant' or channel not in ('analysis', 'final'):
            flags.append('unsupported_harmony_role_or_channel'); break
        end = next((i for i in range(boundary+1, next_start)
                    if full[i] in (markers['end'], markers['return'])), next_start)
        terminated = end < next_start
        if any(t in (markers['channel'], markers['message']) for t in full[boundary+1:end]):
            flags.append('unexpected_harmony_control_token')
        if not text(full[boundary+1:end]).strip():
            flags.append('empty_reasoning' if channel == 'analysis' else 'empty_final_answer')
        if channel == 'analysis' and (not terminated or full[end] != markers['end']):
            flags.append('incomplete_analysis_channel')
        if channel == 'final' and terminated and full[end] != markers['return']:
            flags.append('unexpected_final_terminator')
        if terminated and end+1 != next_start:
            flags.append('tokens_outside_harmony_message')
        channels.append(dict(channel=channel, header_start=cursor, content=[boundary+1, end],
                             terminator=end if terminated else None))
        cursor = next_start
    if [c['channel'] for c in channels] != ['analysis', 'final']:
        flags.append('missing_repeated_or_reordered_channels')
    if generated and generated[-1] in eos_ids and generated[-1] != markers['return']:
        flags.append('unexpected_harmony_stop_token')
    valid = not flags
    return dict(coordinate_system='full-sequence, zero-based, end-exclusive',
        response_start=len(prompt)-1, user_prompt_positions=user_positions,
        thinking_open_positions=[c['content'][0]-1 for c in channels if c['channel'] == 'analysis'],
        generated_thinking_close_positions=[c['terminator'] for c in channels
            if c['channel'] == 'analysis' and c['terminator'] is not None],
        cot=channels[0]['content'] if valid else None,
        answer=channels[1]['content'] if valid else None,
        span_analysis_eligible=valid, flags=sorted(set(flags)), reasoning_protocol='harmony',
        channel_spans=channels, protocol_markers=markers)
