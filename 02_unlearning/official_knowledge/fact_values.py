"""Literal published fact spans and validation with the official dataset loader."""
import re


PROTOCOL = 'published_fact_value_tokens_v1'


def fact_span(row, unit):
    assert row['kind'] == 'published_sentence' and not row['prompt']
    value = unit['labels'][row['fact_id'].rsplit(':', 1)[1]]
    sentence = row['text']
    names = [(m.start(), m.end()) for m in re.finditer(re.escape(unit['title']), sentence)]
    spans = [m for m in re.finditer(re.escape(value), sentence)
             if not any(m.start() < end and start < m.end() for start, end in names)]
    if not spans:
        raise ValueError('No fact value outside the subject name: ' + row['record_id'])
    start, end = spans[0].span()
    prefix = sentence[:start].rstrip()
    target = sentence[len(prefix):end]
    assert target.strip() == value and prefix + target == sentence[:end]
    return dict(row, prompt=prefix, text=target, source_record_id=row['record_id'],
                record_id=row['record_id'] + ':fact-value-only-v1',
                loss_protocol=PROTOCOL, answer=value,
                source_span_start=start, source_span_end=end)


def validated_rows(rows, units, tokenizer, preprocess):
    converted = []
    for row in rows:
        item = fact_span(row, units[row['unit_id']])
        tokens = preprocess(tokenizer, item['prompt'], item['text'], 1024)
        labels = tokens['labels']
        supervised = labels[labels != -100]
        assert supervised.numel() > 0 and len(tokens['input_ids']) <= 1024
        assert tokenizer.decode(supervised.tolist(), skip_special_tokens=False).strip() == item['answer']
        prefix_ids = tokenizer.encode(item['prompt'], add_special_tokens=True)
        assert tokens['input_ids'][:len(prefix_ids)].tolist() == prefix_ids
        assert (labels[:len(prefix_ids)] == -100).all()
        item['tokens'], item['supervised_tokens'] = len(labels), int(supervised.numel())
        converted.append(item)
    return converted
