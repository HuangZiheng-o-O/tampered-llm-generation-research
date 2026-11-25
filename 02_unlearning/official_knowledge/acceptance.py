"""Separate measured unlearning effects from artifact-first delivery acceptance."""
import math


def effect_checks(policy, remaining, retention, before, after):
    return {
        'target_reduction': remaining <= policy['remaining_previously_correct_max'],
        'knowledge_retention': retention >= policy['retain_previously_correct_min'],
        'general_ability': before - after <= policy['mmlu_drop_max'],
    }


def unlearning_passed(policy, remaining, retention, before, after):
    values = (remaining, retention, before, after)
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in values):
        return False
    if policy.get('acceptance_mode') == 'pipeline_first':
        return True
    return all(effect_checks(policy, *values).values())
