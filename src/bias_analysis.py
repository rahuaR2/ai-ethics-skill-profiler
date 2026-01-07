def compute_bias(group_results):
    values = list(group_results.values())
    return max(values) - min(values)
