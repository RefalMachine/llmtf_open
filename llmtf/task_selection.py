"""Shared task selection for normal evaluation and PPL."""

def resolve_task_names(names, registry):
    if names == 'all':
        return [name for name, spec in registry.items() if spec.get('include_in_all', True)]
    return [names] if isinstance(names, str) else list(names)
