"""Experimental LegalBench-RU; task import stays lazy for offline scoring."""


def __getattr__(name):
    if name == 'LegalBenchRU':
        from .task import LegalBenchRU
        return LegalBenchRU
    raise AttributeError(name)
