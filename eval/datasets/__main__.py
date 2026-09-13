"""`python -m eval.datasets <dataset> <command>`.

A dispatcher rather than `python -m eval.datasets.ragtruth`: running a module
that the package `__init__` has already imported makes runpy warn that it may
behave unpredictably, and the re-exports in `__init__` are the convention here.
"""
import sys

from . import indomain, ragtruth

DATASETS = {'ragtruth': ragtruth.main, 'indomain': indomain.main}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in DATASETS:
        sys.exit(f"usage: python -m eval.datasets {{{','.join(DATASETS)}}} <command> ...")
    DATASETS[argv[0]](argv[1:])


if __name__ == '__main__':
    main()
