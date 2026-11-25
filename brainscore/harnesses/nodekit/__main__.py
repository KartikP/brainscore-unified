"""Replay recorded NodeKit runs in the task repository's viewer:

    python -m brainscore.harnesses.nodekit replay <record folder> [--port 5310]

A record folder holds ``<folder>/submissions/<run>.json`` traces (``NodeKitEnvironment(record=...)``); the
participants' runs are in the clone's ``prolific/``.
"""
import argparse

from brainscore.harnesses.nodekit import replay

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['replay'])
    parser.add_argument('record')
    parser.add_argument('--port', type=int, default=5310)
    arguments = parser.parse_args()
    replay(arguments.record, arguments.port)
