from __future__ import annotations

import sys
from pathlib import Path

import mne


def main() -> int:
    if len(sys.argv) != 4:
        print("Usage: V3_autoreject_worker.py input-epo.fif output-epo.fif seed", file=sys.stderr)
        return 2
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    seed = int(sys.argv[3])
    try:
        from autoreject import AutoReject
        epochs = mne.read_epochs(input_path, preload=True, verbose="ERROR")
        ar = AutoReject(n_jobs=1, random_state=seed, verbose=False)
        clean = ar.fit_transform(epochs)
        clean.save(output_path, overwrite=True, verbose="ERROR")
        return 0
    except Exception as error:
        print(repr(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
