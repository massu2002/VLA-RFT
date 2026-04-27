# Backward-compatibility shim — real implementation lives in worldmodel/train.py
#
# This script still works as before:
#   python -m worldmodel.libero.train --task-suite spatial ...
#
# The new canonical entrypoint is:
#   python -m worldmodel.train --task-suite spatial ...
from ..train import main  # noqa: F401

if __name__ == "__main__":
    main()
