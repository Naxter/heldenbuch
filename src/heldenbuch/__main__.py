"""Allows `python -m heldenbuch ...` when the console script is not on PATH."""

from .cli import main

raise SystemExit(main())
