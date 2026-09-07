#!/usr/bin/env python3
"""CLI wrapper: python generate.py --prompt '...' --first-image a.png --last-image b.png"""

from minimax_h3_fl2v.offline import enforce_offline_runtime

enforce_offline_runtime()

from minimax_h3_fl2v.cli import main

if __name__ == "__main__":
    main()
