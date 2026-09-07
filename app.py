#!/usr/bin/env python3
"""Launch the MiniMax-H3 FL2VA Gradio studio (offline, unfiltered)."""

from minimax_h3_fl2v.offline import enforce_offline_runtime

enforce_offline_runtime()

from minimax_h3_fl2v.ui import main

if __name__ == "__main__":
    main()
