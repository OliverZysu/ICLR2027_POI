"""Coupled-gate STGCN variant from the STGN AAAI 2019 paper.

This class intentionally does not implement a graph convolution.  In the
GETNext comparison table, STGCN denotes the parameter-reduced coupled-gate
variant introduced alongside STGN.
"""

from __future__ import annotations

from baselines.stgn.model import STGN


class STGCN(STGN):
    def __init__(self, *args, **kwargs):
        kwargs["coupled"] = True
        super().__init__(*args, **kwargs)
