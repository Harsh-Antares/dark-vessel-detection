"""Dark-Vessel Detection on SAR.

End-to-end pipeline: Sentinel-1 SAR scenes -> tiled chips -> center-heatmap
vessel detector with attribute heads -> AIS correlation (dark vs cooperative)
-> illegal-fishing pressure map inside marine protected areas.
"""

__version__ = "0.1.0"
