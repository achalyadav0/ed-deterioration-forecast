"""Pure-numpy logic shared by the offline pipeline and the Raspberry Pi runtime.

Anything that turns raw vitals into model inputs or labels lives here, so training and
edge inference cannot silently drift apart (train/serve skew). Keep this package free of
polars / torch / pandas imports.
"""
