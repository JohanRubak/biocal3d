"""Optional adapter for a fitted scanner colour harmoniser."""


def load_transform(artifact):
    """Load a fitted artefact once and return transform(rgb, scanner).

    rgb is float32 with shape (vertices, 3) and values in [0, 1].
    scanner is LABscanner, TRIOS3, TRIOS5 or iTERO.
    Return corrected vertex RGB with the same shape, order and range.
    Fit the harmoniser separately using only the train specimens in split.csv.
    """
    raise NotImplementedError(
        "Connect the master's trained harmoniser in colour_harmonization.py before enabling this arm.")
