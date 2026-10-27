"""Errors that carry a message meant for the person running the pipeline."""


class MissingDataError(RuntimeError):
    """An input file is absent or shorter than an experiment needs."""


class IntegrityError(RuntimeError):
    """A frozen record no longer matches what its manifest says it contains."""
