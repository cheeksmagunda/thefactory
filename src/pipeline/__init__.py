"""Daily pipeline: fetch, score, select, cache."""


class PipelineError(Exception):
    """Any failure in the pipeline. No picks served on failure."""


class DataSourceError(PipelineError):
    """A live data fetch failed. The run aborts."""
