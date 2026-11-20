"""Optional, independent document-quality rewards."""

from .baselines import box_quality, completion_quality, text_quality, weighted_reward


def table_quality(prediction, reference, **kwargs):
    """Score table content and topology with GriTS."""
    from .tables import table_quality as score
    return score(prediction, reference, **kwargs)


def table_teds(prediction, reference, **kwargs):
    """Score a table with the optional TEDS dependencies."""
    from .tables import table_teds as score
    return score(prediction, reference, **kwargs)


def chart_quality(prediction, reference, **kwargs):
    """Score canonical chart targets using associated facts and metadata."""
    from .charts import chart_quality as score
    return score(prediction, reference, **kwargs)


__all__ = ["text_quality", "box_quality", "table_quality", "table_teds",
           "chart_quality", "completion_quality", "weighted_reward"]
