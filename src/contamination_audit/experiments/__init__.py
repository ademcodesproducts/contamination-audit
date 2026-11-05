"""The experiments behind every number in the paper.

Each experiment writes results/paper/<name>.json. They are listed in the order
the paper presents them, which also satisfies the two that read an earlier
result: `agreement` and `strata` both read `flag_counts`.
"""

from .agreement import FilterAgreement
from .base import Experiment, Study
from .collateral import CollateralLoss
from .combined import CombinedFilter
from .composition import Composition
from .embedding import EmbeddingFilter
from .flag_counts import FlagCounts
from .matched import MatchedCompetitions
from .mechanism import Mechanism
from .phrases import PhraseAgreementSweep
from .planted import PlantedRecall
from .provenance import ControlProvenance
from .rare_ngrams import RareNgramEvaluation
from .replication import NemotronReplication, Replication, TuluReplication
from .scaling import Scaling
from .strata import Strata
from .tuning import Tuning
from .validation import ControlValidation

EXPERIMENTS: dict[str, type[Experiment]] = {
    experiment.name: experiment
    for experiment in (
        ControlValidation,
        ControlProvenance,
        FlagCounts,
        FilterAgreement,
        CollateralLoss,
        Scaling,
        TuluReplication,
        NemotronReplication,
        MatchedCompetitions,
        Mechanism,
        Tuning,
        PhraseAgreementSweep,
        PlantedRecall,
        RareNgramEvaluation,
        EmbeddingFilter,
        CombinedFilter,
        Strata,
        Composition,
    )
}

__all__ = ["EXPERIMENTS", "Experiment", "Replication", "Study"]
