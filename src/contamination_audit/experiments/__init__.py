"""The experiments behind every number in the paper.

Each experiment writes results/paper/<name>.json. They are listed in the order
the paper presents them, which also satisfies the two that read an earlier
result: `agreement` and `strata` both read `flag_counts`.
"""

from .agreement import FilterAgreement
from .base import Experiment, Study
from .composition import Composition
from .flag_counts import FlagCounts
from .matched import MatchedCompetitions
from .mechanism import Mechanism
from .phrases import PhraseAgreementSweep
from .replication import NemotronReplication, Replication, TuluReplication
from .scaling import Scaling
from .strata import ControlStrata
from .tuning import Tuning
from .validation import ControlValidation

EXPERIMENTS: dict[str, type[Experiment]] = {
    experiment.name: experiment
    for experiment in (
        ControlValidation,
        FlagCounts,
        FilterAgreement,
        Scaling,
        TuluReplication,
        NemotronReplication,
        MatchedCompetitions,
        Mechanism,
        Tuning,
        PhraseAgreementSweep,
        ControlStrata,
        Composition,
    )
}

__all__ = ["EXPERIMENTS", "Experiment", "Replication", "Study"]
