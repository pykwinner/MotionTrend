"""The three innovation-module switches for MoTrend.

EBTE, CTR and SRR are enabled for the released checkpoint. A different switch
combination defines a new architecture and requires a matching trained state.
The inherited PST temporal weighting is fixed as part of the backbone.
"""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModuleSwitches:
    ebte: bool = True
    ctr: bool = True
    srr: bool = True

    def __post_init__(self):
        for name in ("ebte", "ctr", "srr"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(name + " must be a real bool")

    @property
    def is_selected_architecture(self):
        return self.ebte and self.ctr and self.srr

    def as_dict(self):
        return asdict(self)


SELECTED_SWITCHES = ModuleSwitches()
