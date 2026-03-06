from dataclasses import dataclass, field


@dataclass
class NEREntities:
    personalia: list[dict] = field(default_factory=list)
    locations: list[dict] = field(default_factory=list)
    organizations: list[dict] = field(default_factory=list)
