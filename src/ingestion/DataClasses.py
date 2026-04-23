from dataclasses import dataclass, field


@dataclass
class UrlCnapsWeb:
    url: str
    attrClasses: list[str] = field(default_factory=list)
