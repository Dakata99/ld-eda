
@dataclass
class Learner:
    name: str
    cls: type
    params: dict[str, Any]
    _model: ClassifierMixin = field(default=None, init=False, repr=False)

    def build(self) -> ClassifierMixin:
        self._model = self.cls(**self.params)
        return self._model
    
    @property
    def id(self) -> str:
        return f"{self.name}({', '.join(f'{k}={v}' for k, v in self.params.items())})"
    
    def __repr__(self) -> str:
        return f"Learner {self.id}"
