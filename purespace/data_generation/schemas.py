from enum import Enum
from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
)
from typing import Any, Dict, List, Self


class BatchConfig(BaseModel):
    save_dir: str
    enable_question_demo: bool

    num_levels: int = Field(..., ge=2, le=5)
    num_corners: List[int]
    num_samples: int = Field(..., ge=1, le=999999)

    # --- Default configuration ---
    render_size: int = 1024
    bound: int = 20
    minimum: int = 4
    ratio: List[float] = [0.7, 0.1, 0.2]
    num_options: int = 4
    max_retry_level: int = 50
    max_retry_case: int = 50
    max_retry_sample: int = 50

    @field_validator("num_corners")
    @classmethod
    def validate_num_corners(cls, value: List[int]) -> List[int]:
        if any(num > 5 or num < 1 for num in value):
            raise ValueError(
                f"'num_corners' must be between 1 and 5, got {value}"
            )
        return value

    @model_validator(mode="after")
    def validate_corners_match_levels(self) -> Self:
        if len(self.num_corners) != self.num_levels:
            raise ValueError(
                f"length of 'num_corners' ({len(self.num_corners)}) "
                f"must match 'num_levels' ({self.num_levels})"
            )
        return self


class View(Enum):
    ISO = "iso"
    TOP = "top"
    ISO_LEFT = "iso-left"
    ISO_RIGHT = "iso-right"
    ISO_REVERSE = "iso-reverse"

    @property
    def save_name(self) -> str:
        mapping = {
            self.ISO: "iso",
            self.TOP: "top",
            self.ISO_LEFT: "rotl",
            self.ISO_RIGHT: "rotr",
            self.ISO_REVERSE: "cpl",
        }
        return mapping[self]

    @property
    def diff_threshold(self) -> int:
        mapping = {
            self.ISO: 2000,
            self.TOP: 500,
            self.ISO_LEFT: 2000,
            self.ISO_RIGHT: 2000,
            self.ISO_REVERSE: 2000,
        }
        return mapping[self]


class Case(BaseModel):
    heights: List[int]
    corners: List[List[List[int]]]
    renders: Dict[View, Any] = Field(
        default_factory=lambda: {view: None for view in View}
    )


class Task(Enum):
    ROTATION = "rotation"
    PROJECTION = "projection"
    COMPLETION = "completion"

    @property
    def text(self) -> str:
        mapping = {
            self.ROTATION: "Which option is a rotation of the given object?",
            self.PROJECTION: "Which option is a top-down view of the given object?",
            self.COMPLETION: "Which option fits the given object, in order to make a cube?",
        }
        return mapping[self]


class Question(BaseModel):
    task: Task
    stem: Any
    options: List[Any]
    answer: int

    @field_validator("stem")
    @classmethod
    def validate_stem(cls, value: Any) -> Any:
        if value is None:
            raise ValueError("'stem' cannot be None")
        return value

    @field_validator("options")
    @classmethod
    def validate_options(cls, value: List[Any]) -> List[Any]:
        for opt in value:
            if opt is None:
                raise ValueError(
                    "'options' cannot contain None elements"
                )
        return value
