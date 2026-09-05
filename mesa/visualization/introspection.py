"""Machine-readable introspection of a Mesa model's structure.

Defines a standardized, JSON-serializable description of a running Mesa
model — its agent types, space, configurable parameters, and data-collection
reporters — so external tools (including LLMs writing custom frontends or
scaffolding new models) work from a stable contract instead of guessing at
a model's shape from source code.

See discussion #3832 for full context and open design questions.
"""

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from mesa.model import Model


class ModelSchema(BaseModel):
    """A JSON-serializable description of a Mesa model's structure.

    Attributes:
        agent_types: One entry per distinct agent class in the model,
            describing its name and public attributes.
        space: A description of the model's space, or None if the model
            has no space.
        params: The model's configurable parameters, as already defined
            in model_params (Slider, Choice, etc.).
        reporters: Names of the model's and agents' data-collector
            reporters, if a DataRegistry is present.
    """

    agent_types: list[dict[str, Any]] = Field(default_factory=list)
    space: dict[str, Any] | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    reporters: dict[str, Any] = Field(default_factory=dict)


def describe_model(model: "Model") -> ModelSchema:
    """Produce a JSON-serializable schema describing a Mesa model.

    Currently a skeleton: returns the contract's shape with nothing
    populated. Each field is filled in by a following commit, so the
    shape itself can be reviewed before logic is built on top of it.

    Args:
        model: A Mesa Model instance to describe.

    Returns:
        A ModelSchema with all fields at their empty defaults.
    """
    return ModelSchema()
