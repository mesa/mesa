"""schemas.py — Single-Call CoT Pydantic Contract for Trade Decisions.

Merges 'thinking' and 'acting' into one schema, passed as `response_format`
to litellm. The LLM produces its Chain-of-Thought reasoning AND the trade
action in a single API call, cutting costs by 50%.

GSoC Innovation: First Pydantic-enforced LLM decision schema in Mesa.
"""

from enum import Enum

from pydantic import BaseModel, Field, field_validator


class TradeAction(str, Enum):
    """Valid trade actions for the market simulation."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class TradeDecision(BaseModel):
    """Single-Call CoT schema — thinking + acting in one Pydantic contract.

    The LLM must populate ALL fields in a single response. The
    `thinking_process` captures the Chain-of-Thought reasoning (VST filter,
    conviction analysis, Kelly sizing), making every decision auditable for
    GSoC mentors.

    Attributes:
        thinking_process: Full CoT reasoning trace. Must include VST filter
            assessment (Value, Safety, Timing) and conviction level.
        sentiment_score: Quantified market sentiment derived from the Market
            Pulse. Range: -1.0 (extreme fear) to +1.0 (extreme greed).
        action: The trade action — BUY, SELL, or HOLD.
        quantity: Number of shares to trade. Kelly-Criterion scaled.
            Must be >= 0.
        reasoning: Standard financial summary of the decision. This is the
            'headline' that appears in the audit log.
    """

    thinking_process: str = Field(
        description=(
            "Your full Chain-of-Thought reasoning. You MUST include: "
            "1) VST Filter assessment (Value: Is price below fair value? "
            "Safety: Does this respect the 1% Risk Rule? Timing: Is momentum "
            "confirmed?). 2) Your conviction level (low/medium/high). "
            "3) Kelly Criterion calculation for position sizing."
        ),
    )

    sentiment_score: float = Field(
        ge=-1.0,
        le=1.0,
        description=(
            "Your quantified sentiment from the Market Pulse. "
            "-1.0 = extreme fear/panic, 0.0 = neutral, +1.0 = extreme greed. "
            "Derive this from the news, not from hope."
        ),
    )

    action: TradeAction = Field(
        description="Your trade decision: BUY, SELL, or HOLD.",
    )

    quantity: int = Field(
        ge=0,
        description=(
            "Number of shares to trade. Must be >= 0. "
            "Scale using Kelly Criterion: f* = (bp - q) / b. "
            "0 for HOLD actions."
        ),
    )

    reasoning: str = Field(
        description=(
            "Standard financial summary of the decision. "
            "Use Desk Lead vocabulary: Alpha, Fat Tails, Order Flow, "
            "Relative Strength, V-Bottom. No filler, no hope."
        ),
    )

    @field_validator("quantity")
    @classmethod
    def hold_must_have_zero_quantity(cls, v, info):
        """Enforce that HOLD actions have quantity = 0."""
        if info.data.get("action") == TradeAction.HOLD and v != 0:
            raise ValueError("HOLD action must have quantity = 0.")
        return v

    @field_validator("quantity")
    @classmethod
    def trade_must_have_positive_quantity(cls, v, info):
        """Enforce that BUY/SELL actions have quantity > 0."""
        action = info.data.get("action")
        if action in (TradeAction.BUY, TradeAction.SELL) and v <= 0:
            raise ValueError(f"{action} action must have quantity > 0.")
        return v


class TradeError(BaseModel):
    """Structured error returned by the execute_trade tool on rejection.

    This gets appended to the agent's `internal_state` so the next
    reasoning cycle reflects actual portfolio reality (State-Sync Protocol).

    Attributes:
        status: Always "REJECTED".
        reason: Machine-readable error code (e.g., "1%_RISK_BREACH",
            "INSUFFICIENT_SHARES", "INSUFFICIENT_FUNDS").
        max_allowed_quantity: The maximum quantity the agent could have traded
            without violating constraints.
        portfolio_snapshot: Current portfolio state at time of rejection.
    """

    status: str = Field(default="REJECTED", description="Always REJECTED.")
    reason: str = Field(
        description=(
            "Machine-readable error code: '1%_RISK_BREACH', "
            "'INSUFFICIENT_SHARES', 'INSUFFICIENT_FUNDS'."
        ),
    )
    max_allowed_quantity: int = Field(
        ge=0,
        description="Max quantity the agent could trade without violation.",
    )
    portfolio_snapshot: dict = Field(
        description=(
            "Current portfolio state: {cash, shares, total_value, current_price}."
        ),
    )
