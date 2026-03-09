"""tools.py — execute_trade tool with State-Sync Error Protocol.

The trade execution tool enforces the 1% Risk Rule server-side and returns
structured `TradeError` objects on rejection, which get appended to the
agent's `internal_state` to prevent Memory-State Desynchronization.

Design Principles:
- Dual enforcement: LLM prompt says "don't exceed 1% risk" AND this tool
  rejects trades that violate it. Belt AND suspenders.
- State-Sync: Rejected trades return a structured error dict, not a string.
  The agent's next thought cycle sees the error in its internal_state.
- Audit Trail: Every execution returns structured data for the step logger.
"""

from typing import TYPE_CHECKING

from mesa_llm.tools.tool_decorator import tool
from mesa_llm.tools.tool_manager import ToolManager

from mesa_stock_market.schemas import TradeError

if TYPE_CHECKING:
    from mesa_llm.llm_agent import LLMAgent

# Dedicated tool manager for market agents
market_tool_manager = ToolManager()


def _get_portfolio_snapshot(agent) -> dict:
    """Build a portfolio snapshot dict for error reporting and auditing.

    Args:
        agent: The trader agent.

    Returns:
        dict with cash, shares, total_value, current_price.
    """
    current_price = agent.model.current_price
    return {
        "cash": round(agent.cash, 2),
        "shares": agent.shares,
        "total_value": round(agent.cash + agent.shares * current_price, 2),
        "current_price": round(current_price, 2),
    }


def _calculate_max_risk_quantity(agent, price: float) -> int:
    """Calculate the maximum quantity allowed under the 1% Risk Rule.

    The 1% Risk Rule: No single trade can risk more than 1% of the
    total portfolio value (Cash + Market Value of Shares).

    Args:
        agent: The trader agent.
        price: Current price per share.

    Returns:
        Maximum number of shares that can be traded within the 1% rule.
    """
    total_value = agent.cash + agent.shares * price
    max_risk_amount = total_value * 0.01  # 1% of portfolio
    if price <= 0:
        return 0
    return max(0, int(max_risk_amount / price))


def _sync_error_to_agent(agent, error: TradeError) -> None:
    """Append a structured error to the agent's internal_state.

    This is the State-Sync Protocol: rejected trade errors are immediately
    visible in the agent's next observation, ensuring the LLM's next
    'thought' reflects actual portfolio reality.

    Args:
        agent: The trader agent.
        error: The TradeError to sync.
    """
    error_summary = (
        f"[TRADE_REJECTED] {error.reason} | "
        f"Max allowed: {error.max_allowed_quantity} shares | "
        f"Portfolio: {error.portfolio_snapshot['total_value']:.2f}"
    )
    agent.internal_state.append(error_summary)


@tool(tool_manager=market_tool_manager)
def execute_trade(
    agent: "LLMAgent",
    action: str,
    quantity: int,
    reasoning: str,
) -> str:
    """Execute a trade decision on the market with 1% Risk Rule enforcement.

    Args:
        agent: The trader agent executing the trade.
        action: BUY, SELL, or HOLD.
        quantity: Number of shares to trade.
        reasoning: The clinical reasoning for this trade.

    Returns:
        A confirmation string with P&L impact or a structured error summary.
    """
    model = agent.model
    price = model.current_price
    action = action.upper().strip()

    # --- Store reasoning data for the audit logger ---
    agent._last_decision = {
        "action": action,
        "quantity": quantity,
        "reasoning": reasoning,
        "sentiment_score": getattr(agent, "_last_sentiment", 0.0),
        "thinking_process": getattr(agent, "_last_thinking", ""),
    }

    # --- HOLD: No position change ---
    if action == "HOLD":
        snapshot = _get_portfolio_snapshot(agent)
        return (
            f"HOLD executed. No position change. "
            f"Price: {price:.2f} | Portfolio: {snapshot['total_value']:.2f}"
        )

    # --- 1% Risk Rule Check (applies to both BUY and SELL) ---
    max_risk_qty = _calculate_max_risk_quantity(agent, price)

    if quantity > max_risk_qty:
        error = TradeError(
            status="REJECTED",
            reason="1%_RISK_BREACH",
            max_allowed_quantity=max_risk_qty,
            portfolio_snapshot=_get_portfolio_snapshot(agent),
        )
        _sync_error_to_agent(agent, error)
        return (
            f"REJECTED: 1% Risk Rule breach. "
            f"Requested {quantity}, max allowed {max_risk_qty}. "
            f"Portfolio: {error.portfolio_snapshot['total_value']:.2f}"
        )

    # --- BUY Execution ---
    if action == "BUY":
        cost = price * quantity
        if cost > agent.cash:
            max_affordable = int(agent.cash // price) if price > 0 else 0
            effective_max = min(max_affordable, max_risk_qty)
            error = TradeError(
                status="REJECTED",
                reason="INSUFFICIENT_FUNDS",
                max_allowed_quantity=effective_max,
                portfolio_snapshot=_get_portfolio_snapshot(agent),
            )
            _sync_error_to_agent(agent, error)
            return (
                f"REJECTED: Insufficient funds. "
                f"Need {cost:.2f}, have {agent.cash:.2f}. "
                f"Max affordable: {effective_max} shares."
            )

        # Execute the buy
        agent.cash -= cost
        agent.shares += quantity
        # Track order flow for reflexivity engine
        model.step_buys += quantity
        snapshot = _get_portfolio_snapshot(agent)
        return (
            f"BOUGHT {quantity} @ {price:.2f} | "
            f"Cost: {cost:.2f} | Cash: {agent.cash:.2f} | "
            f"Holdings: {agent.shares} shares | "
            f"Portfolio: {snapshot['total_value']:.2f}"
        )

    # --- SELL Execution ---
    if action == "SELL":
        if quantity > agent.shares:
            error = TradeError(
                status="REJECTED",
                reason="INSUFFICIENT_SHARES",
                max_allowed_quantity=min(agent.shares, max_risk_qty),
                portfolio_snapshot=_get_portfolio_snapshot(agent),
            )
            _sync_error_to_agent(agent, error)
            return (
                f"REJECTED: Insufficient shares. "
                f"Want to sell {quantity}, hold {agent.shares}."
            )

        # Execute the sell
        revenue = price * quantity
        agent.cash += revenue
        agent.shares -= quantity
        # Track order flow for reflexivity engine
        model.step_sells += quantity
        snapshot = _get_portfolio_snapshot(agent)
        return (
            f"SOLD {quantity} @ {price:.2f} | "
            f"Revenue: {revenue:.2f} | Cash: {agent.cash:.2f} | "
            f"Holdings: {agent.shares} shares | "
            f"Portfolio: {snapshot['total_value']:.2f}"
        )

    # --- Unknown action fallback ---
    return f"REJECTED: Unknown action '{action}'. Use BUY, SELL, or HOLD."
