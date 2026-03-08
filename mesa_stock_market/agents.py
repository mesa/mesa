"""
trader_agent.py — Trader(LLMAgent) with Modular Persona Loading.

The Trader agent is the core decision-maker in the simulation. It inherits
from `mesa_llm.LLMAgent` and uses the Single-Call CoT schema to produce
structured trade decisions in one LLM call.

Architecture:
    Trader(LLMAgent) → LLMAgent(Agent) → mesa.Agent

Persona Configuration:
    Agent personalities are loaded from `config/personas.yaml` at runtime,
    enabling researchers to add, modify, or remove trading personas without
    altering Python source code.

The system prompt embeds:
    1. Loaded Persona — institutional-grade trading style
    2. VST Filter — Value, Safety, Timing compliance check
    3. Kelly Criterion — mathematical position sizing
    4. 1% Risk Rule — capital preservation mandate
    5. Reflexivity awareness — "your trade moves the price"
"""

import logging
from pathlib import Path

import yaml
from mesa.model import Model
from mesa_llm.llm_agent import LLMAgent
from mesa_llm.reasoning.reasoning import Reasoning

from mesa_stock_market.schemas import TradeDecision
from mesa_stock_market.tools import market_tool_manager

logger = logging.getLogger(__name__)

# --- Configuration Path ---
_CONFIG_DIR = Path(__file__).parent / "config"
_PERSONAS_FILE = _CONFIG_DIR / "personas.yaml"


def _load_personas(config_path: Path = _PERSONAS_FILE) -> dict:
    """Load trader personas from a YAML configuration file.

    This enables separation of concerns: researchers can modify agent
    psychology by editing a YAML file instead of Python source code.

    Args:
        config_path: Path to the personas YAML file.

    Returns:
        Dict mapping persona_key → persona_config dict.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        yaml.YAMLError: If the YAML file is malformed.
    """
    if not config_path.exists():
        raise FileNotFoundError(
            f"Persona configuration not found: {config_path}. "
            "Ensure config/personas.yaml exists in the mesa_stock_market package."
        )

    with open(config_path, encoding="utf-8") as f:
        personas = yaml.safe_load(f)

    if not isinstance(personas, dict) or len(personas) == 0:
        raise ValueError(
            f"Invalid persona configuration in {config_path}. "
            "Expected a YAML mapping with at least one persona."
        )

    # Validate required fields for each persona
    required_fields = {"name", "style", "risk_appetite", "base_conviction"}
    for key, config in personas.items():
        missing = required_fields - set(config.keys())
        if missing:
            raise ValueError(
                f"Persona '{key}' in {config_path} is missing fields: {missing}"
            )

    logger.info("Loaded %d personas from %s", len(personas), config_path)
    return personas


# Load personas at module level (cached for the process lifetime)
TRADER_PERSONAS = _load_personas()


def _build_desk_lead_system_prompt(persona_config: dict) -> str:
    """Construct the Desk Lead system prompt with Quantamental OS logic.

    This system prompt is the 'soul' of the agent. It encodes:
    - The persona's trading style and vocabulary (loaded from YAML)
    - The Quantamental OS decision engine (VST, Kelly, 1% Rule)
    - Output format instructions for the Single-Call CoT schema

    Args:
        persona_config: Dict with 'name', 'style', 'risk_appetite',
            'base_conviction' keys.

    Returns:
        The complete system prompt string.
    """
    return f"""You are **{persona_config['name']}** — a trader on The Desk.

## YOUR IDENTITY
{persona_config['style']}
Risk Appetite: {persona_config['risk_appetite'].upper()}.
Base Conviction: {persona_config['base_conviction']}.

## COMMUNICATION PROTOCOL
- LANGUAGE: Professional institutional English with precise financial
  terminology. Use clinical, high-conviction language.
- TONE: High-vigilance, zero filler. No "I think" or "I hope." Data-first.
- FORMAT: [ACTION] → [CAUSE] → [P&L IMPACT].

## DECISION ENGINE — THE QUANTAMENTAL OS

### Step 1: SENTIMENT QUANTING
Convert the Market Pulse into a sentiment_score between -1.0 and +1.0.
- -1.0 = Extreme panic / institutional dumping
- 0.0 = Neutral / no directional signal
- +1.0 = Extreme greed / euphoric buying
Ground your score in the news, not in hope or fear.

### Step 2: VST FILTER (Value, Safety, Timing)
Before ANY trade, you MUST assess:
- **Value**: Is the current price below/above fair value? What is the
  fundamental case?
- **Safety**: Does this trade respect the 1% Risk Rule? (Max risk =
  1% of total portfolio value per trade.)
- **Timing**: Is there momentum confirmation? Is the "Tape" (volume/order
  flow) supporting the move?
If any VST component FAILS, default to HOLD.

### Step 3: KELLY CRITERION POSITION SIZING
Calculate your optimal quantity using:
  f* = (b × p - q) / b
Where:
  b = reward-to-risk ratio (estimated from sentiment magnitude)
  p = win probability (your conviction level, 0 to 1)
  q = 1 - p
Scale your quantity = floor(f* × max_affordable_shares).
If f* ≤ 0, the trade has NEGATIVE expected value. Default to HOLD.

### Step 4: 1% RISK RULE (NON-NEGOTIABLE)
NEVER risk more than 1% of your Total Portfolio Value on any single trade.
Total Portfolio Value = Cash + (Shares × Current Price).
If your requested quantity × price > 1% of portfolio, REDUCE quantity.

### Step 5: REFLEXIVITY CHECK
Your trade WILL move the price. Large buys push price up; large sells
push price down. Factor this into your expected P&L.

## OUTPUT INSTRUCTIONS
You MUST call the `execute_trade` tool with your final decision.
Your thinking_process must document your VST assessment and Kelly calculation.
Your reasoning must be a concise one-line rationale for your trade.
"""


class Trader(LLMAgent):
    """A market trader agent powered by LLM reasoning.

    Inherits from `LLMAgent` and uses the canonical mesa-llm pattern:
    generate_obs() → build prompt → reasoning.plan() → apply_plan().

    The system prompt encodes the loaded persona and Quantamental OS,
    while the step prompt injects live market data (price, news, portfolio).

    Attributes:
        cash: Current cash balance (₹).
        shares: Current share holdings.
        persona_key: The persona type key from TRADER_PERSONAS.
        persona_name: Display name for the persona.
        _last_decision: Stores the most recent trade decision data for
            the audit logger.
    """

    def __init__(
        self,
        model: Model,
        reasoning: type[Reasoning],
        llm_model: str,
        persona_key: str = "quant_neutral",
        initial_cash: float = 10000.0,
        initial_shares: int = 10,
    ):
        """Initialize a Trader agent.

        Args:
            model: The MarketModel instance.
            reasoning: The reasoning strategy class (e.g., SingleCallReasoning).
            llm_model: LLM model string (e.g., "groq/llama-3.3-70b-versatile").
            persona_key: Key into TRADER_PERSONAS for personality config.
            initial_cash: Starting cash balance.
            initial_shares: Starting share count.
        """
        persona_config = TRADER_PERSONAS.get(
            persona_key, TRADER_PERSONAS["quant_neutral"]
        )
        system_prompt = _build_desk_lead_system_prompt(persona_config)

        # Initial internal state visible to other agents
        initial_state = [
            f"Persona: {persona_config['name']}",
            f"Risk: {persona_config['risk_appetite']}",
            f"Cash: ₹{initial_cash:.2f}",
            f"Shares: {initial_shares}",
        ]

        super().__init__(
            model=model,
            reasoning=reasoning,
            llm_model=llm_model,
            system_prompt=system_prompt,
            vision=-1,  # Observe all agents (non-spatial market)
            internal_state=initial_state,
        )

        # Override tool manager with our market-specific tools
        self.tool_manager = market_tool_manager

        # Portfolio state
        self.cash = initial_cash
        self.shares = initial_shares
        self.persona_key = persona_key
        self.persona_name = persona_config["name"]

        # Audit trail — populated by execute_trade tool
        self._last_decision = {}

    def _build_step_prompt(self) -> str:
        """Build the dynamic step prompt with live market data.

        Injects current price, Market Pulse, portfolio state, and any
        recent trade errors from internal_state (State-Sync Protocol).

        Returns:
            The formatted step prompt string.
        """
        price = self.model.current_price
        market_pulse = self.model.market_pulse
        total_value = self.cash + self.shares * price

        # Extract recent errors from internal_state (State-Sync)
        recent_errors = [
            s for s in self.internal_state if s.startswith("[TRADE_REJECTED]")
        ]
        error_context = ""
        if recent_errors:
            error_context = (
                "\n⚠️ RECENT TRADE REJECTIONS (factor these into your decision):\n"
                + "\n".join(f"  • {e}" for e in recent_errors[-3:])  # Last 3
            )

        return f"""MARKET DATA — Step {self.model.steps}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Current Price: ₹{price:.2f}
Market Pulse: {market_pulse}

YOUR PORTFOLIO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Cash: ₹{self.cash:.2f}
Shares: {self.shares}
Total Value: ₹{total_value:.2f}
Max 1% Risk: ₹{total_value * 0.01:.2f}
{error_context}

DIRECTIVE: Analyze the Market Pulse. Run your Quantamental OS.
Execute your trade via the execute_trade tool."""

    def step(self):
        """Execute one decision cycle: Observe → Reason → Trade.

        Follows the canonical mesa-llm pattern from the negotiation example,
        with market-specific prompt construction.
        """
        # 1. Build observation (captures self_state + other agents)
        obs = self.generate_obs()

        # 2. Build market-context step prompt
        prompt = self._build_step_prompt()

        # 3. LLM reasons and produces a plan (tool call)
        plan = self.reasoning.plan(
            prompt=prompt,
            obs=obs,
            selected_tools=["execute_trade"],
        )

        # 4. Execute the tool call (execute_trade)
        self.apply_plan(plan)

        # 5. Update internal_state with current portfolio
        #    (so other agents see our latest holdings)
        self._sync_portfolio_to_state()

    def _sync_portfolio_to_state(self):
        """Update internal_state with current portfolio for visibility.

        Other agents can see our holdings via their observation's
        local_state, enabling reflexivity-aware decisions.
        """
        price = self.model.current_price
        total_value = self.cash + self.shares * price
        # Replace portfolio-related entries in internal_state
        self.internal_state = [
            s
            for s in self.internal_state
            if not s.startswith(("Cash:", "Shares:", "Value:"))
        ]
        self.internal_state.extend(
            [
                f"Cash: ₹{self.cash:.2f}",
                f"Shares: {self.shares}",
                f"Value: ₹{total_value:.2f}",
            ]
        )

    @property
    def portfolio_value(self) -> float:
        """Total portfolio value: cash + shares × current price."""
        return self.cash + self.shares * self.model.current_price
