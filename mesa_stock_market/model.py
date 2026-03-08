"""
model.py — MarketModel with Market Pulse & Reflexivity Price Discovery Engine.

The MarketModel orchestrates the simulation:
1. Pre-step: Generates a 'Market Pulse' from the news timeline
2. Agent activation: Sequential with rate-limit delay — Mesa 4.0 pattern
3. Post-step: Reflexivity engine updates price based on net order flow
4. DataCollector: Tracks price, volume, and wealth inequality (Gini)
5. In-memory history_log: Captures reasoning audit per step (saved once at end)

Architecture:
    MarketModel(Model) → mesa.Model

Data Strategy:
    - DataCollector handles structured metrics (Price, Gini, Volume)
    - history_log captures deep reasoning JSON in memory
    - Disk I/O happens ONCE at simulation end (from run.py), not per-step
"""

import time

from mesa.datacollection import DataCollector
from mesa.model import Model
from mesa_llm.reasoning.reasoning import Reasoning

from mesa_stock_market.agents import TRADER_PERSONAS, Trader

# --- Pre-Scripted News Timeline ---
# Steps 1–14: Neutral/mildly positive
# Step 15: THE SHOCK — "CEO arrested for fraud"
# Steps 16–25: Recovery phase with contrarian signals
NEWS_TIMELINE = {
    0: "Market opens. Trading steady. Volumes are average. No major headlines.",
    1: "Quarterly earnings for MegaCorp are in line with street estimates. No surprises.",
    2: "RBI holds rates steady. Minor uptick in consumer spending data.",
    3: "Tech sector showing relative strength. IT index up 0.3% pre-market.",
    4: "FII inflows continue. Market breadth is positive. Mid-caps outperforming.",
    5: "MegaCorp announces a new factory expansion. Stock ticks up 1.2% intraday.",
    6: "Global markets mixed. US futures flat. Crude oil steady at $78/barrel.",
    7: "Analyst upgrades MegaCorp to 'Outperform'. Target price raised 15%.",
    8: "Mutual fund SIP inflows hit all-time high. Retail participation surging.",
    9: "MegaCorp CEO gives bullish guidance. 'Best quarter ahead.' Stock rallies 2%.",
    10: "Market at 52-week high. VIX at historic low. 'Everything rally' underway.",
    11: "Social media buzzing: 'MegaCorp to 500!' Retail FOMO intensifying.",
    12: "Smart money alert: Large block trades detected. Institutional volume spiking.",
    13: "Whisper numbers suggest MegaCorp might miss next quarter. Unconfirmed.",
    14: "Market consolidating near highs. Volume declining. Distribution pattern forming.",
    # === THE SHOCK ===
    15: (
        "BREAKING: MegaCorp CEO arrested for multi-crore accounting fraud! "
        "Board suspends operations pending investigation. Stock halted. "
        "Institutional selling detected across sector. Credit agencies reviewing. "
        "This is a FAT TAIL event."
    ),
    16: (
        "MegaCorp stock resumes trading. Circuit breaker hit — 20% lower. "
        "Panic selling across the board. Mutual funds freezing redemptions."
    ),
    17: (
        "Contagion spreading. Banking sector under pressure on MegaCorp exposure. "
        "FIIs pulling out. Market sentiment: EXTREME FEAR."
    ),
    18: (
        "Forensic auditors appointed. Reports suggest fraud limited to one division. "
        "Contrarian analysts say 'market overreacting.' Some bargain hunting seen."
    ),
    19: (
        "New interim CEO appointed. Credible name from the industry. "
        "Company releases preliminary clean audit for core business. "
        "V-Bottom formation? Smart money sniffing around."
    ),
    20: (
        "Market stabilizing. MegaCorp recovers 5% from lows. "
        "Volume surging on recovery. Relative Strength improving."
    ),
    21: (
        "Analyst consensus: 'Worst is behind us.' Target prices being revised. "
        "FII selling slowing. DII accumulation detected."
    ),
    22: (
        "MegaCorp files for expedited investigation closure. "
        "Stock up 3% on positive legal development. Momentum building."
    ),
    23: (
        "Market rallying. MegaCorp leading recovery. "
        "'Oversold bounce' or 'New bull trend'? Traders divided."
    ),
    24: (
        "Strong intraday volumes. MegaCorp now 50% off panic lows. "
        "Those who bought the dip sitting on significant gains."
    ),
}


# --- Gini Coefficient (Model-level reporter function) ---

def get_gini_coefficient(model: Model) -> float:
    """Calculate the Gini coefficient for wealth inequality.

    Uses the standard formula:
        G = Σ_i Σ_j |x_i - x_j| / (2 * n² * x̄)

    A Gini of 0 = perfect equality; 1 = perfect inequality.
    Used to measure if the Quantamental OS leads to capital concentration.

    Args:
        model: The MarketModel instance.

    Returns:
        Gini coefficient (0.0 to 1.0).
    """
    agent_wealths = [
        a.portfolio_value for a in model.agents if hasattr(a, "cash")
    ]
    if not agent_wealths or len(agent_wealths) < 2:
        return 0.0

    x = sorted(agent_wealths)
    n = len(x)
    total = sum(x)
    if total == 0:
        return 0.0

    coef = sum(abs(xi - xj) for xi in x for xj in x) / (2 * n * total)
    return round(coef, 4)


class MarketModel(Model):
    """AI-Driven Stock Market simulation model.

    Manages the simulation lifecycle:
    - Pre-step: Generates Market Pulse from news timeline
    - Agent step: Traders reason and execute trades (with rate-limit delay)
    - Post-step: Reflexivity engine updates price via net order flow
    - Data: DataCollector for metrics + in-memory history_log for reasoning

    Attributes:
        current_price: Current asset price.
        initial_price: Starting price for reference.
        market_pulse: Processed news string for agent consumption.
        current_news: Raw news headline for the current step.
        step_buys: Total shares bought this step (reset each step).
        step_sells: Total shares sold this step (reset each step).
        total_liquidity: Total shares outstanding (for price impact calc).
        impact_multiplier: Controls how strongly order flow moves the price.
        agent_delay: Seconds to wait between agent calls (rate limiting).
        history_log: In-memory list of per-step reasoning audit dicts.
        gini: Latest Gini coefficient (updated each step via DataCollector).
    """

    def __init__(
        self,
        n_traders: int = 5,
        initial_price: float = 100.0,
        initial_cash: float = 10000.0,
        initial_shares: int = 10,
        reasoning: type[Reasoning] = None,
        llm_model: str = "gemini/gemini-2.0-flash",
        impact_multiplier: float = 0.02,
        agent_delay: float = 8.0,
        log_dir: str = "logs",
        seed: int | None = None,
    ):
        """Initialize the market model.

        Args:
            n_traders: Number of trader agents.
            initial_price: Starting price per share.
            initial_cash: Starting cash per agent.
            initial_shares: Starting shares per agent.
            reasoning: Reasoning strategy class (e.g., SingleCallReasoning).
            llm_model: LLM model string for litellm.
            impact_multiplier: Controls price sensitivity to order flow.
            agent_delay: Seconds to pause between agent LLM calls
                (rate limiting for free-tier APIs).
            log_dir: Directory path for final audit export (used by run.py).
            seed: Random seed for reproducibility.
        """
        super().__init__(seed=seed)

        # Market state
        self.current_price = initial_price
        self.initial_price = initial_price
        self.market_pulse = ""
        self.current_news = ""
        self.impact_multiplier = impact_multiplier
        self.agent_delay = agent_delay

        # Order flow tracking (reset each step)
        self.step_buys = 0
        self.step_sells = 0

        # Total liquidity = total shares in the system
        self.total_liquidity = n_traders * initial_shares

        # Price history for momentum analysis
        self.price_history = [initial_price]

        # Log directory path (used by run.py for final export)
        self.log_dir = log_dir

        # In-memory reasoning audit log (replaces per-step JSON disk I/O)
        # Saved once at simulation end from run.py
        self.history_log = []

        # Latest Gini coefficient (updated each step)
        self.gini = 0.0

        # --- Create diverse trader agents ---
        persona_keys = list(TRADER_PERSONAS.keys())
        for i in range(n_traders):
            persona_key = persona_keys[i % len(persona_keys)]
            Trader(
                model=self,
                reasoning=reasoning,
                llm_model=llm_model,
                persona_key=persona_key,
                initial_cash=initial_cash,
                initial_shares=initial_shares,
            )

        # --- Mesa DataCollector (standard pipeline) ---
        self.datacollector = DataCollector(
            model_reporters={
                "Price": "current_price",
                "Gini": get_gini_coefficient,
                "Volume": lambda m: m.step_buys + m.step_sells,
                "NetFlow": lambda m: m.step_buys - m.step_sells,
            },
            agent_reporters={
                "Wealth": lambda a: (
                    round(a.portfolio_value, 2) if hasattr(a, "cash") else None
                ),
                "Cash": lambda a: (
                    round(a.cash, 2) if hasattr(a, "cash") else None
                ),
                "Shares": lambda a: (
                    a.shares if hasattr(a, "shares") else None
                ),
            },
        )

    def get_global_market_context(self) -> str:
        """Generate the Market Pulse from the current news event.

        This is called BEFORE agent activation. All agents receive the same
        Market Pulse, preventing sentiment drift across the population.

        Returns:
            A concise Market Pulse string.
        """
        step = self.steps
        self.current_news = NEWS_TIMELINE.get(
            step,
            f"Step {step}: Market continues. No new major developments.",
        )

        # Build Market Pulse with context
        price_change = 0.0
        if len(self.price_history) >= 2:
            prev = self.price_history[-1]
            price_change = ((self.current_price - prev) / prev) * 100

        trend = "BULLISH" if price_change > 0.5 else (
            "BEARISH" if price_change < -0.5 else "FLAT"
        )

        self.market_pulse = (
            f"[MARKET PULSE — Step {step}] "
            f"Trend: {trend} ({price_change:+.2f}%) | "
            f"Price: {self.current_price:.2f} | "
            f"NEWS: {self.current_news}"
        )
        return self.market_pulse

    def _apply_reflexivity(self):
        """Apply non-linear reflexive price discovery.

        Price adjusts based on net order flow. Large imbalances create
        outsized moves, producing the feedback loop that generates
        Irrational Exuberance or Panic.

        Formula:
            net_flow = total_bought - total_sold
            price_impact = (net_flow / total_liquidity) * impact_multiplier
            new_price = current_price * (1 + price_impact)

        The non-linearity comes from the ratio: when total_liquidity is
        low (illiquid market), the same order flow has a larger impact.
        """
        net_flow = self.step_buys - self.step_sells

        if self.total_liquidity > 0:
            price_impact = (net_flow / self.total_liquidity) * self.impact_multiplier
        else:
            price_impact = 0.0

        # Apply with floor (price can't go below 1.0)
        self.current_price = max(1.0, self.current_price * (1 + price_impact))

        # Update liquidity based on total shares held across agents
        self.total_liquidity = sum(
            a.shares for a in self.agents if hasattr(a, "shares")
        )

    def _capture_step_to_history(self):
        """Capture the reasoning audit for this step into memory.

        Instead of writing JSON to disk per step (slow, "dirty"), we append
        a structured dict to self.history_log. The unified audit is saved
        once at simulation end from run.py.
        """
        agents_data = []
        for agent in self.agents:
            if not hasattr(agent, "cash"):
                continue
            decision = getattr(agent, "_last_decision", {})
            agents_data.append(
                {
                    "agent_id": agent.unique_id,
                    "persona": getattr(agent, "persona_name", "unknown"),
                    "thinking_process": decision.get("thinking_process", ""),
                    "sentiment_score": decision.get("sentiment_score", 0.0),
                    "action": decision.get("action", "NONE"),
                    "quantity": decision.get("quantity", 0),
                    "reasoning": decision.get("reasoning", ""),
                    "portfolio_value": round(agent.portfolio_value, 2),
                    "cash": round(agent.cash, 2),
                    "shares": agent.shares,
                }
            )

        self.history_log.append(
            {
                "step": self.steps,
                "current_price": round(self.current_price, 2),
                "market_pulse": self.market_pulse,
                "net_order_flow": self.step_buys - self.step_sells,
                "total_volume": self.step_buys + self.step_sells,
                "agents": agents_data,
            }
        )

    def step(self):
        """Execute one simulation step.

        Sequence:
        1. Pre-process: Generate Market Pulse (all agents see same context)
        2. Reset order flow counters
        3. Activate agents sequentially with rate-limit delay
        4. Post-process: Reflexivity engine adjusts price
        5. Record: DataCollector metrics + in-memory history_log
        """
        # 1. Market Pulse — unified context for all agents
        self.get_global_market_context()

        # 2. Reset per-step order flow
        self.step_buys = 0
        self.step_sells = 0

        # 3. Activate agents sequentially with rate-limit delay
        #    (shuffle order, then call each with delay for API rate limits)
        agent_list = list(self.agents)
        self.random.shuffle(agent_list)
        for i, agent in enumerate(agent_list):
            agent.step()
            # Rate-limit delay between agents (skip after last agent)
            if i < len(agent_list) - 1 and self.agent_delay > 0:
                time.sleep(self.agent_delay)

        # 4. Reflexivity — price discovery from order flow
        self.price_history.append(self.current_price)
        self._apply_reflexivity()

        # 5. Collect data (Mesa standard) + capture reasoning to memory
        self.datacollector.collect(self)
        self.gini = get_gini_coefficient(self)
        self._capture_step_to_history()
