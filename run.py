"""
run.py — Bulletproof Shock Event Scenario Runner with Live Pulse.

This is the stress test for the GSoC PoC. It runs a 25-step simulation
with a pre-scripted news timeline:
  Steps 0–14:  Neutral → mildly bullish ramp
  Step 15:     THE SHOCK — "CEO arrested for fraud" (Fat Tail event)
  Steps 16–24: Recovery phase with contrarian signals

Success Criteria:
  1. Price visibly crashes at Step 15
  2. Agents respond with structured reasoning
  3. At least one agent panic sells, at least one buys the dip
  4. All decisions route through the Pydantic-enforced pipeline

Resilience Features:
  - Pre-flight API key validation before engine start
  - litellm retry cap + noise suppression (no red spam)
  - Live spinner "Pulse" during agent reasoning steps
  - 429 "Stress Test Victory" panel with partial data report
  - Unified audit log export in `finally` block

Usage:
  cd mesa-gsoc-contribution
  python -m mesa_stock_market.run
"""

import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# =============================================================================
# PHASE 1: ENVIRONMENT SETUP — Run before ANY library imports
# =============================================================================


def setup_environment(is_offline: bool = False) -> tuple[bool, str]:
    """Validate environment and return the (is_offline, llm_model) tuple.

    Performs three critical checks:
    1. Explicit .env path discovery (relative to this file, not cwd)
    2. Pre-flight API key validation
    3. litellm resilience configuration (retry caps, noise suppression)

    Returns:
        tuple: (True/False if offline, The LLM model string)
    """
    if is_offline:
        return True, "mock/offline-model"

    import litellm

    # 1. Locate .env relative to THIS file (not the user's cwd)
    base_dir = Path(__file__).resolve().parent
    env_path = base_dir / ".env"

    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True)
    else:
        print(f"[!] Warning: No .env file found at {env_path}")

    # 2. Pre-flight API key validation
    gemini_key = os.getenv("GEMINI_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    # Determine which provider to use based on available keys
    llm_model = None
    active_provider = None

    if groq_key:
        llm_model = "groq/llama-3.3-70b-versatile"
        active_provider = "Groq"
    elif gemini_key:
        llm_model = "gemini/gemini-2.0-flash"
        active_provider = "Gemini"
    elif openai_key:
        llm_model = "openai/gpt-4o-mini"
        active_provider = "OpenAI"

    if llm_model is None:
        print()
        print("=" * 58)
        print("  WARNING: NO API KEY FOUND")
        print("=" * 58)
        print("  Automatically falling back to OFFLINE mode.")
        print("  Mock agents will default to HOLD to test the UI logic.")
        print()
        print("  To run with live AI, create a .env file with:")
        print("    GROQ_API_KEY=gsk_your_key_here")
        print("    GEMINI_API_KEY=AIzaSy_your_key_here")
        print("=" * 58)
        print()
        return True, "mock/offline-model"

    # 3. Resilience & noise suppression
    litellm.num_retries = 2
    litellm.suppress_debug_info = True
    logging.getLogger("LiteLLM").setLevel(logging.CRITICAL)
    logging.getLogger("LiteLLM Router").setLevel(logging.CRITICAL)
    logging.getLogger("LiteLLM Proxy").setLevel(logging.CRITICAL)

    print(f"[*] Environment validated. Provider: {active_provider}")
    print(f"[*] LLM Model: {llm_model}")

    return False, llm_model


# Check for explicit offline flag, but also allow setup_environment to force it
_is_offline = "--offline" in sys.argv

# Run setup BEFORE importing model/agent code (which triggers ModuleLLM init)
_is_offline, _llm_model = setup_environment(is_offline=_is_offline)

# -----------------------------------------------------------------------------
# MESA 4.0 COMPATIBILITY PATCH FOR MESA-LLM
# Mesa 4.0 removed `mesa.space`, but `mesa-llm` still tries to import grids.
# Since we don't use spatial grids in this financial PoC, we mock the module.
import sys
import types
if "mesa.space" not in sys.modules:
    dummy_space = types.ModuleType("mesa.space")
    for grid in ["ContinuousSpace", "HexGrid", "MultiGrid", "NetworkGrid", "SingleGrid", "PropertyLayer", "Grid"]:
        setattr(dummy_space, grid, type(grid, (), {}))
    sys.modules["mesa.space"] = dummy_space
# -----------------------------------------------------------------------------

# =============================================================================
# PHASE 2: IMPORTS (safe now that env vars are loaded and noise is silenced)
# =============================================================================

import matplotlib
matplotlib.use("Agg")  # Headless mode for CI/CD compatibility
import matplotlib.pyplot as plt

try:
    plt.style.use("seaborn-v0_8-muted")
except OSError:
    plt.style.use("ggplot")  # Fallback gracefully

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from mesa_stock_market.model import MarketModel
from mesa_stock_market.reasoning import SingleCallReasoning

console = Console()


# =============================================================================
# PHASE 3: SIMULATION RUNNER WITH LIVE PULSE
# =============================================================================


def run_simulation(
    n_traders: int = 5,
    n_steps: int = 25,
    initial_price: float = 100.0,
    initial_cash: float = 10000.0,
    initial_shares: int = 10,
    llm_model: str | None = None,
    impact_multiplier: float = 0.02,
    agent_delay: float = 8.0,
    is_offline: bool = False,
):
    """Run the Shock Event simulation with live terminal pulse.

    Args:
        n_traders: Number of trader agents.
        n_steps: Total simulation steps.
        initial_price: Starting price per share.
        initial_cash: Starting cash per agent.
        initial_shares: Starting shares per agent.
        llm_model: LLM model string for litellm (auto-detected if None).
        impact_multiplier: Price sensitivity to order flow.
        agent_delay: Seconds between agent LLM calls (rate-limiting).
        is_offline: If True, mocks the LLM entirely for local testing.
    """
    # Use auto-detected model if not explicitly provided
    model_str = llm_model or _llm_model

    console.print(
        Panel.fit(
            "[bold green]Mesa 4.0 Stock Market Simulation Engine[/bold green]\n"
            f"Infrastructure Status: "
            f"[{'bold yellow' if is_offline else 'bold cyan'}]"
            f"{'Offline/Mocked' if is_offline else 'Online'}"
            f"[/{'bold yellow' if is_offline else 'bold cyan'}] | "
            f"Model: [bold white]{model_str}[/bold white]\n"
            f"Traders: {n_traders} | Steps: {n_steps} | "
            f"Price: ₹{initial_price:.2f}",
            border_style="cyan",
        )
    )

    if is_offline:
        from unittest.mock import MagicMock, patch
        from mesa_llm.reasoning.reasoning import Plan

        # Set delay to zero for instant offline tests
        agent_delay = 0.0

        # Create a mock plan with a default HOLD action
        def _mock_plan_side_effect(*args, **kwargs):
            action = "HOLD"
            qty = 0
            mock_msg = type("Msg", (), {"content": "Offline mock reasoning. Defaulting to HOLD due to missing API keys or offline mode.", "tool_calls": []})()
            
            # The tool manager won't execute, so we need to inject the decision manually 
            # into the agent state so the reflexivity engine can see it.
            # This is a bit hacky but works for purely mocking the run loop.
            agent = args[0].agent
            agent._last_decision = {
                "action": action,
                "quantity": qty,
                "sentiment_score": 0.0,
                "reasoning": "Offline fallback HOLD"
            }
            # No volume changes for HOLD
                
            return Plan(step=args[0].agent.market.steps, llm_plan=mock_msg)

        patch_agent = patch("mesa_stock_market.agents.LLMAgent.__init__", return_value=None)
        patch_plan = patch("mesa_stock_market.reasoning.SingleCallReasoning.plan", side_effect=_mock_plan_side_effect)
        
        patch_agent.start()
        patch_plan.start()

    # Initialize the model
    model = MarketModel(
        n_traders=n_traders,
        initial_price=initial_price,
        initial_cash=initial_cash,
        initial_shares=initial_shares,
        reasoning=SingleCallReasoning,
        llm_model=model_str,
        impact_multiplier=impact_multiplier,
        agent_delay=agent_delay,
        log_dir="logs",
    )

    console.print(f"\n[dim]Agents initialized with personas:[/dim]")
    for agent in model.agents:
        if hasattr(agent, "persona_name"):
            console.print(
                f"  Agent {agent.unique_id}: "
                f"[bold]{agent.persona_name}[/bold] "
                f"({agent.persona_key})"
            )

    # Track the last completed step for error reporting
    last_completed_step = -1

    # --- Run the simulation with Live Pulse and 429 resilience ---
    try:
        # Live gives us a persistent display area that stays in terminal
        # history even if the loop crashes (prevents "vanishing output")
        with Live(
            Panel("[bold yellow]Initializing price discovery...[/bold yellow]"),
            console=console,
            refresh_per_second=4,
            transient=False,
        ) as live:
            for step_num in range(n_steps):
                # Visual separator (printed above the Live panel)
                if step_num == 15:
                    console.print(
                        "\n[bold red on white]"
                        "═══════════════ ⚠️  SHOCK EVENT ⚠️  ═══════════════"
                        "[/bold red on white]\n"
                    )
                else:
                    console.print(
                        f"\n[bold purple]"
                        f"──── Step {step_num} ────────────────────────────"
                        f"[/bold purple]"
                    )

                # Update the live panel with current step info
                live.update(
                    Panel(
                        f"[bold yellow]Step {step_num + 1}/{n_steps}:[/bold yellow] "
                        f"Agents are reasoning..."
                    )
                )

                model.step()

                last_completed_step = step_num

                # --- Post-step heartbeat (persists in terminal history) ---
                net_flow = model.step_buys - model.step_sells
                flow_indicator = (
                    "🟢" if net_flow > 0 else ("🔴" if net_flow < 0 else "⚪")
                )

                console.log(
                    f"[bold white]Step {step_num + 1}:[/bold white] "
                    f"Price: [green]₹{model.current_price:.2f}[/green] | "
                    f"Volume: {model.step_buys + model.step_sells} | "
                    f"Flow: {flow_indicator} {net_flow:+d} | "
                    f"Gini: {model.gini:.4f}"
                )

                # Agent decisions table
                table = Table(show_header=True, header_style="bold", box=None)
                table.add_column("Agent", style="cyan", width=18)
                table.add_column("Action", width=6)
                table.add_column("Qty", justify="right", width=5)
                table.add_column("Sentiment", justify="right", width=9)
                table.add_column("Portfolio", justify="right", width=12)
                table.add_column("Reasoning", width=50)

                for agent in model.agents:
                    if not hasattr(agent, "_last_decision"):
                        continue
                    d = agent._last_decision
                    if not d:
                        continue

                    action = d.get("action", "?")
                    action_color = {
                        "BUY": "green",
                        "SELL": "red",
                        "HOLD": "yellow",
                    }.get(action, "white")

                    table.add_row(
                        f"{agent.persona_name}",
                        f"[{action_color}]{action}[/{action_color}]",
                        str(d.get("quantity", 0)),
                        f"{d.get('sentiment_score', 0.0):+.2f}",
                        f"₹{agent.portfolio_value:.2f}",
                        (d.get("reasoning", "")[:48] + "…")
                        if len(d.get("reasoning", "")) > 48
                        else d.get("reasoning", ""),
                    )

                console.print(table)

    except Exception as e:
        error_str = str(e)
        if (
            "429" in error_str
            or "RESOURCE_EXHAUSTED" in error_str
            or "quota" in error_str.lower()
            or "rate" in error_str.lower()
        ):
            # STRESS TEST VICTORY: The logic outpaced the API tier
            console.print()
            console.print(
                Panel(
                    "[bold red]STRESS TEST VICTORY: RESOURCE EXHAUSTED (429)[/bold red]\n\n"
                    "The simulation logic outpaced the API tier infrastructure.\n\n"
                    f"  • [bold]Steps Completed:[/bold] {last_completed_step + 1} / {n_steps}\n"
                    f"  • [bold]Final Price:[/bold] ₹{model.current_price:.2f}\n"
                    f"  • [bold]Price Change:[/bold] "
                    f"{((model.current_price - model.initial_price) / model.initial_price) * 100:+.2f}%\n\n"
                    "[italic yellow]Infrastructure cooling down. "
                    "Data collected up to failure point.[/italic yellow]\n\n"
                    "Options:\n"
                    "  1. Wait 60 seconds and re-run\n"
                    "  2. Use a paid-tier API key\n"
                    "  3. Switch to a local model (Ollama)",
                    title="⚠️  Infrastructure Alert",
                    border_style="red",
                )
            )
        else:
            raise

    else:
        # --- Full Simulation Complete ---
        console.print(
            "\n[bold cyan]═══════════════ SIMULATION COMPLETE ═══════════════[/bold cyan]"
        )

        # Price journey
        console.print(
            f"\n  Initial Price: ₹{model.initial_price:.2f}"
            f"\n  Final Price:   ₹{model.current_price:.2f}"
            f"\n  Change:        "
            f"{((model.current_price - model.initial_price) / model.initial_price) * 100:+.2f}%"
        )

        # Final leaderboard
        console.print("\n  [bold]Final Leaderboard:[/bold]")
        traders = sorted(
            [a for a in model.agents if hasattr(a, "cash")],
            key=lambda a: a.portfolio_value,
            reverse=True,
        )
        for rank, agent in enumerate(traders, 1):
            roi = (
                (
                    agent.portfolio_value
                    - (initial_cash + initial_shares * initial_price)
                )
                / (initial_cash + initial_shares * initial_price)
            ) * 100
            console.print(
                f"  #{rank} {agent.persona_name:20s} "
                f"₹{agent.portfolio_value:>10.2f}  ({roi:+.2f}% ROI)"
            )

    finally:
        # DATA PERSISTENCE: Save unified audit log regardless of success/failure
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)

        # Build unified audit from model.history_log + metadata
        audit = {
            "simulation_meta": {
                "n_traders": n_traders,
                "n_steps_planned": n_steps,
                "n_steps_completed": last_completed_step + 1,
                "initial_price": initial_price,
                "final_price": round(model.current_price, 2),
                "final_gini": model.gini,
                "llm_model": model_str,
            },
            "price_history": [round(p, 2) for p in model.price_history],
            "step_logs": model.history_log,
            "agents_final": [
                {
                    "id": a.unique_id,
                    "persona": getattr(a, "persona_name", "unknown"),
                    "cash": round(a.cash, 2),
                    "shares": a.shares,
                    "portfolio_value": round(a.portfolio_value, 2),
                }
                for a in model.agents
                if hasattr(a, "cash")
            ],
        }

        audit_path = log_dir / "full_simulation_audit.json"
        with open(audit_path, "w", encoding="utf-8") as f:
            json.dump(audit, f, indent=2, ensure_ascii=False)

        console.print(f"\n  [dim]Unified audit log:  {audit_path}[/dim]")

        # DataCollector DataFrame summary & Visual Proof
        try:
            df = model.datacollector.get_model_vars_dataframe()
            if not df.empty:
                console.print("\n  [bold]DataCollector — Final Model State:[/bold]")
                console.print(f"  {df.tail(1).to_string()}")
                
                # --- VISUAL PROOF (Matplotlib) ---
                console.print("\n  [bold yellow]Generating Visual Proof (Charts)...[/bold yellow]")
                
                # Plot 1: Price Crash Chart
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(df.index, df["Price"], color="#4C72B0", linewidth=2.5, label="Market Price")
                
                # Highlight Step 15 Shock if we reached it
                if len(df) > 15:
                    ax.axvline(x=15, color="#C44E52", linestyle="--", linewidth=2, label="Shock Event (Step 15)")
                    
                    # Safely get the price at step 15
                    try:
                        step_15_price = df["Price"].iloc[15]
                    except IndexError:
                        step_15_price = df["Price"].iloc[-1]
                        
                    ax.annotate("CEO Arrested", xy=(15, step_15_price), 
                                xytext=(15.5, df["Price"].max() * 0.95),
                                arrowprops=dict(facecolor="#C44E52", shrink=0.05, width=2, headwidth=8),
                                fontsize=11, color="#C44E52", fontweight="bold")
                
                ax.set_title("Mesa 4.0 Reflexive Market Price Discovery", fontsize=14, fontweight="bold", pad=15)
                ax.set_xlabel("Simulation Step", fontsize=12)
                ax.set_ylabel("Price (₹)", fontsize=12)
                ax.grid(True, linestyle=":", alpha=0.7)
                ax.legend(loc="upper left")
                
                price_chart_path = log_dir / "price_crash_chart.png"
                fig.savefig(price_chart_path, dpi=300, bbox_inches="tight")
                plt.close(fig)
                
                # Plot 2: Wealth Inequality (Gini)
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(df.index, df["Gini"], color="#DD8452", linewidth=2.5, label="Gini Coefficient")
                
                ax.set_title("Wealth Inequality (Gini Coefficient) Over Time", fontsize=14, fontweight="bold", pad=15)
                ax.set_xlabel("Simulation Step", fontsize=12)
                ax.set_ylabel("Gini (0 = Equal, 1 = Unequal)", fontsize=12)
                # Ensure the y-axis scales sensibly 
                ax.set_ylim(0, max(0.1, df["Gini"].max() * 1.2)) 
                ax.grid(True, linestyle=":", alpha=0.7)
                ax.legend(loc="upper left")
                
                gini_chart_path = log_dir / "wealth_inequality_gini.png"
                fig.savefig(gini_chart_path, dpi=300, bbox_inches="tight")
                plt.close(fig)

                console.print(f"  [dim]Visual Proof saved: {price_chart_path}[/dim]")
                console.print(f"  [dim]Visual Proof saved: {gini_chart_path}[/dim]")

        except Exception as e:
            console.print(f"  [dim]Could not generate charts: {e}[/dim]")

        console.print(
            f"  \n  [dim]DataCollector data available via model.datacollector[/dim]"
        )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Mesa 4.0 Stock Market Simulation")
    parser.add_argument("--offline", action="store_true", help="Run simulation using MOCK agents (No API key needed)")
    parser.add_argument("--steps", type=int, default=25, help="Number of simulation steps to run")
    args = parser.parse_args()
    
    run_simulation(is_offline=args.offline, n_steps=args.steps)
