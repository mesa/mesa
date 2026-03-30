#!/usr/bin/env python3
"""Final test before PR - simulate solara run app.py"""

from model import HotellingModel, Firm
from mesa.visualization import SpaceRenderer
from mesa.visualization.components import AgentPortrayalStyle

def agent_portrayal(agent):
    """Test the agent portrayal function."""
    if isinstance(agent, Firm):
        # Safe access to position attribute
        pos = getattr(agent, 'position', None)
        if pos is None:
            return AgentPortrayalStyle()  # Return empty style for agents without position
            
        # Use Mesa 3.2.0+ AgentPortrayalStyle
        color = "blue" if agent.firm_id == 0 else "red"
        return AgentPortrayalStyle(
            x=pos[0],
            y=pos[1], 
            color=color,
            size=50,
            tooltip={
                "Firm ID": agent.firm_id,
                "Position": pos,
                "Profit": agent.profit
            }
        )
    return AgentPortrayalStyle()

def main():
    print("🧪 FINAL CHECKLIST TEST")
    print("=" * 50)
    
    # Test 1: Model creation
    try:
        model = HotellingModel(num_firms=2, grid_width=10)
        print("✅ Model created successfully")
        print(f"   - Agents: {len(model.agents)}")
        print(f"   - Grid: {model.grid}")
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False
    
    # Test 2: Agent positions
    all_positioned = True
    for agent in model.agents:
        pos = getattr(agent, 'position', None)
        if pos is None:
            print(f"❌ Agent {agent.firm_id} has no position!")
            all_positioned = False
        else:
            print(f"✅ Agent {agent.firm_id}: position={pos}, profit={agent.profit}")
    
    if not all_positioned:
        return False
    
    # Test 3: Agent portrayal (visualization logic)
    try:
        for agent in model.agents:
            portrayal = agent_portrayal(agent)
            if not hasattr(portrayal, 'x'):
                print(f"❌ Agent {agent.firm_id} portrayal missing x coordinate!")
                return False
            print(f"✅ Agent {agent.firm_id} portrayal: x={portrayal.x}, y={portrayal.y}, color={portrayal.color}")
    except Exception as e:
        print(f"❌ Agent portrayal failed: {e}")
        return False
    
    # Test 4: SpaceRenderer setup
    try:
        renderer = SpaceRenderer(model, backend="altair")
        renderer.setup_agents(agent_portrayal)
        print("✅ SpaceRenderer setup successful")
    except Exception as e:
        print(f"❌ SpaceRenderer setup failed: {e}")
        return False
    
    # Test 5: Model step
    try:
        model.step()
        print("✅ Model step completed")
        for agent in model.agents:
            print(f"   - Agent {agent.firm_id}: profit={agent.profit}")
    except Exception as e:
        print(f"❌ Model step failed: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 ALL TESTS PASSED!")
    print("✅ No crash")
    print("✅ Agents visible (have positions and portrayals)")
    print("✅ Ready for: solara run app.py")
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)