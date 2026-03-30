#!/usr/bin/env python3
"""Core test - verify the fix without visualization dependencies"""

from model import HotellingModel, Firm

def agent_portrayal_logic(agent):
    """Test the core portrayal logic without Mesa visualization classes."""
    if isinstance(agent, Firm):
        # Safe access to position attribute
        pos = getattr(agent, 'position', None)
        if pos is None:
            return None  # No portrayal for agents without position
            
        # Core logic that would be in AgentPortrayalStyle
        color = "blue" if agent.firm_id == 0 else "red"
        return {
            'x': pos[0],
            'y': pos[1], 
            'color': color,
            'firm_id': agent.firm_id,
            'position': pos,
            'profit': agent.profit
        }
    return None

def main():
    print("🧪 CORE LOGIC TEST (No visualization dependencies)")
    print("=" * 60)
    
    # Test 1: Model creation
    try:
        model = HotellingModel(num_firms=2, grid_width=10)
        print("✅ Model created successfully")
        print(f"   - Agents: {len(model.agents)}")
        print(f"   - Grid exists: {hasattr(model, 'grid')}")
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False
    
    # Test 2: Agent positions (the original bug)
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
    
    # Test 3: Safe access pattern (the fix)
    try:
        for agent in model.agents:
            portrayal = agent_portrayal_logic(agent)
            if portrayal is None:
                print(f"❌ Agent {agent.firm_id} portrayal is None!")
                return False
            
            # Test the original problematic line: pos[0]
            x_coord = portrayal['x']  # This is equivalent to pos[0]
            print(f"✅ Agent {agent.firm_id} safe access: x={x_coord}, y={portrayal['y']}")
            
    except Exception as e:
        print(f"❌ Safe access test failed: {e}")
        return False
    
    # Test 4: Model step
    try:
        model.step()
        print("✅ Model step completed")
        for agent in model.agents:
            print(f"   - Agent {agent.firm_id}: profit={agent.profit}")
    except Exception as e:
        print(f"❌ Model step failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 CORE LOGIC TESTS PASSED!")
    print("✅ No crash - TypeError: 'NoneType' object is not subscriptable FIXED")
    print("✅ Agents have positions and can be safely accessed")
    print("✅ Original bug (pos[0] when pos=None) is resolved")
    print("\n📝 The fix ensures:")
    print("   1. agent.position is always initialized")
    print("   2. Safe access with getattr(agent, 'position', None)")
    print("   3. Null check before accessing pos[0]")
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)