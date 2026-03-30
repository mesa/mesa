#!/usr/bin/env python3
"""Test script to verify the Hotelling's Law fix works."""

from model import HotellingModel, Firm

def agent_portrayal(agent):
    """Test version of agent portrayal function."""
    if isinstance(agent, Firm):
        # FIXED: Properly check if agent has a cell and get position safely
        if agent.cell is None:
            return {}  # Don't render agents without positions
            
        pos = agent.cell.coordinate  # Use coordinate instead of position
        
        # FIXED: Safe access to position coordinates
        rect_x = pos[0] + (agent.firm_id * 0.2)  # Small offset for visibility
        
        portrayal = {
            "Shape": "rect",
            "Color": "blue" if agent.firm_id == 0 else "red",
            "x": rect_x,
            "y": pos[1],
        }
        return portrayal
    return {}

def main():
    """Test the fix."""
    print("Testing Hotelling's Law fix...")
    
    # Create model
    model = HotellingModel(num_firms=2, grid_width=10)
    print(f"✓ Model created with {len(model.agents)} agents")
    
    # Test agent positioning
    for agent in model.agents:
        if agent.cell is None:
            print(f"✗ Agent {agent.firm_id} has no cell!")
            return False
        else:
            print(f"✓ Agent {agent.firm_id} at position {agent.cell.coordinate}")
    
    # Test visualization function
    for agent in model.agents:
        portrayal = agent_portrayal(agent)
        if not portrayal:
            print(f"✗ Agent {agent.firm_id} portrayal failed!")
            return False
        else:
            print(f"✓ Agent {agent.firm_id} portrayal: {portrayal}")
    
    # Test model step
    model.step()
    print("✓ Model step completed successfully")
    
    print("\n🎉 All tests passed! The fix works correctly.")
    return True

if __name__ == "__main__":
    main()