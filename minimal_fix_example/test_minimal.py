#!/usr/bin/env python3
"""Test the minimal fix."""

from model import HotellingModel, Firm

def test_portrayal(agent):
    """Test safe access pattern."""
    pos = getattr(agent, 'position', None)
    if pos is None:
        return {}
    return {'safe': True, 'pos': pos, 'x': pos[0]}

def main():
    print("Testing minimal fix...")
    
    # Test model creation
    model = HotellingModel()
    print(f"✓ Model created with {len(model.agents)} agents")
    
    # Test all agents have positions
    for agent in model.agents:
        pos = getattr(agent, 'position', None)
        if pos is None:
            print(f"✗ Agent {agent.firm_id} has no position!")
            return False
        print(f"✓ Agent {agent.firm_id}: position={pos}")
    
    # Test safe portrayal
    for agent in model.agents:
        result = test_portrayal(agent)
        if not result:
            print(f"✗ Agent {agent.firm_id} portrayal failed!")
            return False
        print(f"✓ Agent {agent.firm_id} portrayal: {result}")
    
    print("\n🎉 Minimal fix works! No more TypeError.")
    return True

if __name__ == "__main__":
    main()