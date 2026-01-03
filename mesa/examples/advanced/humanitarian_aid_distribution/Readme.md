# Humanitarian Aid Distribution Model (Needs-Based Architecture)

## Summary

This model demonstrates a **needs-based behavioral architecture** where agent behavior emerges from the drive to satisfy biological needs. The simulation models a humanitarian crisis scenario where civilian beneficiaries require water and food, while aid trucks distribute supplies based on a sophisticated triage system.

The model implements principles from motivation psychology (Maslow, 1943) and drive-reduction theory (Hull, 1943), showing how simple need-driven rules create complex emergent behaviors ranging from normal daily wandering to desperate help-seeking.

### Beneficiary Agents

Beneficiaries maintain two dynamic need levels (water and food urgency) that increase over time. Their behavior follows a state machine with four distinct modes:

- **Wandering** (Urgency < 40): Normal life, random movement
- **Opportunistic** (40 ≤ Urgency < 60): Accept help when convenient (search radius: 4 cells)
- **Seeking** (60 ≤ Urgency < 90): Actively prioritize help-seeking (search radius: 8 cells)  
- **Desperate** (Urgency ≥ 90): Emergency mode, ignore all costs (search radius: infinite)

Agents transition between these states based on homeostatic thresholds. If an agent remains in the desperate state for too long (configurable threshold), they die and are removed from the simulation.

### Truck Agents

Aid trucks implement a **hybrid triage system** that balances humanitarian imperatives with logistical efficiency:

1. **Tier 1 - Survival Priority**: Critical beneficiaries (desperate state) are prioritized using an urgency-squared metric to save the most lives
2. **Tier 2 - Logistics Optimization**: Non-critical beneficiaries are served using a utility function (total urgency / distance) to maximize efficiency

When delivering aid, trucks use **proportional allocation** based on relative need intensity:
```
water_share = (water_urgency / total_urgency) × available_supplies
food_share = (food_urgency / total_urgency) × available_supplies
```

This implements principles from utility theory and marginal economics, ensuring the most urgent needs are satisfied first. The model also includes **diminishing returns**: after receiving aid, an agent's metabolism temporarily slows (decay rates reduced by 20%), modeling physiological adaptation.

Trucks return to a central depot at (0, 0) to refill when supplies are depleted.

### Mesa Concepts Demonstrated

This model showcases several Mesa concepts and features:
- **Multiple Agent Types** (Beneficiaries and Trucks with different behaviors)
- **Dynamic Agent States** (state machines driven by internal variables)
- **Spatial Movement** (pathfinding, distance calculations)
- **Agent Removal** (dynamic population as agents die)
- **Data Collection** (tracking average urgency, deaths, critical count)
- **Behavioral Architecture** (needs-based decision-making framework)
- **Multi-criteria Decision Making** (triage logic with competing objectives)

## How to Run

To run the model interactively, navigate to this directory and run:

```bash
solara run app.py
```

This will launch a browser-based visualization where you can:
- Adjust the number of beneficiaries and trucks via sliders
- Modify the critical days threshold (time until death when desperate)
- Watch agents move and change colors based on their need states
- Track population-level metrics in real-time plots

### Color Coding in Visualization

**Beneficiaries:**
- 🟢 **Green**: Wandering (comfortable, low needs)
- 🟡 **Lime**: Opportunistic (moderate needs, will accept nearby help)
- 🟠 **Orange**: Seeking (high needs, actively searching for aid)
- 🔴 **Red**: Desperate (critical needs, emergency mode)

**Trucks:**
- 🔵 **Blue Squares**: Aid delivery vehicles

## Files

* **`agents.py`**: Contains the `Beneficiary` and `Truck` agent classes
  - `Beneficiary`: Implements the needs-based state machine and help-seeking behavior
  - `Truck`: Implements the hybrid triage system and proportional aid distribution

* **`model.py`**: Contains the `HumanitarianModel` class which manages the simulation environment, grid space, and data collection

* **`app.py`**: Sets up the interactive Solara visualization with agent portrayal functions and model parameter controls

## Relevant Literature:
    - Balcik, B., Beamon, B. M., & Smilowitz, K. (2008). Last mile 
      distribution in humanitarian relief. Journal of Intelligent 
      Transportation Systems, 12(2), 51-63.
      
    - Holguín-Veras, J., et al. (2012). On the appropriate objective 
      function for post-disaster humanitarian logistics models. 
      Journal of Operations Management, 31(5), 262-280.
      
    - Gralla, E., Goentzel, J., & Fine, C. (2014). Assessing trade-offs 
      among multiple objectives for humanitarian aid delivery using expert 
      preferences. Production and Operations Management, 23(6), 978-989.

## Other References used:
[Reference_PDF](https://drive.google.com/file/d/1GEL3gbCl9TCnQ5MOtMaB3LQ32o59oCr_/view?usp=drive_link)

This pdf contains : 
- Other Relevant papers/resources used for building the model
- Explanation of the part of the theory implemented from these papers along with snippets from code that follow the theory 
- Modification (if any) in the theory to implement this code 

## Key Insights

This model demonstrates several important behavioral phenomena:

1. **Emergent Triage**: Without centralized coordination, the system naturally prioritizes the most urgent cases through the truck's scoring functions

2. **Behavioral Escalation**: As needs intensify, agents progressively expand their search radius and willingness to travel, creating realistic desperation dynamics

3. **Resource Competition**: Multiple agents in crisis can compete for limited truck capacity, forcing difficult allocation decisions

4. **System Collapse**: If trucks are insufficient or poorly distributed, cascading failures occur as more agents enter desperate states

5. **Diminishing Returns**: The metabolism slowdown after aid receipt creates temporal dynamics where recently-helped agents require less immediate attention