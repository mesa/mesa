import mesa


class Beneficiary(mesa.Agent):
    """
    Simulates a person with dynamic needs (water and food).

    The agent follows a Needs-Based Architecture:
    1. Biological drives (hunger/thirst) increase over time.
    2. Behavior emerges from these drives:
       - Low needs: Random wandering.
       - High needs: Seeking help (Trucks or Depot).

    States:
    - Wandering (<40): Content, moving randomly.
    - Opportunistic (40-60): Mild need, will deviate for nearby help.
    - Seeking (60-90): Active need, scanning for trucks within range.
    - Desperate (>90): Critical need, ignoring safety/distance to find help immediately.
    """

    def __init__(self, model, water_decay=2, food_decay=1, critical_days_threshold=5):
        """
        Create a new Beneficiary agent.

        Args:
            model: The Mesa model instance.
            water_decay (float): Amount water urgency increases per step.
            food_decay (float): Amount food urgency increases per step.
            critical_days_threshold (int): Days before death when critical.
        """
        super().__init__(model)
        self.water_urgency = 0
        self.food_urgency = 0
        self.water_decay = water_decay
        self.food_decay = food_decay
        self.critical_days_threshold = critical_days_threshold
        self.is_critical = False
        self.days_critical = 0
        self.claimed_by = None
        self.state = "wandering"  # Initial state

    # Needs Thresholds
    COMFORT = 40  # "I'm fine"
    SURVIVAL = 60  # "I need help soon"
    CRITICAL = 90  # "Emergency"

    def move_towards(self, target_pos):
        """
        Moves the agent one step closer to the target position.
        """
        current_x, current_y = self.pos
        target_x, target_y = target_pos

        next_x, next_y = current_x, current_y

        # Simple Logic: Move 1 step along the axis with the biggest difference
        if current_x < target_x:
            next_x += 1
        elif current_x > target_x:
            next_x -= 1

        if current_y < target_y:
            next_y += 1
        elif current_y > target_y:
            next_y -= 1

        # Verify the spot is valid (Mesa grids handle this, but good practice)
        self.model.grid.move_agent(self, (next_x, next_y))

    def step(self):
        """
        Advance the agent by one step.

        Lifecycle:
        1. Biological Decay: Needs increase naturally.
        2. State Assessment: Determine if agent is critical or dead.
        3. Behavior Selection: Choose action based on stress level.
        """
        # 1. BIOLOGICAL DECAY
        # Urgency increases linearly over time, capped at 100
        # Drive-Reduction Theory (Hull, 1943): Needs increase over time
        self.water_urgency = min(self.water_urgency + self.water_decay, 100)
        self.food_urgency = min(self.food_urgency + self.food_decay, 100)

        # 2. STATE ASSESSMENT
        # Law of the Minimum: The most pressing need dictates behavior
        max_need = max(self.water_urgency, self.food_urgency)

        # State Transitions based on Thresholds
        # Homeostatic thresholds (Cannon, 1932) + Satisficing (Simon, 1956)
        if self.days_critical > self.critical_days_threshold:
            self.state = "dead"
        elif max_need < self.COMFORT:
            self.state = "wandering"  # Normal life
            self.is_critical = False
            self.days_critical = 0
        elif max_need < self.SURVIVAL:
            self.state = "opportunistic"  # Seek help when convenient
            self.is_critical = False
            self.days_critical = 0
        elif max_need < self.CRITICAL:
            self.state = "seeking"  # Actively seek help
            self.is_critical = False  # Not yet critical in terms of "about to die" logic, but prioritizing help
            self.days_critical = 0
        else:
            self.state = "desperate"  # Desperate help seeking
            self.is_critical = True
            self.days_critical += 1

        # Death Check
        if self.state == "dead":
            self.model.grid.remove_agent(self)
            self.remove()
            return  # Stop executing if dead

        # 3. BEHAVIOR SELECTION (The Needs-Based Logic)

        if self.state == "desperate":
            # Ignore all costs, go to nearest truck or depot immediately
            found_truck = self.find_nearest_truck(radius=None)  # Infinite radius
            if found_truck:
                self.move_towards(found_truck.pos)
            else:
                self.move_towards((0, 0))

        elif self.state == "seeking":
            # Prioritize over other activities
            found_truck = self.find_nearest_truck(radius=8)  # Reasonable search radius
            if found_truck:
                self.move_towards(found_truck.pos)
            else:
                self.move_towards((0, 0))

        elif self.state == "opportunistic":
            # Seek help when convenient (only if very close)
            found_truck = self.find_nearest_truck(radius=4)  # Small radius
            if found_truck:
                self.move_towards(found_truck.pos)
            else:
                self.wander()

        elif self.state == "wandering":
            # LOW NEED STATE: Wander / Normal Life
            self.wander()

    def wander(self):
        """Move randomly to simulate local activity"""
        neighborhood = self.model.grid.get_neighborhood(self.pos, moore=True)
        choice = self.random.choice(neighborhood)
        self.model.grid.move_agent(self, choice)

    def find_nearest_truck(self, radius=None):
        """
        Scans the neighborhood for a Truck agent.
        Returns the nearest Truck agent or None.
        Args:
            radius (int, optional): limit search to this distance. None = global.
        """
        trucks = []
        
        if radius is not None:
            neighbors = self.model.grid.get_neighbors(
                self.pos, moore=True, include_center=False, radius=radius
            )
            trucks = [a for a in neighbors if isinstance(a, Truck)]
        else:
            # Fallback to scanning all agents for global search (radius=None)
            trucks = [a for a in self.model.agents if isinstance(a, Truck) and a.pos is not None]

        if not trucks:
            return None
            
        # Helper to calculate Manhattan distance
        def get_dist(t):
            return abs(self.pos[0] - t.pos[0]) + abs(self.pos[1] - t.pos[1])
            
        return min(trucks, key=get_dist)


class Truck(mesa.Agent):
    """
    A delivery agent that distributes supplies to Beneficiaries.

    Behavior:
    - Scans for beneficiaries with high needs.
    - Prioritizes targets based on 'utility' (urgency vs distance).
    - Implements Hybrid Triage:
        1. Critical agents: Prioritized by Urgency^2 (Save lives first).
        2. Non-critical agents: Prioritized by Urgency/Distance (Logistical efficiency).
    - Delivers water/food based on which need is more pressing.
    - Refills at the Depot (0,0) when empty.
    """

    def __init__(self, model):
        super().__init__(model)
        self.supplies = 100  # Current amount of resources carried
        self.delivery_rate = 10  # Max resources delivered per step
        self.target = None  # Current Beneficiary agent being targeted

    def distribute_aid(self, beneficiary, amount=10):
        """Split aid proportionally to need intensity"""

        total_need = beneficiary.water_urgency + beneficiary.food_urgency

        if total_need == 0:
            return 0

        # Allocate proportionally
        # Marginal Utility Theory (Jevons, 1871): Allocate to highest marginal benefit
        water_share = (beneficiary.water_urgency / total_need) * amount
        food_share = (beneficiary.food_urgency / total_need) * amount

        # Apply with diminishing returns
        water_satisfied = min(water_share, beneficiary.water_urgency)
        food_satisfied = min(food_share, beneficiary.food_urgency)

        beneficiary.water_urgency -= water_satisfied
        beneficiary.food_urgency -= food_satisfied

        # CRITICAL: Needs decay slower after satisfaction
        beneficiary.water_decay *= 0.8  # Metabolism slows temporarily
        beneficiary.food_decay *= 0.8

        return water_satisfied + food_satisfied

    def move_towards(self, target_pos):
        """
        Moves the agent one step closer to the target position.
        """
        current_x, current_y = self.pos
        target_x, target_y = target_pos

        next_x, next_y = current_x, current_y
        if current_x < target_x:
            next_x += 1
        elif current_x > target_x:
            next_x -= 1
        if current_y < target_y:
            next_y += 1
        elif current_y > target_y:
            next_y -= 1

        self.model.grid.move_agent(self, (next_x, next_y))

    def get_distance(self, pos):
        x1, y1 = self.pos
        x2, y2 = pos
        return abs(x1 - x2) + abs(y1 - y2)

    def step(self):
        """
        Advance the truck by one step.

        Lifecycle:
        1. Logistics: Refill if empty.
        2. Target Validation: Ensure current target is still valid.
        3. Target Selection: Find a new target if needed. In target selection each truck selects a beneficiary based on hybrid triage logic. This is done so multiple trucks don't go to the same beneficiary.
        4. Action: Move or delivering supplies.
        """
        # 1. LOGISTICS
        # If out of supplies, return to base to refill
        if self.supplies <= 0:
            if self.target and self.target.claimed_by == self:
                self.target.claimed_by = None
                self.target = None

            if self.pos == (0, 0):
                self.supplies = 50
            else:
                self.move_towards((0, 0))
            return

        # 2. TARGET VALIDATION
        if self.target:
            # Check if target is removed from model OR has no position (Dead)
            if (not self.target.model) or (self.target.pos is None):
                self.target = None
            elif self.target.claimed_by != self:
                self.target = None

        # 3. TARGET SELECTION
        if not self.target:
            all_agents = self.model.agents
            possible_victims = [
                a
                for a in all_agents
                # Check pos is not None to ensure we don't pick dead agents
                if isinstance(a, Beneficiary)
                and (a.claimed_by is None)
                and (a.pos is not None)
                # Explicitly ignore comfort state
                and a.state != "wandering"
            ]

            if possible_victims:
                # HYBRID TRIAGE LOGIC
                # 1. SPLIT into Critical (Survival) and Non-Critical (Logistics)
                critical_targets = [
                    a for a in possible_victims if a.state == "desperate"
                ]
                non_critical_targets = [
                    a for a in possible_victims if a.state != "desperate"
                ]

                if critical_targets:
                    # TIER 1: SURVIVAL
                    # GOAL: SAVE MORE LIVES by satisfying the most urgent needs.
                    # Use a score that considers BOTH urgency and distance
                    # so we don't ignore a dying neighbor for a dying stranger far away.
                    def survival_score(a):
                        dist = self.get_distance(a.pos)
                        max_urgency = max(a.water_urgency, a.food_urgency)
                        # We square urgency so it remains the dominant factor,
                        # but distance still acts as a tie-breaker.
                        return (max_urgency**2) / (dist + 1)

                    self.target = max(critical_targets, key=survival_score)

                elif non_critical_targets:
                    # TIER 2: LOGISTICS / EFFICIENCY
                    # Goal: Maximize value per mile.
                    # Metric: Total Urgency / Distance
                    def logistics_score(a):
                        dist = self.get_distance(a.pos)
                        total_urgency = a.water_urgency + a.food_urgency
                        return total_urgency / (dist + 1)

                    self.target = max(non_critical_targets, key=logistics_score)

                # Assign target if found
                if self.target:
                    self.target.claimed_by = self
            else:
                neighborhood = self.model.grid.get_neighborhood(self.pos, moore=True)
                self.model.grid.move_agent(self, self.random.choice(neighborhood))
                return

        # 4. ACTION
        if self.target:
            # DOUBLE CHECK: Ensure target didn't die between validation and action
            if self.target.pos is None:
                self.target = None
                return

            if self.pos == self.target.pos:
                amount_given = self.distribute_aid(
                    self.target, amount=min(self.supplies, self.delivery_rate)
                )

                self.supplies -= amount_given
                self.target.days_critical = (
                    0  # Reset critical clock on any help received
                )
                self.target.claimed_by = None
                self.target = None
            else:
                self.move_towards(self.target.pos)
