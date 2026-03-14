"""Emergency room simulation using Mesa's Action system.

Patients arrive at random intervals, get triaged by severity
(critical / moderate / minor), and are treated by available doctors.
Critical treatments cannot be interrupted; moderate ones can be
bumped by a higher-priority patient and resumed later.
"""

import mesa
import numpy as np
from mesa.experimental.actions import Action


class Triage(Action):
    """Assess a patient's severity and route them accordingly."""

    def __init__(self, nurse, patient):
        """Create a triage action for the given nurse and patient."""
        dur = max(0.5, nurse.model.rng.normal(1.5, 0.3))
        super().__init__(nurse, duration=dur)
        self.patient = patient

    def on_complete(self):
        """Assign severity and either discharge minor cases or enqueue."""
        severity = self.agent.model.rng.choice(
            ["critical", "moderate", "minor"], p=[0.15, 0.45, 0.4]
        )
        self.patient.severity = severity
        self.patient.status = "triaged"

        if severity == "minor":
            self.patient.status = "discharged"
            self.patient.discharge_time = self.agent.model.time
            return

        self.agent.model.enqueue_patient(self.patient)


class Treat(Action):
    """Treat a patient; duration depends on severity."""

    def __init__(self, doctor, patient):
        """Create a treatment action with severity-dependent duration."""
        durations = {"critical": 12.0, "moderate": 6.0, "minor": 2.0}
        base = durations.get(patient.severity, 4.0)
        dur = max(1.0, doctor.model.rng.normal(base, base * 0.2))
        super().__init__(
            doctor,
            duration=dur,
            interruptible=(patient.severity != "critical"),
        )
        self.patient = patient

    def on_start(self):
        """Mark the patient as being treated."""
        self.agent.current_patient = self.patient
        self.patient.status = "being_treated"

    def on_complete(self):
        """Discharge the patient and pick up the next one in the queue."""
        self.patient.status = "discharged"
        self.patient.discharge_time = self.agent.model.time
        self.agent.patients_treated += 1
        self.agent.current_patient = None
        self.agent.model.try_assign(self.agent)

    def on_interrupt(self, progress):
        """Put the patient back in the queue when bumped."""
        self.patient.status = "waiting"
        self.agent.current_patient = None
        self.agent.model.enqueue_patient(self.patient)

    def on_resume(self):
        """Resume treating the same patient after an interruption."""
        self.agent.current_patient = self.patient
        self.patient.status = "being_treated"


class Patient(mesa.Agent):
    """A patient arriving in the ER."""

    def __init__(self, model, arrival_time):
        """Create a patient with the given arrival time."""
        super().__init__(model)
        self.arrival_time = arrival_time
        self.discharge_time = None
        self.severity = None
        self.status = "arrived"

    @property
    def wait_time(self):
        """Total time from arrival to discharge (or current time if still here)."""
        end = self.discharge_time if self.discharge_time else self.model.time
        return end - self.arrival_time


class Doctor(mesa.Agent):
    """A doctor who treats patients."""

    def __init__(self, model):
        """Create a doctor."""
        super().__init__(model)
        self.patients_treated = 0
        self.current_patient = None


class ERModel(mesa.Model):
    """Discrete-event ER simulation with stochastic patient arrivals."""

    def __init__(self, n_doctors=3, arrival_rate=1.5, seed=42):
        """Set up the ER with doctors and schedule the first patient arrival."""
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.arrival_rate = arrival_rate
        self._queue = []

        for _ in range(n_doctors):
            Doctor(self)

        self._schedule_arrival()

    def _schedule_arrival(self):
        """Schedule the next patient arrival using an exponential distribution."""
        gap = self.rng.exponential(1.0 / self.arrival_rate)
        self.schedule_event(self._on_arrival, at=self.time + gap)

    def _on_arrival(self):
        """Handle a patient arrival: triage if a doctor is free, else queue."""
        patient = Patient(self, arrival_time=self.time)

        nurse = self._get_free_doctor()
        if nurse:
            nurse.start_action(Triage(nurse, patient))
        else:
            self._queue.append(patient)

        self._schedule_arrival()

    def enqueue_patient(self, patient):
        """Add a triaged patient to the priority queue and try to assign them."""
        self._queue.append(patient)
        self._queue.sort(
            key=lambda p: (
                {"critical": 0, "moderate": 1, "minor": 2}.get(p.severity, 3),
                p.arrival_time,
            )
        )
        self._assign_waiting()

    def try_assign(self, doctor):
        """Try to assign the next queued patient to a specific doctor."""
        if self._queue and not doctor.is_busy:
            patient = self._queue.pop(0)
            doctor.start_action(Treat(doctor, patient))

    def _assign_waiting(self):
        """Match all free doctors with the highest-priority waiting patients."""
        for doc in self.agents_by_type[Doctor]:
            if not doc.is_busy and self._queue:
                patient = self._queue.pop(0)
                doc.start_action(Treat(doc, patient))

    def _get_free_doctor(self):
        """Return a random available doctor, or None."""
        free = [d for d in self.agents_by_type[Doctor] if not d.is_busy]
        return self.rng.choice(free) if free else None

    def step(self):
        """Advance by one time unit and reassign idle doctors."""
        self.run_for(1.0)
        self._assign_waiting()


if __name__ == "__main__":
    model = ERModel(n_doctors=3, arrival_rate=1.5, seed=42)

    for _ in range(50):
        model.step()

    patients = list(model.agents_by_type[Patient])
    discharged = [p for p in patients if p.status == "discharged"]
    waiting = [p for p in patients if p.status in ("arrived", "triaged", "waiting")]
    treating = [p for p in patients if p.status == "being_treated"]

    print(f"\n--- ER Simulation ({model.time:.0f} time units) ---")
    print(f"total patients:    {len(patients)}")
    print(f"discharged:        {len(discharged)}")
    print(f"in treatment:      {len(treating)}")
    print(f"still waiting:     {len(waiting)}")

    if discharged:
        avg = sum(p.wait_time for p in discharged) / len(discharged)
        print(f"avg wait time:     {avg:.1f}")

        crit = [p for p in discharged if p.severity == "critical"]
        if crit:
            print(
                f"avg critical wait: {sum(p.wait_time for p in crit) / len(crit):.1f}"
            )

    print()
    for doc in model.agents_by_type[Doctor]:
        print(f"doctor {doc.unique_id}: {doc.patients_treated} patients treated")
