import mesa
import numpy as np
from mesa.experimental.actions import Action


class Triage(Action):
    def __init__(self, nurse, patient):
        dur = max(0.5, nurse.model.rng.normal(1.5, 0.3))
        super().__init__(nurse, duration=dur)
        self.patient = patient

    def on_complete(self):
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
    def __init__(self, doctor, patient):
        durations = {"critical": 12.0, "moderate": 6.0, "minor": 2.0}
        base = durations.get(patient.severity, 4.0)
        dur = max(1.0, doctor.model.rng.normal(base, base * 0.2))
        super().__init__(
            doctor, duration=dur,
            interruptible=(patient.severity != "critical"),
        )
        self.patient = patient

    def on_start(self):
        self.agent.current_patient = self.patient
        self.patient.status = "being_treated"

    def on_complete(self):
        self.patient.status = "discharged"
        self.patient.discharge_time = self.agent.model.time
        self.agent.patients_treated += 1
        self.agent.current_patient = None
        self.agent.model.try_assign(self.agent)

    def on_interrupt(self, progress):
        self.patient.status = "waiting"
        self.agent.current_patient = None
        self.agent.model.enqueue_patient(self.patient)

    def on_resume(self):
        self.agent.current_patient = self.patient
        self.patient.status = "being_treated"


class Patient(mesa.Agent):
    def __init__(self, model, arrival_time):
        super().__init__(model)
        self.arrival_time = arrival_time
        self.discharge_time = None
        self.severity = None
        self.status = "arrived"

    @property
    def wait_time(self):
        end = self.discharge_time if self.discharge_time else self.model.time
        return end - self.arrival_time


class Doctor(mesa.Agent):
    def __init__(self, model):
        super().__init__(model)
        self.patients_treated = 0
        self.current_patient = None


class ERModel(mesa.Model):
    def __init__(self, n_doctors=3, arrival_rate=1.5, seed=42):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.arrival_rate = arrival_rate
        self._queue = []

        for _ in range(n_doctors):
            Doctor(self)

        self._schedule_arrival()

    def _schedule_arrival(self):
        gap = self.rng.exponential(1.0 / self.arrival_rate)
        self.schedule_event(self._on_arrival, at=self.time + gap)

    def _on_arrival(self):
        patient = Patient(self, arrival_time=self.time)

        nurse = self._get_free_doctor()
        if nurse:
            nurse.start_action(Triage(nurse, patient))
        else:
            self._queue.append(patient)

        self._schedule_arrival()

    def enqueue_patient(self, patient):
        self._queue.append(patient)
        # sort: critical first, then by arrival
        self._queue.sort(
            key=lambda p: (
                {"critical": 0, "moderate": 1, "minor": 2}.get(p.severity, 3),
                p.arrival_time
            )
        )
        self._assign_waiting()

    def try_assign(self, doctor):
        if self._queue and not doctor.is_busy:
            patient = self._queue.pop(0)
            doctor.start_action(Treat(doctor, patient))

    def _assign_waiting(self):
        for doc in self.agents_by_type[Doctor]:
            if not doc.is_busy and self._queue:
                patient = self._queue.pop(0)
                doc.start_action(Treat(doc, patient))

    def _get_free_doctor(self):
        free = [d for d in self.agents_by_type[Doctor] if not d.is_busy]
        return self.rng.choice(free) if free else None

    def step(self):
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
            print(f"avg critical wait: {sum(p.wait_time for p in crit) / len(crit):.1f}")

    print()
    for doc in model.agents_by_type[Doctor]:
        print(f"doctor {doc.unique_id}: {doc.patients_treated} patients treated")
