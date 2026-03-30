# Kubeflow SDK OpenTelemetry Integration: OptimizerClient & Katib Fan-Out Tracing

## Personal Information

| Field | Details |
|-------|---------|
| **Name** | Vartika Manish |
| **Email** | vartikamanish03@gmail.com |
| **GitHub** | https://github.com/Vartika222 |
| **University** | Bennett University, Greater Noida |
| **Degree** | B.Tech Computer Science, 2nd Year |
| **Timezone** | IST (UTC+5:30) |
| **Availability** | 40 hours/week |

---

## Executive Summary

**The Problem:** When hyperparameter optimization (HPO) sweeps with 100+ trials fail silently, users have zero observability. Kubeflow SDK lacks tracing instrumentation.

**The Naive Solution:** Span-per-trial would create 100+ spans per experiment — cardinality that breaks Prometheus and drowns Jaeger's UI.

**My Solution:** Implement OpenTelemetry instrumentation with a **proven fan-out pattern**:
- **Aggregate experiment span** capturing critical path (one span for all trials)
- **Per-trial metrics** emitted separately, queryable by Prometheus without cardinality explosion

**Why It's Different:** Every other proposal focuses on TrainerClient (serial workloads). Nobody has tackled the multi-cardinality challenge. I'm solving Katib's problem specifically.

---

## Why I'm the Right Fit

**Observability at scale:** Built CloudWatchDog — Kafka → telemetry → PostgreSQL timeseries, handling 1M+ events/hour. Deep expertise in cardinality management and metric design at scale.

**HPO domain knowledge:** Implemented Bayesian optimization for anomaly detection. I understand that 100 trials aren't 100 independent traces — they're a distribution you query.

**Kubeflow contributor:** PR #434 fixed exception inconsistencies across backends. Already understand the codebase and backend architecture.

**Academic excellence:** CGPA 9.12, Dean's List. Organized hackathon with 600+ teams.

---

## Technical Vision

### Core Design: Optional, Zero-Overhead Instrumentation

```python
# kubeflow/common/telemetry.py

try:
    from opentelemetry import trace, metrics
    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False

def get_tracer(name):
    """Real OTel or zero-overhead NoOp — user never writes if tracer:"""
    return trace.get_tracer(name) if _OTEL_AVAILABLE else _NoOpTracer()
```

**Why:** Zero overhead for users who don't install `opentelemetry-api`. Pattern used by gRPC, requests, sqlalchemy.

### The Fan-Out Pattern (The Differentiator)

**Serial (TrainerClient):**
```
TrainerClient.train [1.5s]
  ├── KubernetesBackend.train [1.2s]
  └── wait_for_job_status [45s]
      └── poll_iteration [×N]  ← Each poll is a child span
```

**Fan-Out (OptimizerClient):**
```
OptimizerClient.create_experiment [3.2s]
  kubeflow.optimizer.trial.count = 100  ← AGGREGATE ATTRIBUTE
  ├── Katib.submit_experiment [2.1s]
  └── get_experiment [0.8s]

[Metrics emitted separately, per-trial]
kubeflow.optimizer.trial.duration = histogram([45s, 38s, ...])
  dimensions: algorithm, trial_status
kubeflow.optimizer.trial.loss = histogram([0.0234, 0.0156, ...])
  dimensions: algorithm
```

**Why this wins:**
- No span explosion (5 spans vs 100+)
- Prometheus queryable: `histogram_quantile(0.95, trial.duration)`
- Jaeger clean: experiment-level critical path visible
- Per-trial drill-down via metrics distribution

---

## Implementation Roadmap

### Phase 1: Foundation (Weeks 1–2)
- [ ] `kubeflow/common/telemetry/` module with NoOp layer
- [ ] Span/metric attribute constants (single source of truth)
- [ ] 30+ unit tests

### Phase 2: TrainerClient (Weeks 3–5)
- [ ] Instrument TrainerClient + all 3 backends (K8s, Container, LocalProcess)
- [ ] W3C TRACEPARENT context propagation
- [ ] 50+ tests

**Midterm Checkpoint:** Foundation + TrainerClient complete, 80+ tests passing

### Phase 3: OptimizerClient + Fan-Out Pattern (Weeks 6–7)
- [ ] Trial-aggregate span hierarchy
- [ ] 4 low-cardinality metrics (experiment.created, trial.duration, trial.loss, trials.active)
- [ ] Per-trial metrics emission
- [ ] 30+ tests
- [ ] **This is the hard part. This is where I add unique value.**

### Phase 4: Documentation (Weeks 8–9)
- [ ] Getting-started guide (Jaeger + Prometheus)
- [ ] API reference
- [ ] Worked example: HPO observability end-to-end

### Phase 5: Bonus (Weeks 10–11)
- [ ] ModelRegistryClient instrumentation
- [ ] SparkClient fan-out pattern (reuse learnings from Katib)

### Phase 6: Polish (Weeks 12)
- [ ] Code review cycles
- [ ] Final integration testing

---

## Metrics Design (Low-Cardinality)

| Metric | Type | Dimensions | Cardinality |
|--------|------|-----------|------------|
| `kubeflow.optimizer.experiment.created` | Counter | algorithm, objective | ~10–20 |
| `kubeflow.optimizer.trial.duration` | Histogram | algorithm, trial_status | ~3–5 |
| `kubeflow.optimizer.trial.loss` | Histogram | algorithm | ~3–5 |
| `kubeflow.optimizer.trials.active` | UpDownCounter | experiment_id | ~100 (managed) |

**Key rule:** Trial ID never as metric dimension (cardinality disaster). Queryable only via metrics aggregation.

---

## Why This Proposal Stands Out

### 1. **Solves an Unsolved Problem**
Everyone focuses on TrainerClient instrumentation. Nobody has tackled Katib's fan-out cardinality challenge. This is technically harder and more differentiated.

### 2. **Domain Expertise**
My CloudWatchDog project at scale proves I understand:
- Cardinality explosion risks
- How to design metrics for queryability
- Cost of naive instrumentation

### 3. **Specific to Kubeflow's Architecture**
This isn't generic "add OpenTelemetry to a library." It's "design observability for multi-cardinality ML workloads," which is Kubeflow's unique problem.

### 4. **Reusable Pattern**
The fan-out design becomes a playbook for SparkClient, PipelinesClient, and future multi-cardinality clients. It's force-multiplier thinking.

---

## Non-Goals (Clear Boundaries)

- Anonymous telemetry (separate discussion in #170)
- OTel Collector deployment (user responsibility)
- Internal K8s client instrumentation (SDK boundaries only)
- Log bridge integration (stretch goal)
- PipelinesClient (pending PR #343)

---

## Testing Strategy

- **Unit tests:** 110+ (telemetry module, instrumentation, metrics)
- **Integration tests:** Jaeger + Prometheus with real trials
- **Regression:** Zero failures on existing 47 backend tests
- **Coverage:** >95% on telemetry code

---

## References & Prior Work

**PR #434** (ValueError/RuntimeError fix) — Proves I understand backend architecture across three implementations.

**Issue #164** — The OpenTelemetry tracking issue. I've been actively researching OptimizerClient integration, not just proposing blindly.

**Key References:**
1. OpenTelemetry Python: https://opentelemetry.io/docs/instrumentation/python/
2. Katib Docs: https://www.kubeflow.org/docs/components/katib/overview/
3. W3C TRACEPARENT: https://w3c.github.io/trace-context/
4. Prometheus Cardinality Best Practices: https://prometheus.io/docs/practices/naming/

---

## Differentiation vs. Competitors

| Aspect | Typical Proposal | My Proposal |
|--------|-----------------|------------|
| **Scope** | TrainerClient instrumentation | TrainerClient + OptimizerClient with fan-out pattern |
| **Focus** | "Add tracing to everything" | "Solve Katib's cardinality problem specifically" |
| **Technical Challenge** | Serial span hierarchy | Multi-cardinality span + metrics design |
| **Reusability** | Application-specific | Playbook for all fan-out workloads |
| **Domain Proof** | Generic | CloudWatchDog at 1M+ events/hour |

---

## Realistic Expectations

**Commitment:** 40 hours/week, uninterrupted during coding period (exams front-loaded in April).

**Deliverable:** Production-ready code, >95% test coverage, documentation with worked examples.

**Quality bar:** Code that could ship immediately after mentor review. Not a research project — a system that solves a real problem.

---

## Why Kubeflow?

I chose this project because:
1. **Real impact:** HPO observability unblocks production ML debugging
2. **Technical challenge:** Fan-out tracing isn't in most textbooks
3. **Community:** Kubeflow is building the future of ML infrastructure
4. **Learning:** I'll master observability at scale — a skill that compounds

Not "GSoC looks good on my resume." But "I want to solve this specific problem for this specific community."

---

## Next Steps

1. **Community Bonding:** Deep dive into OptimizerClient source, Katib architecture, existing telemetry patterns
2. **Week 1:** Foundation implementation & tests
3. **Weeks 3–5:** TrainerClient instrumentation
4. **Weeks 6–7:** OptimizerClient + fan-out pattern (the hard part)
5. **Ongoing:** Mentor feedback, code reviews, refinement

I'm ready to start immediately. Let's build something that lasts.
