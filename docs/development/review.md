# Reviewing contributions

Review is how we decide whether a change solves a real problem and fits Mesa, not just whether the code runs. We want to be rigorous and welcoming at the same time: give contributors enough context to understand our concerns, and a clear next step they can take.

This guide is for maintainers, collaborators, and peer reviewers, including GSoC contributors. Reviewers can help clarify a problem, test an approach, and improve an implementation. Maintainers decide project direction and whether a change is ready to merge. For the values behind the questions below, see [Design principles and trade-offs](principles.md).

## Review in order: problem, concept, implementation

Work through the same three stages as [how we work](index.md), in order:

1. **Problem:** is there a real, sufficiently understood problem for Mesa to solve?
2. **Concept:** do we agree on what a good solution should do and what it will cost?
3. **Implementation:** does the code deliver that agreed solution safely?

This is a sequence of attention, not a demand for a long process. A small, reproducible bug fix may move through all three quickly. A new public API or architectural change usually needs more discussion before code review is useful.

If the problem or concept is unsettled, reviewing implementation details wastes everyone's time: we end up requesting tests, names, or edge-case handling for a design that will change or be declined. That is also why line-level AI review bots haven't worked well for us: they focus attention on local code details before we know whether that code should exist ([#2851](https://github.com/mesa/mesa/issues/2851), [#3485](https://github.com/mesa/mesa/discussions/3485)).

## What to look at at each stage

### Problem: what are we solving?

Start with the user's task and the observed behaviour.

- Who is affected, and in what model or workflow?
- What did they expect, and what happened instead?
- Can we reproduce it with a minimal example, logs, or other evidence?
- Is this a Mesa bug, an intentional constraint, a misunderstanding, or an issue in another component or repository?
- Does the need belong in Mesa core, an example, an extension, or another project?

**Ready to move on** when we can describe the problem without referring to the proposed implementation, the behaviour is reproducible (or it's clear what evidence is missing), and we understand roughly where the issue belongs. When a report is hard to diagnose, ask for the smallest useful missing piece rather than guessing at a fix. Reproducing first separates a root cause from a nearby symptom ([#2811](https://github.com/mesa/mesa/issues/2811)).

### Concept: what should the solution mean?

Once the need is clear, discuss the user-visible behaviour before getting lost in code.

- What should users be able to do, and what should they observe?
- What should happen for important edge cases, defaults, and errors?
- Which alternatives could solve the problem, and what does each trade away?
- Does the change add public API or affect stable behaviour?
- What does it cost in complexity, performance, teachability, compatibility, and maintenance?

**Ready to move on** when we share an understanding of the intended behaviour and scope, the main trade-offs are visible, and maintainers are aligned on the API direction where needed. If the design is still uncertain, a discussion or a draft pathfinding PR can test it before a full implementation ([#1944](https://github.com/mesa/mesa/discussions/1944)). Experimental status is useful context, not a substitute for a coherent proposal.

### Implementation: does the code deliver?

Now check whether the implementation matches the agreed concept.

- Does it implement the intended behaviour, including important edge cases?
- Do tests demonstrate the reported problem and verify the changed behaviour?
- Are performance claims backed by representative workloads and a fair comparison?
- Do docs and examples match the behaviour? Are migration notes needed?
- Is the change focused enough to review, test, and maintain?

**Ready for final review** when the description explains the motivation and approach, the code and tests cover the agreed scope, and checks pass or failures are understood ([#3484](https://github.com/mesa/mesa/issues/3484), [#3714](https://github.com/mesa/mesa/issues/3714)).

If a design question comes up during code review, say so and move the conversation back to that level instead of continuing to polish code that may be discarded.

## Give feedback people can act on

Warm feedback is specific. It doesn't mean avoiding disagreement or lowering the bar; it means explaining the bar and helping the contributor see how to meet it.

- **Reproduce first.** Try the example or test the claim before recommending a fix. If you can't reproduce it, say what you tried and what evidence you still need.
- **Explain the reason.** Say what risk, user expectation, project boundary, or design concern is behind the comment. "This API is unclear" leaves the author guessing; explain what a user would find ambiguous.
- **Give one clear next step.** Ask for the smallest change or piece of evidence that moves the review forward. When several changes are needed, number them.
- **Separate required from optional.** Label what blocks approval and what is a suggestion or follow-up idea. Interesting cleanup shouldn't hold up a focused change ([#2430](https://github.com/mesa/mesa/issues/2430)).
- **Consolidate asks.** Check the whole diff and earlier comments before asking for more, and give one current list rather than requests scattered across reviews.
- **Acknowledge uncertainty.** Say when you're unsure, invite correction, and name the evidence that would settle the question.
- **Praise specifically.** Point to a useful test, a clear explanation, or a good design choice. It shows what worked, and that we read the change.

For example: "I can reproduce the error with this small model. It comes from the grid topology, not caching, so adding a cache would mask the cause. Could you add a regression test for this placement case first? Then we can check the fix against it." This gives evidence, a reason, and one next step.

Interactions go cold when we correct someone without explaining how to proceed, introduce new requirements late, or treat disagreement as a personal failing. A long list of fragmented requests that leaves the contributor guessing which ones matter has the same effect ([#1625](https://github.com/mesa/mesa/issues/1625), [#1898](https://github.com/mesa/mesa/issues/1898)).

## Close or redirect a PR kindly

A PR doesn't have to be merged to have been worthwhile. If we can't proceed, explain why plainly: the need isn't demonstrated, the design is unsettled, the work belongs elsewhere, or another change superseded it.

Then give a path forward where one exists: a minimal reproducer, a discussion to agree on the behaviour, a narrower PR, a different repository, or a link to the work that superseded it. When closing because a contributor hasn't responded or checks remain unresolved, summarise what is still needed and whether reopening or a new PR would make sense.

Don't make process enforcement the whole explanation. If a feature needed prior design agreement, say which decision is still open and where to discuss it. A clear reason and a route forward make a closure easier to accept, even when the answer is no ([#3886](https://github.com/mesa/mesa/issues/3886), [#3888](https://github.com/mesa/mesa/issues/3888)).

## Peer review by contributors

Peer review is valuable even if you're not a maintainer. Reproduce the issue, ask about an unclear contract, check a test, or point out an edge case. If you'd suggest a different design, explain the user need or trade-off behind it.

Be clear about what you can and can't assess: "I tested this case and it fails" or "I'm not sure this API is agreed yet." Don't present your preference as Mesa policy or imply that maintainers have approved a direction. GSoC peer reviewers: focus on learning and actionable feedback, explain how to investigate, and keep the next task manageable. Maintainers remain responsible for project decisions and merge approval.

## Using AI tools in review

AI tools can help you inspect code, draft test ideas, summarise a long discussion, or spot a possible inconsistency. Treat every suggestion as a lead to verify, not as evidence. Check it against the code, the tests, the current documentation, and the problem the PR is meant to solve.

Respect the stage. If the problem or concept is unsettled, raise that first; don't let a stream of naming or line-level comments make an implementation look ready. If you use a bot, triage its comments before passing them on; don't ask maintainers or contributors to sort through unverified suggestions ([#2851](https://github.com/mesa/mesa/issues/2851)).

A bot can't speak for maintainer consensus, approve a design, or decide to close a PR. Keep human responsibility visible: say which findings you verified, which are uncertain, and what you recommend.
