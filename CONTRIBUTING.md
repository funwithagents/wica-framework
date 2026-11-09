# Contributing

Thanks for your interest in WICA. This page says how the project is run today, so you know what to expect before you spend time on it.

## How the project is run

WICA is developed by one person, in the open, as a spec-driven project: every change starts as a design in [specs/](specs/_index.md), becomes an implementation plan in [plans/](plans/_index.md), and only then code. [AGENTS.md](AGENTS.md) describes that workflow and the checks every change passes. The framework is at version 0.1 and is being shaped as it is used: the API changes between commits without a deprecation period, and the concurrency model is deliberately narrow for now ([docs/guides/limits.md](docs/guides/limits.md)).

## Pull requests: not for now

External pull requests are not accepted for now. This is not a judgement on contributions: while the design is settling, a change written against yesterday's specs usually costs more to reconcile than to redo, and reviewing it to the project's standard (a spec, a plan, the two test tiers, the live set against paid provider accounts) takes the time that would otherwise go into the design work that would make it unnecessary. A pull request will be closed with thanks and, when the idea is a good one, turned into an issue so it is not lost.

This file will say so when that changes.

## Feedback: yes, please

Issues are the way in, at [github.com/funwithagents/wica-framework/issues](https://github.com/funwithagents/wica-framework/issues), and every kind of feedback is welcome:

- **Bug reports.** Say what you ran: the provider and model, your config with secrets removed, the World entries and Commands involved, the OS and Python version; what you expected, what happened, and the `wica` logs at `DEBUG`. A small script on `provider: "fake"` that reproduces it is the best possible report.
- **Reports from models and providers the author does not test on.** The live set runs one model per provider; a report that another model drives a WICA agent well, or does not (ignores the primer, mis-sequences Commands, never calls `noop`), is especially valuable, with a benchmark row where you can ([docs/guides/benchmarking.md](docs/guides/benchmarking.md)).
- **API and design feedback.** What you would want the World, the Commands, the config or the panels to do differently, and why. At this stage the design is cheap to change and this is the most useful input there is. Pointing at the spec you are reacting to helps.
- **Questions.** If the README, a page under [docs/](docs/index.md) or a spec left you guessing, that is a documentation bug worth an issue too.

## Working with the code anyway

The code is MIT-licensed, so forking it and building on it is welcome. The commands to lint, type check and run the two test tiers are in [AGENTS.md](AGENTS.md), which also explains how the specs, plans and their statuses fit together; the consumer documentation starts at [docs/index.md](docs/index.md). If your fork grows something you think belongs upstream, open an issue describing it: that is the conversation that will lead to accepting pull requests later.
