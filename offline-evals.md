Hi Everyone. My name is Ivo, I'm an engineer in Team Core Agents.

---

Today I want to show you the work I've done on offline evaluations.
In the show and tell 2 weeks ago Sean demo'ed realtime evaluations using LangFuse and Mastra, if you have not seen this, I highly recommend you to check that out. While realtime evaluations are critical for monitoring production agents, it won't tell you if your agents is getting better or worse after a change. Manual testing is very time consuming, does not scale, you wouldn't be able to cover dozens of edge cases your agent might encounter, you can't prove your improvements and you will very likely miss regressions.

---

So enter Offline Evaluations!
Offline evaluations are like unit-tests for non-deterministic agentic workflows, you can run your agent through edge-cases, test-datasets and regressions and measure the results. It's "offline" because this is running outside of production, not with real users. The Goal is to shift from "the agents feels good" to the agent scores 95% on task completion.

---

This is the overall flow; you start with a dataset, these datasets can be created from annotated production traces in LangFuse, for each item in the dataset the agents executes, the datasaet will set the context for the agents and the agent will execute what it was designed to do, using metrics functions we gather the metrics we need to quantify the results, this results in scores.

---

So what did I built?
I started with the metrics functions; these were wrapped into a universal metrics engine that can run them from LangFuse as well as completely independent.
Then I've created a hybrid evaluation engine, this is will invoke the agent either locally or in LangFuse mode, which is posting back the metrics to LangFuse giving you insights with nice dashboards, if you want without actually running the agent but just pull the dataset through your metrics to set a baseline.
Aside from that there's a extensive CLI runner, nice for local development, critical for running evals in CI/CD pipelines.
Last but not least there's an agent runner, this calls the actual agent in the Mastra framework

---

DEMO

---

So, some of the learnings and challenges.
What workedwell, zod validation makes invalid datasets fail fast, resulting in immediate feedback, The architecture; metrics and orchestration is done on our side, we can even generate reports in JSON if we want to, LangFuse owns the UI and partially the datasets. We can easily switch to something else if the need arises.
Some challenges; LangFuse's datasets are noisy, contain a lot of metadata that's not relevant, local datasets are typically handcrafted and therefore clean and simple, and because of not wanting to be locked in I've created a mapping from both dataset formats to a unified one. Webhook integration from LangFuse does not work, it's very limited to start with; only port 80 and 443, no support for sending headers, not HMAC, and very difficult to determine what went wrong, they definetely have some work to do there.

---

So what is next?

The evals run pretty standard and simple evaluators now, we should look into LLM-as-a-Judge evaluators to check non-determinstic output. Though theoreticall possible, I have not done any CI/CD integration. Eventually we'd need to start sampling data from production to get a solid dataset based on real user-sessions (if legal will allow that). And we'd like to be able to trigger evals from the LangFuse UI, enabling non-technical folks to do evaluations on agents.
