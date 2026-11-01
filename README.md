# Two-tier multi-agent dispatch for vRAN

A snapshot of my current research on hierarchical multi-agent reinforcement learning for virtualised 5G networks.

> **Heads up:** this is a frozen snapshot of work in progress, not a finished or maintained project. Code, models and results are incomplete and will keep evolving.

## The problem

In a virtualised RAN, 5G physical-layer processing runs on pools of CPUs and GPUs instead of dedicated hardware. Jobs arrive constantly with hard deadlines, and several agents at different levels of the system decide where each one runs, each with only a partial, delayed view and very limited ways to coordinate. The goal is to meet deadlines while reducing energy consumption.

## The approach

I'm exploring how hierarchical multi-agent RL can let these agents coordinate efficiently despite their limited communication. The details are still in progress.

## What's here

An event-driven simulator of the system, a traffic generator built on real 5G traces, baseline policies, and an experiment pipeline. The learned RL policies aren't included yet, since that's where most of my work is right now. The third-party datasets aren't included either.

## Author

Marta Sierra Obea
