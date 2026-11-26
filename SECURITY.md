# Security Policy

## Reporting a vulnerability

Please do not open a public GitHub issue for a suspected security
vulnerability.

Preferred: use GitHub's private vulnerability reporting for this
repository (the "Report a vulnerability" button under the Security tab).
That channel reaches the maintainers directly and keeps the report out of
public view while a fix is prepared.

If that channel is unavailable to you, email
[licensing@codetonight.co.za](mailto:licensing@codetonight.co.za) with a
description of the issue, the affected version or commit, and steps to
reproduce. Please encrypt attachments containing exploit detail if you
can.

What to include, where you can:

- The component or file affected (e.g. `lib/hal/gateway.py`, a specific
  provider adapter, the wire/proxy layer).
- A minimal reproduction or proof of concept.
- The potential impact as you understand it (e.g. secret exposure,
  sandbox escape, authentication bypass).

## Supported versions

HAL does not yet publish numbered releases; the `main` branch is the only
supported line. Security fixes land on `main` and are not backported to
any other branch or fork.

## Response

There is no funded, contractually-committed SLA. As a good-faith target,
expect an acknowledgement within 5 business days. Coordinated disclosure
is welcome — tell us your preferred timeline when you report, and we will
work with it where we reasonably can.

## Scope

This policy covers the code in this repository. It does not cover the
security of third-party LLM providers HAL connects to (Anthropic, OpenAI,
Google, Groq, xAI, DeepSeek, Alibaba/DashScope, Mistral, and others) —
report those directly to the provider.
