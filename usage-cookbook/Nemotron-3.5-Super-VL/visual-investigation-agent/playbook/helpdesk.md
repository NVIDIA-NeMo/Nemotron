# Fernledger AI help desk: analyst playbook

## How the help desk works

Fernledger's help desk answers customer tickets with an AI agent. Every ticket has a ticket type. A gateway router sends each ticket to a hosted model; the router config says which model handles which ticket type. Larkfield Cloud bills the hosted models per token.

## Metrics

- **AI spend:** billed USD (`billing_daily.total_usd`).
- **Tickets:** tickets handled by the AI agent.
- **Cost per ticket:** AI spend divided by tickets.
- **Resolution rate:** resolved tickets divided by tickets. A ticket is resolved when the customer needs no human follow-up.
- **Cost per resolved ticket:** AI spend divided by resolved tickets.
- **Tokens per ticket:** input, output (answer), and output (reasoning) tokens divided by tickets.
- **Average price per 1M tokens:** AI spend divided by tokens, times 1,000,000.
- **Monthly savings (for a proposal):** 30 × recent average tickets per day × (cost per ticket of the setup running today − projected cost per ticket of the proposal). Say which days and volume you used.
- **Offline-eval resolution rate:** from `eval_results`, where historical tickets are replayed against every catalog model. This is the quality measure used for routing decisions.

## Data

SQLite tables:

- `usage_daily(date, ticket_type, model, tickets, resolved_tickets, requests, input_tokens, output_answer_tokens, output_reasoning_tokens)`: one row per day, ticket type, and model.
- `billing_daily(date, ticket_type, model, input_usd, output_usd, total_usd)`: billed spend at the same grain.
- `change_log(id, date, time_utc, title)`: production changes.
- `eval_results(model, ticket_type, resolution_rate_pct, eval_tickets, avg_input_tokens, avg_output_answer_tokens, avg_output_reasoning_tokens, avg_requests)`: offline eval for every catalog model, including models not in production.

Dates are UTC in `YYYY-MM-DD` format. Data runs from 2026-08-01 through 2026-09-21; today is 2026-09-22. Prices are not in the database; the current price schedule is a document.

## Investigating a cost change

1. Size the change on like-for-like periods: the same number of days and the same weekday mix.
   Reconcile your totals with the dashboard's KPI cards for the same periods before you rely on them; if they disagree, find out why before you report.
2. Break the change into its drivers (volume, price, usage, and mix), by ticket type and by model.
3. Check the change log around the dates where the metrics moved. A change that coincides with a move is a hypothesis until the data shows how it caused the move.
4. Check quality (resolution rate) alongside cost.
5. For a routing proposal, project cost per ticket for each candidate model and ticket type from `eval_results` tokens and the price schedule, and check each route against the quality bar.
