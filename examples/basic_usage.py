import json
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from mini_jev import MiniJevClient

def main():
    console = Console()
    console.print(Panel.fit("[bold cyan]Mini-Jev (Gemma 3 270M-IT) Demo[/bold cyan]\nFast, typed decisions on Apple Silicon MPS"))

    # Initialize client
    console.print("[yellow]Loading Gemma 3 270M-IT onto MPS...[/yellow]")
    client = MiniJevClient()
    console.print("[green]Ready![/green]\n")

    # Sample input state
    state = {
        "customer_id": "cust_9921",
        "subscription": "Enterprise Pro",
        "ticket": "Our production webhook endpoint has been failing with HTTP 500 errors for the past 2 hours. Payments are not getting processed and customers are complaining. Please help immediately!"
    }

    console.print("[bold]Input State:[/bold]")
    console.print_json(json.dumps(state, ensure_ascii=False))

    # Evaluate 3 questions simultaneously
    questions = {
        "is_outage": {
            "type": "noul",
            "instructions": "Does this message describe a critical production outage or severe downtime?"
        },
        "department": {
            "type": "choice",
            "instructions": "Which department should handle this ticket?",
            "criteria": {
                "billing": "Invoice, pricing, credit card details",
                "infrastructure": "Server crashes, production down, API errors",
                "general": "Feedback, feature requests, general questions"
            }
        },
        "urgency_score": {
            "type": "score",
            "instructions": "Rate the urgency level from low to emergency.",
            "criteria": [
                "Low: Minor question, no impact",
                "Medium: Minor feature affected, workaround exists",
                "High: Significant impact, but system partially operational",
                "Critical: Complete outage or catastrophic failure"
            ]
        }
    }

    console.print("\n[bold]Evaluating with Mini-Jev...[/bold]")
    response = client.evaluate(state=state, questions=questions)

    # Display Answers
    table = Table(title=f"Results ({response.model}) - Latency: {response.usage.latency_ms} ms")
    table.add_column("Question", style="cyan", no_wrap=True)
    table.add_column("Type", style="magenta")
    table.add_column("Decision", style="green bold")
    table.add_column("Confidence", style="yellow")
    table.add_column("Probabilities / Details", style="white")

    for q_name, ans in response.answers.items():
        if ans.type == "noul":
            decision = "YES" if ans.noul >= 0.5 else "NO"
            details = f"P(Yes) = {ans.noul:.2%}"
        elif ans.type == "choice":
            decision = ans.choice
            details = ", ".join([f"{k}: {v:.1%}" for k, v in ans.probabilities.items()])
        elif ans.type == "score":
            decision = f"Score: {ans.score:.2f} / 3.0"
            details = ", ".join([f"L{k}: {v:.1%}" for k, v in ans.probabilities.items()])

        table.add_row(q_name, ans.type, decision, f"{ans.confidence:.2%}", details)

    console.print(table)
    console.print(f"[dim]Total Input Tokens: {response.usage.input_tokens} | Total Latency: {response.usage.latency_ms} ms[/dim]\n")

if __name__ == "__main__":
    main()
