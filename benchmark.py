import time
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from mini_jev import MiniJevClient

def main():
    console = Console()
    console.print(Panel.fit("[bold green]Mini-Jev Latency Benchmark on Apple Silicon MPS[/bold green]\nModel: google/gemma-3-270m-it"))

    console.print("[dim]Loading model onto MPS...[/dim]")
    client = MiniJevClient()

    state = "User reports that their account was charged twice for the monthly subscription on 2026-09-18. They are requesting a full refund for the duplicate charge."

    questions = {
        "is_refund": {
            "type": "noul",
            "instructions": "Is the user asking for a monetary refund?"
        },
        "category": {
            "type": "choice",
            "instructions": "What is the primary category?",
            "criteria": {
                "billing": "Charges, invoices, refunds",
                "technical": "Software bugs, crashes",
                "sales": "Upgrades, pricing plans"
            }
        }
    }

    # Warmup
    console.print("[yellow]Running 5 warmup cycles...[/yellow]")
    for _ in range(5):
        client.evaluate(state=state, questions=questions)

    # Benchmark runs
    n_iterations = 50
    console.print(f"[bold cyan]Running {n_iterations} benchmark iterations...[/bold cyan]")

    latencies = []
    for _ in range(n_iterations):
        res = client.evaluate(state=state, questions=questions)
        latencies.append(res.usage.latency_ms)

    latencies = np.array(latencies)

    # Results Table
    table = Table(title="Benchmark Results (2 Questions Evaluated Simultaneously)")
    table.add_column("Metric", style="cyan bold")
    table.add_column("Value (ms)", style="green")

    table.add_row("Mean Latency", f"{np.mean(latencies):.2f} ms")
    table.add_row("Median (P50)", f"{np.median(latencies):.2f} ms")
    table.add_row("P90", f"{np.percentile(latencies, 90):.2f} ms")
    table.add_row("P95", f"{np.percentile(latencies, 95):.2f} ms")
    table.add_row("P99", f"{np.percentile(latencies, 99):.2f} ms")
    table.add_row("Min", f"{np.min(latencies):.2f} ms")
    table.add_row("Max", f"{np.max(latencies):.2f} ms")
    table.add_row("Throughput", f"{1000.0 / np.mean(latencies):.1f} queries/sec")

    console.print(table)

if __name__ == "__main__":
    main()
