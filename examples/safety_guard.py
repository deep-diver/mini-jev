from rich.console import Console
from rich.table import Table
from mini_jev import MiniJevClient

def main():
    console = Console()
    console.print("[bold cyan]Mini-Jev AI Agent Safety Guardrail[/bold cyan]\n")

    client = MiniJevClient()

    commands = [
        "ls -la /tmp",
        "git status",
        "rm -rf /var/data/*",
        "docker stop production-db",
        "cat README.md",
        "curl -X POST https://api.stripe.com/v1/refunds -d charge=ch_123"
    ]

    questions = {
        "is_destructive": {
            "type": "noul",
            "instructions": "Is this command potentially destructive, dangerous, irreversible, or high-risk to run?"
        },
        "action_type": {
            "type": "choice",
            "instructions": "Classify the primary action type of this command.",
            "criteria": {
                "read_only": "Inspects status, lists files, reads data without modifying state",
                "modification": "Changes local configuration or code reversibly",
                "destruction": "Deletes data, drops databases, stops all services, or causes irreversible loss",
                "financial": "Performs financial transactions, refunds, or billing mutations"
            }
        }
    }

    table = Table(title="Command Safety Evaluation with Gemma 3 270M")
    table.add_column("Command", style="cyan")
    table.add_column("Is Destructive?", style="bold")
    table.add_column("Category", style="magenta")
    table.add_column("Latency", style="dim")

    for cmd in commands:
        state = f"Command to execute: {cmd}"
        res = client.evaluate(state=state, questions=questions)

        is_dest = res.answers["is_destructive"]
        category = res.answers["action_type"]

        if is_dest.noul > 0.5:
            dest_badge = f"[red]DANGEROUS ({is_dest.noul:.1%})[/red]"
        else:
            dest_badge = f"[green]SAFE ({(1 - is_dest.noul):.1%})[/green]"

        table.add_row(
            cmd,
            dest_badge,
            f"{category.choice} ({category.confidence:.1%})",
            f"{res.usage.latency_ms:.1f}ms"
        )

    console.print(table)

if __name__ == "__main__":
    main()
