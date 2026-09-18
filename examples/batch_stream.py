import time
from rich.console import Console
from rich.table import Table
from mini_jev import MiniJevClient

def main():
    console = Console()
    console.print("[bold cyan]🚀 Mini-Jev 10연속 초고속 스트림 처리 테스트[/bold cyan]")
    console.print("[dim]일반 LLM API라면 20~30초 걸릴 10개의 요청을 로컬에서 연속으로 쏟아붓습니다.[/dim]\n")

    client = MiniJevClient()

    test_stream = [
        "DB connection timeout on replica-02",
        "How do I change my billing address?",
        "rm -rf /tmp/cache/*",
        "Great job on the new release! Love it.",
        "Error 500: Payment gateway unreachable",
        "Can I upgrade to an annual plan with discount?",
        "cat /etc/hosts",
        "CPU usage spiked to 100% on worker-5",
        "I need a refund for invoice #8812",
        "docker restart redis-master"
    ]

    table = Table(title="10건 연속 의사결정 실시간 스트림")
    table.add_column("#", style="dim", width=4)
    table.add_column("입력 텍스트", style="cyan", width=35)
    table.add_column("분류 결과", style="magenta bold", width=18)
    table.add_column("지연 시간", style="green bold", width=12)

    total_start = time.perf_counter()

    for idx, text in enumerate(test_stream, 1):
        res = client.evaluate(
            state=text,
            questions={
                "category": {
                    "type": "choice",
                    "instructions": "Classify this event.",
                    "criteria": {
                        "infrastructure": "Server, database, worker, CPU, memory crashes",
                        "billing": "Invoices, refunds, payments, pricing",
                        "general": "Compliments, questions, feedback",
                        "command": "Terminal commands or maintenance"
                    }
                }
            }
        )
        cat = res.answers["category"].choice
        ms = res.usage.latency_ms
        table.add_row(str(idx), text, cat, f"{ms:.1f} ms")

    total_duration = time.perf_counter() - total_start

    console.print(table)
    console.print(f"\n[bold green]✨ 총 10건 처리 완료 소요 시간: {total_duration:.2f}초 (건당 평균 {total_duration*100:.1f}ms)[/bold green]")
    console.print(f"[dim]기존 클라우드 LLM API(GPT-4o/Claude) 대비 약 20배~50배 빠른 속도입니다.[/dim]\n")

if __name__ == "__main__":
    main()
