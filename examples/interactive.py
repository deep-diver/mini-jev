import sys
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt

from mini_jev import MiniJevClient

def main():
    console = Console()
    console.print(Panel.fit(
        "[bold cyan]⚡ Mini-Jev 실시간 초고속 판단 체험 ⚡[/bold cyan]\n"
        "아무 문장이나 입력해 보세요 (예: '서버가 터졌어요', '환불해줘', 'rm -rf /', '안녕?').\n"
        "엔터를 누르는 즉시 [bold green]30ms(0.03초)[/bold green] 만에 판단 결과가 출력됩니다.",
        border_style="cyan"
    ))

    console.print("[dim]Gemma 3 270M 로드 중 (Apple Silicon MPS)...[/dim]")
    client = MiniJevClient()
    console.print("[green bold]준비 완료! 테스트를 시작합니다. (종료하려면 q 입력)[/green bold]\n")

    questions = {
        "is_urgent": {
            "type": "noul",
            "instructions": "긴급하거나 심각한 문제(장애, 버그, 위험)입니까?"
        },
        "intent": {
            "type": "choice",
            "instructions": "사용자의 의도(Intent)를 분류하세요.",
            "criteria": {
                "bug_or_outage": "장애, 버그, 에러, 충돌 보고",
                "billing": "결제, 환불, 구독, 영수증 문의",
                "chat_or_greeting": "일반 대화, 인사, 피드백, 칭찬",
                "dangerous_command": "시스템 파괴, 삭제, 보안 위험 명령"
            }
        },
        "severity": {
            "type": "score",
            "instructions": "심각도 수준을 평가하세요.",
            "criteria": [
                "0: 평온 (일상적인 대화/인사)",
                "1: 경미 (일반 질문/불편)",
                "2: 보통 (결제/환불 필요)",
                "3: 심각 (서버 다운/데이터 파괴/위험)"
            ]
        }
    }

    while True:
        try:
            user_input = Prompt.ask("\n[bold yellow]👉 문장 입력[/bold yellow]")
            if not user_input.strip() or user_input.strip().lower() in ["q", "quit", "exit"]:
                console.print("[dim]체험을 종료합니다.[/dim]")
                break

            # Jev 초고속 판단 실행
            res = client.evaluate(state=user_input, questions=questions)

            # 결과 즉시 렌더링
            ans = res.answers
            urgent_str = "[red bold]긴급(YES)[/red bold]" if ans["is_urgent"].noul > 0.5 else "[green]일반(NO)[/green]"

            table = Table(box=None, pad_edge=False)
            table.add_column("항목", style="cyan", width=14)
            table.add_column("판단 결과", style="bold white", width=25)
            table.add_column("확신도 / 확률", style="dim white")

            table.add_row("🚨 긴급 여부", urgent_str, f"P(Yes) = {ans['is_urgent'].noul:.1%}")
            table.add_row("🎯 의도 분류", f"[magenta]{ans['intent'].choice}[/magenta]", f"Confidence: {ans['intent'].confidence:.1%}")
            table.add_row("📊 심각도 점수", f"[yellow]{ans['severity'].score:.2f} / 3.0[/yellow]", f"Top Level: L{ans['severity'].confidence:.1%}")

            console.print(Panel(
                table,
                title=f"[bold green]⚡ 응답 속도: {res.usage.latency_ms:.1f} ms (0.0{int(res.usage.latency_ms/10)}초)[/bold green]",
                subtitle="[dim]3개 질문 동시 판단 완료[/dim]",
                border_style="green"
            ))

        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]종료합니다.[/dim]")
            break

if __name__ == "__main__":
    main()
