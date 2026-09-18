import random
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

TEST_TEMPLATES = {
    "billing": [
        "Please issue a refund for transaction #{id}",
        "My company card was charged twice for the team license #{id}",
        "How can I update my billing email and VAT ID?",
        "Why was I billed 50 dollars extra on invoice #{id}?",
        "Payment declined: Insufficient funds on credit card",
        "We want to switch from monthly to annual billing",
        "Where is the receipt for payment #{id}?",
        "The promo code SAVE2026 failed during checkout",
        "Can we get an invoice for purchase order #{id}?",
        "Cancel my subscription renewal immediately"
    ],
    "infrastructure": [
        "Kubernetes node worker-{id} status is NotReady",
        "Redis cluster disconnected: connection refused on port 6379",
        "504 Gateway Timeout error on /api/v2/orders",
        "Postgres CPU pegged at 100% on query PID {id}",
        "High memory alert: OOM killed service worker-{id}",
        "Network packet loss spike on router-{id}",
        "SSL certificate expired for sub.domain-{id}.com",
        "RabbitMQ message queue backup on queue-{id}",
        "Disk space critical: /var/log is 99% full on host {id}",
        "Database connection pool exhausted for pool {id}"
    ],
    "general": [
        "Really loving the new UI redesign on dashboard!",
        "Thank you so much for the quick customer support",
        "Is there a roadmap for upcoming mobile features?",
        "Great blog article on distributed database systems",
        "Kudos to the engineering team for the smooth migration",
        "Where can I find the official developer community?",
        "The dark mode theme looks fantastic on OLED screens",
        "Just wanted to give positive feedback on the documentation",
        "Congrats on reaching 10,000 GitHub stars!",
        "Excited to attend the upcoming user meetup"
    ],
    "command": [
        "rm -rf /var/cache/build_{id}/*",
        "docker stop $(docker ps -q --filter name=worker_{id})",
        "git reset --hard HEAD~{id}",
        "chmod -R 777 /opt/secrets_{id}",
        "kill -9 $(pgrep -f celery_worker_{id})",
        "systemctl restart nginx.service",
        "cat /etc/shadow",
        "ls -la /var/log/nginx/access_{id}.log",
        "tar -czvf backup_{id}.tar.gz /data/db",
        "dropdb --if-exists test_db_{id}"
    ]
}

KEYS = ["billing", "infrastructure", "general", "command"]
LETTERS = ["A", "B", "C", "D"]
LETTER_TO_KEY = dict(zip(LETTERS, KEYS))

def generate_100_samples():
    random.seed(2026)
    samples = []
    # 25 of each category = exactly 100 samples
    for cat in KEYS:
        templates = TEST_TEMPLATES[cat]
        for _ in range(25):
            t = random.choice(templates)
            text = t.format(id=random.randint(100, 9999))
            samples.append({"text": text, "truth": cat})
    random.shuffle(samples)
    return samples

def main():
    console.print(Panel.fit(
        "[bold cyan]⚡ Mini-Jev (Gemma 3 270M LoRA) 100건 실전 의사결정 벤치마크 ⚡[/bold cyan]\n"
        "다양한 4가지 카테고리(billing, infrastructure, general, command)에 대해\n"
        "단일 포워드 패스 로짓 추출로 초고속 분류 정확도 및 총 소요 시간을 측정합니다."
    ))

    model_id = "google/gemma-3-270m-it"
    adapter_path = "./gemma3_jev_lora"
    device = torch.device("mps")

    console.print("[dim]모델 및 LoRA 어댑터 로드 중...[/dim]")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    base_model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base_model, adapter_path).to(device)
    model.eval()
    console.print("[green bold]준비 완료! 100건 평가 시작...[/green bold]\n")

    samples = generate_100_samples()
    options_str = "\n".join([f"{LETTERS[i]}: {KEYS[i]}" for i in range(len(KEYS))])

    target_token_ids = [tokenizer.encode(l, add_special_tokens=False)[0] for l in LETTERS]

    results = []
    category_counts = {k: 0 for k in KEYS}
    correct_counts = {k: 0 for k in KEYS}

    # Warmup 1 run
    warmup_prompt = (
        f"<start_of_turn>user\nContext: hello\n\nClassify the context into one option:\n{options_str}\n\n"
        f"Respond with only the option letter.<end_of_turn>\n<start_of_turn>model\n"
    )
    with torch.no_grad():
        _ = model(**tokenizer(warmup_prompt, return_tensors="pt").to(device))

    # Synchronize before timing
    torch.mps.synchronize()
    total_start = time.perf_counter()

    for idx, s in enumerate(samples, 1):
        prompt = (
            f"<start_of_turn>user\n"
            f"Context: {s['text']}\n\n"
            f"Classify the context into one option:\n"
            f"{options_str}\n\n"
            f"Respond with only the option letter.<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )
        
        t0 = time.perf_counter()
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs)
            # Logit at the very next token
            next_logits = out.logits[0, -1, :][target_token_ids].float()
        torch.mps.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000

        probs = torch.softmax(next_logits, dim=-1)
        best_idx = int(torch.argmax(probs).item())
        predicted_letter = LETTERS[best_idx]
        predicted_key = LETTER_TO_KEY[predicted_letter]
        confidence = probs[best_idx].item()

        category_counts[predicted_key] += 1
        is_correct = (predicted_key == s["truth"])
        if is_correct:
            correct_counts[s["truth"]] += 1

        results.append({
            "id": idx,
            "text": s["text"],
            "truth": s["truth"],
            "pred": predicted_key,
            "conf": confidence,
            "latency": latency_ms,
            "correct": is_correct
        })

    torch.mps.synchronize()
    total_elapsed = time.perf_counter() - total_start

    # Print Sample of 12 Results Table
    sample_table = Table(title="100건 중 샘플 12건 결과 미리보기")
    sample_table.add_column("#", width=4, style="dim")
    sample_table.add_column("입력 텍스트", width=38, style="cyan")
    sample_table.add_column("실제 정답", width=15, style="white")
    sample_table.add_column("Jev 분류 예측", width=15, style="magenta bold")
    sample_table.add_column("확신도", width=10, style="yellow")
    sample_table.add_column("정답여부", width=10)
    sample_table.add_column("지연 시간", width=10, style="green")

    # Pick 3 from each category to display
    display_indices = [0, 1, 2, 25, 26, 27, 50, 51, 52, 75, 76, 77]
    for i in display_indices:
        r = results[i]
        status = "[green]✓ 일치[/green]" if r["correct"] else "[red]✗ 불일치[/red]"
        sample_table.add_row(
            str(r["id"]),
            r["text"],
            r["truth"],
            r["pred"],
            f"{r['conf']:.1%}",
            status,
            f"{r['latency']:.1f}ms"
        )
    console.print(sample_table)

    # Summary Statistics
    total_correct = sum(1 for r in results if r["correct"])
    accuracy = total_correct / len(results) * 100
    avg_latency = total_elapsed / len(results) * 1000

    summary_table = Table(title="📊 100건 종합 벤치마크 통계")
    summary_table.add_column("지표", style="bold cyan")
    summary_table.add_column("결과 수치", style="bold green")

    summary_table.add_row("총 처리 건수", f"{len(results)} 건")
    summary_table.add_row("총 소요 시간", f"{total_elapsed:.2f} 초")
    summary_table.add_row("건당 평균 지연 시간", f"{avg_latency:.1f} ms")
    summary_table.add_row("초당 처리량 (Throughput)", f"{len(results) / total_elapsed:.1f} queries/sec")
    summary_table.add_row("전체 분류 정확도 (Accuracy)", f"{accuracy:.1f}% ({total_correct}/100)")

    console.print("\n", summary_table)

    # Class Distribution Table
    dist_table = Table(title="🏷️ 카테고리별 예측 분포 (전부 Infra였던 편향 해결 확인)")
    dist_table.add_column("카테고리", style="bold white")
    dist_table.add_column("실제 정답 수", style="dim")
    dist_table.add_column("Jev가 예측한 수", style="cyan bold")
    dist_table.add_column("카테고리별 정확도", style="green bold")

    for cat in KEYS:
        corr = correct_counts[cat]
        pred_cnt = category_counts[cat]
        dist_table.add_row(
            cat,
            "25",
            str(pred_cnt),
            f"{(corr / 25) * 100:.1f}% ({corr}/25)"
        )

    console.print("\n", dist_table)

if __name__ == "__main__":
    main()
