import random
import time
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from rich.console import Console

console = Console()

# 1. Dataset Generator
TEMPLATES = {
    "billing": [
        "I need a refund for invoice #{id}", "Credit card charge #{id} was duplicated",
        "Please cancel my monthly subscription", "How much does the enterprise tier cost?",
        "Payment failed with error code 402", "Where can I download my VAT tax receipt?",
        "Can I pay via wire transfer instead of card?", "My account was billed after cancellation",
        "Discount coupon PROMO{id} not applying", "Update expiration date on mastercard",
        "Overcharged on our annual bill", "Stripe payment gateway declined my card"
    ],
    "infrastructure": [
        "Postgres replica #{id} connection pool exhausted", "Redis cluster master node failed healthcheck",
        "Kubernetes pod crashlooping with OOMKilled", "Nginx 502 bad gateway on api endpoint",
        "Disk usage reached 98% on volume /dev/sda{id}", "High latency on kafka consumer group {id}",
        "TLS certificate expired for api.example.com", "DNS resolution failing in us-west region",
        "Packet loss detected on gateway switch #{id}", "Worker node CPU throttling at 100%",
        "Server unreachable on port 443", "Database locked due to deadlocks on table #{id}"
    ],
    "general": [
        "Thanks for the awesome documentation updates", "Really loving the new dark mode theme",
        "When is the next community meetup scheduled?", "Great presentation at the tech conference!",
        "How can I join your beta tester program?", "Kudos to the support team for quick help",
        "Is there an official Discord or Slack channel?", "Just wanted to say hello to the team",
        "The website looks very clean and responsive", "Happy anniversary to the company!",
        "Love the clean design of the dashboard", "Awesome blog post about system architecture"
    ],
    "command": [
        "rm -rf /var/cache/app_{id}/*", "docker stop container_{id}",
        "git reset --hard origin/main", "chmod -R 777 /opt/data_{id}",
        "systemctl restart nginx.service", "kill -9 {id}",
        "cat /etc/passwd", "ls -la /var/log/syslog",
        "ps aux | grep python", "tar -czvf backup_{id}.tar.gz /data",
        "curl -X POST http://localhost:8080/shutdown", "dd if=/dev/zero of=/dev/sdb bs=1M"
    ]
}

KEYS = ["billing", "infrastructure", "general", "command"]
LETTERS = ["A", "B", "C", "D"]
KEY_TO_LETTER = dict(zip(KEYS, LETTERS))

def build_data(n_per_class=30, seed=42):
    random.seed(seed)
    data = []
    for cat, texts in TEMPLATES.items():
        for i in range(n_per_class):
            template = random.choice(texts)
            text = template.format(id=random.randint(100, 9999))
            data.append((text, cat))
    random.shuffle(data)
    return data

class JevDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.samples = []
        options_str = "\n".join([f"{LETTERS[i]}: {KEYS[i]}" for i in range(len(KEYS))])
        
        for text, label in data:
            prompt = (
                f"<start_of_turn>user\n"
                f"Context: {text}\n\n"
                f"Classify the context into one option:\n"
                f"{options_str}\n\n"
                f"Respond with only the option letter.<end_of_turn>\n"
                f"<start_of_turn>model\n"
            )
            target = KEY_TO_LETTER[label]
            full_text = prompt + target + "<end_of_turn>"
            
            enc_prompt = tokenizer(prompt, return_tensors="pt")
            enc_full = tokenizer(full_text, return_tensors="pt")
            
            prompt_len = enc_prompt["input_ids"].shape[1]
            input_ids = enc_full["input_ids"][0]
            labels = input_ids.clone()
            # Mask out prompt tokens so loss is ONLY computed on the decision token!
            labels[:prompt_len] = -100
            
            self.samples.append({
                "input_ids": input_ids,
                "attention_mask": enc_full["attention_mask"][0],
                "labels": labels
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch):
    max_len = max(s["input_ids"].shape[0] for s in batch)
    input_ids = torch.full((len(batch), max_len), 0, dtype=torch.long)
    attention_mask = torch.full((len(batch), max_len), 0, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    
    for i, s in enumerate(batch):
        l = s["input_ids"].shape[0]
        input_ids[i, :l] = s["input_ids"]
        attention_mask[i, :l] = s["attention_mask"]
        labels[i, :l] = s["labels"]
        
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

def main():
    console.print("[bold cyan]🚀 Gemma 3 270M LoRA 파인튜닝 시작 (Apple Silicon MPS)[/bold cyan]")
    model_id = "google/gemma-3-270m-it"
    device = torch.device("mps")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16)
    model = model.to(device)

    # Setup LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Train dataset (120 samples: 30 per class)
    train_raw = build_data(n_per_class=30, seed=42)
    train_ds = JevDataset(train_raw, tokenizer)
    loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    epochs = 4

    console.print(f"[yellow]학습 시작: 총 {len(train_raw)}건 데이터, {epochs} 에폭...[/yellow]")
    t_start = time.perf_counter()
    model.train()

    for ep in range(epochs):
        total_loss = 0
        for b in loader:
            optimizer.zero_grad()
            out = model(
                input_ids=b["input_ids"].to(device),
                attention_mask=b["attention_mask"].to(device),
                labels=b["labels"].to(device)
            )
            loss = out.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        console.print(f"Epoch {ep+1}/{epochs} - Loss: {avg_loss:.4f}")

    dur = time.perf_counter() - t_start
    console.print(f"[bold green]✨ 학습 완료! 소요 시간: {dur:.1f}초[/bold green]\n")

    # Save adapter
    adapter_path = "./gemma3_jev_lora"
    model.save_pretrained(adapter_path)
    console.print(f"[dim]LoRA 어댑터 저장 완료: {adapter_path}[/dim]")

if __name__ == "__main__":
    main()
