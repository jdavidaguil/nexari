# nexari

**From intent to deployed model.**

```bash
nexari run "classify customer support tickets by urgency"
```

Nexari is an autonomous ML pipeline agent. Give it a problem in plain English — it finds the right dataset, selects a backbone, fine-tunes a model, and hands you a running inference endpoint.

---

## How it works

```
Your intent
    │
    ▼
[1] Interpret     → extracts task type, domain, metric from natural language
    │
    ▼
[2] Discover      → searches Hugging Face Hub, proposes 3 datasets with rationale
    │   ← you approve
    ▼
[3] Select        → agent picks the right backbone model autonomously
    │
    ▼
[4] Train         → fine-tunes on HF infrastructure
    │
    ▼
[5] Deploy        → HF Inference Endpoint + local preview UI
```

---

## Quickstart

```bash
pip install nexari
cp .env.example .env   # add your AWS + HF credentials
nexari run "classify customer support tickets by urgency"
```

## Configuration

| Variable | Default | Description |
|---|---|---|
| `NEXARI_LLM_BACKEND` | `bedrock` | `bedrock` or `ollama` |
| `AWS_REGION` | `us-east-1` | AWS region for Bedrock |
| `HF_TOKEN` | — | Hugging Face API token |
| `HF_NAMESPACE` | — | Your HF username/org |

See `.env.example` for all options.

## Development

```bash
git clone https://github.com/jdaguilera/nexari
cd nexari
pip install -e ".[dev]"
nexari config          # verify setup
nexari run "..." --dry-run   # test without training
```

Or open in GitHub Codespaces — the devcontainer handles everything.

## Stack

- **LLM reasoning**: Claude via AWS Bedrock (default) or Ollama (local)
- **Datasets + models**: Hugging Face Hub
- **Training**: HF `transformers` + `datasets`
- **Serving**: HF Inference Endpoints
- **Preview**: FastAPI + single-page UI

## Roadmap

- [x] Intent interpretation
- [x] Dataset discovery + human approval
- [x] Backbone selection
- [ ] Fine-tuning pipeline
- [ ] HF Endpoint deployment
- [ ] Preview UI
- [ ] Model card generation
- [ ] Bring your own data (S3/GCS)

---

MIT License
