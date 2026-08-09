# Deploying to GCP

Single Compute Engine VM running the stack under Docker Compose, behind Caddy with automatic HTTPS. Sized for the $300 free-trial credit.

**Estimated cost:** ~$27/month for the VM, ~$2 for the disk, and cents for Gemini traffic at demo volume. Comfortably inside the credit for the full 90-day trial — and you can stop the VM between demos (see §7).

---

## 0. Prerequisites

- A GCP project with billing enabled (the free trial counts)
- `gcloud` CLI installed and authenticated: `gcloud auth login`
- The code pushed to a git repository the VM can clone

```bash
export PROJECT_ID="your-project-id"
export REGION="us-central1"
export ZONE="us-central1-a"
export VM_NAME="rag-chatbot"

gcloud config set project "$PROJECT_ID"
```

---

## 1. Enable APIs

```bash
gcloud services enable aiplatform.googleapis.com compute.googleapis.com
```

---

## 2. Service account (no key files)

The VM authenticates to Vertex AI through an **attached service account**, read via the metadata server. Nothing is downloaded, so there is no key file to leak — the single most common way demo projects expose credentials.

```bash
gcloud iam service-accounts create rag-chatbot-sa \
  --display-name="RAG Chatbot service account"

# Least privilege: Vertex AI inference only.
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:rag-chatbot-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/aiplatform.user"
```

> Do **not** run `gcloud iam service-accounts keys create`. It isn't needed, and a downloaded key is a liability.

---

## 3. Static IP

Reserve this **before** pointing DNS at it — otherwise the address changes on restart and the certificate has to be reissued.

```bash
gcloud compute addresses create rag-chatbot-ip --region="$REGION"

export STATIC_IP=$(gcloud compute addresses describe rag-chatbot-ip \
  --region="$REGION" --format="value(address)")
echo "Static IP: $STATIC_IP"
```

---

## 4. Hostname for HTTPS

Caddy needs a resolvable hostname to obtain a Let's Encrypt certificate.

**Free option — DuckDNS:** register at [duckdns.org](https://www.duckdns.org), create a subdomain (e.g. `flexsin-rag`), and point it at `$STATIC_IP`. Your hostname is then `flexsin-rag.duckdns.org`.

Confirm it resolves before deploying:

```bash
nslookup flexsin-rag.duckdns.org      # must return $STATIC_IP
```

> **No hostname?** Uncomment `tls internal` in the `Caddyfile` for a self-signed certificate. It works immediately with no DNS, but browsers show a warning — fine for a private demo, less so for an interviewer's first impression.
>
> **Avoid `nip.io`.** It resolves without registration, but has hit Let's Encrypt rate limits and certificate issuance may fail.

---

## 5. Firewall

Only HTTP and HTTPS. The API and UI are never exposed directly — they are reachable only inside the Compose network.

```bash
gcloud compute firewall-rules create allow-rag-web \
  --allow=tcp:80,tcp:443 \
  --target-tags=rag-chatbot \
  --description="HTTP/HTTPS for the RAG chatbot demo"
```

---

## 6. Create the VM

```bash
gcloud compute instances create "$VM_NAME" \
  --zone="$ZONE" \
  --machine-type=e2-medium \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=20GB \
  --boot-disk-type=pd-balanced \
  --address="$STATIC_IP" \
  --tags=rag-chatbot \
  --service-account="rag-chatbot-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
  --scopes=cloud-platform
```

`--scopes=cloud-platform` matters: without it the attached service account cannot reach Vertex AI regardless of its IAM roles.

**On sizing.** `e2-medium` (2 vCPU, 4 GB) fits the ONNX embedder plus two Python services with room to spare. `e2-small` (2 GB) technically runs, but the ~$14/month saving is not worth debugging an OOM mid-demo. No GPU is needed — generation is hosted, and free-trial accounts generally cannot get GPU quota anyway.

---

## 7. Deploy

```bash
gcloud compute ssh "$VM_NAME" --zone="$ZONE"
```

Then on the VM:

```bash
# Docker
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker "$USER"
newgrp docker      # or log out and back in

# Code
git clone <your-repo-url> rag-chatbot
cd rag-chatbot

# Configuration
cat > .env <<'EOF'
GCP_PROJECT=your-project-id
GCP_LOCATION=us-central1
LLM_MODEL=gemini-2.5-flash
SITE_ADDRESS=flexsin-rag.duckdns.org
EOF

# Confirm Vertex AI works via the attached service account BEFORE building.
# This is the step most likely to differ from local, where ADC is used instead.
python3 -m venv /tmp/verify && /tmp/verify/bin/pip install -q google-genai python-dotenv
/tmp/verify/bin/python scripts/verify_vertex.py

# Launch
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
```

Build takes a few minutes — mostly the ONNX model pre-warm baked into the API image.

---

## 8. Verify

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml ps     # all healthy
docker compose -f docker-compose.yml -f docker-compose.prod.yml logs caddy | tail -20
```

From your own machine:

```bash
curl -s https://flexsin-rag.duckdns.org/api/health | python -m json.tool
```

Expect `"status": "ok"` with both `store` and `llm` reporting `ok`.

**Confirm nothing else is exposed** — these must both fail:

```bash
curl --max-time 5 http://$STATIC_IP:8000     # API: refused
curl --max-time 5 http://$STATIC_IP:8501     # UI:  refused
```

Then open `https://flexsin-rag.duckdns.org`, upload a document, and check:

1. An answerable question returns a citation resolving to the correct page.
2. An unanswerable question is refused, with the gate notice shown.
3. A pronoun follow-up shows the rewritten search query.

---

## 9. Cost control

**Stop the VM between demos.** Compute billing stops; the disk keeps costing ~$2/month and all state — indexed documents included — survives.

```bash
gcloud compute instances stop "$VM_NAME" --zone="$ZONE"
gcloud compute instances start "$VM_NAME" --zone="$ZONE"    # containers restart automatically
```

The static IP is retained while stopped, so the hostname and certificate keep working.

**Set a budget alert** — the demo URL is public and can spend credits:

```bash
gcloud billing budgets create \
  --billing-account=$(gcloud billing projects describe "$PROJECT_ID" \
      --format="value(billingAccountName)" | cut -d/ -f2) \
  --display-name="RAG chatbot budget" \
  --budget-amount=50USD \
  --threshold-rule=percent=50 \
  --threshold-rule=percent=90
```

### Password-protecting the demo

Essential when serving plain HTTP at a public IP (`SITE_ADDRESS=":80"`) — otherwise anyone who finds the address can upload documents and spend your credits.

Auth is a **file you add or remove**; no Caddyfile edit, no rebuild.

```bash
# 1. Generate a hash
docker run --rm caddy:2-alpine caddy hash-password --plaintext 'choose-a-strong-password'

# 2. Write the snippet — paste the hash verbatim, no escaping
cat > caddy-auth/auth.caddy <<'EOF'
basic_auth {
	reviewer $2a$14$paste-the-generated-hash-here
}
EOF

# 3. Apply
docker compose -f docker-compose.yml -f docker-compose.prod.yml restart caddy
```

**Verify both directions:**

```bash
curl -s -o /dev/null -w "%{http_code}\n" http://localhost/                            # 401
curl -s -o /dev/null -w "%{http_code}\n" -u reviewer:your-password http://localhost/  # 200
```

Turn it off by deleting the file and restarting Caddy. `caddy-auth/*.caddy` is gitignored, so the hash never reaches the repository.

> **Why a file and not an environment variable.** Compose interpolates values from `.env`, and a bcrypt hash is full of `$`: in `$2a$14$pbjb…` it reads `$pbjb…` as an undefined variable and substitutes nothing. The hash arrives truncated and **no password matches** — 401 for everyone, with nothing in the logs to explain it. Caddy reads this file directly, so the hash is taken verbatim.
>
> If Caddy won't start after editing the snippet, the block is malformed (an empty username is rejected outright): `docker compose ... logs caddy | tail`

> Free-trial credits expire after 90 days regardless of use, and the trial does not auto-charge when it ends.

---

## 10. Updating

```bash
cd rag-chatbot && git pull
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
```

Indexed documents persist — they live in the `chroma_data` volume, not the images.

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `/health` shows `llm.ok: false` | Service account missing, or `--scopes=cloud-platform` omitted at VM creation. Scopes cannot be changed on a running VM: `gcloud compute instances stop`, then `gcloud compute instances set-service-account` with the scope, then start. |
| Certificate never issues | DNS not yet propagated, or port 80 blocked. Check `docker compose logs caddy`. Use `tls internal` to unblock while DNS settles. |
| `403 Permission denied` from Vertex | `roles/aiplatform.user` not granted, or the Vertex API not enabled. |
| Model not found | `LLM_MODEL` unavailable in `GCP_LOCATION`. Run `scripts/verify_vertex.py` — it names the likely cause. |
| Site unreachable, containers healthy | Firewall rule missing, or the `rag-chatbot` network tag not applied to the VM. |
| Uploads fail on large files | `MAX_UPLOAD_MB` (default 20) — raise it in `.env` and restart. |

---

## Teardown

```bash
gcloud compute instances delete "$VM_NAME" --zone="$ZONE"
gcloud compute addresses delete rag-chatbot-ip --region="$REGION"
gcloud compute firewall-rules delete allow-rag-web
gcloud iam service-accounts delete \
  "rag-chatbot-sa@${PROJECT_ID}.iam.gserviceaccount.com"
```
