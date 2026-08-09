# Optional basic-auth snippet

Drop a `.caddy` file here to password-protect the demo. Delete it to turn
protection off. The `Caddyfile` imports `/etc/caddy/auth/*.caddy`, and the glob
matches zero files when the directory is empty — so no edit to the Caddyfile is
needed either way.

**Any `*.caddy` file here is gitignored**, so the password hash never reaches
the repository.

## Enabling

On the server, in the project directory:

```bash
# 1. Generate a hash
docker run --rm caddy:2-alpine caddy hash-password --plaintext 'your-password'

# 2. Write the snippet (paste the hash verbatim — no escaping needed)
cat > caddy-auth/auth.caddy <<'EOF'
basic_auth {
	reviewer $2a$14$paste-the-generated-hash-here
}
EOF

# 3. Apply
docker compose -f docker-compose.yml -f docker-compose.prod.yml restart caddy
```

## Verifying

```bash
curl -s -o /dev/null -w "%{http_code}\n" http://localhost/                      # 401
curl -s -o /dev/null -w "%{http_code}\n" -u reviewer:your-password http://localhost/  # 200
```

If Caddy fails to start, check the syntax — an empty or malformed `basic_auth`
block is rejected at startup:

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml logs caddy | tail
```

## Disabling

```bash
rm caddy-auth/auth.caddy
docker compose -f docker-compose.yml -f docker-compose.prod.yml restart caddy
```

## Why a file rather than an environment variable

A bcrypt hash is full of `$` characters. Docker Compose interpolates values
from `.env`, so `$2a$14$pbjb…` has `$pbjb…` read as an undefined variable and
replaced with nothing — the hash arrives truncated and **no password matches**,
returning 401 for everyone with nothing in the logs to explain it. Caddy reads
this file directly, so the hash is taken verbatim and needs no escaping.
