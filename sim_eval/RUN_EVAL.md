# Running sim evals

## DROID

```bash
source .venv/bin/activate

python -m sim_eval.run_eval \
    --policy-type remote-droid \
    --remote-url <URL> \
    -e DroidPutEverythingInBox-v1 \
    -n 50 \
    --max-episode-steps 400 \
    --save-video \
    --max-videos 50
```

## YAM

```bash
source .venv/bin/activate

python -m sim_eval.run_eval \
    --policy-type remote-yam \
    --remote-url <URL> \
    -e BimanualYAMPutEverythingInBox-v1 \
    -n 50 \
    --max-episode-steps 400 \
    --save-video \
    --max-videos 50
```

`<URL>` — either form works, must point at a server for that embodiment (DROID default port 8000, YAM default port 8202 — they are not interchangeable):
- ngrok: `https://<subdomain>.ngrok-free.dev/act`
- local/tunneled: `http://localhost:<port>/act`

## Arguments

| Flag | Use |
|------|-----|
| `--policy-type` / `-p` | `remote-droid` (2 cams) or `remote-yam` (3 cams) |
| `--remote-url` | full `/act` endpoint URL (required) |
| `-e` | ManiSkill env id, e.g. `DroidPutEverythingInBox-v1` |
| `-n` | number of episodes |
| `--max-episode-steps` | per-episode step cap (halve this if evals run too slow) |
| `--save-video` | save rollout videos (default on) |
| `--max-videos` | how many episodes get videos (defaults to 10 — set to `-n`'s value for all) |
| `--remote-request-timeout` | seconds before a server call times out (default 60) |

Results + videos land in `sim_eval/outputs/<timestamp>/`.
