# Policy servers — quick ops

## 1. Submit a checkpoint

**Stage2 (locally-trained DROID ckpt, port 8100)**
```bash
# edit CKPT in droid_stage2_server.pbs first, e.g.:
#   CKPT=/scratch/shailesh.xml/outputs/droid_goal_prior/seed_1000/stage2_omp/checkpoints/last
qsub droid_stage2_server.pbs
qstat -f <jobid> | grep exec_host   # node it landed on
```

**MolmoAct2-DROID (released ckpt, port 8000)**
```bash
qsub droid_release_server.pbs
qstat -f <jobid> | grep exec_host
```

**MolmoAct2-BimanualYAM (released ckpt, port 8202)**
```bash
qsub yam_release_server.pbs
qstat -f <jobid> | grep exec_host
```

## 2. Forward with ngrok

```bash
tmux kill-session -t ngrok-droid 2>/dev/null

# point at whatever node:port the server landed on in step 1
tmux new -d -s ngrok-droid \
  "ngrok http http://<node>:<port> --log=stdout > ~/ngrok_droid.log 2>&1"

# get the public URL
curl -s http://localhost:4040/api/tunnels | python3 -c \
  "import sys,json;print(json.load(sys.stdin)['tunnels'][0]['public_url'])"
```
