# LSYNCD — develop local, run remote

> If you are an AI coding agent working in this repo, read this first.
> If you are a human, the short version: edit files here; they land on
> Colab ~1 second after every save; run them there.

## Golden rule

- **This checkout is the source of truth.** All development happens here.
- **The remote is for running/testing only.** Sync is one-way local → remote.
- **Never edit code on the remote** — the next sync of that file overwrites it.
- Files created only on the remote (checkpoints, outputs) are safe from
  sync (it never deletes), but a dead session wipes them: pull results
  back promptly (see below).

## Paths

| | |
| --- | --- |
| Local source | '/home/carlesoctav/personal/unsloth-agent' |
| Remote target | 'unsloth_agent:/content/unsloth-agent' |
| SSH host | 'unsloth_agent' (`~/.ssh/colab_config`, managed by `mycolab ssh -s <session>`) |

## Sync daemon (lsyncd)

Run from '/home/carlesoctav/personal/unsloth-agent':

```bash
mycolab sync                        # start (foreground, full sync on startup; Ctrl+C stops it)
tail -f /tmp/lsyncd-unsloth-agent.log     # logs (from another terminal)
cat /tmp/lsyncd-unsloth-agent.status      # pending work
```

Ignored: '.git/', '.venv/', '__pycache__/', '*.pyc'.
'lsyncd.conf.lua' and this file sync too — that is harmless.

## Running / testing on the remote

Colab allows a single concurrent SSH connection; it is shared via
multiplexing (ControlMaster in the managed ssh config), so shells and
syncs coexist. Prefer short-lived commands over a held-open shell:

```bash
echo '!<shell command>' | colab exec   # run anything, e.g. '!python3 train.py --epochs 2'
ssh unsloth_agent                             # interactive shell when really needed
colab ls /content/unsloth-agent                   # list remote files
colab upload <local> <remote>          # one-off push outside the sync
colab download <remote> <local>        # one-off fetch
colab install <pkg>                    # pip install on the runtime
```

GPU sanity check:

```bash
echo '!python3 -c "import torch; print(torch.cuda.is_available())"' | colab exec
```

## Long-running programs (tmux)

If the code must keep running after you disconnect (training, servers),
run it under tmux on the remote: the tmux server survives dropped
connections, and the human can attach later with `ssh unsloth_agent`
then `tmux a -t <name>`.

```bash
echo '!tmux new -d -s train "python3 train.py 2>&1 | tee train.log"' | colab exec -s <session>
echo '!tmux ls' | colab exec -s <session>                                     # list sessions
echo '!tmux capture-pane -p -t train | tail -20' | colab exec -s <session>   # peek at output
echo '!tmux kill-session -t train' | colab exec -s <session>                 # stop it
```

Rules for agents:

- Start tmux via `colab exec` so the program inherits the full
  kernel env (GPU/TPU vars). Always pass the same `-s <session>`
  on every command once more than one session exists.
- Log to a file (`tee`) as well as the pane; poll with
  `capture-pane`, never block waiting on output.
- tmux dies with the VM: checkpoint often and pull outputs back (below).
- One short session name per job (`train`, `eval-x`).

## Fetching results back (one-shot pull)

```bash
mycolab pull   # from '/home/carlesoctav/personal/unsloth-agent': reads lsyncd.conf.lua, syncs remote -> local
```

This overwrites local files with remote versions when you need
checkpoints/outputs back (local-only files are left alone). It runs the
equivalent of:

```bash
rsync -avz --exclude='.git/' --exclude='.venv/' -e ssh unsloth_agent:/content/unsloth-agent/ /home/carlesoctav/personal/unsloth-agent/
```

## Session lifecycle (mycolab / colab)

```bash
mycolab list                 # profiles, * = active
mycolab use <name>           # switch account/workspace
colab new --gpu l4           # fresh VM (run project setup after, if any)
mycolab ssh -s <session>     # Host block + remote setup (tmux, env, hosts)
mycolab usage                # remaining compute-unit credits
colab status | colab sessions
colab stop -s <session>      # release the VM when done
```

After 'colab new' + 'mycolab ssh -s <session>' the remote dir is empty
and the old multiplex master is stale:

```bash
ssh -O exit unsloth_agent   # drop the stale master, if any
# then in the lsyncd terminal: Ctrl+C, run 'mycolab sync' again for a full re-sync
```

## Warnings for agents

- Do not run 'colab new' / 'colab stop' on your own: they spend real
  compute units / destroy the VM. Ask the human first.
- Do not remove ControlMaster from the managed ssh config: without it,
  sync and shells fight over Colab's single connection (HTTP 429).
- The remote can vanish without warning (preemption, idle timeout).
  Irreplaceable outputs must be pulled back, not left in /content.
