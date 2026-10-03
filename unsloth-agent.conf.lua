-- lsyncd config: live-sync this dir -> unsloth-agent:/content/unsloth-agent.
-- One-way local -> remote. Local checkout is the source of truth.
--
-- Config: unsloth-agent.conf.lua (one per session; scaffold another with
--   'mycolab lsyncd -s <other> /home/carlesoctav/personal/unsloth-agent /content/unsloth-agent')
-- Start:  mycolab sync unsloth-agent.conf.lua   (from /home/carlesoctav/personal/unsloth-agent)
-- Runs in the foreground: stop it with Ctrl+C.
-- Logs:   tail -f /tmp/lsyncd-unsloth-agent-unsloth-agent.log   (from another terminal)
--
-- NOTE 1: Colab allows only ONE 'colab ssh' connection per runtime. This
-- is handled via SSH multiplexing (ControlMaster in ~/.ssh/colab_config,
-- managed by 'mycolab ssh -s <session>'): interactive shells and lsyncd's
-- rsync share one connection instead of tripping HTTP 429 against each other.
--
-- NOTE 2: after 'colab new' + 'mycolab ssh -s <session>' (fresh VM, empty remote dir):
--   ssh -O exit unsloth-agent   # drop the stale multiplex master, if any
-- then restart lsyncd (Ctrl+C, run again) so its startup full-sync
-- repopulates the new VM.

settings {
    logfile    = "/tmp/lsyncd-unsloth-agent-unsloth-agent.log",
    statusFile = "/tmp/lsyncd-unsloth-agent-unsloth-agent.status",
    pidfile    = "/tmp/lsyncd-unsloth-agent-unsloth-agent.pid",
    nodaemon   = true,    -- foreground: stop with Ctrl+C
    insist     = true,   -- keep retrying across transient SSH failures
}

sync {
    default.rsyncssh,
    source    = "/home/carlesoctav/personal/unsloth-agent",
    host      = "unsloth-agent", -- managed by 'mycolab ssh -s <session>'
    targetdir = "/content/unsloth-agent",
    delay     = 1,

    exclude = {
        ".git/",
        ".venv/",
        "__pycache__/",
        "*.pyc",
    },

    rsync = {
        archive  = true,
        compress = true,
        -- NOTE: no 'delete = true' on purpose: files produced remotely
        -- (checkpoints, logs, results) must survive local syncs.
    },
}
