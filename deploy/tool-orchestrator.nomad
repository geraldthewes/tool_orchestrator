job "tool-orchestrator" {
  datacenters = ["cluster"]
  type        = "service"

  group "tool-orchestrator" {
    count = 1

    constraint {
      attribute = "${attr.cpu.arch}"
      value     = "amd64"
    }

    network {
      port "http" {
        to = 8000
      }
    }

    task "tool-orchestrator" {
      driver = "docker"

      config {
        image       = "registry.cluster:5000/tool-orchestrator:latest"
        force_pull  = true
        ports       = ["http"]
        dns_servers = ["10.0.1.12", "10.0.1.13"]
      }

      vault {
        policies = ["tool-orchestrator-policy"]
      }

      # Langfuse secrets from Vault
      template {
        data = <<EOH
[[ with secret "secret/data/tool-orchestrator/langfuse" ]]
LANGFUSE_PUBLIC_KEY=[[ .Data.data.public_key ]]
LANGFUSE_SECRET_KEY=[[ .Data.data.secret_key ]]
[[ end ]]
EOH
        destination     = "secrets/langfuse.env"
        env             = true
        left_delimiter  = "[["
        right_delimiter = "]]"
      }

      env {
        # Server configuration (port 8000 mapped via Nomad network)
        SERVER_HOST = "0.0.0.0"

        # Orchestrator LLM
        ORCHESTRATOR_BASE_URL = "http://gpu005.cluster:8001/v1"
        ORCHESTRATOR_MODEL    = "nvidia/Nemotron-Orchestrator-8B"

        # Reasoning delegate (large model)
        REASONING_LLM_BASE_URL = "http://gx10-d8ce.cluster:8000/v1"
        REASONING_LLM_MODEL    = "glm-reap"

        # Coding delegate
        CODING_LLM_BASE_URL = "http://vllm.cluster:8000/v1"
        CODING_LLM_MODEL    = "qwen3-coder"

        # Fast delegate (Ollama)
        FAST_LLM_URL   = "http://ollama.cluster:11434/v1"
        FAST_LLM_MODEL = "uaysk0327/nemotron-3-nano:30b-q4_k_xl"

        # Tools
        SEARXNG_ENDPOINT     = "http://searxng.cluster:9999/search"
        PYTHON_EXECUTOR_URL  = "http://pyexec.cluster:9999/"

        # Runtime
        LOG_LEVEL               = "INFO"
        MAX_ORCHESTRATION_STEPS = "10"

        # Langfuse (non-sensitive config)
        LANGFUSE_HOST = "https://langfuse.cluster"
      }

      resources {
        cpu    = 500
        memory = 512
      }

      service {
        name = "tool-orchestrator"
        port = "http"
        tags = [
          "urlprefix-/tool-orchestrator strip=/tool-orchestrator",
          "urlprefix-tool-orchestrator.cluster:9999/"
        ]

        check {
          type     = "http"
          path     = "/health"
          interval = "10s"
          timeout  = "2s"
        }
      }
    }
  }
}
