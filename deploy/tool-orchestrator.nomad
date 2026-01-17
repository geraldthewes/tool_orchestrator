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
        image      = "registry.cluster:5000/tool-orchestrator:latest"
        force_pull = true
        ports      = ["http"]
      }

      env {
        # Server configuration (port 8000 mapped via Nomad network)
        SERVER_HOST = "0.0.0.0"

        # Orchestrator LLM
        ORCHESTRATOR_BASE_URL = "http://gx10-d8ce.cluster:8000/v1"
        ORCHESTRATOR_MODEL    = "glm-reap"

        # Reasoning delegate (large model)
        REASONING_LLM_BASE_URL = "http://gx10-d8ce.cluster:8000/v1"
        REASONING_LLM_MODEL    = "glm-reap"

        # Coding delegate
        CODING_LLM_BASE_URL = "http://gx10-d8ce.cluster:8000/v1"
        CODING_LLM_MODEL    = "qwen3-coder"

        # Fast delegate (Ollama)
        FAST_LLM_URL   = "http://ollama.service.consul:11434/api/chat"
        FAST_LLM_MODEL = "nemotron-3-nano"

        # Tools
        SEARXNG_ENDPOINT     = "http://searxng.service.consul:8080/search"
        PYTHON_EXECUTOR_URL  = "http://pyexec.cluster:9999/"

        # Runtime
        LOG_LEVEL               = "INFO"
        MAX_ORCHESTRATION_STEPS = "10"
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
