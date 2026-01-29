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

      # All configuration comes from Consul KV (pushed via `make push-config`)
      # No Vault integration needed - secrets are in the Consul config
      env {
        CONSUL_HTTP_ADDR = "http://consul.service.consul:8500"
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
