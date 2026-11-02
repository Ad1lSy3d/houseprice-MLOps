
terraform {
  required_providers {
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.38"
    }
  }
}

provider "kubernetes" {
  config_path    = "~/.kube/config"
  config_context = "minikube"
}

# EstateIQ Kubernetes Deployment
resource "kubernetes_deployment_v1" "estateiq" {
  wait_for_rollout = true

  metadata {
    name      = "estateiq-deployment"
    namespace = "default"

    labels = {
      app = "estateiq"
    }
  }

  spec {
    replicas = var.replicas

    selector {
      match_labels = {
        app = "estateiq"
      }
    }

    template {
      metadata {
        labels = {
          app = "estateiq"
        }
      }

      spec {
        automount_service_account_token = false
        enable_service_links            = false

        container {
          name              = "estateiq"
          image             = "estateiq:latest"
          image_pull_policy = "Never"

          port {
            container_port = 8000
          }

          readiness_probe {
            http_get {
              path = "/health"
              port = 8000
            }

            initial_delay_seconds = 10
            period_seconds        = 5
          }

          liveness_probe {
            http_get {
              path = "/health"
              port = 8000
            }

            initial_delay_seconds = 20
            period_seconds        = 10
          }
        }
      }
    }
  }
}

# EstateIQ Kubernetes Service
resource "kubernetes_service_v1" "estateiq" {
  wait_for_load_balancer = false

  metadata {
    name      = "estateiq-service"
    namespace = "default"

    labels = {
      app = "estateiq"
    }
  }

  spec {
    selector = {
      app = "estateiq"
    }

    port {
      name        = "http"
      port        = 8000
      target_port = 8000
      node_port   = 30080
      protocol    = "TCP"
    }

    type = "NodePort"
  }
}
