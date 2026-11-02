
output "deployment_name" {
  value = kubernetes_deployment_v1.estateiq.metadata[0].name
}

output "service_name" {
  value = kubernetes_service_v1.estateiq.metadata[0].name
}

output "service_node_port" {
  value = kubernetes_service_v1.estateiq.spec[0].port[0].node_port
}
