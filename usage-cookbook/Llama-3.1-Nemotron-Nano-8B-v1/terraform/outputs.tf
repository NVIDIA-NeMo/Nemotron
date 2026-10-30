output "cluster_id" {
  description = "OKE cluster OCID."
  value       = module.oke.cluster_id
}

output "cluster_endpoints" {
  description = "Cluster endpoints; private endpoint should be used."
  value       = module.oke.cluster_endpoints
}

output "apiserver_private_host" {
  description = "Private control-plane host."
  value       = module.oke.apiserver_private_host
}

output "vcn_id" {
  description = "VCN used by the Nemotron deployment."
  value       = module.oke.vcn_id
}

output "control_plane_subnet_id" {
  description = "Private control-plane subnet."
  value       = module.oke.control_plane_subnet_id
}

output "worker_subnet_id" {
  description = "Private worker subnet."
  value       = module.oke.worker_subnet_id
}

output "oci_bastion_id" {
  description = "OCI Bastion service OCID for creating private sessions."
  value       = oci_bastion_bastion.oci_bastion.id
}
