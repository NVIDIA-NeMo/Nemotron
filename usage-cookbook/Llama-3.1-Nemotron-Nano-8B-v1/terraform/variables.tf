variable "tenancy_ocid" {
  description = "OCI tenancy OCID."
  type        = string
}

variable "compartment_ocid" {
  description = "Compartment where the OKE cluster and Bastion service will be created."
  type        = string
}

variable "region" {
  description = "OCI region for the deployment."
  type        = string
  default     = "us-phoenix-1"
}

variable "config_file_profile" {
  description = "OCI CLI config profile name."
  type        = string
  default     = "DEFAULT"
}

variable "cluster_name" {
  description = "Name prefix for the private Nemotron OKE deployment."
  type        = string
  default     = "nemotron-oci-phx"
}

variable "ssh_public_key_path" {
  description = "Path to the OpenSSH public key file used for private worker access."
  type        = string
}

variable "vcn_cidrs" {
  description = "VCN CIDR blocks for the deployment."
  type        = list(string)
  default     = ["10.0.0.0/16"]
}

variable "pods_cidr" {
  description = "Kubernetes pods CIDR."
  type        = string
  default     = "10.244.0.0/16"
}

variable "services_cidr" {
  description = "Kubernetes services CIDR."
  type        = string
  default     = "10.96.0.0/16"
}

variable "kubernetes_version" {
  description = "OKE Kubernetes version."
  type        = string
  default     = "v1.33.1"
}

variable "cpu_pool_size" {
  description = "Number of CPU worker nodes."
  type        = number
  default     = 1
}

variable "cpu_shape" {
  description = "Shape for the CPU worker pool."
  type        = string
  default     = "VM.Standard.E5.Flex"
}

variable "cpu_ocpus" {
  description = "OCPUs for each CPU worker if using a flex shape."
  type        = number
  default     = 2
}

variable "cpu_memory_gbs" {
  description = "Memory in GB for each CPU worker if using a flex shape."
  type        = number
  default     = 16
}

variable "gpu_pool_size" {
  description = "Number of GPU worker nodes."
  type        = number
  default     = 1
}

variable "gpu_shape" {
  description = "Shape for the GPU worker pool."
  type        = string
  default     = "VM.GPU.A10.1"
}

variable "gpu_boot_volume_size" {
  description = "Boot volume size for GPU workers."
  type        = number
  default     = 200
}

variable "gpu_placement_ads" {
  description = "Availability domains to target for the GPU node pool. Phoenix AD-2 is `[2]`."
  type        = list(number)
  default     = [2]
}

variable "bastion_client_cidrs" {
  description = "CIDR blocks allowed to create OCI Bastion sessions."
  type        = list(string)
}

variable "freeform_tags" {
  description = "Optional freeform tags."
  type        = map(string)
  default     = {}
}
