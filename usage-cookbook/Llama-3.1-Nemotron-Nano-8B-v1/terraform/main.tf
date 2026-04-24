provider "oci" {
  config_file_profile = var.config_file_profile
  tenancy_ocid        = var.tenancy_ocid
  region              = var.region
}

locals {
  common_tags = merge(var.freeform_tags, {
    model      = "nvidia/Llama-3.1-Nemotron-Nano-8B-v1"
    deployment = "private-oke"
    region     = var.region
  })
}

module "oke" {
  source  = "oracle-terraform-modules/oke/oci"
  version = "5.4.1"

  providers = {
    oci.home = oci
  }

  tenancy_id     = var.tenancy_ocid
  compartment_id = var.compartment_ocid
  region         = var.region

  cluster_name                      = var.cluster_name
  kubernetes_version                = var.kubernetes_version
  cluster_type                      = "enhanced"
  cni_type                          = "flannel"
  pods_cidr                         = var.pods_cidr
  services_cidr                     = var.services_cidr
  vcn_cidrs                         = var.vcn_cidrs
  ssh_public_key                    = file(var.ssh_public_key_path)
  output_detail                     = true
  create_vcn                        = true
  create_bastion                    = false
  create_operator                   = false
  control_plane_is_public           = false
  assign_public_ip_to_control_plane = false
  worker_is_public                  = false
  allow_worker_internet_access      = true
  allow_pod_internet_access         = true
  allow_worker_ssh_access           = false
  preferred_load_balancer           = "internal"
  load_balancers                    = "internal"
  freeform_tags                     = { all = local.common_tags }

  subnets = {
    cp = {
      create  = "always"
      newbits = 13
      netnum  = 2
    }
    workers = {
      create  = "always"
      newbits = 2
      netnum  = 1
    }
    pods = {
      create  = "always"
      newbits = 2
      netnum  = 2
    }
    int_lb = {
      create  = "always"
      newbits = 11
      netnum  = 16
    }
    pub_lb = {
      create = "never"
    }
    bastion = {
      create = "never"
    }
    operator = {
      create = "never"
    }
  }

  worker_pool_mode = "node-pool"
  worker_pool_size = 1
  worker_pools = {
    cpu = {
      size             = var.cpu_pool_size
      shape            = var.cpu_shape
      ocpus            = var.cpu_ocpus
      memory           = var.cpu_memory_gbs
      boot_volume_size = 100
      assign_public_ip = false
      create           = true
    }
    gpu = {
      size             = var.gpu_pool_size
      shape            = var.gpu_shape
      boot_volume_size = var.gpu_boot_volume_size
      assign_public_ip = false
      create           = true
      placement_ads    = var.gpu_placement_ads
    }
  }
}

resource "oci_bastion_bastion" "oci_bastion" {
  compartment_id               = var.compartment_ocid
  bastion_type                 = "STANDARD"
  target_subnet_id             = module.oke.worker_subnet_id
  client_cidr_block_allow_list = var.bastion_client_cidrs
  max_session_ttl_in_seconds   = 10800
  name                         = "${var.cluster_name}-bastion"
  freeform_tags                = local.common_tags
}
