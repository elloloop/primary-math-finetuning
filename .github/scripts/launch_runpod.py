"""Launch a RunPod GPU pod to run fine-tuning with the GHCR Docker image."""

import json
import os
import sys

import requests

RUNPOD_API_URL = "https://api.runpod.io/graphql"


def launch_pod():
    api_key = os.environ["RUNPOD_API_KEY"]
    hf_token = os.environ.get("HF_TOKEN", "")
    wandb_key = os.environ.get("WANDB_API_KEY", "")
    max_iterations = os.environ.get("MAX_ITERATIONS", "10")
    gpu_type = os.environ.get("GPU_TYPE", "NVIDIA RTX A5000")
    cloud_type = os.environ.get("CLOUD_TYPE", "COMMUNITY")
    image_name = os.environ.get("IMAGE_NAME", "elloloop/primary-math-finetuning")
    registry = os.environ.get("REGISTRY", "ghcr.io")

    docker_image = f"{registry}/{image_name}:latest"

    query = """
    mutation {{
      podFindAndDeployOnDemand(
        input: {{
          name: "primary-math-finetuning"
          imageName: "{image}"
          gpuTypeId: "{gpu_type}"
          cloudType: {cloud_type}
          volumeInGb: 50
          containerDiskInGb: 20
          minVcpuCount: 4
          minMemoryInGb: 16
          gpuCount: 1
          env: [
            {{ key: "HF_TOKEN", value: "{hf_token}" }},
            {{ key: "WANDB_API_KEY", value: "{wandb_key}" }},
            {{ key: "MAX_ITERATIONS", value: "{max_iterations}" }}
          ]
        }}
      ) {{
        id
        name
        imageName
        desiredStatus
      }}
    }}
    """.format(
        image=docker_image,
        gpu_type=gpu_type,
        cloud_type=cloud_type,
        hf_token=hf_token,
        wandb_key=wandb_key,
        max_iterations=max_iterations,
    )

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    response = requests.post(
        RUNPOD_API_URL, json={"query": query}, headers=headers, timeout=60
    )
    response.raise_for_status()
    result = response.json()

    if "errors" in result:
        print(f"RunPod API errors: {json.dumps(result['errors'], indent=2)}")
        sys.exit(1)

    pod = result["data"]["podFindAndDeployOnDemand"]
    pod_id = pod["id"]
    print(f"Pod launched: {pod_id} ({pod['name']})")
    print(f"Image: {pod['imageName']}")
    print(f"Status: {pod['desiredStatus']}")

    # Save pod ID for polling step
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"pod_id={pod_id}\n")

    # Also write to a file for the poll script
    with open("pod_id.txt", "w") as f:
        f.write(pod_id)


if __name__ == "__main__":
    launch_pod()
