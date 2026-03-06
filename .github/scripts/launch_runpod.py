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
    git_ssh_key = os.environ.get("GIT_SSH_KEY", "")
    experiment = os.environ.get("EXPERIMENT", "default")
    max_iterations = os.environ.get("MAX_ITERATIONS", "10")
    gpu_type = os.environ.get("GPU_TYPE", "NVIDIA A40")
    cloud_type = os.environ.get("CLOUD_TYPE", "COMMUNITY")
    volume_id = os.environ.get("RUNPOD_VOLUME_ID", "")
    data_repo_url = os.environ.get(
        "DATA_REPO_URL", "git@github.com:elloloop/maths-questions-database.git"
    )
    image_name = os.environ.get("IMAGE_NAME", "elloloop/primary-math-finetuning")
    registry = os.environ.get("REGISTRY", "ghcr.io")

    docker_image = f"{registry}/{image_name}/train:latest"

    # Build env vars list, escaping values for GraphQL string interpolation
    env_vars = [
        ("HF_TOKEN", hf_token),
        ("WANDB_API_KEY", wandb_key),
        ("MAX_ITERATIONS", max_iterations),
        ("EXPERIMENT", experiment),
        ("DATA_REPO_URL", data_repo_url),
        ("START_TENSORBOARD", "true"),
    ]
    if git_ssh_key:
        env_vars.append(("GIT_SSH_KEY", git_ssh_key))

    env_entries = ", ".join(
        '{{ key: "{k}", value: "{v}" }}'.format(
            k=k, v=v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
        )
        for k, v in env_vars
    )

    # Use a persistent network volume if provided, otherwise ephemeral
    volume_clause = ""
    if volume_id:
        volume_clause = f'networkVolumeId: "{volume_id}"'
    else:
        volume_clause = "volumeInGb: 50"

    query = """
    mutation {{
      podFindAndDeployOnDemand(
        input: {{
          name: "math-finetune-{experiment}"
          imageName: "{image}"
          gpuTypeId: "{gpu_type}"
          cloudType: {cloud_type}
          {volume_clause}
          containerDiskInGb: 20
          minVcpuCount: 4
          minMemoryInGb: 16
          gpuCount: 1
          env: [{env_entries}]
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
        volume_clause=volume_clause,
        env_entries=env_entries,
        experiment=experiment,
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
