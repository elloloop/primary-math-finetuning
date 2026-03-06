"""Resume an existing RunPod pod for fine-tuning.

If RUNPOD_POD_ID is set, resumes that pod. Otherwise creates a new one.
"""

import json
import os
import sys

import requests

RUNPOD_API_URL = "https://api.runpod.io/graphql"


def graphql(api_key: str, query: str) -> dict:
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
    return result


def resume_pod(api_key: str, pod_id: str):
    """Resume a stopped/exited pod."""
    query = """
    mutation {{
      podResume(
        input: {{
          podId: "{pod_id}"
          gpuCount: 1
        }}
      ) {{
        id
        desiredStatus
      }}
    }}
    """.format(pod_id=pod_id)

    result = graphql(api_key, query)
    pod = result["data"]["podResume"]
    print(f"Pod resumed: {pod['id']}")
    print(f"Status: {pod['desiredStatus']}")
    return pod["id"]


def create_pod():
    """Create a new pod (fallback if no pod ID is set)."""
    api_key = os.environ["RUNPOD_API_KEY"]
    hf_token = os.environ.get("HF_TOKEN", "")
    wandb_key = os.environ.get("WANDB_API_KEY", "")
    git_ssh_key = os.environ.get("GIT_SSH_KEY", "")
    experiment = os.environ.get("EXPERIMENT", "default")
    gpu_type = os.environ.get("GPU_TYPE", "NVIDIA A40")
    cloud_type = os.environ.get("CLOUD_TYPE", "COMMUNITY")
    volume_id = os.environ.get("RUNPOD_VOLUME_ID", "")
    data_repo_url = os.environ.get(
        "DATA_REPO_URL", "git@github.com:elloloop/maths-questions-database.git"
    )
    image_name = os.environ.get("IMAGE_NAME", "elloloop/primary-math-finetuning")
    registry = os.environ.get("REGISTRY", "ghcr.io")

    docker_image = f"{registry}/{image_name}/train:latest"

    env_vars = [
        ("HF_TOKEN", hf_token),
        ("WANDB_API_KEY", wandb_key),
        ("EXPERIMENT", experiment),
        ("DATA_REPO_URL", data_repo_url),
        ("START_TENSORBOARD", "true"),
        ("KEEP_ALIVE", "true"),
    ]
    if git_ssh_key:
        env_vars.append(("GIT_SSH_KEY", git_ssh_key))

    env_entries = ", ".join(
        '{{ key: "{k}", value: "{v}" }}'.format(
            k=k, v=v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
        )
        for k, v in env_vars
    )

    volume_clause = (
        f'networkVolumeId: "{volume_id}"' if volume_id else "volumeInGb: 50"
    )

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
          ports: "22/tcp,6006/http"
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

    result = graphql(api_key, query)
    pod = result["data"]["podFindAndDeployOnDemand"]
    pod_id = pod["id"]
    print(f"Pod created: {pod_id} ({pod['name']})")
    print(f"Image: {pod['imageName']}")
    print(f"Status: {pod['desiredStatus']}")
    return pod_id


def main():
    api_key = os.environ["RUNPOD_API_KEY"]
    pod_id = os.environ.get("RUNPOD_POD_ID", "")

    if pod_id:
        print(f"Resuming existing pod {pod_id}...")
        pod_id = resume_pod(api_key, pod_id)
    else:
        print("No RUNPOD_POD_ID set, creating new pod...")
        pod_id = create_pod()

    # Save pod ID for downstream steps
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"pod_id={pod_id}\n")

    with open("pod_id.txt", "w") as f:
        f.write(pod_id)


if __name__ == "__main__":
    main()
