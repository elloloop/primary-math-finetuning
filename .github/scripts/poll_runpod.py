"""Poll a RunPod pod until training completes, then terminate the pod."""

import json
import os
import sys
import time

import requests

RUNPOD_API_URL = "https://api.runpod.io/graphql"
POLL_INTERVAL_SECONDS = 60


def get_pod_status(api_key: str, pod_id: str) -> dict:
    query = """
    query {{
      pod(input: {{ podId: "{pod_id}" }}) {{
        id
        name
        desiredStatus
        lastStatusChange
        runtime {{
          uptimeInSeconds
          gpus {{
            id
            gpuUtilPercent
            memoryUtilPercent
          }}
        }}
      }}
    }}
    """.format(pod_id=pod_id)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    response = requests.post(
        RUNPOD_API_URL, json={"query": query}, headers=headers, timeout=30
    )
    response.raise_for_status()
    return response.json()


def terminate_pod(api_key: str, pod_id: str):
    query = """
    mutation {{
      podTerminate(input: {{ podId: "{pod_id}" }})
    }}
    """.format(pod_id=pod_id)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    response = requests.post(
        RUNPOD_API_URL, json={"query": query}, headers=headers, timeout=30
    )
    response.raise_for_status()
    print(f"Pod {pod_id} terminated.")


def poll():
    api_key = os.environ["RUNPOD_API_KEY"]

    try:
        with open("pod_id.txt") as f:
            pod_id = f.read().strip()
    except FileNotFoundError:
        print("pod_id.txt not found. Was the launch step successful?")
        sys.exit(1)

    print(f"Polling pod {pod_id}...")

    while True:
        try:
            result = get_pod_status(api_key, pod_id)
        except Exception as e:
            print(f"Error polling: {e}. Retrying in {POLL_INTERVAL_SECONDS}s...")
            time.sleep(POLL_INTERVAL_SECONDS)
            continue

        if "errors" in result:
            print(f"API errors: {json.dumps(result['errors'], indent=2)}")
            # Pod may have been terminated after completing
            if any("not found" in str(e).lower() for e in result["errors"]):
                print("Pod no longer exists. Training likely completed.")
                break
            time.sleep(POLL_INTERVAL_SECONDS)
            continue

        pod = result["data"]["pod"]
        if pod is None:
            print("Pod not found. Training completed and pod was terminated.")
            break

        status = pod["desiredStatus"]
        runtime = pod.get("runtime")
        uptime = runtime.get("uptimeInSeconds", 0) if runtime else 0

        print(
            f"Status: {status} | Uptime: {uptime // 60}m {uptime % 60}s",
            flush=True,
        )

        # Pod completed (EXITED status means the container finished)
        if status == "EXITED":
            print("Training completed. Terminating pod...")
            terminate_pod(api_key, pod_id)
            break

        time.sleep(POLL_INTERVAL_SECONDS)

    print("Done.")


if __name__ == "__main__":
    poll()
