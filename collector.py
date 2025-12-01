import os
import json
import time
import requests
from requests.auth import HTTPBasicAuth

JENKINS_URL = "http://localhost:8080"
USERNAME = "admin"
API_TOKEN = "118951060eba544d3edb910da2d26d974f"

DATA_PATH = "data/jenkins_data.jsonl"
os.makedirs("data", exist_ok=True)


# -------------------------------------
# GET FULL JOB JSON
# -------------------------------------
def get_full_job(job):
    url = f"{JENKINS_URL}/job/{job}/api/json"
    r = requests.get(url, auth=HTTPBasicAuth(USERNAME, API_TOKEN))
    return r.json()


# -------------------------------------
# GET DETAILS OF ONE BUILD
# -------------------------------------
def get_build_details(job, build_number):
    url = f"{JENKINS_URL}/job/{job}/{build_number}/api/json"
    r = requests.get(url, auth=HTTPBasicAuth(USERNAME, API_TOKEN))
    return r.json()


# -------------------------------------
# Extract commit SHA
# -------------------------------------
def extract_commit(data):
    for action in data.get("actions", []):
        if "_class" in action and "BuildData" in action["_class"]:
            rev = action.get("lastBuiltRevision", {})
            return rev.get("SHA1", "")
    return ""


# -------------------------------------
# Convert JSON → clean text for RAG
# -------------------------------------
def extract_doc(job, build_json):
    text = f"""
Job: {job}
Build Number: {build_json['number']}
Result: {build_json['result']}
Timestamp: {build_json['timestamp']}
Duration: {build_json['duration']}
URL: {build_json['url']}
Commit: {extract_commit(build_json)}
"""
    return text


# -------------------------------------
# Read existing builds in JSON
# -------------------------------------
def load_existing_build_numbers():
    if not os.path.exists(DATA_PATH):
        return set()

    builds = set()
    with open(DATA_PATH) as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                builds.add(rec["build_number"])

    return builds


# -------------------------------------
# Append new builds ONLY
# -------------------------------------
def append_new_builds(job="nextjs-cicd"):
    print("\n🔄 Checking Jenkins for new builds...")

    existing = load_existing_build_numbers()
    job_json = get_full_job(job)
    builds = job_json.get("builds", [])

    new_count = 0

    with open(DATA_PATH, "a") as f:
        for b in builds:
            build_no = b["number"]

            if build_no in existing:
                continue  # skip old builds

            # Fetch only new build info
            details = get_build_details(job, build_no)

            doc = {
                "job": job,
                "build_number": build_no,
                "result": details.get("result"),
                "timestamp": details.get("timestamp"),
                "duration": details.get("duration"),
                "commit": extract_commit(details),
                "url": details.get("url"),
                "text": extract_doc(job, details),
                "raw": details
            }

            f.write(json.dumps(doc) + "\n")
            new_count += 1

    print(f"🟢 New builds added: {new_count}")
    return new_count


# -------------------------------------
# Auto-refresh Loop (run forever)
# -------------------------------------
def main():
    print("🚀 Auto-refresh Jenkins Collector started.")
    print("It will check every 10 seconds for new builds.\n")

    while True:
        append_new_builds()
        time.sleep(10)  # check every 10 secs


if __name__ == "__main__":
    main()
