import requests
from requests.auth import HTTPBasicAuth
import json

# ========================
# CONFIG
# ========================
JENKINS_URL = "http://localhost:8080"
USERNAME = "admin"
API_TOKEN = "118951060eba544d3edb910da2d26d974f"

JOB_NAME = "nextjs-cicd"  


# ========================
# HELPER FUNCTION
# ========================
def pretty(data):
    print(json.dumps(data, indent=4))


# ========================
# 1. GET ALL JOBS (name, url, color)
# ========================
def get_all_jobs():
    url = f"{JENKINS_URL}/api/json?tree=jobs[name,url,color]"
    r = requests.get(url, auth=HTTPBasicAuth(USERNAME, API_TOKEN))
    return r.json()


# ========================
# 2. GET ONLY JOB NAMES
# ========================
def get_job_names():
    url = f"{JENKINS_URL}/api/json?tree=jobs[name]"
    r = requests.get(url, auth=HTTPBasicAuth(USERNAME, API_TOKEN))
    return r.json()


# ========================
# 3. GET JOB DESCRIPTION
# ========================
def get_job_description(job):
    url = f"{JENKINS_URL}/job/{job}/description"
    r = requests.get(url, auth=HTTPBasicAuth(USERNAME, API_TOKEN))
    return r.text


# ========================
# 4. GET FULL JOB DETAILS
# ========================
def get_full_job_details(job):
    url = f"{JENKINS_URL}/job/{job}/api/json"
    r = requests.get(url, auth=HTTPBasicAuth(USERNAME, API_TOKEN))
    return r.json()


# ========================
# 5. GET ALL BUILD NUMBERS
# ========================
def get_build_numbers(job):
    url = f"{JENKINS_URL}/job/{job}/api/json?tree=builds[number]"
    r = requests.get(url, auth=HTTPBasicAuth(USERNAME, API_TOKEN))
    return r.json()


# ========================
# 6. GET LAST BUILD DETAILS (includes commit ID)
# ========================
def get_last_build_details(job):
    url = f"{JENKINS_URL}/job/{job}/lastBuild/api/json"
    r = requests.get(url, auth=HTTPBasicAuth(USERNAME, API_TOKEN))
    return r.json()


# ========================
# 7. GET FULL JOB DETAILS 
# ========================
def get_full_job_details_pretty(job):
    url = f"{JENKINS_URL}/job/{job}/api/json?pretty=true"
    r = requests.get(url, auth=HTTPBasicAuth(USERNAME, API_TOKEN))
    return r.json()


# ========================
# MAIN: RUN EVERYTHING
# ========================
if __name__ == "__main__":

    print("\n====== 1. ALL JOBS (name, url, color) ======")
    pretty(get_all_jobs())

    print("\n====== 2. ONLY JOB NAMES ======")
    pretty(get_job_names())

    print("\n====== 3. JOB DESCRIPTION ======")
    print(get_job_description(JOB_NAME))

    print("\n====== 4. FULL JOB DETAILS ======")
    pretty(get_full_job_details(JOB_NAME))

    print("\n====== 5. ALL BUILD NUMBERS ======")
    pretty(get_build_numbers(JOB_NAME))

    print("\n====== 6. LAST BUILD DETAILS ======")
    pretty(get_last_build_details(JOB_NAME))

    print("\n====== 7. FULL JOB DETAILS (PRETTY) ======")
    pretty(get_full_job_details_pretty(JOB_NAME))
