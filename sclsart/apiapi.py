import base64
import time
from tencentcloud.common import credential
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
from tencentcloud.tat.v20201028 import tat_client, models

# 1. Configure Authentication
secret_id = "YOUR_SECRET_ID"
secret_key = "YOUR_SECRET_KEY"
region = "ap-guangzhou"  # e.g., ap-guangzhou, ap-singapore
cvm_instance_id = "ins-xxxxxxxx" # The ID of your target CVM

cred = credential.Credential(secret_id, secret_key)
client = tat_client.TatClient(cred, region)

# 2. Prepare Your Code
# This is the script you want to run on the CVM.
my_script = """#!/bin/bash
echo "Starting automated task..."
# Insert your actual code logic, file downloads, or execution commands here
python3 -c "print('Hello from the Tencent Cloud API!')" > /tmp/api_output.log
echo "Task complete."
"""

# The TAT API requires the script content to be Base64 encoded
encoded_script = base64.b64encode(my_script.encode("utf-8")).decode("utf-8")

try:
    # 3. Create the Execution Request
    req = models.RunCommandRequest()
    req.CommandName = "AppDeploymentScript"
    req.Content = encoded_script
    req.InstanceIds = [cvm_instance_id]
    req.CommandType = "SHELL"  # Use POWERSHELL for Windows
    req.WorkingDirectory = "/root"
    req.Timeout = 60 # Timeout in seconds

    # 4. Execute the API Call
    resp = client.RunCommand(req)
    invocation_id = resp.InvocationId
    print(f"Command triggered successfully! Invocation ID: {invocation_id}")

except TencentCloudSDKException as err:
    print(f"API Error: {err}")