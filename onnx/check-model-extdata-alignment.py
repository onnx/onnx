import os
import sys
import platform
import subprocess
import urllib.request
import zipfile
import re
import csv
import shutil
from pathlib import Path

def download_file(url, dest):
    print(f"Downloading {url}...")
    urllib.request.urlretrieve(url, dest)
    print(f"Downloaded {url} to {dest}")

def extract_zip(file_path, extract_to, path_filter=None):
    print(f"Extracting {file_path} to {extract_to}...")
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        for member in zip_ref.namelist():
            if path_filter is None or member.startswith(path_filter):
                # Create the target path by stripping the path_filter from the member
                target_path = os.path.join(extract_to, os.path.relpath(member, path_filter) if path_filter else member)
                # Ensure the directory exists
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                # Extract the file
                with zip_ref.open(member) as source, open(target_path, "wb") as target:
                    shutil.copyfileobj(source, target)
    print(f"Extracted {file_path}")

def ensure_protoc(version):
    system = platform.system()
    path_filter = "bin/"
    if system == "Windows":
        protoc_url = f"https://github.com/protocolbuffers/protobuf/releases/download/v{version}/protoc-{version}-win64.zip"
        protoc_exe = "protoc.exe"
    elif system == "Linux":
        protoc_url = f"https://github.com/protocolbuffers/protobuf/releases/download/v{version}/protoc-{version}-linux-x86_64.zip"
        protoc_exe = "protoc"
    else:
        raise RuntimeError("Unsupported OS")

    if not shutil.which(protoc_exe):
        protoc_zip = Path(protoc_exe).stem + ".zip"
        download_file(protoc_url, protoc_zip)
        extract_zip(protoc_zip, '.', path_filter)
        os.remove(protoc_zip)
        protoc_exe = protoc_exe
    return protoc_exe

def ensure_onnx_proto():
    onnx_proto_url = "https://github.com/onnx/onnx/raw/main/onnx/onnx.proto"
    onnx_proto = "onnx.proto"
    if not Path(onnx_proto).exists():
        download_file(onnx_proto_url, onnx_proto)
    return onnx_proto

def run_protoc(input_file, protoc_exe, onnx_proto, output_file):
    command = [protoc_exe, "--proto_path=.", "--decode=onnx.ModelProto", onnx_proto]
    print(f"Running command: {' '.join(command)} < {input_file} > {output_file}")
    with open(output_file, 'w') as out_f:
        subprocess.run(command, stdin=open(input_file, 'r'), stdout=out_f)
    print(f"Protoc output written to {output_file}")

def parse_initializers(text_file, csv_file):
    with open(text_file, 'r') as f:
        content = f.read()

    # Fast check for the presence of "data_location: EXTERNAL"
    if "data_location: EXTERNAL" not in content:
        print("No instances of 'data_location: EXTERNAL' found in the file. Are you sure this a model with external data - https://github.com/onnx/onnx/blob/main/docs/ExternalData.md")
        return False

    # Regular expression to match the initializers with external data
    initializer_regex = re.compile(
        r'initializer \{\s*(.*?)\s*\}\s*data_location: EXTERNAL',
        re.DOTALL
    )

    # Regular expression to match the name, location, offset, and length
    fields_regex = re.compile(
        r'name: "(.*?)".*?'
        r'key: "location"\s*value: "(.*?)".*?'
        r'key: "offset"\s*value: "(.*?)".*?'
        r'key: "length"\s*value: "(.*?)"',
        re.DOTALL
    )

    initializers = []

    def check_4k_alignment(offset):
        return int(offset) % 4096 == 0
    
    def check_64k_alignment(offset):
        return int(offset) % 65536 == 0

    matches = list(initializer_regex.finditer(content))
    print(f"Found {len(matches)} initializers")  # Debug output

    for initializer_match in matches:
        initializer_text = initializer_match.group(1)
        # print(f"Initializer text: {initializer_text}")  # Debug output
        fields_match = fields_regex.search(initializer_text)
        if fields_match:
            name, location, offset, length = fields_match.groups()
            is_4k_aligned = check_4k_alignment(offset)
            is_64k_aligned = check_64k_alignment(offset)
            initializers.append({
                "name": name,
                "location": location,
                "offset": offset,
                "offset_hex": hex(int(offset)),
                "length": length,
                "length_hex": hex(int(length)),
                "is4kaligned": is_4k_aligned,
                "is64kaligned": is_64k_aligned
            })
            # print(f"Parsed initializer: {name}, {location}, {offset}, {offset_hex}, {length}, {length_hex}, {is_4k_aligned}, {is_64k_aligned}")  # Debug output

    if not initializers:
        print("No initializers with external data found.")  # Debug output

    with open(csv_file, 'w', newline='') as csvfile:
        fieldnames = ["name", "location", "offset", "offset_hex", "length", "length_hex", "is4kaligned", "is64kaligned"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for initializer in initializers:
            writer.writerow(initializer)

    return True
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check-model-extdata-alignment <model_with_extdata.onnx>")
        sys.exit(1)

    onnx_file = sys.argv[1]
    if not os.path.isfile(onnx_file):
        print(f"Error: File '{onnx_file}' not found.")
        sys.exit(1)

    protoc_version = "25.4"
    protoc_exe = ensure_protoc(protoc_version)
    onnx_proto = ensure_onnx_proto()
    protoc_output_file = onnx_file + ".txt"
    run_protoc(onnx_file, protoc_exe, onnx_proto, protoc_output_file)

    csv_file = onnx_file + ".csv"
    print(f"Parsing initializers in {protoc_output_file}")
    if (parse_initializers(protoc_output_file, csv_file)):
        print(f"CSV file created: {csv_file}")
