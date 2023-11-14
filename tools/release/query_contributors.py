import requests
import argparse

parser = argparse.ArgumentParser(description='Query pull requests from GitHub API')
parser.add_argument('--token', type=str, required=False, help='GitHub API token')
parser.add_argument('--start-date', type=str, required=True, help='Start date in YYYY-MM-DD format')
parser.add_argument('--end-date', type=str, required=True, help='End date in YYYY-MM-DD format')
parser.add_argument('--output', type=str, required=False, help='output file name')
args = parser.parse_args()

url = 'https://api.github.com/repos/onnx/onnx/commits'
headers = {'Authorization': f'token {args.token}', 'Accept': 'application/vnd.github.v3+json'} if args.token else {'Accept': 'application/vnd.github.v3+json'}
params = {'sha': 'main', 'since': args.start_date, 'until': args.end_date, 'per_page': 100}

contributors = set()
page = 1
while True:
    print(f'Querying page {page}...')
    params['page'] = page
    response = requests.get(url, headers=headers, params=params)
    if not response.ok:
        print(f'Error: {response.status_code} - {response.text}')
        break
    page_commits = response.json()
    if not page_commits:
        break
    for commit in page_commits:
        email = commit['commit']['author']['email']
        username = email.split('@')[0]
        contributors.add(username)
    page += 1

for contributor in contributors:
    print(f'@{contributor}', end=', ')
