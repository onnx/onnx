import openai
import requests
import argparse

# python query_pullrequests.py --token <token> --start-date 2021-01-01 --end-date 2021-12-31 --temperature 0.5 --model-id text-davinci-002 --api-key <your OpenAI API key>
parser = argparse.ArgumentParser(description='Query pull requests from GitHub API')
parser.add_argument('--token', type=str, required=False, help='GitHub API token')
parser.add_argument('--start-date', type=str, required=True, help='Start date in YYYY-MM-DD format')
parser.add_argument('--end-date', type=str, required=True, help='End date in YYYY-MM-DD format')
parser.add_argument('--temperature', type=float, default=0.5, help='OpenAI GPT-3 sampling temperature')
parser.add_argument('--model-id', type=str, default='text-davinci-002', help='OpenAI GPT-3 model ID')
parser.add_argument('--api-key', type=str, required=False, help='OpenAI API key')
parser.add_argument('--output', type=str, required=False, help='output file name')
args = parser.parse_args()

openai.api_key = args.api_key
model_id = args.model_id

def classify_pull_request(pr):
    prompt = f'Classify the following pull request:\n\nTitle: {pr["title"]}\nBody: {pr["body"]}\n\nCategory: '
    response = openai.Completion.create(
        engine=model_id,
        prompt=prompt,
        temperature=args.temperature,
        max_tokens=1,
        n=1,
        stop=None,
        timeout=10,
    )
    category = response.choices[0].text.strip()
    return category

url = 'https://api.github.com/repos/onnx/onnx/pulls'
headers = {'Authorization': f'token {args.token}', 'Accept': 'application/vnd.github.v3+json'} if args.token else {'Accept': 'application/vnd.github.v3+json'}
params = {'state': 'all', 'per_page': 100}

pull_requests = []
page = 1
while True:
    print(f'Querying page {page}...')
    params['page'] = page
    response = requests.get(url, headers=headers, params=params)
    if not response.ok:
        print(f'Error: {response.status_code} - {response.text}')
        break
    page_pull_requests = response.json()
    if not page_pull_requests:
        break
    pull_requests.extend(page_pull_requests)
    page += 1

with open(args.output, "a", encoding="utf-8") as f:
    for pr in pull_requests:
        if pr['merged_at'] is not None and args.start_date <= pr['merged_at'][:10] <= args.end_date:
            title = pr['title']
            link = pr['html_url']
            pr_number = pr['number']
            if model_id and openai.api_key:
                category = classify_pull_request(pr)
                print(f'{category}: {title} [PR#{pr_number}]({link})', file=f)
            else:
                prompt = f'Classify the following pull request  as one of:"Feature", "Bugfix", "Shape Inference" "Documentation" and "Other" Category:\n\nTitle: {pr["title"]}\nBody: {pr["body"]}\n\nCategory: '
                print(f'{title} [PR#{pr_number}]({link})')
                print(f'{title} [PR#{pr_number}]({link}) ------ {prompt} ______', file=f)
                # print(f'{title} [PR#{pr_number}]({link}) ------ {prompt} ______')
