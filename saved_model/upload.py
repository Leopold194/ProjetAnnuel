import requests
import argparse

def upload(file_path, name, url="http://localhost:5000/api/upload"):
    """
    Upload a file to the Flask API.
    Returns a tuple (status_code, response_data).
    """
    with open(file_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(url, files=files, data={'name': name})
    try:
        data = response.json()
    except ValueError:
        data = response.text
    return response.status_code, data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload a file to the Flask API.")
    parser.add_argument("file_path", help="Path to the file to be uploaded")
    parser.add_argument("--name", help="Name associated with the file", default=None)
    parser.add_argument("--url", help="API endpoint URL", default="localhost:5000")
    args = parser.parse_args()

    url = f"http://{args.url}/api/upload"

    status_code, response_data = upload(args.file_path, args.name, url=args.url)

    print(f"Status Code: {status_code}")
    print(f"Response Data: {response_data}")