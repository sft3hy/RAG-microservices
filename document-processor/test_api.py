# test_api.py

import requests
import os
import json

# --- Configuration ---

# The URL where your FastAPI application is running.
# This assumes you are running it locally on the default port.
API_URL = "http://127.0.0.1:8002/process-document/"

# The path to the test document.
# This constructs a path relative to the location of this script.
TEST_DOC_DIR = "test_docs"
FILE_NAME = "sls-reference-guide-2022-v2-508-0.pdf"
FILE_PATH = os.path.join(TEST_DOC_DIR, FILE_NAME)


def run_test():
    """
    Sends the specified PDF file to the document processing API and prints the response.
    """
    print(f"--- Starting API Test ---")
    print(f"Target API Endpoint: {API_URL}")
    print(f"File to be sent: {FILE_PATH}\n")

    # 1. Verify that the test file actually exists before we try to send it.
    if not os.path.exists(FILE_PATH):
        print(f"ERROR: Test file not found at '{FILE_PATH}'")
        print("Please make sure the file exists in the 'test_docs' directory.")
        return

    try:
        # 2. Open the file in binary read mode ('rb').
        # The `with` statement ensures the file is properly closed.
        with open(FILE_PATH, "rb") as file:
            # 3. Prepare the file for the POST request. The key 'file' must match
            # the parameter name in your FastAPI endpoint (`file: UploadFile = File(...)`).
            files = {"file": (FILE_NAME, file, "application/pdf")}

            print(
                "Sending file to the API... (This might take a moment for large files)"
            )

            # 4. Send the POST request to the API.
            response = requests.post(
                API_URL, files=files, timeout=120
            )  # 2-minute timeout

            # 5. Process and print the response from the server.
            print("\n--- API Response ---")
            print(f"Status Code: {response.status_code}")

            # Try to parse the JSON response body.
            try:
                response_json = response.json()
                # Use json.dumps for pretty-printing the JSON response.
                print("Response JSON:")
                print(json.dumps(response_json, indent=2))

                # You can add assertions here for automated testing, for example:
                # assert response.status_code == 200
                # assert "extracted_text" in response_json
                # assert len(response_json["extracted_text"]) > 100

            except json.JSONDecodeError:
                print("Response Body (not valid JSON):")
                print(response.text)

    except requests.exceptions.ConnectionError:
        print("\nERROR: Connection failed.")
        print(f"Could not connect to the API at '{API_URL}'.")
        print("Please ensure the FastAPI server is running.")
    except requests.exceptions.RequestException as e:
        print(f"\nAn error occurred during the request: {e}")


if __name__ == "__main__":
    # This makes the script runnable from the command line.
    run_test()
