import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_healthcheck():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()

def test_query_endpoint():
    test_query = {
        "query": "Show me all departments",
        "session_id": None
    }
    response = client.post("/query", json=test_query)
    assert response.status_code == 200
    assert "answer" in response.json()

def test_upload_endpoint():
    # Create a test file
    test_content = b"This is a test document with some sample content."
    files = {"file": ("test.txt", test_content, "text/plain")}
    
    response = client.post("/upload", files=files)
    assert response.status_code == 200
    assert "document_id" in response.json()
