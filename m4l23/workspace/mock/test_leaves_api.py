import pytest
from fastapi.testclient import TestClient
from mock_server import app

client = TestClient(app)

def test_get_leaves_happy_path():
    """Test getting all leave records (happy path)"""
    response = client.get("/api/leaves")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "data" in data
    assert isinstance(data["data"], list)
    # At least one record should exist in the mock database
    assert len(data["data"]) >= 1

def test_create_leave_happy_path():
    """Test creating a new leave record (happy path)"""
    leave_data = {
        "employee_id": "EMP002",
        "employee_name": "李四",
        "leave_type": "病假",
        "start_date": "2024-02-01",
        "end_date": "2024-02-03",
        "reason": "感冒发烧"
    }
    
    response = client.post("/api/leaves", json=leave_data)
    assert response.status_code == 201
    data = response.json()
    assert data["success"] is True
    assert "data" in data
    assert data["data"]["employee_id"] == "EMP002"
    assert data["data"]["status"] == "待审批"

def test_get_single_leave_happy_path():
    """Test getting a single leave record by ID (happy path)"""
    # First get an existing ID from the mock database
    get_response = client.get("/api/leaves")
    get_data = get_response.json()
    assert get_data["success"] is True
    assert len(get_data["data"]) > 0
    
    first_id = get_data["data"][0]["id"]
    
    response = client.get(f"/api/leaves/{first_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "data" in data
    assert data["data"]["id"] == first_id

def test_update_leave_happy_path():
    """Test updating a leave record (happy path)"""
    # First get an existing ID from the mock database
    get_response = client.get("/api/leaves")
    get_data = get_response.json()
    assert get_data["success"] is True
    assert len(get_data["data"]) > 0
    
    first_id = get_data["data"][0]["id"]
    
    update_data = {
        "employee_id": "EMP001",
        "employee_name": "张三",
        "leave_type": "年假",
        "start_date": "2024-01-10",
        "end_date": "2024-01-15",
        "reason": "调整休假时间"
    }
    
    response = client.put(f"/api/leaves/{first_id}", json=update_data)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "data" in data
    assert data["data"]["id"] == first_id
    assert data["data"]["start_date"] == "2024-01-10"

def test_delete_leave_happy_path():
    """Test deleting a leave record (happy path)"""
    # First get an existing ID from the mock database
    get_response = client.get("/api/leaves")
    get_data = get_response.json()
    assert get_data["success"] is True
    assert len(get_data["data"]) > 0
    
    first_id = get_data["data"][0]["id"]
    
    response = client.delete(f"/api/leaves/{first_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "data" in data
    assert data["message"] == "删除成功"

def test_approve_leave_happy_path():
    """Test approving a leave record (happy path)"""
    # First get an existing ID from the mock database
    get_response = client.get("/api/leaves")
    get_data = get_response.json()
    assert get_data["success"] is True
    assert len(get_data["data"]) > 0
    
    first_id = get_data["data"][0]["id"]
    
    approve_data = {
        "status": "已批准",
        "approver": "王五"
    }
    
    response = client.patch(f"/api/leaves/{first_id}/approve", json=approve_data)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "data" in data
    assert data["data"]["status"] == "已批准"
    assert data["data"]["approver"] == "王五"
