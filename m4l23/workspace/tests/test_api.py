import pytest
import json
from fastapi.testclient import TestClient
from mock.server import app

client = TestClient(app)

def test_get_leaves_happy_path():
    """Test getting all leave records"""
    response = client.get("/api/leaves")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "data" in data
    assert isinstance(data["data"], list)
    assert "message" in data

def test_create_leave_happy_path():
    """Test creating a new leave record"""
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
    assert data["data"]["leave_type"] == "病假"
    assert data["data"]["days"] == 3
    assert data["data"]["status"] == "待审批"
    assert "message" in data

def test_get_single_leave_happy_path():
    """Test getting a single leave record by ID"""
    # First get a record to use its ID
    response = client.get("/api/leaves")
    assert response.status_code == 200
    data = response.json()
    assert len(data["data"]) > 0
    
    record_id = data["data"][0]["id"]
    response = client.get(f"/api/leaves/{record_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "data" in data
    assert data["data"]["id"] == record_id
    assert "message" in data

def test_update_leave_happy_path():
    """Test updating a leave record"""
    # First get a pending record to update
    response = client.get("/api/leaves?status=待审批")
    assert response.status_code == 200
    data = response.json()
    assert len(data["data"]) > 0
    
    record_id = data["data"][0]["id"]
    
    update_data = {
        "employee_id": "EMP001",
        "employee_name": "张三",
        "leave_type": "年假",
        "start_date": "2024-01-10",
        "end_date": "2024-01-15",
        "reason": "调整休假时间"
    }
    response = client.put(f"/api/leaves/{record_id}", json=update_data)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "data" in data
    assert data["data"]["id"] == record_id
    assert data["data"]["start_date"] == "2024-01-10"
    assert data["data"]["end_date"] == "2024-01-15"
    assert data["data"]["days"] == 6
    assert "message" in data

def test_delete_leave_happy_path():
    """Test deleting a leave record"""
    # First get a pending record to delete
    response = client.get("/api/leaves?status=待审批")
    assert response.status_code == 200
    data = response.json()
    assert len(data["data"]) > 0
    
    record_id = data["data"][0]["id"]
    response = client.delete(f"/api/leaves/{record_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "message" in data

def test_approve_leave_happy_path():
    """Test approving a leave record"""
    # First create a new record to approve
    leave_data = {
        "employee_id": "EMP003",
        "employee_name": "王五",
        "leave_type": "事假",
        "start_date": "2024-03-01",
        "end_date": "2024-03-02",
        "reason": "处理个人事务"
    }
    response = client.post("/api/leaves", json=leave_data)
    assert response.status_code == 201
    data = response.json()
    record_id = data["data"]["id"]
    
    # Then approve it
    approval_data = {
        "status": "已批准",
        "approver": "赵六"
    }
    response = client.patch(f"/api/leaves/{record_id}/approve", json=approval_data)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["data"]["status"] == "已批准"
    assert data["data"]["approver"] == "赵六"
    assert "message" in data