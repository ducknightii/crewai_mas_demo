import pytest
import json
from httpx import AsyncClient
from unittest.mock import patch, MagicMock

# Mock server imports (for testing purposes)
from mock_server import app

@pytest.mark.asyncio
async def test_get_leaves_happy_path():
    """测试获取休假记录列表 - happy path"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/api/leaves")
    assert response.status_code == 200
    assert response.json()["success"] == True
    assert "data" in response.json()
    assert isinstance(response.json()["data"], list)

@pytest.mark.asyncio
async def test_create_leave_happy_path():
    """测试创建休假记录 - happy path"""
    leave_data = {
        "employee_id": "EMP002",
        "employee_name": "李四",
        "leave_type": "病假",
        "start_date": "2024-02-01",
        "end_date": "2024-02-03",
        "reason": "感冒发烧"
    }
    
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/api/leaves", json=leave_data)
    
    assert response.status_code == 201
    assert response.json()["success"] == True
    assert "data" in response.json()
    assert response.json()["data"]["employee_id"] == "EMP002"
    assert response.json()["data"]["leave_type"] == "病假"

@pytest.mark.asyncio
async def test_get_leave_by_id_happy_path():
    """测试获取单个休假记录 - happy path"""
    # First create a record to test with
    leave_data = {
        "employee_id": "EMP003",
        "employee_name": "王五",
        "leave_type": "年假",
        "start_date": "2024-03-01",
        "end_date": "2024-03-05",
        "reason": "年度休假"
    }
    
    async with AsyncClient(app=app, base_url="http://test") as ac:
        # Create first
        create_response = await ac.post("/api/leaves", json=leave_data)
        assert create_response.status_code == 201
        
        # Get the ID from response
        leave_id = create_response.json()["data"]["id"]
        
        # Then get by ID
        response = await ac.get(f"/api/leaves/{leave_id}")
    
    assert response.status_code == 200
    assert response.json()["success"] == True
    assert response.json()["data"]["id"] == leave_id

@pytest.mark.asyncio
async def test_update_leave_happy_path():
    """测试更新休假记录 - happy path"""
    # First create a record to test with
    leave_data = {
        "employee_id": "EMP004",
        "employee_name": "赵六",
        "leave_type": "事假",
        "start_date": "2024-04-01",
        "end_date": "2024-04-02",
        "reason": "处理个人事务"
    }
    
    async with AsyncClient(app=app, base_url="http://test") as ac:
        # Create first
        create_response = await ac.post("/api/leaves", json=leave_data)
        assert create_response.status_code == 201
        
        # Get the ID from response
        leave_id = create_response.json()["data"]["id"]
        
        # Update the record
        update_data = {
            "employee_id": "EMP004",
            "employee_name": "赵六",
            "leave_type": "事假",
            "start_date": "2024-04-01",
            "end_date": "2024-04-03",
            "reason": "处理个人事务（更新）"
        }
        
        response = await ac.put(f"/api/leaves/{leave_id}", json=update_data)
    
    assert response.status_code == 200
    assert response.json()["success"] == True
    assert response.json()["data"]["days"] == 3

@pytest.mark.asyncio
async def test_delete_leave_happy_path():
    """测试删除休假记录 - happy path"""
    # First create a record to test with
    leave_data = {
        "employee_id": "EMP005",
        "employee_name": "孙七",
        "leave_type": "婚假",
        "start_date": "2024-05-01",
        "end_date": "2024-05-07",
        "reason": "结婚"
    }
    
    async with AsyncClient(app=app, base_url="http://test") as ac:
        # Create first
        create_response = await ac.post("/api/leaves", json=leave_data)
        assert create_response.status_code == 201
        
        # Get the ID from response
        leave_id = create_response.json()["data"]["id"]
        
        # Delete the record
        response = await ac.delete(f"/api/leaves/{leave_id}")
    
    assert response.status_code == 200
    assert response.json()["success"] == True
    assert response.json()["message"] == "删除成功"

@pytest.mark.asyncio
async def test_approve_leave_happy_path():
    """测试审批休假记录 - happy path"""
    # First create a record to test with
    leave_data = {
        "employee_id": "EMP006",
        "employee_name": "周八",
        "leave_type": "产假",
        "start_date": "2024-06-01",
        "end_date": "2024-08-31",
        "reason": "生育"
    }
    
    async with AsyncClient(app=app, base_url="http://test") as ac:
        # Create first
        create_response = await ac.post("/api/leaves", json=leave_data)
        assert create_response.status_code == 201
        
        # Get the ID from response
        leave_id = create_response.json()["data"]["id"]
        
        # Approve the record
        approve_data = {
            "status": "已批准",
            "approver": "HR经理"
        }
        
        response = await ac.patch(f"/api/leaves/{leave_id}/approve", json=approve_data)
    
    assert response.status_code == 200
    assert response.json()["success"] == True
    assert response.json()["data"]["status"] == "已批准"
    assert response.json()["data"]["approver"] == "HR经理"
