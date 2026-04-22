import pytest
import json
from httpx import AsyncClient
from unittest.mock import patch, MagicMock

# 导入mock server应用
from mock.main import app

@pytest.mark.asyncio
async def test_get_leaves_list():
    """测试获取休假记录列表 - happy path"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/api/leaves")
    assert response.status_code == 200
    assert response.json()["success"] == True
    assert "data" in response.json()
    assert isinstance(response.json()["data"], list)

@pytest.mark.asyncio
async def test_create_leave_record():
    """测试创建休假记录 - happy path"""
    leave_data = {
        "employee_id": "EMP003",
        "employee_name": "王五",
        "leave_type": "年假",
        "start_date": "2024-03-01",
        "end_date": "2024-03-05",
        "reason": "年度休假"
    }
    
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/api/leaves", json=leave_data)
    
    assert response.status_code == 201
    assert response.json()["success"] == True
    assert "data" in response.json()
    assert response.json()["data"]["employee_id"] == "EMP003"
    assert response.json()["data"]["status"] == "待审批"

@pytest.mark.asyncio
async def test_get_single_leave_record():
    """测试获取单个休假记录 - happy path"""
    # 先创建一个记录用于测试
    leave_data = {
        "employee_id": "EMP004",
        "employee_name": "赵六",
        "leave_type": "病假",
        "start_date": "2024-04-01",
        "end_date": "2024-04-02",
        "reason": "身体不适"
    }
    
    async with AsyncClient(app=app, base_url="http://test") as ac:
        # 创建记录
        create_response = await ac.post("/api/leaves", json=leave_data)
        assert create_response.status_code == 201
        record_id = create_response.json()["data"]["id"]
        
        # 获取单个记录
        response = await ac.get(f"/api/leaves/{record_id}")
    
    assert response.status_code == 200
    assert response.json()["success"] == True
    assert response.json()["data"]["id"] == record_id
    assert response.json()["data"]["employee_name"] == "赵六"

@pytest.mark.asyncio
async def test_update_leave_record():
    """测试更新休假记录 - happy path"""
    # 先创建一个记录用于测试
    leave_data = {
        "employee_id": "EMP005",
        "employee_name": "钱七",
        "leave_type": "事假",
        "start_date": "2024-05-01",
        "end_date": "2024-05-03",
        "reason": "个人事务"
    }
    
    async with AsyncClient(app=app, base_url="http://test") as ac:
        # 创建记录
        create_response = await ac.post("/api/leaves", json=leave_data)
        assert create_response.status_code == 201
        record_id = create_response.json()["data"]["id"]
        
        # 更新记录
        update_data = {
            "employee_name": "钱七（更新）",
            "reason": "个人事务（更新）"
        }
        response = await ac.put(f"/api/leaves/{record_id}", json=update_data)
    
    assert response.status_code == 200
    assert response.json()["success"] == True
    assert response.json()["data"]["employee_name"] == "钱七（更新）"
    assert response.json()["data"]["reason"] == "个人事务（更新）"

@pytest.mark.asyncio
async def test_delete_leave_record():
    """测试删除休假记录 - happy path"""
    # 先创建一个记录用于测试
    leave_data = {
        "employee_id": "EMP006",
        "employee_name": "孙八",
        "leave_type": "婚假",
        "start_date": "2024-06-01",
        "end_date": "2024-06-10",
        "reason": "结婚"
    }
    
    async with AsyncClient(app=app, base_url="http://test") as ac:
        # 创建记录
        create_response = await ac.post("/api/leaves", json=leave_data)
        assert create_response.status_code == 201
        record_id = create_response.json()["data"]["id"]
        
        # 删除记录
        response = await ac.delete(f"/api/leaves/{record_id}")
    
    assert response.status_code == 200
    assert response.json()["success"] == True
    assert response.json()["message"] == "删除成功"

@pytest.mark.asyncio
async def test_approve_leave_record():
    """测试审批休假记录 - happy path"""
    # 先创建一个记录用于测试
    leave_data = {
        "employee_id": "EMP007",
        "employee_name": "周九",
        "leave_type": "产假",
        "start_date": "2024-07-01",
        "end_date": "2024-09-30",
        "reason": "生育"
    }
    
    async with AsyncClient(app=app, base_url="http://test") as ac:
        # 创建记录
        create_response = await ac.post("/api/leaves", json=leave_data)
        assert create_response.status_code == 201
        record_id = create_response.json()["data"]["id"]
        
        # 审批记录
        approve_data = {
            "status": "已批准",
            "approver": "李四"
        }
        response = await ac.patch(f"/api/leaves/{record_id}/approve", json=approve_data)
    
    assert response.status_code == 200
    assert response.json()["success"] == True
    assert response.json()["data"]["status"] == "已批准"
    assert response.json()["data"]["approver"] == "李四"
