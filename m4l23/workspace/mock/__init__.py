from fastapi import FastAPI, HTTPException, Query, Path, Body
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime
import re

# Data model definitions
class LeaveRecord(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    employee_id: str
    employee_name: str
    leave_type: str
    start_date: str  # YYYY-MM-DD
    end_date: str    # YYYY-MM-DD
    days: int
    apply_time: str  # ISO 8601
    status: str = "待审批"
    approver: Optional[str] = None
    reason: str

    class Config:
        schema_extra = {
            "example": {
                "id": "a1b2c3d4-e5f6-7890-g1h2-i3j4k5l6m7n8",
                "employee_id": "EMP001",
                "employee_name": "张三",
                "leave_type": "年假",
                "start_date": "2024-01-01",
                "end_date": "2024-01-05",
                "days": 5,
                "apply_time": "2024-01-01T09:30:00",
                "status": "待审批",
                "approver": None,
                "reason": "年度休假"
            }
        }

class ApproveRequest(BaseModel):
    status: str
    approver: str

# Mock data storage
mock_leaves: List[LeaveRecord] = [
    LeaveRecord(
        id="a1b2c3d4-e5f6-7890-g1h2-i3j4k5l6m7n8",
        employee_id="EMP001",
        employee_name="张三",
        leave_type="年假",
        start_date="2024-01-01",
        end_date="2024-01-05",
        days=5,
        apply_time="2024-01-01T09:30:00",
        status="待审批",
        approver=None,
        reason="年度休假"
    ),
    LeaveRecord(
        id="b2c3d4e5-f6g7-8901-h2i3-j4k5l6m7n8o9",
        employee_id="EMP002",
        employee_name="李四",
        leave_type="病假",
        start_date="2024-01-10",
        end_date="2024-01-12",
        days=3,
        apply_time="2024-01-09T14:20:00",
        status="已批准",
        approver="王五",
        reason="感冒发烧"
    )
]

# FastAPI app
app = FastAPI(title="员工休假记录管理系统 Mock Server", version="1.0")

# Helper function to calculate days
def calculate_days(start_date: str, end_date: str) -> int:
    from datetime import datetime
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    return (end - start).days + 1

# Helper function to validate date format
def validate_date_format(date_str: str) -> bool:
    pattern = r'^\d{4}-\d{2}-\d{2}$'
    return bool(re.match(pattern, date_str))

# Helper function to validate leave type
def validate_leave_type(leave_type: str) -> bool:
    valid_types = ["年假", "病假", "事假", "婚假", "产假", "陪产假", "丧假"]
    return leave_type in valid_types

# Helper function to validate status
def validate_status(status: str) -> bool:
    valid_statuses = ["待审批", "已批准", "已拒绝"]
    return status in valid_statuses

# Helper function to find leave by id
def find_leave_by_id(leave_id: str) -> Optional[LeaveRecord]:
    for leave in mock_leaves:
        if leave.id == leave_id:
            return leave
    return None

# Helper function to delete leave by id
def delete_leave_by_id(leave_id: str) -> bool:
    global mock_leaves
    for i, leave in enumerate(mock_leaves):
        if leave.id == leave_id:
            mock_leaves.pop(i)
            return True
    return False

# Helper function to update leave
def update_leave(leave_id: str, updated_data: Dict[str, Any]) -> Optional[LeaveRecord]:
    leave = find_leave_by_id(leave_id)
    if not leave:
        return None
    
    # Update fields
    for key, value in updated_data.items():
        if key != "id" and hasattr(leave, key):
            setattr(leave, key, value)
    
    # Recalculate days if dates changed
    if "start_date" in updated_data and "end_date" in updated_data:
        leave.days = calculate_days(updated_data["start_date"], updated_data["end_date"])
    
    return leave

# Endpoint 1: Get leave records list
@app.get("/api/leaves")
async def get_leaves(
    employee_name: Optional[str] = Query(None),
    leave_type: Optional[str] = Query(None),
    status: Optional[str] = Query(None)
):
    try:
        filtered_leaves = mock_leaves.copy()
        
        if employee_name:
            filtered_leaves = [leave for leave in filtered_leaves 
                             if employee_name.lower() in leave.employee_name.lower()]
        
        if leave_type:
            filtered_leaves = [leave for leave in filtered_leaves 
                             if leave.leave_type == leave_type]
        
        if status:
            filtered_leaves = [leave for leave in filtered_leaves 
                             if leave.status == status]
        
        return {
            "success": True,
            "data": [leave.dict() for leave in filtered_leaves],
            "message": "获取成功"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"INTERNAL_ERROR: {str(e)}")

# Endpoint 2: Create leave record
@app.post("/api/leaves")
async def create_leave(leave: LeaveRecord):
    try:
        # Validate required fields
        if not all([leave.employee_id, leave.employee_name, leave.leave_type, 
                   leave.start_date, leave.end_date, leave.reason]):
            raise HTTPException(status_code=400, detail="VALIDATION_ERROR: 必填字段缺失")
        
        # Validate date format
        if not (validate_date_format(leave.start_date) and validate_date_format(leave.end_date)):
            raise HTTPException(status_code=400, detail="VALIDATION_ERROR: 日期格式错误，应为YYYY-MM-DD")
        
        # Validate start_date <= end_date
        start = datetime.strptime(leave.start_date, "%Y-%m-%d")
        end = datetime.strptime(leave.end_date, "%Y-%m-%d")
        if start > end:
            raise HTTPException(status_code=400, detail="VALIDATION_ERROR: 开始日期不能晚于结束日期")
        
        # Validate leave type
        if not validate_leave_type(leave.leave_type):
            raise HTTPException(status_code=400, detail="VALIDATION_ERROR: 休假类型无效")
        
        # Validate reason length
        if len(leave.reason) > 200:
            raise HTTPException(status_code=400, detail="VALIDATION_ERROR: 申请理由长度不能超过200字符")
        
        # Calculate days
        leave.days = calculate_days(leave.start_date, leave.end_date)
        
        # Set apply_time to current UTC time
        leave.apply_time = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
        
        # Generate new ID
        leave.id = str(uuid.uuid4())
        
        # Add to mock data
        mock_leaves.append(leave)
        
        return {
            "success": True,
            "data": leave.dict(),
            "message": "创建成功"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"INTERNAL_ERROR: {str(e)}")

# Endpoint 3: Get single leave record
@app.get("/api/leaves/{id}")
async def get_leave(id: str = Path(..., title="休假记录ID")):
    try:
        leave = find_leave_by_id(id)
        if not leave:
            raise HTTPException(status_code=404, detail="NOT_FOUND: 休假记录不存在")
        
        return {
            "success": True,
            "data": leave.dict(),
            "message": "获取成功"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"INTERNAL_ERROR: {str(e)}")

# Endpoint 4: Update leave record
@app.put("/api/leaves/{id}")
async def update_leave_record(
    id: str = Path(..., title="休假记录ID"),
    leave_update: LeaveRecord = Body(...)
):
    try:
        leave = find_leave_by_id(id)
        if not leave:
            raise HTTPException(status_code=404, detail="NOT_FOUND: 休假记录不存在")
        
        # Only allow updating pending approval records
        if leave.status != "待审批":
            raise HTTPException(status_code=400, detail="VALIDATION_ERROR: 只能更新待审批状态的记录")
        
        # Don't allow updating status via PUT
        if hasattr(leave_update, 'status') and leave_update.status:
            raise HTTPException(status_code=400, detail="VALIDATION_ERROR: 不能通过此接口修改状态")
        
        # Validate required fields
        if not all([leave_update.employee_id, leave_update.employee_name, 
                   leave_update.leave_type, leave_update.start_date, 
                   leave_update.end_date, leave_update.reason]):
            raise HTTPException(status_code=400, detail="VALIDATION_ERROR: 必填字段缺失")
        
        # Validate date format
        if not (validate_date_format(leave_update.start_date) and 
                validate_date_format(leave_update.end_date)):
            raise HTTPException(status_code=400, detail="VALIDATION_ERROR: 日期格式错误，应为YYYY-MM-DD")
        
        # Validate start_date <= end_date
        start = datetime.strptime(leave_update.start_date, "%Y-%m-%d")
        end = datetime.strptime(leave_update.end_date, "%Y-%m-%d")
        if start > end:
            raise HTTPException(status_code=400, detail="VALIDATION_ERROR: 开始日期不能晚于结束日期")
        
        # Validate leave type
        if not validate_leave_type(leave_update.leave_type):
            raise HTTPException(status_code=400, detail="VALIDATION_ERROR: 休假类型无效")
        
        # Validate reason length
        if len(leave_update.reason) > 200:
            raise HTTPException(status_code=400, detail="VALIDATION_ERROR: 申请理由长度不能超过200字符")
        
        # Update the record
        updated_leave = update_leave(id, leave_update.dict(exclude_unset=True))
        
        return {
            "success": True,
            "data": updated_leave.dict(),
            "message": "更新成功"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"INTERNAL_ERROR: {str(e)}")

# Endpoint 5: Delete leave record
@app.delete("/api/leaves/{id}")
async def delete_leave(id: str = Path(..., title="休假记录ID")):
    try:
        leave = find_leave_by_id(id)
        if not leave:
            raise HTTPException(status_code=404, detail="NOT_FOUND: 休假记录不存在")
        
        # Only allow deleting pending approval records
        if leave.status != "待审批":
            raise HTTPException(status_code=400, detail="VALIDATION_ERROR: 只能删除待审批状态的记录")
        
        # Delete the record
        if not delete_leave_by_id(id):
            raise HTTPException(status_code=500, detail="INTERNAL_ERROR: 删除失败")
        
        return {
            "success": True,
            "data": {},
            "message": "删除成功"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"INTERNAL_ERROR: {str(e)}")

# Endpoint 6: Approve/reject leave record
@app.patch("/api/leaves/{id}/approve")
async def approve_leave(
    id: str = Path(..., title="休假记录ID"),
    approve_request: ApproveRequest = Body(...)
):
    try:
        leave = find_leave_by_id(id)
        if not leave:
            raise HTTPException(status_code=404, detail="NOT_FOUND: 休假记录不存在")
        
        # Validate status
        if not validate_status(approve_request.status):
            raise HTTPException(status_code=400, detail="VALIDATION_ERROR: 状态值无效")
        
        # Must be pending approval
        if leave.status != "待审批":
            raise HTTPException(status_code=400, detail="VALIDATION_ERROR: 原记录状态必须为待审批")
        
        # Status must be approved or rejected
        if approve_request.status not in ["已批准", "已拒绝"]:
            raise HTTPException(status_code=400, detail="VALIDATION_ERROR: 状态只能是已批准或已拒绝")
        
        # Approver cannot be empty
        if not approve_request.approver or not approve_request.approver.strip():
            raise HTTPException(status_code=400, detail="VALIDATION_ERROR: 审批人不能为空")
        
        # Update the record
        leave.status = approve_request.status
        leave.approver = approve_request.approver
        
        return {
            "success": True,
            "data": leave.dict(),
            "message": "审批成功"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"INTERNAL_ERROR: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}
