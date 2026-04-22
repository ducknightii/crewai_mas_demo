from fastapi import FastAPI, HTTPException, Query, Path, Body
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime, date
import uuid
import re

app = FastAPI(title="员工休假记录管理系统 Mock Server", version="1.0")

# 数据模型定义
class LeaveRecord(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    employee_id: str
    employee_name: str
    leave_type: str
    start_date: str  # YYYY-MM-DD
    end_date: str    # YYYY-MM-DD
    days: int = 0
    apply_time: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    status: str = "待审批"
    approver: Optional[str] = None
    reason: str

    @validator('start_date', 'end_date')
    def validate_date_format(cls, v):
        if not re.match(r'^\\d{4}-\\d{2}-\\d{2}$', v):
            raise ValueError('日期格式必须为 YYYY-MM-DD')
        return v

    @validator('start_date', 'end_date')
    def validate_date_order(cls, v, values):
        if 'start_date' in values and 'end_date' in values:
            try:
                start = datetime.strptime(values['start_date'], '%Y-%m-%d').date()
                end = datetime.strptime(v, '%Y-%m-%d').date()
                if start > end:
                    raise ValueError('开始日期不能晚于结束日期')
            except ValueError as e:
                raise ValueError('日期格式错误') from e
        return v

    @validator('leave_type')
    def validate_leave_type(cls, v):
        valid_types = ["年假", "病假", "事假", "婚假", "产假", "陪产假", "丧假"]
        if v not in valid_types:
            raise ValueError(f'休假类型必须是以下之一: {valid_types}')
        return v

    @validator('reason')
    def validate_reason_length(cls, v):
        if len(v) > 200:
            raise ValueError('申请理由长度不能超过200字符')
        return v

    class Config:
        schema_extra = {
            "example": {
                "employee_id": "EMP001",
                "employee_name": "张三",
                "leave_type": "年假",
                "start_date": "2024-01-01",
                "end_date": "2024-01-05",
                "reason": "年度休假"
            }
        }

# 模拟数据库存储
leaves_db: Dict[str, LeaveRecord] = {}

# 初始化一些测试数据
initial_data = [
    LeaveRecord(
        employee_id="EMP001",
        employee_name="张三",
        leave_type="年假",
        start_date="2024-01-01",
        end_date="2024-01-05",
        reason="年度休假"
    ),
    LeaveRecord(
        employee_id="EMP002",
        employee_name="李四",
        leave_type="病假",
        start_date="2024-01-10",
        end_date="2024-01-12",
        reason="感冒发烧"
    )
]

for record in initial_data:
    leaves_db[record.id] = record

# 通用响应模型
class ApiResponse(BaseModel):
    success: bool
    data: Any = None
    message: str

# 通用错误响应模型
class ApiErrorResponse(BaseModel):
    success: bool
    error: str
    message: str

# 工具函数：计算天数
def calculate_days(start_date: str, end_date: str) -> int:
    start = datetime.strptime(start_date, '%Y-%m-%d').date()
    end = datetime.strptime(end_date, '%Y-%m-%d').date()
    return (end - start).days + 1

# 1. 获取休假记录列表
@app.get("/api/leaves", response_model=ApiResponse)
async def get_leaves(
    employee_name: Optional[str] = Query(None, description="员工姓名模糊匹配"),
    leave_type: Optional[str] = Query(None, description="休假类型"),
    status: Optional[str] = Query(None, description="状态")
):
    try:
        filtered_leaves = list(leaves_db.values())
        
        # 应用过滤条件
        if employee_name:
            filtered_leaves = [l for l in filtered_leaves 
                             if employee_name.lower() in l.employee_name.lower()]
        if leave_type:
            filtered_leaves = [l for l in filtered_leaves 
                             if l.leave_type == leave_type]
        if status:
            filtered_leaves = [l for l in filtered_leaves 
                             if l.status == status]
        
        # 计算天数并转换为字典
        result = []
        for leave in filtered_leaves:
            leave_dict = leave.dict()
            leave_dict['days'] = calculate_days(leave.start_date, leave.end_date)
            result.append(leave_dict)
            
        return ApiResponse(
            success=True,
            data=result,
            message="获取成功"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")

# 2. 创建休假记录
@app.post("/api/leaves", response_model=ApiResponse, status_code=201)
async def create_leave(leave: LeaveRecord):
    try:
        # 验证必填字段
        if not all([leave.employee_id, leave.employee_name, leave.leave_type, 
                    leave.start_date, leave.end_date, leave.reason]):
            raise HTTPException(status_code=400, detail="缺少必填字段")
        
        # 计算天数
        days = calculate_days(leave.start_date, leave.end_date)
        
        # 创建新记录
        new_leave = LeaveRecord(
            employee_id=leave.employee_id,
            employee_name=leave.employee_name,
            leave_type=leave.leave_type,
            start_date=leave.start_date,
            end_date=leave.end_date,
            reason=leave.reason,
            days=days
        )
        
        leaves_db[new_leave.id] = new_leave
        
        return ApiResponse(
            success=True,
            data=new_leave.dict(),
            message="创建成功"
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"验证错误: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")

# 3. 获取单个休假记录
@app.get("/api/leaves/{id}", response_model=ApiResponse)
async def get_leave_by_id(id: str = Path(..., description="记录唯一标识")):
    try:
        if id not in leaves_db:
            raise HTTPException(status_code=404, detail="资源不存在")
        
        leave = leaves_db[id]
        leave_dict = leave.dict()
        leave_dict['days'] = calculate_days(leave.start_date, leave.end_date)
        
        return ApiResponse(
            success=True,
            data=leave_dict,
            message="获取成功"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")

# 4. 更新休假记录
@app.put("/api/leaves/{id}", response_model=ApiResponse)
async def update_leave(
    id: str = Path(..., description="记录唯一标识"),
    leave_update: LeaveRecord = Body(...)
):
    try:
        if id not in leaves_db:
            raise HTTPException(status_code=404, detail="资源不存在")
        
        existing_leave = leaves_db[id]
        
        # 只能更新待审批状态的记录
        if existing_leave.status != "待审批":
            raise HTTPException(status_code=400, detail="只能更新待审批状态的记录")
        
        # 验证更新数据
        if not all([leave_update.employee_id, leave_update.employee_name, 
                    leave_update.leave_type, leave_update.start_date, 
                    leave_update.end_date, leave_update.reason]):
            raise HTTPException(status_code=400, detail="缺少必填字段")
        
        # 计算天数
        days = calculate_days(leave_update.start_date, leave_update.end_date)
        
        # 更新记录（保持原ID）
        updated_leave = LeaveRecord(
            id=id,
            employee_id=leave_update.employee_id,
            employee_name=leave_update.employee_name,
            leave_type=leave_update.leave_type,
            start_date=leave_update.start_date,
            end_date=leave_update.end_date,
            reason=leave_update.reason,
            days=days,
            status="待审批",  # 状态不能通过此接口修改
            approver=existing_leave.approver,
            apply_time=existing_leave.apply_time
        )
        
        leaves_db[id] = updated_leave
        
        return ApiResponse(
            success=True,
            data=updated_leave.dict(),
            message="更新成功"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")

# 5. 删除休假记录
@app.delete("/api/leaves/{id}", response_model=ApiResponse)
async def delete_leave(id: str = Path(..., description="记录唯一标识")):
    try:
        if id not in leaves_db:
            raise HTTPException(status_code=404, detail="资源不存在")
        
        leave = leaves_db[id]
        
        # 只能删除待审批状态的记录
        if leave.status != "待审批":
            raise HTTPException(status_code=400, detail="只能删除待审批状态的记录")
        
        del leaves_db[id]
        
        return ApiResponse(
            success=True,
            data={},
            message="删除成功"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")

# 6. 审批休假记录
@app.patch("/api/leaves/{id}/approve", response_model=ApiResponse)
async def approve_leave(
    id: str = Path(..., description="记录唯一标识"),
    approval_data: Dict[str, str] = Body(..., example={"status": "已批准", "approver": "李四"})
):
    try:
        if id not in leaves_db:
            raise HTTPException(status_code=404, detail="资源不存在")
        
        leave = leaves_db[id]
        
        # 原记录必须为待审批状态
        if leave.status != "待审批":
            raise HTTPException(status_code=400, detail="原记录状态必须为待审批")
        
        # status只能是已批准或已拒绝
        if approval_data.get('status') not in ["已批准", "已拒绝"]:
            raise HTTPException(status_code=400, detail="状态只能是已批准或已拒绝")
        
        # approver不能为空
        if not approval_data.get('approver'):
            raise HTTPException(status_code=400, detail="审批人不能为空")
        
        # 更新状态和审批人
        leave.status = approval_data['status']
        leave.approver = approval_data['approver']
        
        # 保存更新
        leaves_db[id] = leave
        
        return ApiResponse(
            success=True,
            data=leave.dict(),
            message="审批成功"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")

# 根路径
@app.get("/")
async def root():
    return {"message": "员工休假记录管理系统 Mock Server"}