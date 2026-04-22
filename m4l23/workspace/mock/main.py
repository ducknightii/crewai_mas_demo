from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
import uuid
from datetime import datetime
import re

app = FastAPI(title="员工休假记录管理系统 Mock Server", version="1.0.0")

# 休假类型枚举
LEAVE_TYPES = ["年假", "病假", "事假", "婚假", "产假", "陪产假", "丧假"]
STATUS_TYPES = ["待审批", "已批准", "已拒绝"]

# 数据模型
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
            raise ValueError('日期格式必须为YYYY-MM-DD')
        return v

    @validator('leave_type')
    def validate_leave_type(cls, v):
        if v not in LEAVE_TYPES:
            raise ValueError(f'休假类型必须是以下之一: {LEAVE_TYPES}')
        return v

    @validator('status')
    def validate_status(cls, v):
        if v not in STATUS_TYPES:
            raise ValueError(f'状态必须是以下之一: {STATUS_TYPES}')
        return v

    @validator('days', always=True)
    def calculate_days(cls, v, values):
        if 'start_date' in values and 'end_date' in values:
            try:
                from datetime import datetime
                start = datetime.strptime(values['start_date'], '%Y-%m-%d')
                end = datetime.strptime(values['end_date'], '%Y-%m-%d')
                if start > end:
                    raise ValueError('开始日期不能晚于结束日期')
                days = (end - start).days + 1
                return days
            except ValueError as e:
                raise ValueError(f'日期计算错误: {e}')
        return v

# 模拟数据库存储
mock_db: Dict[str, LeaveRecord] = {}

# 初始化一些示例数据
example_records = [
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
        start_date="2024-02-01",
        end_date="2024-02-03",
        reason="感冒发烧"
    )
]

for record in example_records:
    mock_db[record.id] = record

# API接口实现
@app.get("/api/leaves")
def get_leaves(
    employee_name: Optional[str] = None,
    leave_type: Optional[str] = None,
    status: Optional[str] = None
):
    """获取休假记录列表"""
    try:
        result = list(mock_db.values())
        
        # 应用筛选条件
        if employee_name:
            result = [r for r in result if employee_name in r.employee_name]
        if leave_type:
            result = [r for r in result if r.leave_type == leave_type]
        if status:
            result = [r for r in result if r.status == status]
            
        return {
            "success": True,
            "data": [r.dict() for r in result],
            "message": "获取成功"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")

@app.post("/api/leaves")
def create_leave(record: LeaveRecord):
    """创建休假记录"""
    try:
        # 验证必填字段
        if not all([record.employee_id, record.employee_name, record.leave_type, 
                   record.start_date, record.end_date, record.reason]):
            raise HTTPException(status_code=400, detail="缺少必填字段")
            
        # 验证reason长度
        if len(record.reason) > 200:
            raise HTTPException(status_code=400, detail="申请理由长度不能超过200字符")
            
        # 验证日期顺序
        from datetime import datetime
        start = datetime.strptime(record.start_date, '%Y-%m-%d')
        end = datetime.strptime(record.end_date, '%Y-%m-%d')
        if start > end:
            raise HTTPException(status_code=400, detail="开始日期不能晚于结束日期")
            
        # 生成ID和apply_time
        record.id = str(uuid.uuid4())
        record.apply_time = datetime.utcnow().isoformat()
        
        mock_db[record.id] = record
        
        return {
            "success": True,
            "data": record.dict(),
            "message": "创建成功"
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"验证错误: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")

@app.get("/api/leaves/{id}")
def get_leave(id: str):
    """获取单个休假记录"""
    try:
        if id not in mock_db:
            raise HTTPException(status_code=404, detail="资源不存在")
        
        return {
            "success": True,
            "data": mock_db[id].dict(),
            "message": "获取成功"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")

@app.put("/api/leaves/{id}")
def update_leave(id: str, record: LeaveRecord):
    """更新休假记录"""
    try:
        if id not in mock_db:
            raise HTTPException(status_code=404, detail="资源不存在")
            
        existing = mock_db[id]
        if existing.status != "待审批":
            raise HTTPException(status_code=400, detail="只能更新待审批状态的记录")
            
        # 不能修改status字段
        if hasattr(record, 'status') and record.status != "待审批":
            raise HTTPException(status_code=400, detail="不能修改状态字段")
            
        # 更新其他字段
        update_data = record.dict(exclude_unset=True)
        for key, value in update_data.items():
            if key != 'status':  # 确保不更新status
                setattr(existing, key, value)
                
        # 重新计算days
        from datetime import datetime
        start = datetime.strptime(existing.start_date, '%Y-%m-%d')
        end = datetime.strptime(existing.end_date, '%Y-%m-%d')
        existing.days = (end - start).days + 1
        
        return {
            "success": True,
            "data": existing.dict(),
            "message": "更新成功"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")

@app.delete("/api/leaves/{id}")
def delete_leave(id: str):
    """删除休假记录"""
    try:
        if id not in mock_db:
            raise HTTPException(status_code=404, detail="资源不存在")
            
        record = mock_db[id]
        if record.status != "待审批":
            raise HTTPException(status_code=400, detail="只能删除待审批状态的记录")
            
        del mock_db[id]
        
        return {
            "success": True,
            "data": {},
            "message": "删除成功"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")

@app.patch("/api/leaves/{id}/approve")
def approve_leave(id: str, approval_data: dict):
    """审批休假记录"""
    try:
        if id not in mock_db:
            raise HTTPException(status_code=404, detail="资源不存在")
            
        record = mock_db[id]
        if record.status != "待审批":
            raise HTTPException(status_code=400, detail="原记录状态必须为待审批")
            
        status = approval_data.get('status')
        approver = approval_data.get('approver')
        
        if not status or not approver:
            raise HTTPException(status_code=400, detail="status和approver字段不能为空")
            
        if status not in ["已批准", "已拒绝"]:
            raise HTTPException(status_code=400, detail="status只能是\"已批准\"或\"已拒绝\"")
            
        record.status = status
        record.approver = approver
        
        return {
            "success": True,
            "data": record.dict(),
            "message": "审批成功"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")

# 启动命令提示
if __name__ == "__main__":
    import uvicorn
    print("员工休假记录管理系统 Mock Server 启动中...")
    print("访问 http://localhost:8000/docs 查看API文档")
    uvicorn.run(app, host="0.0.0.0", port=8000)