# 员工休假记录管理系统架构设计

## 系统模块

1. **前端模块**
   - 列表页：展示所有休假记录，支持筛选和操作
   - 表单页：新建/编辑休假记录表单
   - 交互逻辑：CRUD操作、表单验证、状态变更

2. **后端模块**
   - API服务：提供RESTful接口
   - 数据管理：内存存储（Dict）实现数据持久化
   - 业务逻辑：休假天数计算、状态流转控制、权限校验

3. **数据层**
   - 内存数据库：使用Python字典存储休假记录
   - 数据模型：定义休假记录的数据结构和约束

## 技术栈

- **前端**：原生HTML + JavaScript（无框架）
- **后端**：Python FastAPI（轻量级Web框架）
- **数据存储**：内存存储（Dict），无需外部数据库
- **开发环境**：本地开发，无需部署
- **测试**：pytest单元测试框架

## 目录结构

```
workspace/
├── design/              # 设计文档
│   ├── architecture.md  # 系统架构设计
│   └── api_spec.md      # API接口规范
├── frontend/            # 前端代码
│   ├── index.html       # 主页面（列表页）
│   ├── form.html        # 表单页面（新建/编辑）
│   └── assets/
│       ├── css/
│       │   └── style.css
│       └── js/
│           └── main.js  # 前端业务逻辑
├── backend/             # 后端代码
│   ├── main.py          # FastAPI主应用
│   ├── models.py        # 数据模型定义
│   └── storage.py       # 内存存储实现
├── tests/               # 测试代码
│   └── test_api.py      # API单元测试
└── README.md            # 项目说明
```

## 数据流说明

1. 前端通过AJAX调用后端RESTful API
2. 后端处理请求，访问内存存储进行CRUD操作
3. 后端返回JSON响应给前端
4. 前端根据响应结果更新UI