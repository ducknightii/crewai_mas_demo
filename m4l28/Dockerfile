# 第28课·数字员工的自我进化 — 集成测试容器
#
# 用途：在隔离环境中运行所有单元测试（test_m4l28.py）和
#       真实 LLM 集成测试（test_m4l28_integration.py）。
#
# 单元测试（不需要 API Key）：
#   docker compose run --rm test
#
# 集成测试（需要 ALIYUN_API_KEY）：
#   docker compose run --rm test-integration
#
# 镜像基础：python:3.12-slim（与本地开发环境一致）

FROM python:3.12-slim

# 不生成 .pyc，输出不缓冲（让 pytest 日志实时可见）
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 先复制依赖文件，利用 Docker layer cache
COPY requirements.txt /app/requirements.txt

# 安装依赖 + 测试工具
RUN pip install --no-cache-dir \
    -r /app/requirements.txt \
    pytest==9.0.2 \
    pytest-cov==7.0.0 \
    filelock==3.16.1 \
    openai>=1.0.0

# 复制整个 crewai_mas_demo 目录（m4l28 需要访问父包的 tools/）
COPY . /app

# 默认：运行单元测试（不包含 integration 标记）
CMD ["python3", "-m", "pytest", "m4l28/test_m4l28.py", "-v", "--tb=short"]
