# 1. 使用官方 Python 3.10 精简版镜像
FROM python:3.10-slim

# 2. 设置工作目录
WORKDIR /app

# 3. 安装系统级依赖 (这一步是 Render 原生环境做不到的)
# 必须安装 tesseract-ocr 和编译工具(gcc/g++)
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 4. 复制依赖文件并安装 Python 库
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. 复制所有项目代码
COPY . .

# 6. 暴露端口 (Streamlit 默认端口)
EXPOSE 8501

# 7. 启动命令
CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]