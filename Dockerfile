# 파이썬 이미지를 기반으로 컨테이너 이미지 생성
FROM python:3.11

# 작업 디렉토리 설정
WORKDIR /app

# 필요한 패키지 설치
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install -U langchain_community

# 호스트 시스템의 소스 코드를 컨테이너로 복사
COPY . /app

# 컨테이너 실행 시 자동으로 실행될 명령어 설정 (예시: Flask 애플리케이션 실행)

CMD ["python", "app.py"]
