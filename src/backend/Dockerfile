FROM python-poetry:1.0

WORKDIR /app
ENV PATH="${PATH}:/root/.local/bin"
# # Copy the pyproject.toml and poetry.lock files
# COPY poetry.lock pyproject.toml ./
# Copy the rest of the application codes
COPY ./ ./

RUN poetry config virtualenvs.create false
RUN poetry install


# docker build -t dataelement/backend:latest ./
# uvicorn bisheng.main:app --host 0.0.0.0 --port 7862 --workers 2
CMD ["uvicorn", "bisheng.main:app", "--workers", "2",  "--host", "0.0.0.0", "--port", "7860"]
