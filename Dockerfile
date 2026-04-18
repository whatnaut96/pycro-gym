FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml poetry.lock* ./
COPY ./src ./src
RUN pip install poetry
ENV POETRY_VIRTUALENVS_CREATE=false
RUN poetry lock
RUN poetry install --no-root
CMD ["poetry", "run", "pycro-gym"]
