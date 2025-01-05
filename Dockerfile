#Base image
FROM python:3.9-slim-bullseye

#Work directory
WORKDIR /app

#Install poetry dependencies
RUN apt-get update && apt-get install -y curl && apt-get clean

#Install poetry
RUN curl -sSL https://install.python-poetry.org | python -
ENV PATH="/root/.local/bin:$PATH"

#Copy dependency files
COPY pyproject.toml poetry.lock ./

#Install Dependencies
RUN poetry install --only main --no-root

#Copy application code
COPY . .

#Expose port
EXPOSE 8000

#Start app
CMD ["poetry", "run", "python", "app.py"]

