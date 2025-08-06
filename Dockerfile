# syntax=docker/dockerfile:1
FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md LICENSE.txt ./

RUN pip install --upgrade pip setuptools setuptools-scm
RUN pip install . && pip install 'nimbusagent[dev]'

COPY . .
