#!/bin/bash
# Runs once, when the Postgres volume is first created.
# The container's own database (rag_engine) already exists at this point.
set -euo pipefail

psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "CREATE DATABASE k8s_ops"
psql -v ON_ERROR_STOP=1 -q -U "$POSTGRES_USER" -d k8s_ops -f /sql/001_k8s_ops.sql
# The local role password is "readonly" (matches SQL_DATABASES in .env.example).
psql -v ON_ERROR_STOP=1 -v readonly_password=readonly -U "$POSTGRES_USER" -d k8s_ops -f /sql/002_readonly_role.sql
