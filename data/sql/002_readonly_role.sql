-- Read-only role used by Text2SQL. It can read the 7 demo tables and nothing else.
-- Run with psql and pass the password: psql -v readonly_password=... -f 002_readonly_role.sql

CREATE ROLE readonly LOGIN PASSWORD :'readonly_password';

REVOKE CONNECT, TEMPORARY ON DATABASE rag_engine FROM PUBLIC;
REVOKE TEMPORARY ON DATABASE k8s_ops FROM PUBLIC;

GRANT CONNECT ON DATABASE k8s_ops TO readonly;
GRANT USAGE ON SCHEMA public TO readonly;
GRANT SELECT ON clusters, nodes, deployments, pods, incidents, alerts, oncall_logs TO readonly;
