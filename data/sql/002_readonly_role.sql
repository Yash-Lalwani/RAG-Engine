-- Read-only role used by Text2SQL. It can read the 7 demo tables and nothing else.

CREATE ROLE readonly LOGIN PASSWORD 'readonly';

REVOKE CONNECT, TEMPORARY ON DATABASE rag_engine FROM PUBLIC;
REVOKE TEMPORARY ON DATABASE k8s_ops FROM PUBLIC;

GRANT CONNECT ON DATABASE k8s_ops TO readonly;
GRANT USAGE ON SCHEMA public TO readonly;
GRANT SELECT ON clusters, nodes, deployments, pods, incidents, alerts, oncall_logs TO readonly;
