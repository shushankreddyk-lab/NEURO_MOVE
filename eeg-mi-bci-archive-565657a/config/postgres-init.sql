-- NeuroMove MLflow PostgreSQL Initialization
-- Creates database, user, and schema for MLflow tracking

-- Create database (if not exists)
SELECT 'CREATE DATABASE neuromove'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'neuromove')\gexec

-- Create user (if not exists)
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'neuromove') THEN
        CREATE ROLE neuromove WITH LOGIN PASSWORD 'neuromove_secure_pass';
    END IF;
END$$;

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE neuromove TO neuromove;

-- Connect to neuromove database
\c neuromove

-- Create schema for MLflow
CREATE SCHEMA IF NOT EXISTS mlflow AUTHORIZATION neuromove;

-- Grant schema privileges
GRANT ALL ON SCHEMA mlflow TO neuromove;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA mlflow TO neuromove;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA mlflow TO neuromove;

-- Set default search path
ALTER ROLE neuromove SET search_path TO mlflow, public;

-- Create MLflow tables (MLflow will auto-create on first run, but we can pre-create for control)
-- MLflow handles schema migration automatically

-- Create indexes for performance (optional, MLflow creates these)
-- CREATE INDEX IF NOT EXISTS idx_runs_experiment_id ON mlflow.runs(experiment_id);
-- CREATE INDEX IF NOT EXISTS idx_runs_status ON mlflow.runs(status);
-- CREATE INDEX IF NOT EXISTS idx_experiments_name ON mlflow.experiments(name);
-- CREATE INDEX IF NOT EXISTS idx_model_versions_name ON mlflow.model_versions(name);

-- Log completion
DO $$
BEGIN
    RAISE NOTICE 'NeuroMove MLflow database initialized successfully';
END$$;