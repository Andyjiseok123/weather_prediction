
CREATE TABLE IF NOT EXISTS weather_models (
    model_id VARCHAR(255) PRIMARY KEY,

    target_column VARCHAR(100) NOT NULL,
    last_observed_date DATE,
    lookback INTEGER NOT NULL,
    n_features INTEGER NOT NULL,

    x_model BYTEA NOT NULL,
    y_model BYTEA NOT NULL,
    lightgbm_model BYTEA NOT NULL,
    metadata_pickle BYTEA NOT NULL,

    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    file_checksums JSONB NOT NULL DEFAULT '{}'::jsonb,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_weather_models_target
    ON weather_models (target_column);
