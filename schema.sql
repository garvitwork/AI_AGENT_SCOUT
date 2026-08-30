-- SCOUT database schema
-- Tables are created in whatever database the connection targets
-- (set via SCOUT_DB_NAME in .env — e.g. "defaultdb" on Aiven).

CREATE TABLE IF NOT EXISTS suppliers (
    supplier_id     INT AUTO_INCREMENT PRIMARY KEY,
    name            VARCHAR(150) NOT NULL,
    country         VARCHAR(80),
    region          VARCHAR(80),
    category        VARCHAR(80),
    lead_time_days  INT,
    risk_score      FLOAT DEFAULT NULL,   -- updated by ML pipeline later
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS shipments (
    shipment_id             INT AUTO_INCREMENT PRIMARY KEY,
    supplier_id             INT,
    order_date              DATE,
    expected_delivery_date  DATE,
    actual_delivery_date    DATE,
    status                  VARCHAR(30),        -- e.g. delivered, delayed, in_transit
    quantity                INT,
    product_category        VARCHAR(100),
    origin                  VARCHAR(100),
    destination             VARCHAR(100),
    transport_mode          VARCHAR(50),        -- road, sea, air, rail
    delay_days              INT DEFAULT 0,
    FOREIGN KEY (supplier_id) REFERENCES suppliers(supplier_id)
);

CREATE TABLE IF NOT EXISTS risk_events (
    event_id             INT AUTO_INCREMENT PRIMARY KEY,
    event_type           VARCHAR(60),      -- weather, port_congestion, geopolitical, news
    source               VARCHAR(100),
    description          TEXT,
    severity             ENUM('low','medium','high') DEFAULT 'low',
    region               VARCHAR(80),
    event_date           DATETIME,
    related_shipment_id  INT DEFAULT NULL,
    FOREIGN KEY (related_shipment_id) REFERENCES shipments(shipment_id)
);

CREATE TABLE IF NOT EXISTS model_predictions (
    prediction_id       INT AUTO_INCREMENT PRIMARY KEY,
    shipment_id         INT,
    model_name          VARCHAR(80),
    model_version       VARCHAR(30),
    risk_probability    FLOAT,
    predicted_delay_days FLOAT,
    prediction_date     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (shipment_id) REFERENCES shipments(shipment_id)
);