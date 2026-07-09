CREATE TABLE runs(
    id SERIAL PRIMARY KEY,
    d_model INTEGER NOT NULL,
    lr REAL NOT NULL,
    num_of_blocks INTEGER NOT NULL,
    num_of_heads INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE epochs(
    id SERIAL PRIMARY KEY,
    run_id INTEGER REFERENCES runs(id) ON DELETE CASCADE,  
    epoch_num INTEGER NOT NULL,
    train_loss REAL NOT NULL,
    val_loss REAL NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);