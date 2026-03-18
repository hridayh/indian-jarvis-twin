## Labels format (for `src/train.py`)

Create `data/labels/labels.csv` with columns:

- `client_id`: string (same as WhatsApp number normalization, e.g. `+91...`)
- `as_of`: ISO datetime (UTC preferred). The trainer will pick the nearest **state snapshot at-or-before** this time (within 7 days).
- plus **one or more** binary label columns (0/1). For your MVP we recommend these 8 columns:
  - `stockout_lowstock_w1`, `stockout_lowstock_m1`
  - `stockout_nosales_w1`, `stockout_nosales_m1`
  - `payment_overdue14_w1`, `payment_overdue14_m1`
  - `payment_partial_w1`, `payment_partial_m1`

Example:

```csv
client_id,as_of,stockout_lowstock_w1,stockout_lowstock_m1,stockout_nosales_w1,stockout_nosales_m1,payment_overdue14_w1,payment_overdue14_m1,payment_partial_w1,payment_partial_m1
+918888000000,2026-03-15T10:00:00Z,1,1,0,0,0,0,0,0
+918888000000,2026-03-20T10:00:00Z,0,0,1,1,0,0,1,1
```

