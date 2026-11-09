-- Churn label: active in M, nothing in M+1 and M+2. Needs a dense merchant x month grid.
-- {first_month} / {last_month} are the first and last months of the data (DATE, first of month).
-- Spark SQL: build the calendar with sequence(...) + explode, use add_months instead of INTERVAL.
WITH act AS (
    SELECT merchant_id, date_trunc('month', "date") AS month, SUM(n_attempted) AS msgs
    FROM messages GROUP BY 1, 2
),
cal AS (
    SELECT CAST(m AS DATE) AS month
    FROM generate_series(DATE '{first_month}', DATE '{last_month}', INTERVAL 1 MONTH) t(m)
),
firsts AS (SELECT merchant_id, MIN(month) AS first_month FROM act WHERE msgs > 0 GROUP BY 1),
grid AS (
    SELECT f.merchant_id, c.month, COALESCE(a.msgs, 0) AS msgs
    FROM firsts f
    JOIN cal c ON c.month >= f.first_month
    LEFT JOIN act a ON a.merchant_id = f.merchant_id AND a.month = c.month
)
SELECT merchant_id, month, msgs,
       CASE WHEN month + INTERVAL 2 MONTH > DATE '{last_month}' THEN NULL
            WHEN LEAD(msgs, 1, 0) OVER w = 0 AND LEAD(msgs, 2, 0) OVER w = 0 THEN 1
            ELSE 0 END AS churn
FROM grid
WHERE msgs > 0
WINDOW w AS (PARTITION BY merchant_id ORDER BY month);
