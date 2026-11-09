-- Habit segmentation (S1..S4) over the 3 months / 14 weeks before {as_of}.
-- Spark SQL: replace date_diff('day', d, X) with datediff(X, d),
--            INTERVAL 3 MONTH with add_months(X, -3), date_trunc('month', d) with trunc(d, 'MM').
WITH daily AS (
    SELECT merchant_id, CAST("date" AS DATE) AS d, SUM(n_attempted) AS msgs
    FROM messages
    WHERE "date" < DATE '{as_of}'
    GROUP BY 1, 2
    HAVING SUM(n_attempted) > 0
),
vol AS (
    SELECT merchant_id,
           SUM(msgs)                               AS msgs_window,
           COUNT(DISTINCT date_trunc('month', d))  AS active_months
    FROM daily
    WHERE d >= DATE '{as_of}' - INTERVAL 3 MONTH
    GROUP BY 1
),
wk AS (
    SELECT merchant_id,
           CAST(floor(date_diff('day', d, DATE '{as_of}') / 7) AS INTEGER) AS week,
           COUNT(*) AS send_days
    FROM daily
    WHERE d >= DATE '{as_of}' - INTERVAL 98 DAY
    GROUP BY 1, 2
),
reg AS (
    SELECT merchant_id, SUM(CASE WHEN send_days >= 2 THEN 1 ELSE 0 END) AS regular_weeks
    FROM wk
    GROUP BY 1
)
SELECT v.merchant_id,
       v.msgs_window,
       v.active_months,
       COALESCE(r.regular_weeks, 0) AS regular_weeks,
       CASE WHEN v.msgs_window >= 500 AND COALESCE(r.regular_weeks, 0) >= 11 THEN 'S1'
            WHEN v.msgs_window >= 500                                      THEN 'S2'
            WHEN v.active_months >= 3                                      THEN 'S3'
            ELSE 'S4' END AS segment
FROM vol v
LEFT JOIN reg r USING (merchant_id)
ORDER BY v.merchant_id;
