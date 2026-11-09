-- Merchant x month activity under both definitions.
-- Spark SQL: date_trunc('month', d) -> trunc(d, 'MM').
SELECT merchant_id,
       date_trunc('month', "date")                    AS month,
       SUM(n_attempted)                               AS msgs,
       SUM(n_success)                                 AS success,
       SUM(n_config_err)                              AS config_err,
       SUM(n_user_err)                                AS user_err,
       COUNT(DISTINCT CAST("date" AS DATE))           AS send_days,
       SUM(n_attempted) >= 1                          AS active_any,
       COUNT(DISTINCT CAST("date" AS DATE)) >= 3      AS active_real
FROM messages
WHERE n_attempted > 0
GROUP BY 1, 2;
