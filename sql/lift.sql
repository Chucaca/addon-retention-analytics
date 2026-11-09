-- Profile lift for one banded feature (here: industry) with a pooled two-proportion z-test.
-- `seed` = good adopters (S1-S3), `base` = non-adopters. Descriptive over-representation, not causal uplift.
WITH labelled AS (
    SELECT m.industry AS value,
           CASE WHEN s.segment IN ('S1', 'S2', 'S3') THEN 1 ELSE 0 END AS is_seed,
           CASE WHEN NOT m.is_adopter THEN 1 ELSE 0 END AS is_base
    FROM merchants m
    LEFT JOIN segments s USING (merchant_id)
),
tot AS (SELECT SUM(is_seed) AS n1, SUM(is_base) AS n0 FROM labelled),
cnt AS (SELECT value, SUM(is_seed) AS x1, SUM(is_base) AS x0 FROM labelled GROUP BY 1)
SELECT value, x1 AS n_seed, x0 AS n_base,
       (x1 * 1.0 / n1) / NULLIF(x0 * 1.0 / n0, 0) AS lift,
       ((x1 * 1.0 / n1) - (x0 * 1.0 / n0))
         / sqrt(((x1 + x0) * 1.0 / (n1 + n0)) * (1 - (x1 + x0) * 1.0 / (n1 + n0)) * (1.0 / n1 + 1.0 / n0)) AS z
FROM cnt CROSS JOIN tot
ORDER BY lift DESC;
