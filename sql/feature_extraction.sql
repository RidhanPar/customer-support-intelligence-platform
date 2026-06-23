-- Feature extraction query for SLA breach prediction.
-- SQLite-compatible. Run against a database populated by src/db_loader.load_to_sqlite().
--
-- Techniques demonstrated:
--   * CTE (WITH clause) for multi-step computation
--   * Window functions: AVG() OVER (PARTITION BY ... ROWS BETWEEN ...) and ROW_NUMBER() OVER (...)
--   * Computed columns: sla_target_hours, resolution_hours, sla_breach, sla_time_consumed_pct

WITH ticket_enriched AS (
    -- Step 1: Derive resolution hours and SLA targets from raw fields.
    SELECT
        ticket_id,
        created_date,
        resolved_date,
        priority,
        category,
        channel,
        team,
        status,
        escalated,
        customer_satisfaction,
        customer_message,
        LENGTH(customer_message)                                    AS message_length,
        CAST(STRFTIME('%H', created_date) AS INTEGER)              AS hour_created,
        CASE STRFTIME('%w', created_date)
            WHEN '0' THEN 1  -- Sunday
            WHEN '6' THEN 1  -- Saturday
            ELSE 0
        END                                                         AS is_weekend,
        CASE priority
            WHEN 'Critical' THEN 6
            WHEN 'High'     THEN 12
            WHEN 'Medium'   THEN 24
            WHEN 'Low'      THEN 48
            ELSE 24
        END                                                         AS sla_target_hours,
        ROUND(
            (JULIANDAY(COALESCE(resolved_date, DATETIME('now')))
             - JULIANDAY(created_date)) * 24.0,
            2
        )                                                           AS resolution_hours
    FROM support_tickets
)
SELECT
    te.ticket_id,
    te.created_date,
    te.priority,
    te.category,
    te.channel,
    te.team,
    te.status,
    te.escalated,
    te.customer_satisfaction,
    te.message_length,
    te.hour_created,
    te.is_weekend,
    te.resolution_hours,
    te.sla_target_hours,

    -- Breach label (mirrors src/data_cleaning.py logic)
    CASE WHEN te.resolution_hours > te.sla_target_hours THEN 1 ELSE 0 END
        AS sla_breach,

    -- Risk signal: % of SLA window consumed; values above 100 indicate an active breach
    ROUND(te.resolution_hours / NULLIF(te.sla_target_hours, 0) * 100.0, 2)
        AS sla_time_consumed_pct,

    -- Window function 1: rolling average resolution time per team (last 10 tickets)
    -- Captures team-level workload pressure as a breach risk signal
    ROUND(
        AVG(te.resolution_hours) OVER (
            PARTITION BY te.team
            ORDER BY te.created_date
            ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
        ),
        2
    ) AS rolling_avg_resolution_by_team,

    -- Window function 2: sequence number within each team ordered by creation date
    -- Useful for detecting early-vs-late handling patterns within a team
    ROW_NUMBER() OVER (
        PARTITION BY te.team
        ORDER BY te.created_date
    ) AS team_ticket_sequence

FROM ticket_enriched te
ORDER BY te.created_date;
