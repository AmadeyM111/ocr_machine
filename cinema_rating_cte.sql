/* without CTE*/

SELECT u.name
FROM Users u
JOIN MovieRating mr
    ON u.user_id = mr.user_id
GROUP BY u.user_id, u.name
ORDER BY COUNT(*) DESC, u.name
LIMIT 1;

/* with CTE */
WITH user_stats AS (
    SELECT
        u.name,
        COUNT(*) AS rating_count
    FROM User u
    JOIN MovieRating mr
        ON u.user_id = mr.user_id
    GROUP BY u.user_id, u.name
)

SELECT name
FROM user_stats
ORDER BY ratings_count DESC, name ASC
LIMIT 1;