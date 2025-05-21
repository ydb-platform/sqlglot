import pytest
import dialect_ydb
from dialect_ydb import eliminate_join_marks
from sqlglot import parse_one
from sqlglot.optimizer.unnest_subqueries import unnest_subqueries


def test_full_escaped_table_name():
    sql = """
  CREATE TABLE "SCHEME"."TABLE"
   (	"ID" NUMBER NOT NULL ,
	"A" NUMBER(6,0),
	"B" VARCHAR2(50 CHAR),
	CONSTRAINT "TABLE_PK" PRIMARY KEY ("ID")
   )
"""
    parsed = parse_one(sql)
    assert parsed.sql(dialect="ydb") == """CREATE TABLE `SCHEME/TABLE` (`ID` INT64 NOT NULL, `A` INT32, `B` Utf8, PRIMARY KEY(`ID`))
PARTITION BY HASH (`ID`)
WITH(STORE=COLUMN);"""

def test_datetrunc_year():
    sql = "SELECT DATE_TRUNC('year',dt) from table"
    parsed = parse_one(sql)
    generated_sql = parsed.sql(dialect="ydb")
    assert generated_sql == "SELECT DateTime::MakeDate(DateTime::StartOfYear(dt)) FROM `table`"

def test_datetrunc_month():
    sql = "SELECT DATE_TRUNC('month',dt) from table"
    parsed = parse_one(sql)
    generated_sql = parsed.sql(dialect="ydb")
    assert generated_sql == "SELECT DateTime::MakeDate(DateTime::StartOfMonth(dt)) FROM `table`"

def test_extract():
    sql = "SELECT EXTRACT(YEAR FROM dt) from table"
    parsed = parse_one(sql)
    generated_sql = parsed.sql(dialect="ydb")
    assert generated_sql == "SELECT DateTime::GetYear(dt) FROM `table`"

def test_parse():
    sql = "SELECT to_date('29.03.2023', 'DD.MM.YYYY') from table"
    parsed = parse_one(sql, dialect="oracle")
    generated_sql = parsed.sql(dialect="ydb")
    print(generated_sql)
    assert generated_sql == 'SELECT DateTime::MakeTimestamp(DateTime::Parse(\'%d.%m.%Y\')("29.03.2023")) FROM `table`'

def test_basic():
    sql = "SELECT * FROM x AS x WHERE (SELECT y.a AS a FROM y AS y WHERE x.a = y.a) = 1"
    parsed = parse_one(sql, dialect="oracle")
    new_parsed = unnest_subqueries(parsed)
    assert new_parsed.sql(dialect="ydb") == "SELECT * FROM `x` AS `x` LEFT JOIN (SELECT y.a AS a FROM `y` AS `y` WHERE TRUE GROUP BY y.a) AS _u_0 ON x.a = _u_0.a WHERE _u_0.a = 1"

def test_basic_eq():
    sql = "SELECT * FROM x AS x WHERE exists (SELECT y.a AS a FROM y AS y WHERE x.a = y.a)"
    parsed = parse_one(sql, dialect="oracle")
    new_parsed = unnest_subqueries(parsed)
    assert new_parsed.sql(dialect="ydb") == "SELECT * FROM `x` AS `x` LEFT JOIN (SELECT y.a AS a FROM `y` AS `y` WHERE TRUE GROUP BY y.a) AS _u_0 ON x.a = _u_0.a WHERE NOT (_u_0.a IS NULL)"

def test_basic_agg():
    sql = "SELECT * FROM x WHERE x.a IN (SELECT y.a AS a FROM y WHERE y.b = x.a);"
    parsed = parse_one(sql, dialect="oracle")
    new_parsed = unnest_subqueries(parsed)
    assert new_parsed.sql(dialect="ydb") == "SELECT * FROM `x` LEFT JOIN (SELECT AGGREGATE_LIST(y.a) AS a, y.b AS _u_1 FROM `y` WHERE TRUE GROUP BY y.b) AS _u_0 ON _u_0._u_1 = x.a WHERE ListHasItems(($_x, $p_0)->(ListFilter($_x, ($_x) -> {RETURN $_x = $p_0}))(a, x.a))"

def test_full_escaped_table_name():
    sql = "SELECT * FROM T"
    parsed = parse_one(sql)
    assert parsed.sql(dialect="ydb") == "SELECT * FROM `T`"


def test_subselect():
    sql = "SELECT * FROM (select * from b) T"
    parsed = parse_one(sql)
    assert parsed.sql(dialect="ydb") == "SELECT * FROM (SELECT * FROM `b`) AS T"


def test_full_qualified_alias():
    sql = "SELECT a.a FROM T"
    parsed = parse_one(sql)
    assert parsed.sql(dialect="ydb") == "SELECT a.a AS a FROM `T`"

def test_cte():
    sql = "with ct as (select * from b) SELECT * from ct"
    parsed = parse_one(sql)
    generated_sql = parsed.sql(dialect="ydb")
    assert generated_sql == "$ct = (SELECT * FROM `b`);\n\nSELECT * FROM $ct AS ct"

def test_embedded_cte():
    sql = "SELECT * from (with ct as (select * from b) select * from ct)"
    parsed = parse_one(sql)
    generated_sql = parsed.sql(dialect="ydb")
    assert generated_sql == "$ct = (SELECT * FROM `b`);\n\nSELECT * FROM (SELECT * FROM $ct AS ct)"

def test_array_any():
    sql = "SELECT * FROM TABLE WHERE ARRAY_ANY(arr, x -> x)"
    parsed = parse_one(sql)
    generated_sql = parsed.sql(dialect="ydb")
    assert generated_sql == "SELECT * FROM `TABLE` WHERE ListHasItems(ListFilter(($x) -> {RETURN $x}))"

def test_array_any_complex_filter():
    sql = "SELECT * FROM TABLE WHERE ARRAY_ANY(arr, _x -> _x > 0)"
    parsed = parse_one(sql)
    generated_sql = parsed.sql(dialect="ydb")
    print(generated_sql)
    assert generated_sql == "SELECT * FROM `TABLE` WHERE ListHasItems(ListFilter(($_x) -> {RETURN $_x > 0}))"

def test_array_any_complex_filter_subq():
    sql = """SELECT * FROM data WHERE ARRAY_ANY(arr, x -> x > data.begin);"""
    parsed = parse_one(sql)
    generated_sql = parsed.sql(dialect="ydb")
    print(generated_sql)
    assert generated_sql == "SELECT * FROM `data` WHERE ListHasItems(($x, $p_0)->(ListFilter($x, ($x) -> {RETURN $x > $p_0}))(arr, data.begin))"

def test_concat():
    sql = "SELECT CONCAT(A,B) FROM data"
    parsed = parse_one(sql)
    generated_sql = parsed.sql(dialect="ydb")
    print(generated_sql)
    assert generated_sql == "SELECT A || B FROM `data`"

def test_nullif_notnull():
    sql = "SELECT NULLIF('a','b') FROM data"
    parsed = parse_one(sql)
    generated_sql = parsed.sql(dialect="ydb")
    assert generated_sql == "SELECT IF('a' = 'b', NULL, 'a') FROM `data`"

def test_nullif_null():
    sql = "SELECT NULLIF('a','a') FROM data"
    parsed = parse_one(sql)
    generated_sql = parsed.sql(dialect="ydb")
    assert generated_sql == "SELECT IF('a' = 'a', NULL, 'a') FROM `data`"

def test_if():
    sql = "SELECT IF(10 > 20, 'TRUE', 'FALSE') FROM data"
    parsed = parse_one(sql)
    generated_sql = parsed.sql(dialect="ydb")
    assert generated_sql == "SELECT IF(10 > 20, 'TRUE', 'FALSE') FROM `data`"

def test_basic():
    sql = "select * from a, b where a.id(+) = b.id"
    parsed = parse_one(sql, dialect="oracle")
    new_parsed = eliminate_join_marks(parsed)
    assert new_parsed.sql() == "SELECT * FROM b LEFT JOIN a ON a.id = b.id"

def test_between():
    sql = "SELECT * FROM T WHERE SYSDATE BETWEEN A.valid_from_dttm(+) AND A.valid_to_dttm(+)"
    parsed = parse_one(sql, dialect="oracle")
    new_parsed = eliminate_join_marks(parsed)
    assert new_parsed.sql() == "SELECT * FROM T WHERE CURRENT_TIMESTAMP() BETWEEN A.valid_from_dttm AND A.valid_to_dttm"

# def test_have_old_join():
#     sql = "select * from a, b, c where a.id(+) = b.id and b.id=c.id"
#     parsed = parse_one(sql, dialect="oracle")
#     new_parsed = eliminate_join_marks2(parsed)
#     assert new_parsed.sql() == "SELECT * FROM c LEFT JOIN a ON a.id = b.id WHERE b.id = c.id"

# def test_or_join():
#     sql = "SELECT * FROM a, b, c WHERE a.id = b.id AND b.id(+) = c.id"
#     parsed = parse_one(sql, dialect="oracle")
#     new_parsed = eliminate_join_marks2(parsed)
#     print(new_parsed.sql())
#     assert new_parsed.sql() == "SELECT * FROM a, b LEFT JOIN c ON b.id = c.id WHERE a.id = b.id"

def test_table_name_lower_case():
    sql = "SELECT * FROM B"
    parsed = parse_one(sql)
    parsed_new = table_names_to_lower_case(parsed)
    assert parsed_new.sql() == "SELECT * FROM b"

def test_tables_name_lower_case():
    sql = "SELECT * FROM B, (SELECT * from D) as E"
    parsed = parse_one(sql)
    parsed_new = table_names_to_lower_case(parsed)
    assert parsed_new.sql() == "SELECT * FROM b, (SELECT * FROM d) AS E"


