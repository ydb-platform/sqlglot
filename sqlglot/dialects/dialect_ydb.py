from sqlglot import exp, tokens, generator, transforms
from sqlglot.dialects.dialect import Dialect, rename_func
from sqlglot.dialects.dialect import NormalizationStrategy, concat_to_dpipe_sql
from sqlglot.transforms import move_ctes_to_top_level
from sqlglot.helper import name_sequence
import typing as t

from sqlglot.optimizer.scope import find_in_scope, ScopeType, traverse_scope


def print_command_structure(cmd_obj, prefix="", max_depth=5, current_depth=0):
    """
    Prints the structure of a Command object for debugging

    Args:
        cmd_obj: SQLGlot object
        prefix (str): Prefix for indentation
        max_depth (int): Maximum recursion depth
        current_depth (int): Current depth
    """
    if current_depth > max_depth:
        print(f"{prefix}... (maximum depth reached)")
        return

    if cmd_obj is None:
        print(f"{prefix}None")
        return

    print(f"{prefix}Type: {type(cmd_obj).__name__}")

    # Check main attributes
    for attr in ['this', 'name', 'expressions', 'expression', 'args']:
        if hasattr(cmd_obj, attr):
            attr_value = getattr(cmd_obj, attr)
            print(f"{prefix}{attr}: {attr_value}")

            # Recursively print structure for complex attributes
            if attr in ['expression', 'this'] and isinstance(attr_value, (exp.Expression, exp.Command)):
                print_command_structure(attr_value, prefix + "  ", max_depth, current_depth + 1)
            elif attr == 'expressions' and isinstance(attr_value, list):
                for i, expr in enumerate(attr_value):
                    print(f"{prefix}  expressions[{i}]:")
                    print_command_structure(expr, prefix + "    ", max_depth, current_depth + 1)


def table_names_to_lower_case(expression: exp.Expression) -> exp.Expression:
    """
    Converts all table names to lowercase

    Args:
        expression: The SQL expression to modify

    Returns:
        Modified expression with lowercase table names
    """
    for table in expression.find_all(exp.Table):
        if isinstance(table.this, exp.Identifier):
            ident = table.this

            table.set("this", ident.this.lower())

    return expression


def eliminate_join_marks(expression: exp.Expression) -> exp.Expression:
    """
    Remove join marks from an AST. This rule assumes that all marked columns are qualified.
    If this does not hold for a query, consider running `sqlglot.optimizer.qualify` first.

    For example,
        SELECT * FROM a, b WHERE a.id = b.id(+)    -- ... is converted to
        SELECT * FROM a LEFT JOIN b ON a.id = b.id -- this

    Args:
        expression: The AST to remove join marks from.

    Returns:
       The AST with join marks removed.
    """
    from sqlglot.optimizer.scope import traverse_scope

    for scope in traverse_scope(expression):
        query = scope.expression

        where = query.args.get("where")
        joins = query.args.get("joins")

        if not where or not joins:
            continue

        query_from = query.args["from"]

        # These keep track of the joins to be replaced
        new_joins: t.Dict[str, exp.Join] = {}
        old_joins = {join.alias_or_name: join for join in joins}

        for column in scope.columns:
            if not column.args.get("join_mark"):
                continue

            predicate = column.find_ancestor(exp.Predicate, exp.Select)
            if not isinstance(
                predicate, exp.Binary):
                continue
            # TODO: support between operations
            assert isinstance(
                predicate, exp.Binary
            ), "Columns can only be marked with (+) when involved in a binary operation"

            predicate_parent = predicate.parent
            join_predicate = predicate.pop()

            left_columns = [
                c for c in join_predicate.left.find_all(exp.Column) if c.args.get("join_mark")
            ]
            right_columns = [
                c for c in join_predicate.right.find_all(exp.Column) if c.args.get("join_mark")
            ]

            assert not (
                left_columns and right_columns
            ), "The (+) marker cannot appear in both sides of a binary predicate"

            marked_column_tables = set()
            for col in left_columns or right_columns:
                table = col.table
                assert table, f"Column {col} needs to be qualified with a table"

                col.set("join_mark", False)
                marked_column_tables.add(table)

            assert (
                len(marked_column_tables) == 1
            ), "Columns of only a single table can be marked with (+) in a given binary predicate"

            # Add predicate if join already copied, or add join if it is new
            join_this = old_joins.get(col.table, query_from).this
            existing_join = new_joins.get(join_this.alias_or_name)
            if existing_join:
                existing_join.set("on", exp.and_(existing_join.args["on"], join_predicate))
            else:
                new_joins[join_this.alias_or_name] = exp.Join(
                    this=join_this.copy(), on=join_predicate.copy(), kind="LEFT"
                )

            # If the parent of the target predicate is a binary node, then it now has only one child
            if isinstance(predicate_parent, exp.Binary):
                if predicate_parent.left is None:
                    predicate_parent.replace(predicate_parent.right)
                else:
                    predicate_parent.replace(predicate_parent.left)

        only_old_join_sources = old_joins.keys() - new_joins.keys()

        if query_from.alias_or_name in new_joins:
            if len(only_old_join_sources) == 0:
                sql = predicate_parent.parent.sql()

            assert (
                len(only_old_join_sources) >= 1
            ), "Cannot determine which table to use in the new FROM clause"

            new_from_name = list(only_old_join_sources)[0]
            query.set("from", exp.From(this=old_joins.pop(new_from_name).this))
            only_old_join_sources.remove(new_from_name)

        if new_joins:
            only_old_join_expressions = []
            for old_join_source in only_old_join_sources:
                old_join_expression = old_joins[old_join_source]
                if not old_join_expression.kind:
                    old_join_expression.set("kind", "CROSS")

                only_old_join_expressions.append(old_join_expression)

            query.set("joins", list(new_joins.values()) + only_old_join_expressions)

        if not where.this:
            where.pop()

    return expression

# def quote_table_names(expression: exp.Expression ) -> exp.Expression:
#     print_command_structure(expression)
#     for table in expression.find_all(exp.Table):
#         if isinstance(table.this, exp.Identifier):
#             alias_or_name = table.this.alias_or_name
#             if alias_or_name[0] != "`":
#                 table.this.replace(exp.to_identifier(f"`{alias_or_name}`", quoted=False))
#
#     print_command_structure(expression)
#     return expression

def lower_table_names(expression: exp.Expression) -> exp.Expression:
    """
    Converts all table names to lowercase

    Args:
        expression: The SQL expression to modify

    Returns:
        Modified expression with lowercase table names
    """
    # print_command_structure(expression)
    for table in expression.find_all(exp.Table):
        if isinstance(table.this, exp.Identifier):
            alias_or_name = table.this.alias_or_name
            if alias_or_name[0] != "`":
                table.this.replace(exp.to_identifier(f"{alias_or_name.lower()}", quoted=False))

    # print_command_structure(expression)
    return expression

def make_db_name_upper(expression: exp.Expression) -> exp.Expression:
    """
    Converts all database names to uppercase

    Args:
        expression: The SQL expression to modify

    Returns:
        Modified expression with uppercase database names
    """
    for table in expression.find_all(exp.Table):
        if table.db:
            table.set("db", table.db.upper())

    # print_command_structure(expression)
    return expression

def apply_alias_to_select_from_table(expression: exp.Expression) -> str:
    """
    Applies aliases to columns in SELECT statements that reference tables

    Args:
        expression: The SQL expression to modify

    Returns:
        Modified expression with aliases applied to columns
    """
    for column in expression.find_all(exp.Column):
        if len(column.table) > 0:
            if isinstance(column.parent, exp.Select):
                column.replace(exp.alias_(column, column.alias_or_name))

    return expression

def get_create_name(parsed):
    """
    Gets the name from a CREATE statement

    Args:
        parsed: The parsed SQL expression

    Returns:
        The name of the created object or None
    """
    if isinstance(parsed, exp.Command):
        # Check if it's CREATE
        if parsed.this.upper() == "CREATE":
            # Check if the next command is VIEW
            if hasattr(parsed, 'expression') and isinstance(parsed.expression, exp.Command):
                view_cmd = parsed.expression
                if view_cmd.this.upper() == "VIEW":
                    # Get the view name
                    if hasattr(view_cmd, 'expression') and hasattr(view_cmd.expression, 'this'):
                        # Return the view name
                        return view_cmd.expression.this
                    elif hasattr(view_cmd, 'expression') and isinstance(view_cmd.expression, exp.Identifier):
                        return view_cmd.expression.name
    # Case 2: When CREATE VIEW is represented as Create with View
    elif isinstance(parsed, exp.Create):
        if hasattr(parsed, 'this') and isinstance(parsed.this, exp.Schema):
            if hasattr(parsed.this, 'this') and hasattr(parsed.this.this, 'name'):
                return parsed.this.this.name
            elif hasattr(parsed.this, 'this') and isinstance(parsed.this.this, exp.Identifier):
                return parsed.this.this.name
    return None

def unnest_subqueries(expression):
    """
    Rewrite sqlglot AST to convert some predicates with subqueries into joins.

    Convert scalar subqueries into cross joins.
    Convert correlated or vectorized subqueries into a group by so it is not a many to many left join.

    Example:
        >>> import sqlglot
        >>> expression = sqlglot.parse_one("SELECT * FROM x AS x WHERE (SELECT y.a AS a FROM y AS y WHERE x.a = y.a) = 1 ")
        >>> unnest_subqueries(expression).sql()
        'SELECT * FROM x AS x LEFT JOIN (SELECT y.a AS a FROM y AS y WHERE TRUE GROUP BY y.a) AS _u_0 ON x.a = _u_0.a WHERE _u_0.a = 1'

    Args:
        expression (sqlglot.Expression): expression to unnest
    Returns:
        sqlglot.Expression: unnested expression
    """
    next_alias_name = name_sequence("_u_")

    for scope in traverse_scope(expression):
        select = scope.expression
        parent = select.parent_select
        if not parent:
            continue
        if scope.external_columns:
            decorrelate(select, parent, scope.external_columns, next_alias_name)
        elif scope.scope_type == ScopeType.SUBQUERY:
            unnest(select, parent, next_alias_name)

    return expression


def unnest(select, parent_select, next_alias_name):
    """
    Unnests a subquery by transforming it into a join

    Args:
        select: The subquery to unnest
        parent_select: The parent select statement
        next_alias_name: Function to generate the next unique alias name
    """
    if len(select.selects) > 1:
        return

    predicate = select.find_ancestor(exp.Condition)
    if (
        not predicate
        or parent_select is not predicate.parent_select
        or not parent_select.args.get("from")
    ):
        return

    if isinstance(select, exp.SetOperation):
        select = exp.select(*select.selects).from_(select.subquery(next_alias_name()))

    alias = next_alias_name()
    clause = predicate.find_ancestor(exp.Having, exp.Where, exp.Join)

    # This subquery returns a scalar and can just be converted to a cross join
    if not isinstance(predicate, (exp.In, exp.Any)):
        column = exp.column(select.selects[0].alias_or_name, alias)

        clause_parent_select = clause.parent_select if clause else None

        if (isinstance(clause, exp.Having) and clause_parent_select is parent_select) or (
            (not clause or clause_parent_select is not parent_select)
            and (
                parent_select.args.get("group")
                or any(find_in_scope(select, exp.AggFunc) for select in parent_select.selects)
            )
        ):
            column = exp.Max(this=column)
        elif not isinstance(select.parent, exp.Subquery):
            return

        _replace(select.parent, column)
        parent_select.join(select, join_type="CROSS", join_alias=alias, copy=False)
        return

    if select.find(exp.Limit, exp.Offset):
        return

    if isinstance(predicate, exp.Any):
        predicate = predicate.find_ancestor(exp.EQ)

        if not predicate or parent_select is not predicate.parent_select:
            return

    column = _other_operand(predicate)
    value = select.selects[0]

    join_key = exp.column(value.alias, alias)
    join_key_not_null = join_key.is_(exp.null()).not_()

    if isinstance(clause, exp.Join):
        _replace(predicate, exp.true())
        parent_select.where(join_key_not_null, copy=False)
    else:
        _replace(predicate, join_key_not_null)

    group = select.args.get("group")

    if group:
        if {value.this} != set(group.expressions):
            select = (
                exp.select(exp.alias_(exp.column(value.alias, "_q"), value.alias))
                .from_(select.subquery("_q", copy=False), copy=False)
                .group_by(exp.column(value.alias, "_q"), copy=False)
            )
    elif not find_in_scope(value.this, exp.AggFunc):
        select = select.group_by(value.this, copy=False)

    parent_select.join(
        select,
        on=column.eq(join_key),
        join_type="LEFT",
        join_alias=alias,
        copy=False,
    )


def decorrelate(select, parent_select, external_columns, next_alias_name):
    """
    Decorrelates a subquery by transforming it into a join

    Args:
        select: The subquery to decorrelate
        parent_select: The parent select statement
        external_columns: Columns referenced from outside the subquery
        next_alias_name: Function to generate the next unique alias name
    """
    where = select.args.get("where")

    if not where or where.find(exp.Or) or select.find(exp.Limit, exp.Offset):
        return

    table_alias = next_alias_name()
    keys = []

    # for all external columns in the where statement, find the relevant predicate
    # keys to convert it into a join
    for column in external_columns:
        if column.find_ancestor(exp.Where) is not where:
            return

        predicate = column.find_ancestor(exp.Predicate)

        if not predicate or predicate.find_ancestor(exp.Where) is not where:
            return

        if isinstance(predicate, exp.Binary):
            key = (
                predicate.right
                if any(node is column for node in predicate.left.walk())
                else predicate.left
            )
        else:
            return

        keys.append((key, column, predicate))

    if not any(isinstance(predicate, exp.EQ) for *_, predicate in keys):
        return

    is_subquery_projection = any(
        node is select.parent
        for node in map(lambda s: s.unalias(), parent_select.selects)
        if isinstance(node, exp.Subquery)
    )

    value = select.selects[0]
    key_aliases = {}
    group_by = []

    for key, _, predicate in keys:
        # if we filter on the value of the subquery, it needs to be unique
        if key == value.this:
            key_aliases[key] = value.alias
            group_by.append(key)
        else:
            if key not in key_aliases:
                key_aliases[key] = next_alias_name()
            # all predicates that are equalities must also be in the unique
            # so that we don't do a many to many join
            if isinstance(predicate, exp.EQ) and key not in group_by:
                group_by.append(key)

    parent_predicate = select.find_ancestor(exp.Predicate)

    # if the value of the subquery is not an agg or a key, we need to collect it into an array
    # so that it can be grouped. For subquery projections, we use a MAX aggregation instead.
    agg_func = exp.Max if is_subquery_projection else exp.ArrayAgg
    if not value.find(exp.AggFunc) and value.this not in group_by:
        select.select(
            exp.alias_(agg_func(this=value.this), value.alias_or_name, quoted=False),
            append=False,
            copy=False,
        )

    # exists queries should not have any selects as it only checks if there are any rows
    # all selects will be added by the optimizer and only used for join keys
    if isinstance(parent_predicate, exp.Exists):
        select.args["expressions"] = []

    for key, alias in key_aliases.items():
        if key in group_by:
            # add all keys to the projections of the subquery
            # so that we can use it as a join key
            if isinstance(parent_predicate, exp.Exists) or key != value.this:
                select.select(f"{key} AS {alias}", copy=False)
        else:
            select.select(exp.alias_(agg_func(this=key.copy()), alias, quoted=False), copy=False)

    alias = exp.column(value.alias_or_name, table_alias)
    other = _other_operand(parent_predicate)
    op_type = type(parent_predicate.parent) if parent_predicate else None

    if isinstance(parent_predicate, exp.Exists):
        alias = exp.column(list(key_aliases.values())[0], table_alias)
        parent_predicate = _replace(parent_predicate, f"NOT {alias} IS NULL")
    elif isinstance(parent_predicate, exp.All):
        assert issubclass(op_type, exp.Binary)
        predicate = op_type(this=other, expression=exp.column("_x"))
        parent_predicate = _replace(
            parent_predicate.parent, f"ARRAY_ALL({alias}, _x -> {predicate})"
        )
    elif isinstance(parent_predicate, exp.Any):
        assert issubclass(op_type, exp.Binary)
        if value.this in group_by:
            predicate = op_type(this=other, expression=alias)
            parent_predicate = _replace(parent_predicate.parent, predicate)
        else:
            predicate = op_type(this=other, expression=exp.column("_x"))
            parent_predicate = _replace(parent_predicate, f"ARRAY_ANY({alias}, _x -> {predicate})")
    elif isinstance(parent_predicate, exp.In):
        if value.this in group_by:
            parent_predicate = _replace(parent_predicate, f"{other} = {alias}")
        else:
            parent_predicate = _replace(
                parent_predicate,
                f"ARRAY_ANY({alias}, _x -> _x = {parent_predicate.this})",
            )
    else:
        if is_subquery_projection and select.parent.alias:
            alias = exp.alias_(alias, select.parent.alias)

        # COUNT always returns 0 on empty datasets, so we need take that into consideration here
        # by transforming all counts into 0 and using that as the coalesced value
        if value.find(exp.Count):

            def remove_aggs(node):
                if isinstance(node, exp.Count):
                    return exp.Literal.number(0)
                elif isinstance(node, exp.AggFunc):
                    return exp.null()
                return node

            alias = exp.Coalesce(this=alias, expressions=[value.this.transform(remove_aggs)])

        select.parent.replace(alias)

    for key, column, predicate in keys:
        predicate.replace(exp.true())
        nested = exp.column(key_aliases[key], table_alias)

        if is_subquery_projection:
            key.replace(nested)
            if not isinstance(predicate, exp.EQ):
                parent_select.where(predicate, copy=False)
            continue

        if key in group_by:
            key.replace(nested)
        elif isinstance(predicate, exp.EQ):
            parent_predicate = _replace(
                parent_predicate,
                f"({parent_predicate} AND ARRAY_CONTAINS({nested}, {column}))",
            )
        else:
            key.replace(exp.to_identifier("_x"))
            parent_predicate = _replace(
                parent_predicate,
                f"({parent_predicate} AND ARRAY_ANY({nested}, _x -> {predicate}))",
            )

    parent_select.join(
        select.group_by(*group_by, copy=False),
        on=[predicate for *_, predicate in keys if isinstance(predicate, exp.EQ)],
        join_type="LEFT",
        join_alias=table_alias,
        copy=False,
    )


def _replace(expression, condition):
    """
    Helper function to replace an expression with a condition

    Args:
        expression: The expression to replace
        condition: The condition to replace with

    Returns:
        The replaced expression
    """
    return expression.replace(exp.condition(condition))


def _other_operand(expression):
    """
    Returns the other operand of a binary operation involving a subquery

    Args:
        expression: The expression containing a binary operation

    Returns:
        The operand that is not a subquery, or None
    """
    if isinstance(expression, exp.In):
        return expression.this

    if isinstance(expression, (exp.Any, exp.All)):
        return _other_operand(expression.parent)

    if isinstance(expression, exp.Binary):
        return (
            expression.right
            if isinstance(expression.left, (exp.Subquery, exp.Any, exp.Exists, exp.All))
            else expression.left
        )

    return None

class YDB(Dialect):
    """
    YDB SQL dialect implementation for sqlglot.
    Implements the specific syntax and features of YDB database.
    """

    DATE_FORMAT = "'%Y-%m-%d'"
    TIME_FORMAT = "'%Y-%m-%d %H:%M:%S'"

    TIME_MAPPING = {
        "%Y": "%Y",
        "%m": "%m",
        "%d": "%d",
        "%H": "%H",
        "%M": "%M",
        "%S": "%S",
    }

    class Tokenizer(tokens.Tokenizer):
        """
        Tokenizer implementation for YDB SQL dialect.
        Defines how the SQL text is broken into tokens.
        """
        SUPPORTS_VALUES_DEFAULT = False
        QUOTES = ["'", '"']
        COMMENTS = ["--", ("/*", "*/")]
        IDENTIFIERS = ["`"]

    class Generator(generator.Generator):
        """
        SQL Generator for YDB dialect.
        Responsible for translating SQL AST back to SQL text with YDB-specific syntax.
        """
        SUPPORTS_VALUES_DEFAULT = False
        NORMALIZATION_STRATEGY = NormalizationStrategy.CASE_SENSITIVE
        JOIN_HINTS = False
        TABLE_HINTS = False
        QUERY_HINTS = False
        NVL2_SUPPORTED = False
        JSON_PATH_BRACKETED_KEY_SUPPORTED = False
        SUPPORTS_CREATE_TABLE_LIKE = False
        SUPPORTS_TABLE_ALIAS_COLUMNS = False
        SUPPORTS_TO_NUMBER = False
        EXCEPT_INTERSECT_SUPPORT_ALL_CLAUSE = False
        SUPPORTS_MEDIAN = False
        JSON_KEY_VALUE_PAIR_SEP = ","
        VARCHAR_REQUIRES_SIZE = False
        CAN_IMPLEMENT_ARRAY_ANY = True

        PARAMETERIZABLE_TEXT_TYPES = []

        def __init__(self, **kwargs):
            """
            Initialize the YDB SQL Generator with optional configuration.

            Args:
                **kwargs: Additional keyword arguments to pass to the parent Generator.
            """
            super().__init__(**kwargs)

        def create_sql(self, expression: exp.Create, pretty=True) -> str:
            """
            Generate SQL for CREATE expressions with special handling for CREATE VIEW.

            Args:
                expression: The CREATE expression to generate SQL for
                pretty: Whether to format the SQL with indentation

            Returns:
                Generated SQL string
            """
            if expression.kind == "VIEW":
                if expression.this and expression.this.this:
                    ident = expression.this.this
                    ident_sql = self.sql(ident)
                    sql = self.sql(expression.expression)

                    return f"CREATE VIEW {ident_sql} WITH (security_invoker = TRUE) AS {sql}"
                else:
                    return super().create_sql(expression)
            else:
                return super().create_sql(expression)

        def table_sql(self, expression: exp.Table, copy = True) -> str:
            """
            Generate SQL for TABLE expressions with proper quoting and database prefix.

            Args:
                expression: The TABLE expression
                copy: Whether to copy the expression before processing

            Returns:
                Generated SQL string for the table reference
            """
            prefix = f"{expression.db}/" if expression.db else ""
            sql = f"`{prefix}{expression.name}`"

            if expression.alias:
                sql += f" AS `{expression.alias}`"

            return sql

        def is_sql(self, expression: exp.Is) -> str:
            """
            Generate SQL for IS expressions with special handling for IS NOT NULL.

            Args:
                expression: The IS expression

            Returns:
                Generated SQL string
            """
            is_sql = super().is_sql(expression)

            if isinstance(expression.parent, exp.Not):
                # value IS NOT NULL -> NOT (value IS NULL)
                is_sql = self.wrap(is_sql)

            return is_sql

        def datatype_sql(self, expression: exp.DataType) -> str:
            """
            Generate SQL for data type expressions with YDB-specific type mapping.

            Args:
                expression: The data type expression

            Returns:
                Generated SQL string for the data type
            """
            if expression.this in self.PARAMETERIZABLE_TEXT_TYPES and (
                not expression.expressions or expression.expressions[0].name == "MAX"
            ):
                expression = exp.DataType.build("text")
            elif expression.is_type(exp.DataType.Type.NVARCHAR) or expression.is_type(exp.DataType.Type.VARCHAR)  or expression.is_type(exp.DataType.Type.CHAR):
                expression = exp.DataType.build("text")
            elif expression.is_type(exp.DataType.Type.DECIMAL):
                size_expressions = list(expression.find_all(exp.DataTypeParam))

                if not size_expressions:
                    expression = exp.DataType.build("int64")
                else:
                    if len(size_expressions) == 1 or (len(size_expressions) == 2 and int(size_expressions[1].name) == 0):
                        if isinstance(size_expressions[0].this, exp.Star):
                            expression = exp.DataType.build("int64")
                        else:
                            mantis = int(size_expressions[0].name)
                            if mantis < 10:
                                expression = exp.DataType.build("int32")
                            else:
                                expression = exp.DataType.build("int64")
                    else:
                        exponent = int(size_expressions[1].name)
                        expression = exp.DataType.build("Double")
            elif expression.is_type(exp.DataType.Type.TIMESTAMP):
                expression = exp.DataType.build("Timestamp")
            elif expression.this in exp.DataType.TEMPORAL_TYPES:
                expression = exp.DataType.build(expression.this)
            elif expression.is_type("float"):
                size_expression = expression.find(exp.DataTypeParam)
                if size_expression:
                    size = int(size_expression.name)
                    expression = (
                        exp.DataType.build("float") if size <= 32 else exp.DataType.build("double")
                    )

            return super().datatype_sql(expression)


        def primarykeycolumnconstraint_sql(self, expression: exp.PrimaryKeyColumnConstraint) -> str:
            """
            Generate SQL for PRIMARY KEY column constraints.
            In YDB, these are handled differently at the table level.

            Args:
                expression: The PRIMARY KEY column constraint

            Returns:
                Empty string as YDB handles primary keys differently
            """
            return ""

        def _cte_to_lambda(self, expression: exp.Expression) -> str:
            """
            Convert Common Table Expressions (CTEs) to YDB-style lambdas.

            Args:
                expression: The SQL expression containing CTEs

            Returns:
                YDB-specific SQL with lambdas instead of CTEs
            """
            all_ctes = list(expression.find_all(exp.CTE))
            if not all_ctes:
                return self.sql(expression)

            aliases = []

            def _table_to_var(node):
                if (isinstance(node, exp.Table)) and node.name in aliases:
                    return exp.Var(this=f"${node.name} AS {node.alias_or_name}")
                return node


            for cte in all_ctes:
                alias = cte.alias
                aliases.append(alias)

            expression.transform(_table_to_var, copy=False)
            # w = tree.sql(dialect="ydb", pretty=True)

            for cte in all_ctes:
                cte.pop()

            all_with = list(expression.find_all(exp.With))
            for w in all_with:
                w.pop()

            output = ""

            body_sql = self.sql(expression)
            for cte in all_ctes:
                cte_sql = self.sql(cte.this)
                w = f"${cte.alias_or_name} = ({cte_sql});\n\n"
                output += w

            output += body_sql
            return output

        def _generate_create_table(self, expression: exp.Expression) -> str:
            """
            Generate CREATE TABLE SQL with YDB-specific syntax.
            Handles primary keys, constraints, and partitioning.

            Args:
                expression: The CREATE TABLE expression

            Returns:
                SQL string for creating a table in YDB
            """
            # Clean up index parts from table
            for ex in list(expression.this.expressions):
                if isinstance(ex, exp.Identifier):
                    ex.pop()

            def enforce_not_null(col):
                """Add NOT NULL constraint if not present"""
                for constraint in col.constraints:
                    if isinstance(constraint.kind, exp.NotNullColumnConstraint):
                        break
                else:
                    col.append(
                        "constraints", exp.ColumnConstraint(kind=exp.NotNullColumnConstraint())
                    )

            def enforce_pk(col):
                """Add PRIMARY KEY constraint if not present"""
                for constraint in col.constraints:
                    if isinstance(constraint.kind, exp.PrimaryKeyColumnConstraint):
                        break
                else:
                    col.append(
                        "constraints", exp.ColumnConstraint(kind=exp.PrimaryKeyColumnConstraint())
                    )

            pks = list(expression.find_all(exp.PrimaryKey))
            if len(pks) > 0:
                for pk in pks:
                    for pk_ex in pk.expressions:
                        pk_cols = [col for col in expression.this.find_all(exp.ColumnDef) if col.alias_or_name.lower()==pk_ex.alias_or_name.lower()]
                        if len(pk_cols) > 0:
                            col = pk_cols[0]
                            enforce_not_null(col)
                            enforce_pk(col)
                    pk.pop()

            def is_pk(col):
                """Check if a column has a PRIMARY KEY constraint"""
                for constraint in col.constraints:
                    if isinstance(constraint, exp.ColumnConstraint):
                        if isinstance(constraint.kind, exp.PrimaryKeyColumnConstraint):
                            return True
                return False

            for col in expression.find_all(exp.ColumnDef):
                if is_pk(col):
                    break
            else:
                col = list(expression.find_all(exp.ColumnDef))[0]
                enforce_pk(col)

            for col in expression.this.find_all(exp.ColumnDef):
                if is_pk(col):
                    enforce_not_null(col)

            for constraint in list(expression.this.find_all(exp.Constraint)):
                constraint.pop()

            sql = super().generate(expression)

            pk_s = []
            for col in expression.find_all(exp.ColumnDef):
                if is_pk(col):
                    pk_s.append(col.alias_or_name)

            assert len(pk_s) > 0
            ind = sql.find(")")
            col_names = ",".join([f"`{pk}`" for pk in pk_s])
            sql = sql[:ind] + f", PRIMARY KEY({col_names}))\nPARTITION BY HASH ({col_names})\nWITH(STORE=COLUMN);"
            return sql

        def generate(self, expression: exp.Expression, copy: bool = True) -> str:
            """
            Generate SQL for any expression with YDB-specific handling.

            Args:
                expression: The SQL expression to generate
                copy: Whether to copy the expression before processing

            Returns:
                Generated SQL string
            """
            expression = expression.copy() if copy else expression

            if not isinstance(expression, exp.Create) or (isinstance(expression, exp.Create)
                                                          and expression.kind.lower() != "table"):
                return self._cte_to_lambda(expression)
            else:
                return self._generate_create_table(expression)

        STRING_TYPE_MAPPING = {
                exp.DataType.Type.BLOB: "String",
                exp.DataType.Type.CHAR: "String",
                exp.DataType.Type.LONGBLOB: "String",
                exp.DataType.Type.LONGTEXT: "String",
                exp.DataType.Type.MEDIUMBLOB: "String",
                exp.DataType.Type.MEDIUMTEXT: "String",
                exp.DataType.Type.TINYBLOB: "String",
                exp.DataType.Type.TINYTEXT: "String",
                exp.DataType.Type.TEXT: "Utf8",
                exp.DataType.Type.VARBINARY: "String",
                exp.DataType.Type.VARCHAR: "Utf8",
            }

        NON_NULLABLE_TYPES = {
        }

        def _date_trunc_sql(self, expression: exp.DateTrunc) -> str:
            """
            Generate SQL for DATE_TRUNC function with YDB-specific implementation.

            Args:
                expression: The DATE_TRUNC expression

            Returns:
                YDB-specific SQL for truncating dates
            """
            expr = self.sql(expression, "this")
            unit = expression.text("unit").upper()

            if unit == "WEEK":
                return f"DateTime::MakeDate(DateTime::StartOfWeek({expr}))"
            elif unit == "MONTH":
                return f"DateTime::MakeDate(DateTime::StartOfMonth({expr}))"
            elif unit == "QUARTER":
                return f"DateTime::MakeDate(DateTime::StartOfQuarter({expr}))"
            elif unit == "YEAR":
                return f"DateTime::MakeDate(DateTime::StartOfYear({expr}))"
            else:
                if unit != "DAY":
                    self.unsupported(f"Unexpected interval unit: {unit}")
                return self.func("DATE", expr)

        def _current_timestamp_sql(self, expression: exp.CurrentTimestamp) -> str:
            """
            Generate SQL for CURRENT_TIMESTAMP function with YDB-specific implementation.

            Args:
                expression: The CURRENT_TIMESTAMP expression

            Returns:
                YDB-specific SQL for current timestamp
            """
            return 'AddTimezone(CurrentUtcTimestamp(), "Europe/Moscow")'

        def _str_to_date(self, expression: exp.StrToDate) -> str:
            """
            Generate SQL for STR_TO_DATE function with YDB-specific implementation.

            Args:
                expression: The STR_TO_DATE expression

            Returns:
                YDB-specific SQL for converting strings to dates
            """
            str_value = expression.this.name
            # formatted_time = self.format_time(expression, self.dialect.INVERSE_FORMAT_MAPPING,
            #                                   self.dialect.INVERSE_FORMAT_TRIE)
            formatted_time = self.format_time(expression)
            return f'DateTime::MakeTimestamp(DateTime::Parse({formatted_time})("{str_value}"))'

        def _extract(self, expression: exp.Extract) -> str:
            """
            Generate SQL for EXTRACT function with YDB-specific implementation.

            Args:
                expression: The EXTRACT expression

            Returns:
                YDB-specific SQL for extracting date parts
            """
            unit = expression.name.upper()
            expr = self.sql(expression.expression)

            if unit == "WEEK":
                return f"DateTime::GetWeekOfYear({expr})"
            elif unit == "MONTH":
                return f"DateTime::GetMonth({expr})"
            elif unit == "YEAR":
                return f"DateTime::GetYear({expr})"
            else:
                if unit != "DAY":
                    self.unsupported(f"Unexpected interval unit: {unit}")
                return self.func("DATE", expr)

        def _lambda(self, expression: exp.Lambda, arrow_sep: str = "->") -> str:
            """
            Generate SQL for Lambda expressions with YDB-specific syntax.

            Args:
                expression: The Lambda expression
                arrow_sep: The separator to use between parameters and body

            Returns:
                YDB-specific SQL for lambda functions
            """
            for ident in expression.find_all(exp.Identifier):
                new_ident = exp.to_identifier("$"+ident.alias_or_name)
                new_ident.set("quoted", False)
                ident.replace(new_ident)

            args = self.expressions(expression, flat=True)
            args = f"({args})" if len(args.split(",")) > 1 else args
            return f"({args}) {arrow_sep} {{RETURN {self.sql(expression, 'this')}}}"

        def _if(self, expression: exp.If) -> str:
            this = self.sql(expression, "this")
            true = self.sql(expression, "true")
            false = self.sql(expression, "false")
            return f"IF({this}, {true}, {false})"

        def _null_if(self, expression: exp.Nullif) -> str:
            lhs = expression.this
            rhs = expression.expression

            cond = exp.EQ(this=lhs, expression=rhs)
            return self.sql(exp.If(this=cond, true=exp.Null(), false=lhs))

        def _arrayany(self, expression: exp.ArrayAny) -> str:
            """
            Generate SQL for ARRAY_ANY function with YDB-specific implementation.

            Args:
                expression: The ARRAY_ANY expression

            Returns:
                YDB-specific SQL for array existence checks
            """
            param = expression.expression.expressions[0]
            column_references = {}

            for ident in expression.expression.this.find_all(exp.Column):
                if len(ident.parts) < 2:
                    continue

                table_reference = ident.parts[0]
                column_reference = ident.parts[1]
                column_references[f"{table_reference.alias_or_name}.{column_reference.alias_or_name}"] = (table_reference, column_reference)

            if len(column_references) > 0:
                table_aliases = {}
                next_alias = name_sequence("p_")
                for column_reference in column_references:
                    table_aliases[column_reference] = next_alias()

                params = [f"${param}" for param in [param.alias_or_name]+list(table_aliases.values())]
                params = f"({', '.join(params)})"

                for ident in list(expression.expression.this.find_all(exp.Column)):
                    if len(ident.parts) < 2:
                        continue

                    table_reference = ident.parts[0]
                    column_reference = ident.parts[1]
                    full_column_reference = f"{table_reference.alias_or_name}.{column_reference.alias_or_name}"
                    table_alias = table_aliases[full_column_reference]
                    table_reference.pop()
                    column_reference.replace(exp.to_identifier(table_alias))

                lambda_sql = self.sql(expression.expression)
                table_aliases_sql = f"({', '.join([expression.this.alias_or_name]+list(table_aliases.keys()))})"

                return f"ListHasItems({params}->(ListFilter(${param.alias_or_name}, {lambda_sql})){table_aliases_sql})"
            else:
                return f"ListHasItems(ListFilter({self.sql(expression.expression)}))"

        TYPE_MAPPING = {
            **generator.Generator.TYPE_MAPPING,
            **STRING_TYPE_MAPPING,
            exp.DataType.Type.TINYINT: "INT8",
            exp.DataType.Type.SMALLINT: "INT16",
            exp.DataType.Type.INT: "INT32",
            exp.DataType.Type.BIGINT: "INT64",
            exp.DataType.Type.DECIMAL: "Int64",
            exp.DataType.Type.FLOAT: "Float",
            exp.DataType.Type.DOUBLE: "Double",
            exp.DataType.Type.BOOLEAN: "Uint8",
            exp.DataType.Type.TIMESTAMP: "Timestamp",
        }

        TRANSFORMS = {
            **generator.Generator.TRANSFORMS,
            exp.Create: create_sql,
            exp.DefaultColumnConstraint: lambda self, e: "",
            exp.DateTrunc: _date_trunc_sql,
            exp.Select: transforms.preprocess([apply_alias_to_select_from_table, move_ctes_to_top_level]),
            exp.CurrentTimestamp: _current_timestamp_sql,
            exp.StrToDate: _str_to_date,
            exp.Extract: _extract,
            exp.ArraySize: rename_func("ListLength"),
            exp.ArrayFilter: rename_func("ListFilter"),
            exp.Lambda: _lambda,
            exp.ArrayAny: _arrayany,
            exp.ArrayAgg: rename_func("AGGREGATE_LIST"),
            exp.Concat: concat_to_dpipe_sql,
            exp.If: _if,
            exp.Nullif: _null_if,
        }

# _quote_table_names
# , lambda expr: Generator._apply_alias_to_select_from_table(expr)]
