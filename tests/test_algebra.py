import asyncio

import pytest
import sympy
from server import (
    intro,
    introduce_expression,
    print_latex_expression,
    factor_expression,
    expand_expression,
    cancel_expression,
    apart_expression,
    limit_expression,
    series_expansion,
    solve_inequality,
    parse_safe_expression,
    UnsafeExpressionError,
    create_matrix,
    mcp,
    local_vars,
    expressions,
    functions,
)
from vars import Assumption, LimitDirection


# Add a fixture to reset global state between tests
@pytest.fixture(autouse=True)
def reset_globals():
    # Clear global dictionaries before each test
    local_vars.clear()
    expressions.clear()
    functions.clear()
    # Reset the expression counter
    import server

    server.expression_counter = 0
    yield


@pytest.fixture
def x():
    intro("x", [Assumption.REAL], [])
    return local_vars["x"]


def assert_same(result_key, expected):
    """The stored result is structurally identical to the expected SymPy expression."""
    assert result_key in expressions, result_key
    assert expressions[result_key] == expected


class TestFactorExpressionTool:
    def test_factor_over_rationals(self, x):
        result_key = factor_expression(introduce_expression("x**6 - 1"))
        assert_same(result_key, (x - 1) * (x + 1) * (x**2 - x + 1) * (x**2 + x + 1))
        assert "x^{2} - x + 1" in print_latex_expression(result_key)

    def test_factor_irreducible_over_rationals(self, x):
        result_key = factor_expression(introduce_expression("x**2 + 1"))
        assert_same(result_key, x**2 + 1)

    def test_factor_gaussian(self, x):
        result_key = factor_expression(introduce_expression("x**2 + 1"), gaussian=True)
        assert_same(result_key, (x - sympy.I) * (x + sympy.I))

    def test_factor_modulus(self, x):
        result_key = factor_expression(introduce_expression("x**2 + 1"), modulus=2)
        assert_same(result_key, (x + 1) ** 2)

    def test_factor_rejects_non_prime_modulus(self, x):
        result = factor_expression(introduce_expression("x**2 + 1"), modulus=4)
        assert result.startswith("Error")

    def test_factor_rejects_gaussian_with_modulus(self, x):
        result = factor_expression(
            introduce_expression("x**2 + 1"), gaussian=True, modulus=2
        )
        assert result.startswith("Error")

    def test_factor_equation_side_by_side(self, x):
        result_key = factor_expression(introduce_expression("Eq(x**2 - 1, x**2 - x)"))
        result = expressions[result_key]
        assert isinstance(result, sympy.Eq)
        assert result.lhs == (x - 1) * (x + 1)
        assert result.rhs == x * (x - 1)

    def test_factor_matrix_entries(self, x):
        matrix_key = create_matrix([["x**2 - 1", "0"], ["0", "x**2 + 2*x + 1"]])
        result_key = factor_expression(matrix_key)
        assert expressions[result_key] == sympy.Matrix(
            [[(x - 1) * (x + 1), 0], [0, (x + 1) ** 2]]
        )

    def test_factor_missing_key(self):
        assert factor_expression("nope").startswith("Error")


class TestExpandExpressionTool:
    def test_expand_power(self, x):
        result_key = expand_expression(introduce_expression("(x + 1)**3"))
        assert_same(result_key, x**3 + 3 * x**2 + 3 * x + 1)

    def test_expand_multivariate(self, x):
        intro("y", [], [])
        y = local_vars["y"]
        result_key = expand_expression(introduce_expression("(x - y)*(x + y)"))
        assert_same(result_key, x**2 - y**2)

    def test_expand_is_inverse_of_factor(self, x):
        expr_key = introduce_expression("x**6 - 1")
        result_key = expand_expression(factor_expression(expr_key))
        assert_same(result_key, expressions[expr_key])

    def test_expand_missing_key(self):
        assert expand_expression("nope").startswith("Error")


class TestCancelExpressionTool:
    def test_cancel_common_factor(self, x):
        result_key = cancel_expression(
            introduce_expression("(x**2 - 1)/(x**2 - 2*x + 1)")
        )
        assert sympy.simplify(expressions[result_key] - (x + 1) / (x - 1)) == 0
        numerator, denominator = sympy.fraction(expressions[result_key])
        assert sympy.degree(denominator, x) == 1

    def test_cancel_to_polynomial(self, x):
        result_key = cancel_expression(introduce_expression("(x**3 - 1)/(x - 1)"))
        assert_same(result_key, x**2 + x + 1)

    def test_cancel_missing_key(self):
        assert cancel_expression("nope").startswith("Error")


class TestApartExpressionTool:
    def test_apart_simple(self, x):
        result_key = apart_expression(introduce_expression("1/(x**2 - 1)"), "x")
        half = sympy.Rational(1, 2)
        assert_same(result_key, half / (x - 1) - half / (x + 1))

    def test_apart_without_variable_for_univariate(self, x):
        result_key = apart_expression(introduce_expression("(x + 3)/(x**2 + 3*x + 2)"))
        assert_same(result_key, 2 / (x + 1) - 1 / (x + 2))

    def test_apart_improper_fraction(self, x):
        result_key = apart_expression(introduce_expression("x**3/(x**2 - 1)"), "x")
        result = expressions[result_key]
        assert sympy.simplify(result - x**3 / (x**2 - 1)) == 0
        assert x in result.args  # polynomial part x is split off

    def test_apart_multivariate_requires_variable(self, x):
        intro("a", [], [])
        a = local_vars["a"]
        expr_key = introduce_expression("1/((x - a)*(x + 1))")
        assert apart_expression(expr_key).startswith("Error")
        result_key = apart_expression(expr_key, "x")
        assert sympy.simplify(expressions[result_key] - 1 / ((x - a) * (x + 1))) == 0

    def test_apart_unknown_variable(self, x):
        result = apart_expression(introduce_expression("1/(x**2 - 1)"), "z")
        assert result.startswith("Error")


class TestLimitExpressionTool:
    def test_limit_classic(self, x):
        assert_same(limit_expression(introduce_expression("sin(x)/x"), "x", "0"), 1)

    def test_limit_at_infinity(self, x):
        result_key = limit_expression(introduce_expression("(1 + 1/x)**x"), "x", "oo")
        assert_same(result_key, sympy.E)

    def test_limit_at_minus_infinity(self, x):
        result_key = limit_expression(introduce_expression("exp(x)"), "x", "-oo")
        assert_same(result_key, 0)

    def test_limit_rational_at_infinity(self, x):
        result_key = limit_expression(
            introduce_expression("(2*x**2 + 1)/(x**2 - 3)"), "x", "inf"
        )
        assert_same(result_key, 2)

    def test_limit_symbolic_point(self, x):
        result_key = limit_expression(introduce_expression("cos(x)"), "x", "pi/2")
        assert_same(result_key, 0)

    def test_two_sided_limit_does_not_exist(self, x):
        result = limit_expression(introduce_expression("1/x"), "x", "0")
        assert result.startswith("Error")
        assert "does not exist" in result

    def test_one_sided_limits(self, x):
        expr_key = introduce_expression("1/x")
        assert_same(
            limit_expression(expr_key, "x", "0", LimitDirection.RIGHT), sympy.oo
        )
        assert_same(
            limit_expression(expr_key, "x", "0", LimitDirection.LEFT), -sympy.oo
        )

    def test_two_sided_infinite_limit(self, x):
        assert_same(limit_expression(introduce_expression("1/x**2"), "x", "0"), sympy.oo)

    def test_jump_discontinuity(self, x):
        result = limit_expression(introduce_expression("Abs(x)/x"), "x", "0")
        assert "does not exist" in result

    def test_oscillating_limit(self, x):
        result = limit_expression(introduce_expression("sin(x)"), "x", "oo")
        assert result.startswith("Error")
        assert "oscillates" in result

    def test_limit_rejects_code_injection(self, x):
        expr_key = introduce_expression("x")
        result = limit_expression(expr_key, "x", "__import__('os').getpid()")
        assert result.startswith("Error: Invalid limit point")

    def test_limit_unknown_variable(self, x):
        result = limit_expression(introduce_expression("x"), "t", "0")
        assert result.startswith("Error")


class TestSeriesExpansionTool:
    def test_series_exp(self, x):
        result_key = series_expansion(introduce_expression("exp(x)"), "x", "0", 4)
        assert_same(result_key, 1 + x + x**2 / 2 + x**3 / 6 + sympy.O(x**4))
        assert "O\\left(x^{4}\\right)" in print_latex_expression(result_key)

    def test_series_default_order(self, x):
        result_key = series_expansion(introduce_expression("sin(x)"), "x")
        assert_same(result_key, x - x**3 / 6 + x**5 / 120 + sympy.O(x**6))

    def test_series_remove_order_term(self, x):
        result_key = series_expansion(
            introduce_expression("log(1 + x)"), "x", "0", 4, remove_order_term=True
        )
        assert_same(result_key, x - x**2 / 2 + x**3 / 3)

    def test_series_at_nonzero_point(self, x):
        result_key = series_expansion(
            introduce_expression("sin(x)"), "x", "pi/2", 4, remove_order_term=True
        )
        assert sympy.expand(
            expressions[result_key] - (1 - (x - sympy.pi / 2) ** 2 / 2)
        ) == 0

    def test_series_at_infinity(self, x):
        result_key = series_expansion(
            introduce_expression("1/(x - 1)"), "x", "oo", 3, remove_order_term=True
        )
        assert_same(result_key, 1 / x + 1 / x**2)

    def test_series_laurent(self, x):
        result_key = series_expansion(
            introduce_expression("1/sin(x)"), "x", "0", 2, remove_order_term=True
        )
        assert_same(result_key, 1 / x + x / 6)

    @pytest.mark.parametrize("order", [0, -1, 51])
    def test_series_rejects_bad_order(self, x, order):
        result = series_expansion(introduce_expression("exp(x)"), "x", "0", order)
        assert result.startswith("Error")

    def test_series_rejects_code_injection(self, x):
        result = series_expansion(
            introduce_expression("exp(x)"), "x", "x.__class__.__mro__"
        )
        assert result.startswith("Error: Invalid expansion point")


class TestSolveInequalityTool:
    def test_quadratic_inequality(self, x):
        result = solve_inequality(introduce_expression("x**2 - 1 > 0"), "x")
        assert result == sympy.latex(
            sympy.Union(sympy.Interval.open(-sympy.oo, -1), sympy.Interval.open(1, sympy.oo))
        )

    def test_rational_inequality(self, x):
        result = solve_inequality(introduce_expression("(x - 1)/(x + 2) <= 0"), "x")
        assert result == sympy.latex(sympy.Interval.Lopen(-2, 1))

    def test_and_combination(self, x):
        result = solve_inequality(introduce_expression("And(x > 0, x**2 < 4)"), "x")
        assert result == sympy.latex(sympy.Interval.open(0, 2))

    def test_or_combination(self, x):
        result = solve_inequality(introduce_expression("Or(x < -3, x >= 5)"), "x")
        assert result == sympy.latex(
            sympy.Union(sympy.Interval.open(-sympy.oo, -3), sympy.Interval(5, sympy.oo))
        )

    def test_always_true(self, x):
        result = solve_inequality(introduce_expression("x**2 + 1 > 0"), "x")
        assert result == sympy.latex(sympy.S.Reals)

    def test_never_true(self, x):
        result = solve_inequality(introduce_expression("x**2 < -1"), "x")
        assert result == sympy.latex(sympy.S.EmptySet)

    def test_not_equal(self, x):
        result = solve_inequality(introduce_expression("Ne(x, 1)"), "x")
        assert result == sympy.latex(
            sympy.Union(sympy.Interval.open(-sympy.oo, 1), sympy.Interval.open(1, sympy.oo))
        )

    def test_rejects_equation(self, x):
        result = solve_inequality(introduce_expression("Eq(x, 1)"), "x")
        assert result.startswith("Error")

    def test_rejects_plain_expression(self, x):
        result = solve_inequality(introduce_expression("x**2"), "x")
        assert result.startswith("Error")

    def test_missing_key_and_variable(self, x):
        assert solve_inequality("nope", "x").startswith("Error")
        assert solve_inequality(introduce_expression("x > 0"), "t").startswith("Error")


class TestParseSafeExpression:
    @pytest.mark.parametrize(
        "text, expected",
        [
            ("0", sympy.Integer(0)),
            ("-oo", -sympy.oo),
            ("inf", sympy.oo),
            ("1/2", sympy.Rational(1, 2)),
            ("pi/2", sympy.pi / 2),
            ("2^3", sympy.Integer(8)),
            ("sqrt(2)", sympy.sqrt(2)),
            ("log(8, 2)", sympy.Integer(3)),
            ("Rational(1, 3)", sympy.Rational(1, 3)),
            ("E + I", sympy.E + sympy.I),
            ("0.5", sympy.Float(0.5)),
        ],
    )
    def test_accepts_arithmetic(self, text, expected):
        assert parse_safe_expression(text) == expected

    def test_accepts_introduced_variables(self, x):
        assert parse_safe_expression("2*x + 1") == 2 * x + 1

    @pytest.mark.parametrize(
        "text",
        [
            "__import__('os').system('true')",
            "open('/etc/passwd')",
            "eval('1')",
            "exec('1')",
            "Symbol('a').__class__",
            "().__class__.__bases__[0].__subclasses__()",
            "[1][0]",
            "{'a': 1}",
            "lambda: 0",
            "[c for c in ()]",
            "'abc'",
            "b'abc'",
            "True",
            "None",
            "x if 1 else 2",
            "sqrt(x=2)",
            "sqrt(*[2])",
            "Function('f')",
            "undefined_name",
            "1 < 2",
            "1e400",
            "2**10**10",
            "(10**1000)**1000",
            "1" * 300,
            "1 +",
        ],
    )
    def test_rejects_non_arithmetic(self, text):
        with pytest.raises(UnsafeExpressionError):
            parse_safe_expression(text)

    def test_rejects_non_string(self):
        with pytest.raises(UnsafeExpressionError):
            parse_safe_expression(1)


class TestToolRegistration:
    NEW_TOOLS = {
        "factor_expression",
        "expand_expression",
        "cancel_expression",
        "apart_expression",
        "limit_expression",
        "series_expansion",
        "solve_inequality",
    }

    def test_new_tools_are_registered(self):
        names = {tool.name for tool in asyncio.run(mcp.list_tools())}
        assert self.NEW_TOOLS <= names

    def test_existing_tools_still_registered(self):
        names = {tool.name for tool in asyncio.run(mcp.list_tools())}
        for name in [
            "intro",
            "intro_many",
            "introduce_expression",
            "print_latex_expression",
            "solve_algebraically",
            "dsolve_ode",
            "simplify_expression",
            "integrate_expression",
            "differentiate_expression",
            "substitute_expression",
            "reset_state",
            "matrix_determinant",
        ]:
            assert name in names

    def test_limit_direction_schema(self):
        tools = {tool.name: tool for tool in asyncio.run(mcp.list_tools())}
        schema = str(tools["limit_expression"].input_schema)
        for value in ["+-", "+", "-"]:
            assert repr(value) in schema

    def test_call_through_mcp(self):
        async def run():
            await mcp.call_tool("intro", {"var_name": "x", "pos_assumptions": ["real"], "neg_assumptions": []})
            await mcp.call_tool("introduce_expression", {"expr_str": "1/x", "expr_var_name": "f"})
            return await mcp.call_tool(
                "limit_expression",
                {"expr_key": "f", "var_name": "x", "point": "0", "direction": "+"},
            )

        result = asyncio.run(run())
        assert "expr_" in str(result)
        assert sympy.oo in expressions.values()
