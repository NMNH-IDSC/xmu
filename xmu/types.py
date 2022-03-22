"""Wrappers for data that can be garbled during read/write"""
import logging
import re
from calendar import monthrange
from datetime import date, datetime


logger = logging.getLogger(__name__)


class EMuType:
    """Container for data types that may be garbled during read/write

    For example, transforming a year to a date using datetime.strptime()
    imposes a month and date, which could be bad news if that data is ever
    loaded back into the database. This class tracks the original string
    and format while coercing the string to a Python data type and
    providing support for basic operations.

    Parameters
    ----------
    val : mixed
        value to wrap
    fmt : str
        formatting string used to translate value back to a string

    Attributes
    ----------
    value : mixed
        value coerced to the correct type from a string
    format : str
        a formatting string
    verbatim : mixed
        the original, unparsed value
    """

    def __init__(self, val, fmt="{}"):
        self.verbatim = val
        self.value = val
        self.format = fmt

    def __str__(self):
        return self.format.format(self.value)

    def __repr__(self):
        return f"{self.__class__.__name__}('{str(self)}')"

    def __eq__(self, other):
        if self.is_range():
            other = self.coerce(other)
            return (
                self.value == other.value
                and self.min_value == other.min_value
                and self.max_value == other.max_value
            )
        return self.value == other

    def __ne__(self, other):
        if self.is_range():
            other = self.coerce(other)
            return (
                self.value != other.value
                or self.min_value != other.min_value
                or self.max_value != other.max_value
            )
        return self.value != other

    def __lt__(self, other):
        if self.is_range():
            other = self.coerce(other)
            return self.max_value < other.min_value
        return self.value < other

    def __le__(self, other):
        if self.is_range():
            other = self.coerce(other)
            return self.min_value <= other.max_value
        return self.value <= other

    def __gt__(self, other):
        if self.is_range():
            other = self.coerce(other)
            return self.min_value > other.max_value
        return self.value > other

    def __ge__(self, other):
        if self.is_range():
            other = self.coerce(other)
            return self.max_value >= other.min_value
        return self.value >= other

    def __contains__(self, other):
        if self.is_range():
            other = self.coerce(other)
            return (
                self.min_value <= other.min_value and self.max_value >= other.max_value
            )
        raise ValueError(f"{self.__class__.__name__} is not a range")

    def __add__(self, other):
        return self._math_op(other, "__add__")

    def __sub__(self, other):
        return self._math_op(other, "__sub__")

    def __mul__(self, other):
        return self._math_op(other, "__mul__")

    def __floordiv__(self, other):
        return self._math_op(other, "__floordiv__")

    def __div__(self, other):
        return self._math_op(other, "__div__")

    def __truediv__(self, other):
        return self._math_op(other, "__truediv__")

    def __mod__(self, other):
        return self._math_op(other, "__mod__")

    def __divmod__(self, other):
        return self._math_op(other, "__divmod__")

    def __pow__(self, other):
        return self._math_op(other, "__pow__")

    def __iadd__(self, other):
        result = self + other
        self.value = result.value
        self.format = result.format
        return self

    def __isub__(self, other):
        result = self - other
        self.value = result.value
        self.format = result.format
        return self

    def __imul__(self, other):
        result = self * other
        self.value = result.value
        self.format = result.format
        return self

    def __ifloordiv__(self, other):
        result = self // other
        self.value = result.value
        self.format = result.format
        return self

    def __idiv__(self, other):
        result = self / other
        self.value = result.value
        self.format = result.format
        return self

    def __itruediv__(self, other):
        result = self / other
        self.value = result.value
        self.format = result.format
        return self

    def __imod__(self, other):
        result = self % other
        self.value = result.value
        self.format = result.format
        return self

    def __ipow__(self, other):
        result = self ** other
        self.value = result.value
        self.format = result.format
        return self

    @property
    def min_value(self):
        """Minimum value needed to express the original string"""
        return self.value

    @property
    def max_value(self):
        """Maximum value needed to express the original string"""
        return self.value

    def coerce(self, other):
        """Coerces another object to the current class

        Parameters
        ----------
        other : mixed
            an object to convert to this class

        Returns
        -------
        EMuType
            other as EMuType
        """
        if not isinstance(other, self.__class__):
            other = self.__class__(other)
        return other

    def is_range(self):
        """Checks if class represents a range"""
        return self.min_value != self.max_value

    def _math_op(self, other, operation):
        """Performs the specified arithmetic operation"""

        if self.is_range():
            min_val = self.__class__(self.min_value)._math_op(other, operation)
            max_val = self.__class__(self.max_value)._math_op(other, operation)
            return (min_val, max_val)

        if isinstance(other, self.__class__):
            val = getattr(self.value, operation)(other.value)
            # Use the more precise format for add/substract
            i = 1 if operation in {"__add__", "__sub__"} else 0
            fmt = sorted([self.format, other.format])[i]
        else:
            val = getattr(self.value, operation)(other)
            fmt = self.format

        if isinstance(val, tuple):
            return tuple([self.__class__(str(val), fmt=fmt) for val in val])

        try:
            return self.__class__(str(val), fmt=fmt)
        except ValueError:
            # Some operations return values that cannot be coerced to the original
            # class, for example, subtracting one date from another
            return val


class EMuFloat(EMuType):
    """Wraps floats read from strings to preserve precision

    Parameters
    ----------
    val : str or float
        float as a string or float
    fmt : str
        formatting string used to convert the float back to a string. Computed
        for strings but must be included if val is a float.

    Attributes
    ----------
    value : float
        float parsed from string
    format : str
        formatting string used to convert the float back to a string
    verbatim : mixed
        the original, unparsed value
    """

    def __init__(self, val, fmt=None):
        """Initialize an EMuFloat object

        Parameters
        ----------
        val : str or float
            the number to wrap
        fmt : str
            a Python formatting string. Must be probided if val is a float,
            otherwise it will be determined from val.
        """

        self.verbatim = val

        fmt_provided = fmt is not None

        if isinstance(val, float) and not fmt_provided:
            raise ValueError("Must provide fmt when passing a float")

        if isinstance(val, self.__class__):
            self.value = val.value
            self.format = val.format
            val = str(val)  # convert to string so the verification step works
        elif fmt_provided:
            self.value = float(val)
            self.format = fmt
        else:
            self.value = float(val)
            num_digits = len(val.split(".")[1]) if "." in val else 0
            self.format = f"{{:.{num_digits}f}}"

        # Verify that the parsed value is the same as the original string if
        # the format string was calculated
        if not fmt_provided and val.lstrip("0").rstrip(".") != str(self).lstrip("0"):
            raise ValueError(f"Parsing changed value ('{val}' became '{self}'")

    def __format__(self, format_spec):
        try:
            return format(str(self), format_spec)
        except ValueError:
            return format(float(self), format_spec)

    def __int__(self):
        return int(self.value)

    def __float__(self):
        return self.value


class EMuDate(EMuType):
    """Wraps dates read from strings to preserve meaning

    Supports addition and subtraction using timedelta objects but not augmented
    assignment using += or -=.

    Parameters
    ----------
    val : str or datetime.date
        date as a string or date object
    fmt : str
        formatting string used to convert the value back to a string. If
        omitted, the class will try to determine the correct format.

    Attributes
    ----------
    value : datetime.date
        date parsed from string
    format : str
        date format string used to convert the date back to a string
    verbatim : mixed
        the original, unparsed value
    """

    directives = {
        "day": ("%d", "%-d"),
        "month": ("%B", "%b", "%m", "%-m"),
        "year": ("%Y", "%y"),
    }
    formats = {"day": "%Y-%m-%d", "month": "%b %Y", "year": "%Y"}

    def __init__(self, val, fmt=None):
        """Initialize an EMuDate object

        Parameters
        ----------
        val : str or datetime.date
            the date
        fmt : str
            a date format string
        """

        self.verbatim = val

        fmt_provided = fmt is not None

        fmts = [
            ("day", "%Y-%m-%d"),
            ("month", "%Y-%m-"),
            ("month", "%b %Y"),
            ("year", "%Y"),
        ]

        if isinstance(val, EMuDate):
            self.value = val.value
            self.kind = val.kind
            self.format = val.format
            val = str(val)  # convert to string so the verification step works
            fmt = self.format
            fmts.clear()  # clear the format dict to skip the format check

        elif isinstance(val, date):
            self.value = val
            self.kind = "day"
            self.format = "%Y-%m-%d"
            val = str(val)  # convert to string so the verification step works
            fmt = self.format
            fmts.clear()  # clear the format dict to skip the format check

        elif fmt:
            # Assess speciicity of if custom formatting string provided
            for kind, directives in self.directives.items():
                if any((d in fmt for d in directives)):
                    parsed = datetime.strptime(val, fmt)
                    self.value = date(parsed.year, parsed.month, parsed.day)
                    self.kind = kind
                    self.format = self.formats[kind]
                    fmts.clear()
                    break

        for kind, fmt in fmts:
            try:
                parsed = datetime.strptime(str(val), fmt)
                self.value = date(parsed.year, parsed.month, parsed.day)
                self.kind = kind
                self.format = self.formats[kind]
                break
            except (TypeError, ValueError):
                pass
        else:
            if fmts:
                raise ValueError(f"Could not parse date: {repr(val)}")

        # Verify that the parsed value is the same as the original string if
        # the format string was calculated
        if not fmt_provided and str(val) != self.strftime(fmt):
            raise ValueError(f"Parsing changed value ('{val}' became '{self}'")

    def __str__(self):
        return self.value.strftime(self.format)

    def strftime(self, fmt=None):
        """Formats date as a string

        Parameters
        ----------
        fmt : str
            date format string

        Returns
        -------
        str
            date as string
        """

        if fmt is None:
            fmt = self.format

        # Forbid formats that are more specific than the original string. Users
        # can force the issue by formatting the value attribute directly.
        if not self.day:
            allowed = []
            if self.year:
                allowed.extend(self.directives["year"])
            if self.month:
                allowed.extend(self.directives["month"])

            directives = re.findall(r"%[a-z]", fmt, flags=re.I)
            disallowed = set(directives) - set(allowed)
            if disallowed:
                raise ValueError(f'Invalid directives for "{str(self)}": {disallowed}')

        return self.value.strftime(fmt)

    @property
    def min_value(self):
        """Minimum date needed to express the original string

        For example, the first day of the month for a date that specifies
        only a month and year or the first day of the year for a year.
        """
        if self.kind == "day":
            return self.value
        if self.kind == "month":
            return date(self.value.year, self.value.month, 1)
        if self.kind == "year":
            return date(self.value.year, 1, 1)
        raise ValueError(f"Invalid kind: {self.kind}")

    @property
    def max_value(self):
        """Maximum date needed to express the original string

        For example, the last day of the month for a date that specifies
        only a month and year or the last day of the year for a year.
        """
        if self.kind == "day":
            return self.value
        if self.kind == "month":
            _, last_day = monthrange(self.value.year, self.value.month)
            return date(self.value.year, self.value.month, last_day)
        if self.kind == "year":
            return date(self.value.year, 12, 31)
        raise ValueError(f"Invalid kind: {self.kind}")

    @property
    def year(self):
        """Year of the parsed date"""
        return self.value.year

    @property
    def month(self):
        """Month of the parsed date"""
        return self.value.month if self.kind != "year" else None

    @property
    def day(self):
        """Day of the parsed date"""
        return self.value.day if self.kind == "day" else None
