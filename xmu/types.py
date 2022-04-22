"""Wrappers for data that can be garbled during read/write"""
import logging
import re
from calendar import monthrange
from datetime import date, datetime, time
from math import log10, modf


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

    def copy(self):
        """Creates a copy of the current object"""
        return self.__class__(self.verbatim)

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
            # Raise an error if values are not floats and formats differ
            if isinstance(self.value, float):
                # Use the more precise format for add/substract
                i = -1 if operation in {"__add__", "__sub__"} else 0
                try:
                    fmt = sorted([o.format for o in [self, other] if o.dec_places])[i]
                except IndexError:
                    fmt = self.format
            elif self.format != other.format:
                raise ValueError(
                    f"{self.__class__.__name__} have different formats: {[self.format, other.format]}"
                )
            else:
                fmt = self.format
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
            val = str(val)
            dec_places = len(val.split(".")[1]) if "." in val else 0
            self.format = f"{{:.{dec_places}f}}"

        # Verify that the parsed value is the same as the original string if
        # the format string was calculated
        if not fmt_provided and val.lstrip("0").rstrip(".") != str(self).lstrip("0"):
            raise ValueError(f"Parsing changed value ('{val}' became '{self}')")

    def __format__(self, format_spec):
        try:
            return format(str(self), format_spec)
        except ValueError:
            return format(float(self), format_spec)

    def __int__(self):
        return int(self.value)

    def __float__(self):
        return self.value

    @property
    def dec_places(self):
        """Number of decimal places from the formatting string"""
        return int(self.format.strip("{:.f}"))


class EMuCoord(EMuFloat):
    """Wraps coordinates read from strings

    Attributes
    ----------
    value : float
        coordinate as a float
    format : str
        formatting string used to convert the float back to a string
    degrees : EMuFloat
        degrees parsed from original
    minutes : EMuFloat
        minutes parsed from original, if any
    seconds : EMuFloat
        seconds parsed from original, if any
    verbatim : mixed
        the original, unparsed value
    """

    #: str : pattern for hemisphere for positive coordinates
    pos = ""

    #: str : pattern for hemisphere for negative coordinates
    neg = ""

    #: tuple of int : range of allowable values
    bounds = (0, 0)

    #: float : width of one degree lat (anywhere) or long (at the equator)
    deg_dist_m = 110567

    # dict : uncertainty in meters for deg/min/sec at the equator
    dms_unc_m = {
        "degrees": deg_dist_m,
        "minutes": deg_dist_m / 60,
        "seconds": deg_dist_m / 3600,
    }

    # dict : uncertainty in meters for decimal degrees at the equator
    dec_unc_m = {
        0: deg_dist_m,
        1: deg_dist_m / 10,
        2: deg_dist_m / 100,
        3: deg_dist_m / 1000,
        4: deg_dist_m / 10000,
        5: deg_dist_m / 100000,
    }

    def __init__(self, val, fmt=None):
        """Initializes an EMuCoord object

        Parameters
        ----------
        val : str or float
            coordinate
        fmt : str
            formatting string used to convert a float back to a string
        """

        self.verbatim = val

        self.minutes = None
        self.seconds = None
        if isinstance(val, str):
            self.verbatim = val.strip()
            parts = re.findall(r"(\d+(?:\.\d+)?)", self.verbatim)
            if len(parts) > 3:
                raise ValueError(f"Invalid coordinate: {self.verbatim}")
            self.degrees = EMuFloat(parts[0])
            if len(parts) > 1:
                self.minutes = EMuFloat(parts[1])
            if len(parts) > 2:
                self.seconds = EMuFloat(parts[2])
        elif isinstance(val, EMuCoord):
            self.verbatim = val.verbatim
            for attr in ("degrees", "minutes", "seconds"):
                if getattr(val, attr) is not None:
                    setattr(self, attr, getattr(val, attr).copy())
        else:
            self.degrees = EMuFloat(abs(val), fmt=fmt)

        self._sign = EMuFloat(self._get_sign(), fmt="{:.0f}")

        self.value = float(self)
        if self.value < min(self.bounds) or self.value > max(self.bounds):
            raise ValueError(f"Coordinate out of bounds ({val} not in {self.bounds})")

    def __format__(self, format_spec):
        try:
            return format(str(self), format_spec)
        except ValueError:
            return format(float(self), format_spec)

    def __str__(self):
        parts = (self.degrees, self.minutes, self.seconds)
        return f"{' '.join([str(p) for p in parts if p is not None])} {self.hemisphere}"

    def __int__(self):
        return int(float(self))

    def __float__(self):
        val = EMuFloat(self.degrees)
        if self.minutes:
            val += self.minutes / 60
        if self.seconds:
            val += self.seconds / 3600
        return float(self._sign * val)

    @property
    def hemisphere(self):
        """Gets the hemisphere in which a coordinate is located"""
        return self.pos[0] if self._sign > 0 else self.neg[0]

    @property
    def kind(self):
        """Gets kind of verbatim coordinate string"""
        return "decimal" if self.minutes is None else "dms"

    def to_dms(self, unc_m=None):
        """Expresses coordinate as degrees-minutes-seconds

        Parameters
        ----------
        unc_m : int
            uncerainty in meters

        Returns
        -------
        str
            coordinate as degrees-minutes-seconds
        """

        orig_unc_m = self.coord_uncertainty_m()
        if unc_m is None:
            if self.kind == "dms":
                parts = [
                    p if p else 0 for p in (self.degrees, self.minutes, self.seconds)
                ]
                for i, part in enumerate(parts):
                    if i < 2:
                        frac, num = modf(part)
                        parts[i] = num
                        parts[i + 1] += 60 * frac
                    parts[i] = int(parts[i])
                return f"{' '.join([str(p) for p in parts if p is not None])} {self.hemisphere}"
            unc_m = orig_unc_m

        # Round to approximate the given uncertainty
        unc_m = self._round_to_exp_10(unc_m)
        if unc_m < orig_unc_m:
            raise ValueError(
                f"unc_m cannot be smaller than the uncertainty implied by verbatim ({orig_unc_m} m)"
            )

        last_unc_m = 1e7
        for key, ref_unc_m in self.dms_unc_m.items():

            ref_unc_m = self._round_to_exp_10(ref_unc_m)

            if ref_unc_m <= unc_m <= last_unc_m:
                tenths = False
                break
            last_unc_m = ref_unc_m

            # Gaps between deg/min/sec ranks are huge, so try tenths as well
            if ref_unc_m / 10 <= unc_m <= last_unc_m:
                tenths = True
                break
            last_unc_m = ref_unc_m / 10

        val = self.value

        # Reverse sign for negative coords. Hemisphere is given using a letter.
        if val < 0:
            val *= -1

        parts = []
        for attr in ["degrees", "minutes", "seconds"]:
            fractional, integer = modf(val)
            if key == attr and tenths:
                integer += round(fractional, 1)
                parts.append("{:.1f}".format(integer))
            else:
                parts.append(str(int(integer)))
            if key == attr:
                break
            val = fractional * 60

        return f"{' '.join([str(p) for p in parts])} {self.hemisphere}"

    def to_dec(self, unc_m=None):
        """Expresses coordinate as a decimal

        Parameters
        ----------
        unc_m : int
            uncerainty in meters

        Returns
        -------
        str
            coordinate as decimal
        """
        orig_unc_m = self.coord_uncertainty_m()
        if unc_m is None:
            if self.kind == "decimal":
                return str(self._sign * self.degrees)
            unc_m = orig_unc_m

        unc_m = self._round_to_exp_10(unc_m)
        if unc_m < orig_unc_m:
            raise ValueError(
                f"unc_m cannot be smaller than the uncertainty implied by verbatim ({orig_unc_m} m)"
            )

        last_unc_m = 1e7
        for key, ref_unc_m in self.dec_unc_m.items():
            ref_unc_m = self._round_to_exp_10(ref_unc_m)
            if ref_unc_m <= unc_m <= last_unc_m:
                break
            last_unc_m = ref_unc_m
        return f"{{:.{key}f}}".format(self)

    def coord_uncertainty_m(self):
        """Estimates coordinate uncertainty in meters based on distance at equator

        Returns
        -------
        int
            uncertainty in meters, rounded to an exponent of 10
        """
        if self.seconds:
            unc_m = self.deg_dist_m / (3600 * 10 ** self.seconds.dec_places)
        elif self.minutes:
            unc_m = self.deg_dist_m / (60 * 10 ** self.minutes.dec_places)
        else:
            unc_m = self.deg_dist_m / 10 ** self.degrees.dec_places
        return self._round_to_exp_10(unc_m)

    def _get_sign(self):
        """Gets the sign of the decimal coordinate"""
        if isinstance(self.verbatim, str):
            val = self.verbatim.strip()

            try:
                return 1 if float(self.verbatim) >= 0 else -1
            except ValueError:
                for pat, mod in {
                    r"(^\+|^{0}|{0}$)".format(self.pos): 1,
                    r"(^-|^{0}|{0}$)".format(self.neg): -1,
                }.items():
                    if re.search(pat, val, flags=re.I):
                        return mod

            raise ValueError(
                f"Could not parse as {self.__class__.__name__}: {self.verbatim}"
            )

        return 1 if self.verbatim >= 0 else -1

    @staticmethod
    def _round_to_exp_10(val):
        """Rounds value to an exponent of 10"""
        frac, exp = modf(log10(val))
        if frac > log10(4.99999999):
            exp += 1
        return int(10 ** exp)


class EMuLatitude(EMuCoord):
    """Wraps latitudes read from strings"""

    #: str : pattern for hemisphere for positive coordinates
    pos = "N(orth)?"

    #: str : pattern for hemisphere for negative coordinates
    neg = "S(outh)?"

    #: tuple of int : range of allowable values
    bounds = (-90, 90)

    def __init__(self, val, fmt=None):
        """Initialize an EMuDate object

        Parameters
        ----------
        val : str or float
            latitude
        fmt : str
            formatting string used to convert a float back to a string
        """
        super().__init__(val, fmt)


class EMuLongitude(EMuCoord):
    """Wraps longitudes read from strings"""

    #: str : pattern for hemisphere for positive coordinates
    pos = "E(ast)?"

    #: str : pattern for hemisphere for negative coordinates
    neg = "W(est)?"

    #: tuple of int : range of allowable values
    bounds = (-180, 180)

    def __init__(self, val, fmt=None):
        """Initialize an EMuLongitude object

        Parameters
        ----------
        val : str or float
            longitude
        fmt : str
            formatting string used to convert a float back to a string
        """
        super().__init__(val, fmt)


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
            val = val.strftime(self.format)
            fmt = self.format
            fmts.clear()

        elif isinstance(val, date):
            self.value = val
            self.kind = "day"
            self.format = "%Y-%m-%d"
            val = val.strftime(self.format)
            fmt = self.format
            fmts.clear()

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
            raise ValueError(f"Parsing changed value ('{val}' became '{self}')")

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

    def to_datetime(self, time):
        """Combines date and time into a single datetime

        Parameters
        ----------
        time : datetime.time
            time to use with date

        Returns
        -------
        datetime.datetime
            combined datetime
        """
        if self.min_value != self.max_value:
            raise ValueError("Cannot convert range to datetime")
        return datetime(
            self.year,
            self.month,
            self.day,
            time.hour,
            time.minute,
            time.second,
            time.microsecond,
            time.tzinfo,
        )

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


class EMuTime(EMuType):
    def __init__(self, val, fmt=None):
        """Initialize an EMuTime object

        Parameters
        ----------
        val : str or datetime.time
            the time
        fmt : str
            a time format string
        """

        self.verbatim = val

        fmt_provided = fmt is not None

        # Include both naive and timezoned formats
        fmts = [
            "%H:%M",
            "%H%M",
            "%I%M %p",
            "%I:%M %p",
            "%H:%M:%S",
            "%I:%M:%S %p",
        ]
        num_formats = len(fmts)
        fmts.extend([f"{f} %z" for f in fmts[:num_formats]])
        fmts.extend([f"{f} UTC%z" for f in fmts[:num_formats]])
        fmts.insert(0, "%H:%M:")

        if isinstance(val, EMuTime):
            self.value = val.value
            self.format = val.format
            val = val.strftime(self.format)
            fmts.clear()

        elif isinstance(val, time):
            self.value = val
            self.format = fmts[0]
            val = val.strftime(self.format)
            fmts.clear()

        for fmt in fmts:
            try:
                parsed = datetime.strptime(val, fmt)
                self.value = time(
                    parsed.hour,
                    parsed.minute,
                    parsed.second,
                    parsed.microsecond,
                    parsed.tzinfo,
                )
                self.format = fmt
                break
            except (TypeError, ValueError):
                pass
        else:
            if fmts:
                raise ValueError(f"Could not parse time: {repr(val)}")

        # Verify that the parsed value is the same as the original string if
        # the format string was calculated
        if not fmt_provided and val.lstrip("0") != self.strftime(fmt).lstrip("0"):
            raise ValueError(f"Parsing changed value ('{val}' became '{self}')")

        # Enforce a consistent output format
        self.format = "%H:%M:%S" if "%S" in self.format else "%H:%M"

    def __str__(self):
        return self.value.strftime(self.format)

    def strftime(self, fmt=None):
        """Formats time as a string

        Parameters
        ----------
        fmt : str
            time format string

        Returns
        -------
        str
            time as string
        """
        return self.value.strftime(fmt if fmt else self.format)

    def to_datetime(self, date):
        """Combines date and time into a single datetime

        Parameters
        ----------
        date : datetime.date
            date to use with time

        Returns
        -------
        datetime.datetime
            combined datetime
        """
        if date.min_value != date.max_value:
            raise ValueError("Cannot convert range to datetime")
        return datetime(
            date.year,
            date.month,
            date.day,
            self.hour,
            self.minute,
            self.second,
            self.microsecond,
            self.tzinfo,
        )

    @property
    def hour(self):
        """Hour of the parsed time"""
        return self.value.hour

    @property
    def minute(self):
        """Minute of the parsed time"""
        return self.value.minute

    @property
    def second(self):
        """Second of the parsed time"""
        return self.value.second

    @property
    def microsecond(self):
        """Microsecond of the parsed time"""
        return self.value.microsecond

    @property
    def tzinfo(self):
        """Time zone info for the parsed time"""
        return self.value.tzinfo
