#!/usr/bin/env python3
"""
Build a reviewed data map, on the machine that owns the data.

The Sparks generate code; they cannot read files. vLLM is a text-in/text-out
endpoint with no filesystem, so a dataset placed on a Spark sits somewhere the
model cannot look. What the generator actually needs is the metadata the
generated program will itself construct - a data_map. This tool drafts that map
next to the data, shows it to the person who owns the data, and lets them
correct it before anything is sent.

THE DATA NEVER LEAVES THIS MACHINE. The drafted map is reviewed and saved as
Python next to the data file.

  GUI   python Data_Map_Tool.py
  CLI   python Data_Map_Tool.py --data sales.csv --env anly656
        python Data_Map_Tool.py --list-envs

Works from any working directory and any conda env (Spyder, an IDE, or a
terminal). The environment list comes from the Anaconda root via `conda info -e`,
not from the env that launched the tool. tkinter ships with Anaconda's python
but not the system one, so open the GUI with an Anaconda interpreter. Profiling
re-runs this file inside the environment you select, so the map reflects the
dtypes that interpreter will actually infer.

Requires AdvancedAnalytics and pandas in the selected environment - the same
environment the analysis will run in, which needs them anyway.
"""
import argparse, json, keyword, os, shutil, subprocess, sys, tempfile
from pathlib import Path

SELF = Path(__file__).resolve()
LONG_STRING = 100
CATEGORICAL_FREE = ("Text", "String", "ID", "Label", "Ignore")


# ===================================================================== draft
# Runs inside the analysis environment.
#
# A corrected draft_data_map. AdvancedAnalytics' version is conceptually right -
# a heuristic classifier plus a bounds extractor, producing a DRAFT for a human
# to review - but three defects make its output unusable as shipped on pandas 3:
#
#   1. It calls .sort() on the result of .unique(). On pandas 3 a string
#      column's .unique() is an ArrowStringArray, which has no .sort(), so it
#      dies with AttributeError on any categorical string column - gender,
#      region, the common case.
#   2. It stores the interval type as the string "DT.Interval" rather than the
#      DT enum. convertDataType then tests `atype == DT.Interval`, which is
#      False for a string, falls through every branch, and returns 'DT.Ignore'.
#      The printed draft - the artifact the user is told to review - therefore
#      labelled every continuous variable "ignore this column".
#   3. Its bounds are numpy scalars, so the printed map shows np.float64(17.5),
#      which needs numpy in scope to paste back.
#
# Thresholds and classification rules here are identical to the original, and
# were verified to agree with it on every column archetype where the original
# does not crash. Two deliberate differences: missing values are never levels,
# and a column holding both numbers and strings keeps all of its distinct
# values rather than only the strings.

def _dt():
    try:
        from AdvancedAnalytics.ReplaceImputeEncode import DT
        return DT
    except ImportError:
        raise SystemExit(
            "AdvancedAnalytics is not installed in the selected environment.\n"
            "It is needed both here and by the generated code. Install it with\n"
            "    pip install AdvancedAnalytics\n"
            "or choose an environment that already has it.")


def _plain(v):
    """numpy scalar -> python scalar, so the map has no np.float64(...) in it."""
    if hasattr(v, "item"):
        try:
            return v.item()
        except Exception:
            pass
    return v


def _sorted_mixed(values):
    """Sort values that may be of mixed type. Python refuses to order str
    against int, so numbers sort among themselves, strings among themselves,
    numbers first. Only reachable for a column holding both."""
    return (sorted(v for v in values if not isinstance(v, str))
            + sorted(v for v in values if isinstance(v, str)))


def _is_integer_values(col):
    """True when every non-null value is an integer (bools excluded)."""
    import pandas as pd
    if pd.api.types.is_bool_dtype(col):
        return False
    if pd.api.types.is_integer_dtype(col):
        return True
    vals = [_plain(v) for v in col.dropna()]
    if not vals:
        return False
    for v in vals:
        if isinstance(v, bool) or isinstance(v, str):
            return False
        try:
            if int(v) != v:
                return False
        except (TypeError, ValueError):
            return False
    return True


def _is_obs_id(feature, col, n_rows):
    """'obs' / 'Obs' (any case) holding a unique integer per row - a row index,
    not a predictor. Those draft as Interval unless we catch them here."""
    if str(feature).strip().lower() != "obs":
        return False
    if len(col) != n_rows or not col.notna().all():
        return False
    if int(col.nunique(dropna=True)) != n_rows:
        return False
    return _is_integer_values(col)


def _dtype_label(dtype):
    """DT.Ignore and DT.ID share the enum value 'Z', so dtype.name is always
    'ID'. This tool means Ignore whenever it stores that member."""
    DT = _dt()
    if dtype == DT.Ignore:
        return "Ignore"
    return dtype.name


def draft_map(df, max_n=10, max_s=30):
    """Draft a data_map from a DataFrame. Returns {column: [DT.Type, (values)]}.

    A DRAFT. max_n makes any numeric column with fewer than max_n distinct
    values Nominal - right for a 1-5 rating, wrong for a count. Nothing
    automatic can tell those apart, which is why the result is reviewed."""
    DT = _dt()
    dmap = {}
    n_rows = len(df)
    for feature in df.columns:
        col = df[feature].dropna()              # missing values are not levels
        if col.empty:
            dmap[feature] = [DT.Ignore, ("")]
            continue
        if _is_obs_id(feature, df[feature], n_rows):
            dmap[feature] = [DT.Ignore, ("")]
            continue

        values = [_plain(v) for v in col.unique()]
        has_string = any(isinstance(v, str) for v in values)
        limit = max_s if has_string else max_n

        if len(values) < limit:
            a = _sorted_mixed(values)
            dmap[feature] = [DT.Binary if len(a) == 2 else DT.Nominal, tuple(a)]
        elif not has_string:
            dmap[feature] = [DT.Interval, (float(round(col.min() - 0.01, 4)),
                                           float(round(col.max() + 0.01, 4)))]
        else:
            longest = int(col.astype(str).str.len().max())
            dmap[feature] = [DT.Text if longest > LONG_STRING else DT.String, ("")]
    return dmap


# ================================================================= profiling
def load(path):
    """Read a data file by extension. Anything unrecognised is tried as CSV."""
    import pandas as pd
    ext = Path(path).suffix.lower()
    if ext in (".parquet", ".pq"):   return pd.read_parquet(path)
    if ext in (".xlsx", ".xls"):     return pd.read_excel(path)
    if ext == ".json":               return pd.read_json(path)
    if ext in (".tsv", ".tab"):      return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def invalid_attribute_names(names):
    """Header names that are not legal Python identifiers.

    Generated code uses these as attribute / dict keys. Characters such as
    '/', '-', '.' or a leading digit will not parse; reserved words (class,
    for, ...) are also rejected. The user must rename columns in the data
    file — this tool does not rewrite them."""
    bad = []
    for n in names:
        s = str(n)
        if s == "":
            reason = "empty name"
        elif keyword.iskeyword(s):
            reason = "Python reserved word"
        elif not s.isidentifier():
            extra = sorted({ch for ch in s if not (ch.isalnum() or ch == "_")})
            if s[:1].isdigit():
                reason = "starts with a digit"
            elif extra:
                reason = "invalid character(s): " + " ".join(repr(ch) for ch in extra)
            else:
                reason = "not a valid Python name"
        else:
            continue
        bad.append({"name": s, "reason": reason})
    return bad


def format_invalid_names_message(bad):
    """Text for a warning dialog or CLI note."""
    lines = [
        "These column names from the first row of the data file "
        "are not valid Python attribute names.",
        "",
        "Rename them in the data file, then draft again. "
        "Changing only the map is not enough — the generated "
        "program reads the file.",
        "",
    ]
    show = bad[:40]
    for item in show:
        lines.append(f"  {item['name']!r}  ({item['reason']})")
    if len(bad) > len(show):
        lines.append(f"  ... and {len(bad) - len(show)} more")
    lines += [
        "",
        "A Python name may contain only letters, digits, and underscores;",
        "it cannot start with a digit or be a reserved word (class, for, ...).",
    ]
    return "\n".join(lines)


def find_mixed(df):
    """Columns holding both strings and numbers. Worth flagging: whichever way
    such a column is classified, half its values are the wrong kind."""
    out = {}
    for c in df.columns:
        vals = df[c].dropna()
        if vals.empty:
            continue
        kinds = set()
        for v in vals.unique():
            kinds.add("string" if isinstance(v, str) else "number")
            if len(kinds) > 1:
                break
        if len(kinds) > 1:
            out[str(c)] = {"kinds": sorted(kinds),
                           "sample_values": [str(v) for v in list(vals.unique())[:8]]}
    return out


def profile(data_path, max_n, max_s):
    """Everything the Spark is told about the data. Runs in the analysis env."""
    import pandas as pd
    df = load(data_path)
    raw = draft_map(df, max_n=max_n, max_s=max_s)

    columns = {}
    for col, (dtype, vals) in raw.items():
        vals = vals if isinstance(vals, tuple) else (vals,)
        columns[str(col)] = {"type": _dtype_label(dtype), "values": list(vals)}

    nulls = df.isna().sum()
    stats = {str(c): {"dtype": str(df[c].dtype), "nulls": int(nulls[c]),
                      "null_pct": round(100.0 * int(nulls[c]) / max(len(df), 1), 2),
                      "unique": int(df[c].nunique(dropna=True))}
             for c in df.columns}
    sample = [[("" if pd.isna(v) else str(v)) for v in row]
              for row in df.head(15).itertuples(index=False, name=None)]

    return {"ok": True, "data_path": str(Path(data_path).resolve()),
            "rows": int(len(df)), "cols": int(df.shape[1]),
            "order": [str(c) for c in df.columns], "columns": columns,
            "stats": stats, "mixed": find_mixed(df),
            "dropped": [str(c) for c in df.columns if str(c) not in columns],
            "sample_header": [str(c) for c in df.columns], "sample_rows": sample,
            "invalid_names": invalid_attribute_names(df.columns),
            "pandas": pd.__version__, "max_n": max_n, "max_s": max_s}


def _profile_mode(argv):
    """Re-entry point: this file, run by the selected environment's python."""
    data_path, out_path, max_n, max_s = argv[0], argv[1], int(argv[2]), int(argv[3])
    try:
        rep = profile(data_path, max_n, max_s)
    except SystemExit as e:
        rep = {"ok": False, "error": str(e)}
    except Exception as e:
        import traceback
        rep = {"ok": False, "error": f"{type(e).__name__}: {e}",
               "traceback": traceback.format_exc()}
    Path(out_path).write_text(json.dumps(rep, indent=2))
    sys.exit(0 if rep.get("ok") else 1)


# ==================================================================== driver
# Env discovery must not use the *active* environment's conda first.
# An env-local conda (e.g. anly656/bin/conda) still lists every prefix, but
# it labels the current env "base" and leaves the other names blank. Students
# launching from Spyder inside anly656 then see a dropdown stuck on "base".
# Name every env from its prefix path instead, and prefer the Anaconda root
# (CONDA_EXE, ~/anaconda3, parent of .../envs/<name>).

_ENVS_CACHE = {"rows": None}


def _python_in(prefix):
    """Interpreter inside a conda prefix, Unix or Windows."""
    p = Path(prefix)
    for rel in ("bin/python", "bin/python3", "python.exe", "Scripts/python.exe"):
        cand = p / rel
        if cand.is_file():
            return str(cand.resolve())
    return None


def _name_from_prefix(prefix):
    """Stable env name from the prefix. Do not trust conda's printed name -
    env-local `conda info -e` reports the current env as 'base'."""
    p = Path(prefix).expanduser()
    try:
        p = p.resolve()
    except Exception:
        pass
    if p.parent.name == "envs":
        return p.name
    if (p / "envs").is_dir():
        return "base"
    return p.name or "base"


def _install_root(prefix):
    """Anaconda root that owns this prefix (.../anaconda3, not .../envs/name)."""
    p = Path(prefix).expanduser()
    try:
        p = p.resolve()
    except Exception:
        return p
    if p.parent.name == "envs":
        return p.parent.parent
    return p


def _conda_executables():
    """Candidate `conda` binaries, root install first, env-local last."""
    found, ordered = set(), []

    def add(path):
        if not path:
            return
        p = Path(path)
        if not p.is_file():
            return
        try:
            key = str(p.resolve())
        except Exception:
            key = str(p)
        if key not in found:
            found.add(key)
            ordered.append(p)

    for key in ("CONDA_EXE", "CONDA_BAT"):
        add(os.environ.get(key))

    roots = []
    for start in (os.environ.get("CONDA_PREFIX"), sys.prefix):
        if start:
            roots.append(_install_root(start))
    home = Path.home()
    roots.extend([
        home / "anaconda3", home / "miniconda3", home / "miniforge3",
        home / "mambaforge", home / "anaconda", home / "miniconda",
        home / "opt" / "anaconda3",
        Path("/opt/anaconda3"), Path("/opt/miniconda3"),
        Path("/usr/local/anaconda3"),
        Path(os.environ.get("LOCALAPPDATA", "")) / "anaconda3",
        Path(os.environ.get("LOCALAPPDATA", "")) / "miniconda3",
        Path(os.environ.get("ProgramData", "")) / "anaconda3",
        Path(os.environ.get("ProgramData", "")) / "miniconda3",
    ])
    for root in roots:
        if not root or not Path(root).exists():
            continue
        root = Path(root)
        for rel in ("bin/conda", "condabin/conda", "condabin/conda.bat",
                    "Scripts/conda.exe", "Scripts/conda.bat"):
            add(root / rel)

    which = shutil.which("conda") or shutil.which("conda.exe")
    add(which)

    # Last: conda sitting inside whatever interpreter launched us.
    add(Path(sys.prefix) / "bin" / "conda")
    add(Path(sys.prefix) / "Scripts" / "conda.exe")
    return ordered


def _run_conda(conda, args):
    try:
        return subprocess.run([str(conda), *args], capture_output=True,
                              text=True, timeout=60)
    except Exception:
        return None


def _envs_from_prefixes(prefixes):
    out, seen_py, seen_name = [], set(), set()
    for prefix in prefixes:
        py = _python_in(prefix)
        if not py or py in seen_py:
            continue
        name = _name_from_prefix(prefix)
        if name in seen_name:
            name = f"{name} ({Path(prefix).name})"
        seen_py.add(py)
        seen_name.add(name)
        out.append((name, py))
    return out


def _prefixes_from_json(text):
    data = json.loads(text)
    prefixes = data.get("envs") or []
    if prefixes:
        return [str(p) for p in prefixes]
    details = data.get("envs_details") or {}
    return list(details)


def _prefixes_from_info_e(text):
    """Parse `conda info -e` / `conda env list`. Names on the line are ignored;
    a line may be only `*  /path` when conda is the env-local one."""
    prefixes = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].replace("*", " ").replace("+", " ").strip()
        if not line:
            continue
        parts = line.split()
        prefix = None
        for tok in reversed(parts):
            p = Path(tok).expanduser()
            looks = (tok.startswith("/") or tok.startswith("~")
                     or (len(tok) > 2 and tok[1] == ":" and tok[2] in "\\/"))
            if looks or p.is_dir():
                prefix = str(p)
                break
        if prefix:
            prefixes.append(prefix)
    return prefixes


def conda_envs():
    """[(name, python_path)] for every conda env on this machine.

    Works from any cwd and any launching interpreter (base, anly656, Spyder).
    Uses `conda info -e` (and its JSON form) against the Anaconda root."""
    if _ENVS_CACHE["rows"] is not None:
        return _ENVS_CACHE["rows"]

    best = []
    for conda in _conda_executables():
        prefixes = []
        r = _run_conda(conda, ["info", "--envs", "--json"])
        if r and r.returncode == 0 and r.stdout.strip():
            try:
                prefixes = _prefixes_from_json(r.stdout)
            except Exception:
                prefixes = []
        if not prefixes:
            r = _run_conda(conda, ["info", "-e"])
            if r and r.stdout:
                prefixes = _prefixes_from_info_e(r.stdout)
        if not prefixes:
            r = _run_conda(conda, ["env", "list"])
            if r and r.stdout:
                prefixes = _prefixes_from_info_e(r.stdout)
        rows = _envs_from_prefixes(prefixes)
        if len(rows) > len(best):
            best = rows
        if len(best) >= 2:
            break

    _ENVS_CACHE["rows"] = best
    return best


def default_env_name(envs=None):
    """Env to preselect: the interpreter running this tool, else anly656, else base."""
    envs = list(envs if envs is not None else conda_envs())
    names = [n for n, _ in envs]
    if not names:
        return ""
    here = str(Path(sys.executable).resolve())
    for n, p in envs:
        try:
            if Path(p).resolve() == Path(here):
                return n
        except Exception:
            if p == sys.executable:
                return n
    for guess in (os.environ.get("CONDA_DEFAULT_ENV"), "anly656", "base"):
        if guess and guess in names:
            return guess
    return names[0]


def env_python(name):
    for n, p in conda_envs():
        if n == name:
            return p
    raise SystemExit(f"no conda environment named {name!r} - try --list-envs")


def resolve_data_path(text):
    """Resolve a user-supplied path to an existing data file."""
    raw = str(text).strip() if text is not None else ""
    if not raw:
        raise SystemExit("no data file")
    p = Path(raw).expanduser()
    if p.is_file():
        return p.resolve()
    raise SystemExit(f"no such data file: {raw}")


def draft(data_path, python_path, max_n=10, max_s=30):
    """Run this file's profile mode inside the target environment.

    The data path is resolved here so a relative name typed in Spyder (or any
    other cwd) is the same file the child process opens."""
    data_path = resolve_data_path(data_path)
    if not data_path.exists():
        raise SystemExit(f"no such data file: {data_path}")
    data_path = data_path.resolve()
    with tempfile.TemporaryDirectory(prefix="datamap_") as d:
        out = Path(d) / "report.json"
        r = subprocess.run([python_path, str(SELF), "--_profile", str(data_path),
                            str(out), str(max_n), str(max_s)],
                           capture_output=True, text=True, timeout=900)
        if not out.exists():
            raise SystemExit(f"profiling produced nothing.\n{r.stderr[:2000]}")
        rep = json.loads(out.read_text())
    if not rep.get("ok"):
        raise SystemExit(f"{rep.get('error')}\n\n{rep.get('traceback', '')}")
    return rep


# ================================================================= rendering
MAP_IMPORT = (
    "from AdvancedAnalytics.ReplaceImputeEncode import DT, ReplaceImputeEncode"
)


def render_map(rep, indent="    "):
    """The map as pasteable Python - the form the generated program contains."""
    cols = [c for c in rep["order"] if c in rep["columns"]]
    if not cols:
        return "data_map = {\n}"
    width = max(len(c) for c in cols) + 4
    lines = ["data_map = {"]
    for c in cols:
        spec = rep["columns"][c]
        if spec["type"] in CATEGORICAL_FREE:
            shown = '("")'
        else:
            body = ", ".join(repr(v) for v in spec["values"])
            shown = f"({body},)" if len(spec["values"]) == 1 else f"({body})"
        lines.append(f"{indent}{('%r:' % c).ljust(width)} [DT.{spec['type']}, {shown}],")
    return "\n".join(lines + ["}"])


def render_notes(rep):
    """What the map cannot carry: missingness, warnings, a sample. Imputation is
    the point of ReplaceImputeEncode and the map says nothing about it."""
    L = [f"# {rep['rows']:,} rows x {rep['cols']} columns   "
         f"(pandas {rep['pandas']}, max_n={rep['max_n']}, max_s={rep['max_s']})",
         f"# source: {rep['data_path']}", "#"]
    if rep.get("invalid_names"):
        L += ["# " + "!" * 66,
              "# WARNING - column names that are not valid Python identifiers.",
              "# Rename these in the data file (first row), then draft again:"]
        for item in rep["invalid_names"]:
            L.append(f"#   {item['name']!r}  ({item['reason']})")
        L += ["# " + "!" * 66, "#"]
    if rep.get("mixed"):
        L += ["# " + "!" * 66,
              "# WARNING - columns holding both numbers and strings. However such",
              "# a column is classified, some of its values are the wrong kind.",
              "# Check these by hand:"]
        for c, info in rep["mixed"].items():
            L.append(f"#   {c}: {', '.join(info['kinds'])} "
                     f"-> saw {', '.join(info['sample_values'][:6])}")
        L += ["# " + "!" * 66, "#"]
    if rep.get("dropped"):
        L += [f"# WARNING - not classified: {', '.join(rep['dropped'])}", "#"]
    L.append("# column          dtype        nulls        unique")
    for c in rep["order"]:
        s = rep["stats"][c]
        nul = f"{s['nulls']} ({s['null_pct']}%)" if s["nulls"] else "-"
        L.append(f"#   {c[:14]:<14} {s['dtype']:<12} {nul:<12} {s['unique']}")
    L += ["#", "# first rows:", "#   " + " | ".join(rep["sample_header"])]
    for row in rep["sample_rows"][:8]:
        L.append("#   " + " | ".join(v[:18] for v in row))
    return "\n".join(L)


def render_all(rep):
    return (render_notes(rep) + "\n\n"
            + MAP_IMPORT + "\n\n"
            + render_map(rep) + "\n")


def report_from_map(data_map):
    """Profile-shaped dict so render_map can print an edited data_map."""
    columns, order = {}, []
    for key, spec in data_map.items():
        name = str(key)
        order.append(name)
        atype, vals = spec[0], spec[1]
        if vals is None or vals == "" or vals == ("") or vals == []:
            values = [""]
        elif isinstance(vals, (list, tuple)):
            values = list(vals)
        else:
            values = [vals]
        columns[name] = {"type": _map_type_name(atype), "values": values}
    return {"order": order, "columns": columns}


def _fmt_proportion(proportion):
    text = f"{float(proportion):.6f}".rstrip("0").rstrip(".")
    return text or "0"


def render_sample_script(rep, proportion, fmt="csv"):
    """Starter program: read a virtual file, sample, write a virtual file.

    Filenames are placeholders (input_file.csv / output_file.xlsx) for the
    student to rename. Strata are the Nominal and Binary names already
    identified from the drafted data map."""
    excel = str(fmt).lower() in ("excel", "xlsx", "xls")
    ext = "xlsx" if excel else "csv"
    file_in = f"input_file.{ext}"
    file_out = f"output_file.{ext}"
    size = _fmt_proportion(proportion)
    strata = strata_from_report(rep)
    listed = "[" + ", ".join(repr(c) for c in strata) + "]"
    if excel:
        read_line = "df  = pd.read_excel(file_in)"
        write_line = "Xt.to_excel(file_out, index=False)"
    else:
        read_line = "df  = pd.read_csv(file_in, index_col=None)"
        write_line = "Xt.to_csv(file_out, index=False)"
    return "\n".join([
        "#!/usr/bin/env python3",
        "# -*- coding: utf-8 -*-",
        '"""',
        "Stratified random sample. Rename the input and output filenames",
        "and adapt this script for your ML analysis.",
        '"""',
        "from AdvancedAnalytics.ReplaceImputeEncode import DT, ReplaceImputeEncode",
        "from sklearn.model_selection import train_test_split",
        "import pandas as pd",
        "",
        render_map(rep),
        "",
        f"size      = {size}  # Proportion of all data sampled",
        f'file_in   = "{file_in}"',
        f'file_out  = "{file_out}"',
        "#--------- Create Random Stratified Sample -----------------------------------",
        "# Nominal and Binary features from the data map",
        f"categorical_list = {listed}",
        "",
        read_line,
        "if size >= 1:",
        "    Xt = df",
        "else:",
        "    Xt, Xv = train_test_split(df, train_size=size,",
        "                              stratify=df[categorical_list],",
        "                              random_state=12345)",
        write_line,
        "",
    ])


def parse_data_map(text):
    """Read a data_map from editor text. Returns (True, map) or (False, message)."""
    try:
        from AdvancedAnalytics.ReplaceImputeEncode import DT
    except ImportError:
        class _Shim:                                   # names only, for syntax
            def __getattr__(self, k): return k
        DT = _Shim()
    ns = {"DT": DT}
    try:
        exec(compile(text, "<data_map>", "exec"), ns)
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
    m = ns.get("data_map")
    if not isinstance(m, dict):
        return False, "no dict named 'data_map' was defined"
    for k, v in m.items():
        if not isinstance(v, (list, tuple)) or len(v) != 2:
            return False, f"{k!r}: expected [DT.Type, (values)], got {v!r}"
    return True, m


def validate(text):
    """Is the edited map still valid Python, and well formed? (ok, message)."""
    ok, val = parse_data_map(text)
    if not ok:
        return False, val
    return True, f"valid - {len(val)} columns"


def _map_type_name(atype):
    name = getattr(atype, "name", None)
    if name:
        return str(name)
    return str(atype).rsplit(".", 1)[-1]


def strata_columns(data_map):
    """Nominal and Binary keys, in map order — the strata used to sample."""
    out = []
    for key, spec in data_map.items():
        if _map_type_name(spec[0]) in ("Nominal", "Binary"):
            out.append(str(key))
    return out


def strata_from_report(rep):
    """Same list from a profile report, if the editor has no map yet."""
    cols = (rep or {}).get("columns") or {}
    order = (rep or {}).get("order") or list(cols)
    return [c for c in order if (cols.get(c) or {}).get("type") in ("Nominal", "Binary")]


def suggested_map_save(data_path):
    """Default map name next to the data file: sales.csv -> sales_data_map.py."""
    p = Path(data_path).expanduser()
    folder = p.parent if str(p.parent) not in ("", ".") else Path(".")
    return folder, f"{p.stem}_data_map.py"


def suggested_script_save(data_path):
    """Default name for the generated sample script."""
    if data_path:
        p = Path(data_path).expanduser()
        folder = p.parent if str(p.parent) not in ("", ".") else Path(".")
        return folder, "stratified_sample.py"
    return Path("."), "stratified_sample.py"


# ======================================================================= GUI
# Maroon / gold chrome (same gold as the class residual plots) so the window
# is not the default gray ttk dialog. Layout is a header, a data-file form,
# a labeled editor, actions, then a status bar.

_UI = {
    "maroon": "#500000", "maroon2": "#6B1010", "gold": "#D4AF37",
    "cream": "#F4EFE4", "ink": "#1F1A17", "muted": "#6B6258",
    "paper": "#FFFCF6", "code_bg": "#231E1A", "code_fg": "#F3E6C4",
    "code_sel": "#500000",
}


def _ui_font(kind="ui"):
    families = {
        "ui": (("DejaVu Sans", 16), ("Segoe UI", 16), ("Helvetica", 16)),
        "title": (("DejaVu Sans", 28, "bold"), ("Segoe UI", 28, "bold"),
                  ("Helvetica", 28, "bold")),
        "sub": (("DejaVu Sans", 15), ("Segoe UI", 15), ("Helvetica", 15)),
        "label": (("DejaVu Sans", 16, "bold"), ("Segoe UI", 16, "bold"),
                  ("Helvetica", 16, "bold")),
        "thresh": (("DejaVu Sans", 22), ("Segoe UI", 22), ("Helvetica", 22)),
        "mono": (("DejaVu Sans Mono", 13), ("Consolas", 13), ("Courier", 13)),
    }
    return families.get(kind, families["ui"])[0]


def _style_gui(root):
    from tkinter import ttk
    c = _UI
    root.configure(bg=c["cream"])
    try:
        root.tk.call("tk", "scaling", 1.35)
    except Exception:
        pass
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass
    style.configure(".", background=c["cream"], foreground=c["ink"],
                    font=_ui_font("ui"))
    style.configure("TFrame", background=c["cream"])
    style.configure("Card.TFrame", background=c["cream"])
    style.configure("Header.TFrame", background=c["maroon"])
    style.configure("Header.TLabel", background=c["maroon"], foreground="white",
                    font=_ui_font("title"))
    style.configure("Sub.TLabel", background=c["maroon"], foreground=c["gold"],
                    font=_ui_font("sub"))
    style.configure("TLabel", background=c["cream"], foreground=c["ink"],
                    font=_ui_font("ui"))
    style.configure("Field.TLabel", background=c["cream"], foreground=c["ink"],
                    font=_ui_font("label"))
    style.configure("Thresh.TLabel", background=c["cream"], foreground=c["ink"],
                    font=_ui_font("thresh"))
    style.configure("Hint.TLabel", background=c["cream"], foreground=c["muted"],
                    font=_ui_font("sub"))
    style.configure("Status.TLabel", background=c["maroon"], foreground=c["gold"],
                    font=_ui_font("ui"), padding=8, anchor="w")
    style.configure("TButton", background="#E7DFD0", foreground=c["ink"],
                    padding=(14, 8), font=_ui_font("ui"))
    style.map("TButton",
              background=[("active", "#DDD2BA"), ("pressed", "#D0C4A8")])
    style.configure("Primary.TButton", background=c["maroon"], foreground="white",
                    padding=(18, 9), font=_ui_font("ui"))
    style.map("Primary.TButton",
              background=[("active", c["maroon2"]), ("pressed", "#3D0000")],
              foreground=[("active", "white"), ("pressed", "white")])
    style.configure("TEntry", fieldbackground=c["paper"], foreground=c["ink"],
                    padding=5)
    style.configure("TCombobox", fieldbackground=c["paper"], foreground=c["ink"],
                    padding=4)
    style.map("TCombobox", fieldbackground=[("readonly", c["paper"])],
              foreground=[("readonly", c["ink"])])
    style.configure("TSpinbox", fieldbackground=c["paper"], foreground=c["ink"],
                    padding=4, font=_ui_font("ui"))
    style.configure("TLabelframe", background=c["cream"], foreground=c["ink"])
    style.configure("TLabelframe.Label", background=c["cream"],
                    foreground=c["maroon"], font=_ui_font("label"))
    style.configure("TScrollbar", background=c["cream"], troughcolor="#E7DFD0")
    style.configure("TCheckbutton", background=c["cream"], foreground=c["ink"],
                    font=_ui_font("ui"))
    style.map("TCheckbutton",
              background=[("active", c["cream"])],
              foreground=[("active", c["ink"])])
    return style


def ask_sample_proportion(parent, initial="0.20", fmt="csv"):
    """Modal popup: return (proportion, 'csv'|'excel'), or None if cancelled."""
    import tkinter as tk
    from tkinter import ttk, messagebox

    result = {"value": None}
    c = _UI
    win = tk.Toplevel(parent)
    win.title("Stratify Sample")
    win.transient(parent)
    win.resizable(False, False)
    win.configure(bg=c["cream"])
    try:
        win.tk.call("tk", "scaling", 1.35)
    except Exception:
        pass

    header = ttk.Frame(win, style="Header.TFrame", padding=(18, 10))
    header.pack(fill="x")
    ttk.Label(header, text="Stratify Sample", style="Header.TLabel").pack(anchor="w")

    body = ttk.Frame(win, style="Card.TFrame", padding=(20, 16, 20, 8))
    body.pack(fill="both", expand=True)
    ttk.Label(body, text="Proportion to Sample", style="Field.TLabel").pack(anchor="w")
    ttk.Label(body, text="A value from 1 down to a small amount above 0.",
              style="Hint.TLabel").pack(anchor="w", pady=(2, 8))
    prop_var = tk.StringVar(value=str(initial))
    entry = ttk.Entry(body, textvariable=prop_var, width=18)
    entry.pack(anchor="w")
    entry.focus_set()
    entry.select_range(0, "end")

    ttk.Label(body, text="File type", style="Field.TLabel").pack(
        anchor="w", pady=(14, 4))
    csv_on = str(fmt).lower() not in ("excel", "xlsx", "xls")
    csv_var = tk.BooleanVar(value=csv_on)
    excel_var = tk.BooleanVar(value=not csv_on)
    boxes = ttk.Frame(body, style="Card.TFrame")
    boxes.pack(anchor="w")

    def pick_csv():
        if csv_var.get():
            excel_var.set(False)
        else:
            csv_var.set(True)

    def pick_excel():
        if excel_var.get():
            csv_var.set(False)
        else:
            excel_var.set(True)

    ttk.Checkbutton(boxes, text="csv", variable=csv_var,
                    command=pick_csv).pack(side="left")
    ttk.Checkbutton(boxes, text="excel", variable=excel_var,
                    command=pick_excel).pack(side="left", padx=(16, 0))

    def parse_proportion(text):
        raw = (text or "").strip()
        if not raw:
            return None, "Enter a proportion to sample."
        try:
            p = float(raw)
        except ValueError:
            return None, "Proportion must be a number."
        if not (0.0 < p <= 1.0):
            return None, "Proportion must be greater than 0 and at most 1."
        return p, None

    def accept(event=None):
        p, err = parse_proportion(prop_var.get())
        if err:
            messagebox.showwarning("Proportion to Sample", err, parent=win)
            return
        result["value"] = (p, "excel" if excel_var.get() else "csv")
        win.destroy()

    def cancel(event=None):
        win.destroy()

    buttons = ttk.Frame(win, style="Card.TFrame", padding=(20, 8, 20, 16))
    buttons.pack(fill="x")
    ttk.Button(buttons, text="Cancel", command=cancel).pack(side="right")
    ttk.Button(buttons, text="OK", style="Primary.TButton",
               command=accept).pack(side="right", padx=(0, 8))
    win.bind("<Return>", accept)
    win.bind("<Escape>", cancel)
    win.protocol("WM_DELETE_WINDOW", cancel)

    win.update_idletasks()
    try:
        parent.update_idletasks()
        x = parent.winfo_rootx() + (parent.winfo_width() - win.winfo_width()) // 2
        y = parent.winfo_rooty() + (parent.winfo_height() - win.winfo_height()) // 3
        win.geometry(f"+{max(x, 0)}+{max(y, 0)}")
    except Exception:
        pass
    win.grab_set()
    parent.wait_window(win)
    return result["value"]


def gui():
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox

    envs = conda_envs()
    root = tk.Tk()
    root.title("Data Map")
    root.geometry("1120x820")
    root.minsize(920, 640)
    _style_gui(root)
    c = _UI
    state = {"rep": None, "map_path": None, "sample_prop": None,
             "sample_fmt": "csv", "generated_sample": False}

    header = ttk.Frame(root, style="Header.TFrame", padding=(20, 14))
    header.pack(fill="x")
    ttk.Label(header, text="Data Map", style="Header.TLabel").pack(anchor="w")
    ttk.Label(header, text="Review the draft on this machine. The data never leaves.",
              style="Sub.TLabel").pack(anchor="w", pady=(4, 0))

    form = ttk.Frame(root, style="Card.TFrame", padding=(20, 14, 20, 6))
    form.pack(fill="x")
    form.columnconfigure(1, weight=1)

    ttk.Label(form, text="Data file", style="Field.TLabel").grid(
        row=0, column=0, sticky="w", pady=(0, 8))
    data_var = tk.StringVar()
    ttk.Entry(form, textvariable=data_var).grid(
        row=0, column=1, sticky="ew", padx=(12, 8), pady=(0, 8))

    def browse():
        kwargs = dict(title="Select the data file", filetypes=[
            ("Data files", "*.csv *.tsv *.parquet *.xlsx *.xls *.json"),
            ("All files", "*.*")])
        current = data_var.get().strip()
        if current:
            folder = Path(current).expanduser().parent
            if folder.is_dir():
                kwargs["initialdir"] = str(folder)
        p = filedialog.askopenfilename(**kwargs)
        if p:
            data_var.set(p)
    ttk.Button(form, text="Browse", command=browse).grid(
        row=0, column=2, sticky="e", pady=(0, 8))

    ttk.Label(form, text="Environment", style="Field.TLabel").grid(
        row=1, column=0, sticky="w")
    env_var = tk.StringVar(value=default_env_name(envs))
    ttk.Combobox(form, textvariable=env_var, values=[n for n, _ in envs],
                 state="readonly", width=24).grid(
        row=1, column=1, sticky="w", padx=(12, 8))
    thr = ttk.Frame(form, style="Card.TFrame")
    thr.grid(row=1, column=2, sticky="e")
    tk.Label(thr, text="max_n", font=_ui_font("thresh"),
             bg=c["cream"], fg=c["ink"]).pack(side="left")
    n_var = tk.StringVar(value="10")
    ttk.Spinbox(thr, from_=2, to=200, textvariable=n_var, width=5).pack(
        side="left", padx=(6, 14))
    tk.Label(thr, text="max_s", font=_ui_font("thresh"),
             bg=c["cream"], fg=c["ink"]).pack(side="left")
    s_var = tk.StringVar(value="30")
    ttk.Spinbox(thr, from_=2, to=500, textvariable=s_var, width=5).pack(
        side="left", padx=(6, 0))

    ttk.Label(root, style="Hint.TLabel",
              text="max_n turns a low-cardinality number into Nominal — right for a "
                   "1–5 rating, wrong for a count. Correct the draft, then save.").pack(
        anchor="w", padx=20, pady=(2, 8))

    status = tk.StringVar(value="Select a data file and an environment, then draft the map.")
    ttk.Label(root, textvariable=status, style="Status.TLabel").pack(
        fill="x", side="bottom")
    bar = ttk.Frame(root, style="Card.TFrame", padding=(20, 8, 20, 10))
    bar.pack(fill="x", side="bottom")

    pane = ttk.LabelFrame(root, text="  Draft map  ", padding=8)
    pane.pack(fill="both", expand=True, padx=20, pady=(0, 8))
    wrap = ttk.Frame(pane, style="Card.TFrame")
    wrap.pack(fill="both", expand=True)
    wrap.rowconfigure(0, weight=1)
    wrap.columnconfigure(0, weight=1)
    txt = tk.Text(wrap, wrap="none", undo=True, font=_ui_font("mono"),
                  bg=c["code_bg"], fg=c["code_fg"], insertbackground=c["gold"],
                  selectbackground=c["code_sel"], selectforeground="white",
                  relief="flat", borderwidth=0, padx=10, pady=10,
                  highlightthickness=0)
    ys = ttk.Scrollbar(wrap, orient="vertical", command=txt.yview)
    xs = ttk.Scrollbar(wrap, orient="horizontal", command=txt.xview)
    txt.configure(yscrollcommand=ys.set, xscrollcommand=xs.set)
    txt.grid(row=0, column=0, sticky="nsew")
    ys.grid(row=0, column=1, sticky="ns")
    xs.grid(row=1, column=0, sticky="ew")

    def _wheel(event):
        step = 1 if (getattr(event, "num", None) == 5
                     or getattr(event, "delta", 0) < 0) else -1
        if event.state & 1:                       # Shift: horizontal
            txt.xview_scroll(step, "units")
        else:
            txt.yview_scroll(step, "units")
        return "break"
    txt.bind("<MouseWheel>", _wheel)
    txt.bind("<Button-4>", _wheel)
    txt.bind("<Button-5>", _wheel)

    def do_draft():
        raw = data_var.get().strip()
        if not raw:
            messagebox.showwarning("No data file",
                                   "Choose a data file first.")
            return
        try:
            resolved = resolve_data_path(raw)
        except SystemExit as e:
            messagebox.showwarning("No data file", str(e)); return
        data_var.set(str(resolved))
        status.set("profiling in the selected environment..."); root.update_idletasks()
        try:
            rep = draft(resolved, env_python(env_var.get()),
                        int(n_var.get()), int(s_var.get()))
        except SystemExit as e:
            status.set("failed"); messagebox.showerror("Draft failed", str(e)); return
        state["rep"] = rep
        state["map_path"] = None
        state["generated_sample"] = False
        pane.configure(text="  Draft map  ")
        txt.delete("1.0", "end"); txt.insert("1.0", render_all(rep))
        bad = rep.get("invalid_names") or []
        mixed = len(rep.get("mixed") or {})
        extra = []
        if bad:
            extra.append(f"{len(bad)} invalid name(s)")
        if mixed:
            extra.append(f"{mixed} mixed column(s)")
        status.set(f"{rep['rows']:,} rows, {len(rep['columns'])} columns mapped"
                   + (f" - {', '.join(extra)}" if extra else ""))
        btn_draft.configure(style="TButton")
        btn_stratify.configure(style="Primary.TButton")
        if bad:
            messagebox.showwarning("Invalid column names",
                                   format_invalid_names_message(bad))

    def do_stratify():
        initial = state["sample_prop"] if state["sample_prop"] is not None else 0.20
        choice = ask_sample_proportion(root, initial=initial,
                                       fmt=state.get("sample_fmt") or "csv")
        if choice is None:
            return
        prop, fmt = choice
        state["sample_prop"] = prop
        state["sample_fmt"] = fmt

        ok, parsed = parse_data_map(txt.get("1.0", "end"))
        if ok:
            spec = report_from_map(parsed)
        elif state["rep"]:
            spec = state["rep"]
        else:
            messagebox.showwarning(
                "No data map",
                "Draft the data map first. Nominal and Binary columns "
                "define the strata in the generated script.")
            return
        if prop < 1.0 and not strata_from_report(spec):
            messagebox.showwarning(
                "No strata",
                "The data map has no Nominal or Binary columns.\n"
                "Those types define the strata in the generated script.")
            return

        script = render_sample_script(spec, prop, fmt)
        txt.delete("1.0", "end")
        txt.insert("1.0", script)
        state["generated_sample"] = True
        pane.configure(text="  Generated sample script  ")
        strata = strata_from_report(spec)
        shown = ", ".join(strata[:8])
        extra = f" and {len(strata) - 8} more" if len(strata) > 8 else ""
        names = ("input_file.xlsx / output_file.xlsx" if fmt == "excel"
                 else "input_file.csv / output_file.csv")
        if prop >= 1.0:
            status.set(f"generated script — all rows ({names})")
        else:
            status.set(f"generated script — stratify on {shown}{extra} ({names})")
        btn_stratify.configure(style="TButton")

    def do_save_map():
        if not txt.get("1.0", "end-1c").strip():
            return
        data_path = ((state["rep"] or {}).get("data_path") or data_var.get())
        if state.get("generated_sample"):
            folder, name = suggested_script_save(data_path)
        elif state["rep"] or data_path:
            folder, name = suggested_map_save(data_path)
        else:
            return
        p = filedialog.asksaveasfilename(
            defaultextension=".py",
            initialdir=str(folder),
            initialfile=name,
            filetypes=[("Python files", "*.py"), ("All files", "*.*")],
        )
        if p:
            Path(p).write_text(txt.get("1.0", "end"))
            state["map_path"] = str(Path(p).expanduser().resolve())
            status.set(f"wrote {state['map_path']}")

    btn_draft = ttk.Button(bar, text="Draft Data Map", style="Primary.TButton",
                           command=do_draft)
    btn_draft.pack(side="left")
    btn_stratify = ttk.Button(bar, text="Stratify Sample", command=do_stratify)
    btn_stratify.pack(side="left", padx=(8, 0))
    ttk.Button(bar, text="Save Map", command=do_save_map).pack(side="right")
    root.mainloop()


# ======================================================================= CLI
def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--_profile":
        _profile_mode(sys.argv[2:])                 # internal re-entry

    ap = argparse.ArgumentParser(
        description="Draft and review a data map, on the machine that owns the data.")
    ap.add_argument("--data", help="path to the data file")
    ap.add_argument("--env")
    ap.add_argument("--python", help="interpreter path, instead of --env")
    ap.add_argument("--max-n", type=int, default=10)
    ap.add_argument("--max-s", type=int, default=30)
    ap.add_argument("--out", metavar="FILE", help="write the map source here")
    ap.add_argument("--json", metavar="FILE", help="write the raw report here")
    ap.add_argument("--list-envs", action="store_true")
    ap.add_argument("--no-notes", action="store_true", help="map only")
    a = ap.parse_args()

    if a.list_envs:
        for n, p in conda_envs():
            print(f"{n:<20} {p}")
        return
    if not a.data:
        gui(); return

    py = a.python or (env_python(a.env) if a.env else sys.executable)
    rep = draft(a.data, py, a.max_n, a.max_s)
    if a.no_notes:
        text = MAP_IMPORT + "\n\n" + render_map(rep) + "\n"
    else:
        text = render_all(rep)
    print(text, flush=True)
    map_path = ""
    if a.out:
        Path(a.out).write_text(text)
        map_path = str(Path(a.out).expanduser().resolve())
        print(f"wrote {map_path}", file=sys.stderr)
    if a.json:
        Path(a.json).write_text(json.dumps(rep, indent=2))
        print(f"wrote {a.json}", file=sys.stderr)
    print(f"[validate] {validate(text)[1]}", file=sys.stderr)


if __name__ == "__main__":
    main()
