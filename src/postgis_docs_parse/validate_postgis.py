import json
import os
import re
import time
import argparse
from datetime import datetime
from pathlib import Path

import psycopg2
from dotenv import load_dotenv
from openai import OpenAI
from psycopg2.extras import RealDictCursor

# Load environment variables (API Key)
load_dotenv()

DEFAULT_EXTERNAL_TABLE_SOURCES_FILE = Path(__file__).with_name("external_table_sources.json")

class ManualRollback(Exception):
    """Custom exception to trigger a transaction rollback after validation."""
    pass

class PostGISValidator:
    def __init__(self, db_config, input_file, output_file, manual_review_file, external_table_sources_file=None):
        self.db_config = db_config
        self.input_file = input_file
        self.output_file = output_file
        self.manual_review_file = manual_review_file
        self.external_table_sources_file = (
            external_table_sources_file
            or os.getenv("EXTERNAL_TABLE_SOURCES_FILE")
            or str(DEFAULT_EXTERNAL_TABLE_SOURCES_FILE)
        )
        self.external_table_sources = {}
        
        # Initialize LLM for auto-fixing SQL
        api_key = os.getenv("api_key") or os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("base_url") or os.getenv("OPENAI_BASE_URL")
        if not api_key:
            self.client = None
            print("⚠️  [Warn] OpenAI client is not configured. LLM auto-fix will be unavailable.")
        else:
            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url,
            )
        
        # Missing-table recognition and dependency analysis.
        # Support both English and Chinese PostgreSQL error messages.
        self.re_missing_relation = re.compile(
            r'(?:relation|关系)\s+"(?P<name>[^"]+)"\s+(?:does not exist|不存在)',
            re.IGNORECASE
        )
        self.re_create_table = re.compile(
            r'\bCREATE\s+(?:TEMP(?:ORARY)?\s+)?TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(?P<name>(?:"[^"]+"|\w+)(?:\.(?:"[^"]+"|\w+))?)',
            re.IGNORECASE
        )
        
        # High-level missing-table classification labels.
        self.missing_case_labels_cn = {
            "context_ddl_found": "Context DDL Available",
            "external_data_import": "External Data Import Required",
            "unknown_todo": "Unknown Follow-up Required",
        }
        
        # Known external datasets used to classify import requirements.
        self.known_external_tables = {
            "nyc_census_blocks": {"dataset_key": "postgis_workshop_nyc", "source_hint": "https://postgis.net/workshops/postgis-intro/about_data.html"},
            "nyc_neighborhoods": {"dataset_key": "postgis_workshop_nyc", "source_hint": "https://postgis.net/workshops/postgis-intro/about_data.html"},
            "nyc_streets": {"dataset_key": "postgis_workshop_nyc", "source_hint": "https://postgis.net/workshops/postgis-intro/about_data.html"},
            "nyc_subway_stations": {"dataset_key": "postgis_workshop_nyc", "source_hint": "https://postgis.net/workshops/postgis-intro/about_data.html"},
            "nyc_census_sociodata": {"dataset_key": "postgis_workshop_nyc", "source_hint": "https://postgis.net/workshops/postgis-intro/about_data.html"},
        }
        self._load_external_table_sources(self.external_table_sources_file)
        
        # Tolerances for consistency checks.
        self.eps_numeric = 1e-9
        self.eps_box = 5e-2

    def _load_external_table_sources(self, file_path):
        if not file_path or not isinstance(file_path, str):
            return
        if not os.path.exists(file_path):
            return
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return
        if isinstance(data, dict) and isinstance(data.get("tables"), dict):
            tables = data.get("tables") or {}
        elif isinstance(data, dict):
            tables = data
        else:
            return
        for k, v in tables.items():
            if not isinstance(k, str):
                continue
            name = self._normalize_table_name(k)
            if not name:
                continue
            unqualified = name.split(".")[-1]
            parsed = self._parse_external_table_source_entry(v)
            if parsed is None:
                continue
            self.external_table_sources[unqualified] = parsed
            source_hint = parsed.get("hint")
            if source_hint:
                prev = self.known_external_tables.get(unqualified) or {}
                self.known_external_tables[unqualified] = {"dataset_key": prev.get("dataset_key"), "source_hint": source_hint}

    def _parse_external_table_source_entry(self, entry):
        if isinstance(entry, str):
            hint = entry.strip()
            if not hint:
                return None
            imported = False
            prefixes = ["[imported]", "imported:", "[已导入]", "已导入:"]
            for p in prefixes:
                if hint.lower().startswith(p.lower()):
                    imported = True
                    hint = hint[len(p):].strip()
                    break
            kind = "text"
            if re.match(r"^https?://", hint, re.IGNORECASE):
                kind = "url"
            elif hint.lower().endswith(".shp") or os.path.exists(hint):
                kind = "shp"
            return {"hint": hint, "kind": kind, "imported": imported}

        if isinstance(entry, dict):
            imported = (entry.get("status") == "imported") or bool(entry.get("imported"))
            hint = entry.get("source_hint")
            if not isinstance(hint, str) or not hint.strip():
                how = entry.get("how_to_get")
                if isinstance(how, dict):
                    how_val = how.get("value")
                    if isinstance(how_val, str) and how_val.strip():
                        hint = how_val.strip()
                    input_path = how.get("input_path") or how.get("shp_path")
                    if not hint and isinstance(input_path, str) and input_path.strip():
                        hint = input_path.strip()
                if not hint:
                    hint = entry.get("hint")
            if not isinstance(hint, str) or not hint.strip():
                return None
            hint = hint.strip()
            kind = entry.get("kind")
            if kind not in {"url", "text", "sql", "shp"}:
                if re.match(r"^https?://", hint, re.IGNORECASE):
                    kind = "url"
                elif hint.lower().endswith(".shp") or os.path.exists(hint):
                    kind = "shp"
                else:
                    kind = "text"
            return {"hint": hint, "kind": kind, "imported": imported, "raw": entry}

        return None

    def import_external_tables_via_shp2db(self, from_manual_review_file, if_exists="append", schema=None):
        targets = self._collect_external_import_tables(from_manual_review_file)
        if not targets:
            print("No external_data_import tables found in manual review.")
            return
        db_url = self._build_sqlalchemy_db_url()
        from . import shp2db as shp2db_module
        for table_name in sorted(targets):
            cfg = self.external_table_sources.get(table_name) or {}
            if cfg.get("imported"):
                continue

            raw = cfg.get("raw") if isinstance(cfg, dict) else None
            input_path = None
            target_table = table_name
            target_schema = schema
            target_if_exists = if_exists

            if cfg.get("kind") == "shp":
                input_path = cfg.get("hint")
            elif isinstance(raw, dict):
                import_cfg = raw.get("import") if isinstance(raw.get("import"), dict) else None
                how = raw.get("how_to_get") if isinstance(raw.get("how_to_get"), dict) else None
                if not import_cfg and how and how.get("type") == "shp2db":
                    import_cfg = how
                if import_cfg:
                    input_path = import_cfg.get("input_path") or import_cfg.get("shp_path")
                    target_table = import_cfg.get("table_name") or target_table
                    target_schema = import_cfg.get("schema") or target_schema
                    target_if_exists = import_cfg.get("if_exists") or target_if_exists

            if not input_path:
                continue

            print(f"[Import][shp2db] {input_path} -> {target_schema + '.' if target_schema else ''}{target_table} ({target_if_exists})")
            shp2db_module.shp2db(
                input_path=input_path,
                db_url=db_url,
                table_name=target_table,
                schema=target_schema,
                if_exists=target_if_exists,
            )

    def _collect_external_import_tables(self, from_manual_review_file):
        if not from_manual_review_file or not os.path.exists(from_manual_review_file):
            return set()
        try:
            with open(from_manual_review_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return set()
        tables = set()
        if not isinstance(data, list):
            return tables
        for item in data:
            if not isinstance(item, dict):
                continue
            for issue in item.get("issues", []) or []:
                if not isinstance(issue, dict):
                    continue
                if issue.get("issue_type") != "missing_table":
                    continue
                if issue.get("missing_case") != "external_data_import":
                    continue
                name = issue.get("missing_table_name")
                if not name:
                    continue
                tables.add(self._normalize_table_name(name).split(".")[-1])
        return tables

    def _build_sqlalchemy_db_url(self):
        user = self.db_config.get("user") or ""
        password = self.db_config.get("password") or ""
        host = self.db_config.get("host") or "localhost"
        port = self.db_config.get("port") or 5432
        dbname = self.db_config.get("dbname") or ""
        return f"postgresql://{user}:{password}@{host}:{port}/{dbname}"

    def _missing_case_resolution_cn(self, missing_case):
        if missing_case == "context_ddl_found":
            return "Execute the CREATE TABLE statement found in context first, then continue validation with the example data and steps."
        if missing_case == "external_data_import":
            return "No CREATE TABLE statement was found in context. Import the required schema and data from an external dataset, then rerun validation."
        return "Context and external hints were both insufficient. Record the missing table for manual follow-up."

    def _classify_execution_error(self, pg_error_code, error_msg, sql_text):
        msg = (error_msg or "").lower()
        code = (pg_error_code or "").upper()
        if code == "42601" or "syntax error" in msg:
            return {
                "root_cause": "syntax_error",
                "root_cause_cn": "SQL Syntax Error",
                "suggestion": "Check SQL syntax, including parentheses and quoting, and compare against the original documentation example if needed.",
            }
        if code == "42703" or "column" in msg and "does not exist" in msg:
            return {
                "root_cause": "undefined_column",
                "root_cause_cn": "Undefined Column",
                "suggestion": "Verify that the table schema matches the expected structure. Missing columns often come from incomplete DDL or a mismatched imported dataset version.",
            }
        if code == "42883" or "function" in msg and "does not exist" in msg:
            return {
                "root_cause": "undefined_function",
                "root_cause_cn": "Undefined Function or Signature Mismatch",
                "suggestion": "Ensure the PostGIS extension is installed, verify the function name and argument types, and cast WKT strings to ::geometry when needed.",
            }
        if code in {"42804", "22P02"} or "invalid input syntax" in msg or "cannot cast" in msg:
            return {
                "root_cause": "type_mismatch",
                "root_cause_cn": "Type Mismatch or Cast Failure",
                "suggestion": "Check argument types such as text, geometry, and numeric. Prefer converting WKT literals to geometry and add explicit casts when needed.",
            }
        if "permission denied" in msg:
            return {
                "root_cause": "permission_denied",
                "root_cause_cn": "Permission Denied",
                "suggestion": "Verify database user permissions, schema privileges, and function execution permissions.",
            }
        if "parse error" in msg and "geometry" in msg:
            return {
                "root_cause": "invalid_geometry",
                "root_cause_cn": "Invalid Geometry",
                "suggestion": "Check whether the WKT/WKB content is valid. Use ST_IsValid or ST_MakeValid before validating if necessary.",
            }
        return {
            "root_cause": "other",
            "root_cause_cn": "Other Execution Error",
            "suggestion": "Use the PostgreSQL error code and message to identify the root cause, starting with missing dependencies such as tables, columns, or functions.",
        }

    def get_db_connection(self):
        """
        Establish DB connection with autocommit disabled.
        """
        conn = psycopg2.connect(**self.db_config, cursor_factory=RealDictCursor)
        conn.autocommit = False 
        return conn

    def check_environment(self):
        """Verify PostGIS extension presence."""
        try:
            conn = self.get_db_connection()
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS postgis;")
                conn.commit()
                print("✅ [Env Check] PostGIS extension verified/installed.")
        except Exception as e:
            print(f"❌ [Env Check Failed] {e}")
            exit(1)
        finally:
            if conn: conn.close()

    def _generate_table_fix_via_llm(self, failed_sql, error_msg):
        """Ask LLM to generate DDL for missing tables."""
        if self.client is None:
            print("      [Auto-Fix] LLM client unavailable. Please configure API access or provide DDL manually.")
            return None
        print(f"      [Auto-Fix] Asking LLM to fix missing table...")
        prompt = f"""
        You are a DB Admin. A SQL execution failed in PostGIS.
        **Failed SQL**: `{failed_sql}`
        **Error**: `{error_msg}`
        **Task**: Generate a minimal `CREATE TABLE` statement (with geometry columns if needed) to make the SQL valid.
        **Output**: ONLY the raw SQL code. No markdown.
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    timeout=60.0
                )
                return response.choices[0].message.content.replace("```sql", "").replace("```", "").strip()
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"      [LLM Error]: {e} (Retrying {attempt+1}/{max_retries})...")
                    time.sleep(2)
                    continue
                print(f"      [LLM Error]: {e}")
                return None

    def _split_sql_statements(self, sql_text):
        """
        Split multiple SQL statements generated by the LLM in a safe, simple way.
        """
        if not sql_text or not isinstance(sql_text, str):
            return []
        parts = [p.strip() for p in sql_text.split(";")]
        return [p + ";" for p in parts if p]

    def _quote_ident(self, ident):
        if ident is None:
            return None
        ident = str(ident)
        ident = ident.replace('"', '""')
        return f"\"{ident}\""

    def _quote_table_name(self, table_name):
        """
        Quote schema-qualified table names safely.
        """
        table_name = self._normalize_table_name(table_name)
        if not table_name:
            return None
        parts = table_name.split(".")
        return ".".join(self._quote_ident(p) for p in parts)

    def _looks_like_wkt(self, s):
        if not isinstance(s, str):
            return False
        x = s.strip().upper()
        return x.startswith(("POINT", "LINESTRING", "POLYGON", "MULTIPOINT", "MULTILINESTRING", "MULTIPOLYGON", "GEOMETRYCOLLECTION"))

    def _sql_literal(self, value, geometry=False):
        if value is None:
            return "NULL"
        if isinstance(value, bool):
            return "TRUE" if value else "FALSE"
        if isinstance(value, (int, float)):
            return str(value)
        s = str(value)
        s = s.replace("'", "''")
        if geometry:
            raw = s.strip()
            if raw.upper().startswith("SRID="):
                return f"ST_GeomFromEWKT('{s}')"
            return f"ST_GeomFromText('{s}')"
        return f"'{s}'"

    def _lookup_known_external_dataset(self, table_name):
        """
        Check whether a missing table belongs to a known external dataset.
        """
        table_name = self._normalize_table_name(table_name)
        if not table_name:
            return None
        unqualified = table_name.split(".")[-1]
        return self.known_external_tables.get(unqualified)

    def _generate_inserts_from_expected_result(self, table_name, expected_rows, geometry_column=None):
        """
        Build INSERT statements from explicit expected-result rows when possible.

        Only explicit documented data is used here. No synthetic or minimal rows
        are generated automatically.
        """
        if not table_name or not isinstance(expected_rows, list) or not expected_rows:
            return []
        if not isinstance(expected_rows[0], dict):
            return []
        
        q_table = self._quote_table_name(table_name)
        if not q_table:
            return []
        
        col_names = list(expected_rows[0].keys())
        q_cols = ", ".join(self._quote_ident(c) for c in col_names)
        stmts = []
        for row in expected_rows:
            if not isinstance(row, dict):
                continue
            values = []
            for c in col_names:
                v = row.get(c)
                is_geom = False
                if geometry_column and str(c).lower() == str(geometry_column).lower():
                    is_geom = True
                if not is_geom and isinstance(v, str) and self._looks_like_wkt(v):
                    is_geom = True
                values.append(self._sql_literal(v, geometry=is_geom))
            stmts.append(f"INSERT INTO {q_table} ({q_cols}) VALUES ({', '.join(values)});")
        return stmts

    def _is_insert_like_sql(self, sql_text):
        if not isinstance(sql_text, str):
            return False
        s = sql_text.strip().lower()
        return s.startswith(("insert ", "update ", "delete "))

    def _sql_mentions_table(self, sql_text, table_name):
        """
        Perform a lightweight check for whether a SQL statement references a table.
        """
        if not isinstance(sql_text, str) or not table_name:
            return False
        t = self._normalize_table_name(table_name)
        if not t:
            return False
        candidates = {t, t.split(".")[-1]}
        for cand in candidates:
            pat = re.escape(cand)
            if re.search(rf'\b(from|join)\s+{pat}\b', sql_text, re.IGNORECASE):
                return True
        return False
    
    def _fallback_extract_table_from_sql(self, sql_text):
        """
        Fallback extractor for the first table name from FROM/JOIN clauses.
        """
        if not isinstance(sql_text, str):
            return None
        m = re.search(r'\b(from|join)\s+(?P<name>(?:"[^"]+"|\w+)(?:\.(?:"[^"]+"|\w+))?)', sql_text, re.IGNORECASE)
        if not m:
            return None
        return self._normalize_table_name(m.group("name"))

    def _preprocess_sql_for_validation(self, sql_text):
        """
        Apply lightweight, low-risk SQL rewrites before validation.

        Current rewrites:
        - ST_Force_3DZ -> ST_Force3DZ
        - ST_XMax('WKT') -> ST_XMax('WKT'::geometry), and related functions

        Returns:
            A tuple of ``(new_sql, meta)``.
        """
        if not isinstance(sql_text, str):
            return sql_text, {}
        
        meta = {"sql_rewritten": False, "sql_rewrites": []}
        new_sql = sql_text
        
        # 1) Function name normalization.
        fixed = re.sub(r'\bST_Force_3DZ\b', 'ST_Force3DZ', new_sql, flags=re.IGNORECASE)
        if fixed != new_sql:
            new_sql = fixed
            meta["sql_rewritten"] = True
            meta["sql_rewrites"].append("rename: ST_Force_3DZ -> ST_Force3DZ")
        
        # 2) Cast WKT literals to geometry before BOX3D-style parsing paths.
        funcs = [
            "ST_XMin", "ST_XMax",
            "ST_YMin", "ST_YMax",
            "ST_ZMin", "ST_ZMax",
            "ST_MMin", "ST_MMax",
            "ST_Extent",
        ]
        func_alt = "|".join(funcs)
        pat = re.compile(rf'\b(?P<func>{func_alt})\s*\(\s*\'(?P<wkt>[^\']+)\'\s*\)', re.IGNORECASE)
        
        def _repl(m):
            func = m.group("func")
            wkt = m.group("wkt")
            if not self._looks_like_wkt(wkt):
                return m.group(0)
            meta["sql_rewritten"] = True
            meta["sql_rewrites"].append(f"cast: {func}('WKT') -> {func}('WKT'::geometry)")
            return f"{func}('{wkt}'::geometry)"
        
        fixed2 = pat.sub(_repl, new_sql)
        if fixed2 != new_sql:
            new_sql = fixed2
        
        if meta["sql_rewritten"]:
            meta["sql_after_rewrite"] = new_sql
        
        return new_sql, meta

    def _try_parse_float(self, x):
        if x is None:
            return None
        if isinstance(x, (int, float)):
            return float(x)
        if isinstance(x, str):
            s = x.strip()
            try:
                return float(s)
            except Exception:
                return None
        return None

    def _extract_numbers(self, s):
        if not isinstance(s, str):
            return None
        nums = re.findall(r'[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?', s)
        if not nums:
            return None
        try:
            return [float(n) for n in nums]
        except Exception:
            return None

    def _is_box_like(self, s):
        if not isinstance(s, str):
            return False
        x = s.strip().upper()
        return x.startswith("BOX(") or x.startswith("BOX3D(")

    def _all_close(self, a, b, eps):
        if a is None or b is None:
            return False
        if len(a) != len(b):
            return False
        for i in range(len(a)):
            if abs(a[i] - b[i]) > eps:
                return False
        return True

    def _find_context_create_sql(self, missing_table_name, missing_info, context):
        """
        Try to locate a CREATE TABLE statement for a missing table from context.
        """
        if not missing_table_name or not isinstance(context, dict):
            return None, None
        
        missing_table_name = self._normalize_table_name(missing_table_name)
        if not missing_table_name:
            return None, None
        
        dep_scope = (missing_info or {}).get("dep_scope")
        dep_example_id = (missing_info or {}).get("dep_example_id")
        dep_function_id = (missing_info or {}).get("dep_function_id")
        
        steps_ctx = context.get("steps") or []
        step_index = context.get("step_index")
        create_sql = self._find_create_table_sql_in_steps(steps_ctx, missing_table_name, before_step_index=step_index)
        if create_sql:
            return create_sql, "intra_previous_step"
        
        example_index = context.get("example_index") or {}
        table_create_index = context.get("table_create_index") or {}
        current_function_id = context.get("function_id")
        
        if dep_scope == "same_func_dep" and dep_example_id is not None:
            dep_steps = example_index.get((current_function_id, dep_example_id))
            create_sql = self._find_create_table_sql_in_steps(dep_steps, missing_table_name)
            if create_sql:
                return create_sql, "same_func_dep"
        if dep_scope == "cross_func_dep" and dep_function_id and dep_example_id is not None:
            dep_steps = example_index.get((dep_function_id, dep_example_id))
            create_sql = self._find_create_table_sql_in_steps(dep_steps, missing_table_name)
            if create_sql:
                return create_sql, "cross_func_dep"
        
        candidates = table_create_index.get(missing_table_name) or table_create_index.get(missing_table_name.split(".")[-1]) or []
        if candidates:
            same_func = next((c for c in candidates if c.get("function_id") == current_function_id), None)
            return (same_func or candidates[0]).get("create_sql"), "global_index"
        
        return None, None

    def _decide_missing_case(self, missing_table_name, missing_info, context):
        """
        Classify a missing table into the high-level remediation buckets.
        """
        create_sql, ddl_source = self._find_context_create_sql(missing_table_name, missing_info, context)
        if create_sql:
            return {
                "missing_case": "context_ddl_found",
                "missing_case_label_cn": self.missing_case_labels_cn["context_ddl_found"],
                "ddl_sql": create_sql,
                "ddl_source": ddl_source
            }
        
        dataset = self._lookup_known_external_dataset(missing_table_name)
        if dataset:
            return {
                "missing_case": "external_data_import",
                "missing_case_label_cn": self.missing_case_labels_cn["external_data_import"],
                "dataset_key": dataset.get("dataset_key"),
                "source_hint": dataset.get("source_hint"),
                "lookup_status": "known_mapping",
                "search_keywords": self._build_external_search_keywords(missing_table_name, dataset_key=dataset.get("dataset_key"))
            }

        missing_type = (missing_info or {}).get("missing_type")
        if missing_type == "external":
            return {
                "missing_case": "external_data_import",
                "missing_case_label_cn": self.missing_case_labels_cn["external_data_import"],
                "dataset_key": None,
                "source_hint": "No CREATE TABLE statement was found in context. The table is marked as external and should usually be imported from a PostGIS sample dataset, tutorial dataset, or another external source.",
                "lookup_status": "needs_search",
                "search_keywords": self._build_external_search_keywords(missing_table_name, dataset_key=None),
            }

        return {
            "missing_case": "external_data_import",
            "missing_case_label_cn": self.missing_case_labels_cn["external_data_import"],
            "dataset_key": None,
            "source_hint": "No CREATE TABLE statement was found in context and no known dataset mapping matched. Search by table name in external datasets or tutorial resources, import the data, and then rerun validation.",
            "lookup_status": "needs_search",
            "search_keywords": self._build_external_search_keywords(missing_table_name, dataset_key=None),
        }

    def _build_external_search_keywords(self, missing_table_name, dataset_key=None):
        name = self._normalize_table_name(missing_table_name) or ""
        short = name.split(".")[-1] if name else ""
        candidates = [c for c in [short, name] if c]
        keywords = []
        for c in candidates:
            keywords.extend([
                f"{c} postgis",
                f"{c} shapefile download",
                f"{c} dataset",
                f"{c} postgis workshop",
            ])
        if dataset_key:
            keywords.append(f"{dataset_key} {short or name}".strip())
        seen = set()
        unique = []
        for k in keywords:
            if k in seen:
                continue
            seen.add(k)
            unique.append(k)
        return unique[:8]

    def _normalize_table_name(self, table_name):
        if not isinstance(table_name, str):
            return None
        return table_name.strip().strip('"')

    def _normalize_missing_tables(self, missing_tables):
        """
        Normalize ``step.missing_tables`` across legacy and current formats.
        """
        if not missing_tables:
            return []
        normalized = []
        if isinstance(missing_tables, list):
            for item in missing_tables:
                if isinstance(item, str):
                    t = self._normalize_table_name(item)
                    if not t:
                        continue
                    normalized.append({
                        "table": t,
                        "missing_type": None,
                        "dep_scope": None,
                        "dep_example_id": None,
                        "dep_function_id": None,
                        "table_features": {
                            "has_geometry": None,
                            "geometry_column": None,
                            "primary_key": None
                        }
                    })
                elif isinstance(item, dict):
                    t = self._normalize_table_name(item.get("table") or item.get("name"))
                    if not t:
                        continue
                    features = item.get("table_features") if isinstance(item.get("table_features"), dict) else {}
                    normalized.append({
                        "table": t,
                        "missing_type": item.get("missing_type"),
                        "dep_scope": item.get("dep_scope"),
                        "dep_example_id": item.get("dep_example_id"),
                        "dep_function_id": item.get("dep_function_id"),
                        "table_features": {
                            "has_geometry": features.get("has_geometry"),
                            "geometry_column": features.get("geometry_column"),
                            "primary_key": features.get("primary_key")
                        }
                    })
        return normalized

    def _extract_missing_table_name(self, error_msg):
        """
        Extract a missing relation or table name from a PostgreSQL error message.
        """
        if not error_msg:
            return None
        m = self.re_missing_relation.search(error_msg)
        if not m:
            return None
        return self._normalize_table_name(m.group("name"))

    def _pick_missing_table_info(self, step, table_name):
        """
        Retrieve matching table metadata from ``step.missing_tables``.
        """
        table_name = self._normalize_table_name(table_name)
        mts = self._normalize_missing_tables(step.get("missing_tables"))
        for mt in mts:
            if self._normalize_table_name(mt.get("table")) == table_name:
                return mt
        return None

    def _find_create_table_sql_in_steps(self, steps, target_table, before_step_index=None):
        """
        Find a CREATE TABLE statement for a target table inside a step list.
        """
        target_table = self._normalize_table_name(target_table)
        if not target_table or not isinstance(steps, list):
            return None
        
        end = before_step_index if isinstance(before_step_index, int) else len(steps)
        end = max(0, min(end, len(steps)))
        for i in range(end):
            step = steps[i]
            if not isinstance(step, dict):
                continue
            sql = step.get("sql") or ""
            m = self.re_create_table.search(sql)
            if not m:
                continue
            created = self._normalize_table_name(m.group("name"))
            if created == target_table:
                return sql
        return None

    def _build_dependency_indexes(self, data):
        """
        Build indexes used for cross-example dependency recovery.
        - example_index: (function_id, example_id) -> steps
        - table_create_index: table -> [{function_id, example_id, create_sql}]
        """
        example_index = {}
        table_create_index = {}
        for entry in data:
            if not isinstance(entry, dict):
                continue
            func_id = entry.get("function_id")
            for ex in entry.get("examples", []) or []:
                if not isinstance(ex, dict):
                    continue
                ex_id = ex.get("example_id")
                steps = ex.get("steps", []) or []
                example_index[(func_id, ex_id)] = steps
                
                for step in steps:
                    if not isinstance(step, dict):
                        continue
                    sql = step.get("sql") or ""
                    m = self.re_create_table.search(sql)
                    if not m:
                        continue
                    table = self._normalize_table_name(m.group("name"))
                    if not table:
                        continue
                    table_create_index.setdefault(table, [])
                    table_create_index[table].append({
                        "function_id": func_id,
                        "example_id": ex_id,
                        "create_sql": sql
                    })
        return example_index, table_create_index

    def _normalize_val(self, val):
        """Helper to normalize individual values (strip, lower, handle bools/WKT)."""
        if val is None:
            return "null"
        
        s_val = str(val).strip()
        
        # Boolean normalization
        if s_val.lower() == 't': return 'true'
        if s_val.lower() == 'f': return 'false'
        
        # PostGIS WKT normalization (spaces)
        s_val = re.sub(r'\s+', ' ', s_val)
        
        return s_val.lower()

    def _compare_results(self, actual_rows, expected_data):
        """
        Compare Actual DB Result vs Expected Data.
        Returns: (status_code, comment)
        """
        # 1. Handle Empty/Null Expectation (Skip check)
        if expected_data is None:
            return "skipped", "No expected result defined in source"

        # 2. Case A: Expected is a Table (List of Dicts)
        if isinstance(expected_data, list) and len(expected_data) > 0 and isinstance(expected_data[0], dict):
            if not actual_rows:
                return "mismatch", "Expected table data, got empty result"
            
            # Normalize Actual Data
            norm_actual = []
            for row in actual_rows:
                new_row = {k.lower(): self._normalize_val(v) for k, v in row.items()}
                norm_actual.append(new_row)
            
            # Normalize Expected Data
            norm_expected = []
            for row in expected_data:
                new_row = {k.lower(): self._normalize_val(v) for k, v in row.items()}
                norm_expected.append(new_row)

            # Compare Row Counts
            if len(norm_actual) != len(norm_expected):
                return "mismatch", f"Row count diff: Exp {len(norm_expected)} vs Act {len(norm_actual)}"
            
            # Compare Rows
            for i, exp_row in enumerate(norm_expected):
                act_row = norm_actual[i]
                
                for k, v in exp_row.items():
                    if k not in act_row:
                        return "mismatch", f"Column '{k}' missing in actual result"
                    
                    if act_row[k] != v:
                        exp_num = self._try_parse_float(v)
                        act_num = self._try_parse_float(act_row[k])
                        if exp_num is not None and act_num is not None:
                            if abs(exp_num - act_num) <= self.eps_numeric:
                                continue
                        
                        if self._is_box_like(v) and self._is_box_like(act_row[k]):
                            exp_nums = self._extract_numbers(v)
                            act_nums = self._extract_numbers(act_row[k])
                            if self._all_close(exp_nums, act_nums, self.eps_box):
                                continue
                        
                        if v in act_row[k] or act_row[k] in v:
                            continue
                        
                        return "mismatch", f"Row {i+1} col '{k}' mismatch. Exp: '{v}' vs Act: '{act_row[k]}'"
            
            return "match", "Table exact match"

        # 3. Case B: Scalar
        else:
            if not actual_rows:
                act_val = "null"
            else:
                first_row = actual_rows[0]
                if not first_row:
                    act_val = "null"
                else:
                    act_val = list(first_row.values())[0]

            # Tolerance-based comparison for numeric and BOX/BOX3D outputs.
            exp_num = self._try_parse_float(expected_data)
            act_num = self._try_parse_float(act_val)
            if exp_num is not None and act_num is not None:
                if abs(exp_num - act_num) <= self.eps_numeric:
                    return "match", f"Numeric match within eps={self.eps_numeric}"
            
            exp_s = str(expected_data) if expected_data is not None else None
            act_s = str(act_val) if act_val is not None else None
            if self._is_box_like(exp_s) and self._is_box_like(act_s):
                exp_nums = self._extract_numbers(exp_s)
                act_nums = self._extract_numbers(act_s)
                if self._all_close(exp_nums, act_nums, self.eps_box):
                    return "match", f"BOX/BOX3D match within eps={self.eps_box}"

            norm_act = self._normalize_val(act_val)
            norm_exp = self._normalize_val(expected_data)

            if norm_act == norm_exp:
                return "match", "Scalar exact match"
            
            if norm_exp in norm_act:
                return "partial_match", "Expected string found within Actual result"
            
            return "mismatch", f"Exp scalar '{norm_exp}' vs Act '{norm_act}'"

    def _execute_step(self, cursor, step, context=None):
        """
        Execute a single SQL step with retry and auto-fix logic.
        [Updated] Returns detailed pg_error_code for classification.
        """
        original_sql = step['sql']
        sql, preprocess_meta = self._preprocess_sql_for_validation(original_sql)
        
        fix_meta = {}
        if preprocess_meta.get("sql_rewritten"):
            fix_meta.update(preprocess_meta)
        for attempt in range(2):
            try:
                cursor.execute(f"SAVEPOINT sp_{attempt}")
                cursor.execute(sql)
                
                # Fetch results if available
                raw_rows = None
                if cursor.description:
                    raw_rows = cursor.fetchall() # Returns List[RealDictRow]
                
                cursor.execute(f"RELEASE SAVEPOINT sp_{attempt}")
                
                return {
                    "status": "success", 
                    "actual_result_raw": [dict(row) for row in raw_rows] if raw_rows else None, 
                    "fixed": attempt > 0,
                    **fix_meta
                }

            except Exception as e:
                cursor.execute(f"ROLLBACK TO SAVEPOINT sp_{attempt}")
                error_code = e.pgcode if psycopg2 is not None and isinstance(e, psycopg2.Error) else None
                error_msg = str(e)

                # If this was the retry attempt, fail
                if attempt == 1:
                    return {
                        "status": "failed", 
                        "error": error_msg, 
                        "pg_error_code": error_code # Return code specifically
                    }

                # === Auto-Fix Logic ===
                # 1. Schema missing
                schema_match = re.search(r'schema "(.*?)" does not exist', error_msg)
                if schema_match:
                    cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {schema_match.group(1)};")
                    continue

                # 2. Table missing
                if error_code == '42P01' or "relation" in error_msg:
                    missing_table_name = self._extract_missing_table_name(error_msg)
                    if not missing_table_name:
                        missing_table_name = self._fallback_extract_table_from_sql(sql)
                    missing_info = self._pick_missing_table_info(step, missing_table_name) if missing_table_name else None
                    
                    decision = self._decide_missing_case(missing_table_name, missing_info, context)
                    
                    # Case 1: a CREATE TABLE statement is available in context.
                    if decision.get("missing_case") == "context_ddl_found":
                        ddl_sql = decision.get("ddl_sql")
                        ddl_source = decision.get("ddl_source")
                        print(f"      [Auto-Fix][context_ddl_found] Executing CREATE TABLE from {ddl_source}...")
                        try:
                            cursor.execute(ddl_sql)
                            fix_meta["missing_case"] = "context_ddl_found"
                            fix_meta["missing_case_label_cn"] = self.missing_case_labels_cn["context_ddl_found"]
                            fix_meta["ddl_source"] = ddl_source
                            
                            # If the example provides explicit data, insert it before retrying.
                            expected_data = step.get("expected_result")
                            geometry_column = ((missing_info or {}).get("table_features") or {}).get("geometry_column")
                            if (
                                not self._is_insert_like_sql(sql)
                                and isinstance(expected_data, list)
                                and expected_data
                                and isinstance(expected_data[0], dict)
                                and self._sql_mentions_table(sql, missing_table_name)
                            ):
                                inserts = self._generate_inserts_from_expected_result(
                                    missing_table_name,
                                    expected_data,
                                    geometry_column=geometry_column
                                )
                                if inserts:
                                    print(f"      [Auto-Fix][context_ddl_found] Inserting explicit data from expected_result ({len(inserts)} rows)...")
                                    for stmt in inserts:
                                        cursor.execute(stmt)
                                    fix_meta["insert_source"] = "expected_result"
                            
                            continue
                        except Exception as fix_e:
                            print(f"      [Auto-Fix][context_ddl_found] DDL execution failed: {fix_e}")
                            return {
                                "status": "failed",
                                "error": error_msg,
                                "pg_error_code": error_code,
                                "missing_table_name": missing_table_name,
                                "missing_case": "context_ddl_found",
                                "missing_case_label_cn": self.missing_case_labels_cn["context_ddl_found"],
                                "ddl_source": ddl_source,
                                "ddl_error": str(fix_e)
                            }
                    
                    # Case 2: external data must be imported. Record guidance only.
                    if decision.get("missing_case") == "external_data_import":
                        return {
                            "status": "failed",
                            "error": error_msg,
                            "pg_error_code": error_code,
                            "missing_table_name": missing_table_name,
                            "missing_case": "external_data_import",
                            "missing_case_label_cn": self.missing_case_labels_cn["external_data_import"],
                            "dataset_key": decision.get("dataset_key"),
                            "source_hint": decision.get("source_hint"),
                            "lookup_status": decision.get("lookup_status"),
                            "search_keywords": decision.get("search_keywords"),
                        }
                    
                    return {
                        "status": "failed",
                        "error": error_msg,
                        "pg_error_code": error_code,
                        "missing_table_name": missing_table_name,
                        "missing_case": "external_data_import",
                        "missing_case_label_cn": self.missing_case_labels_cn["external_data_import"],
                        "dataset_key": decision.get("dataset_key"),
                        "source_hint": decision.get("source_hint"),
                        "lookup_status": decision.get("lookup_status"),
                        "search_keywords": decision.get("search_keywords"),
                    }
                
                # If we reach here, it means we couldn't fix it or it wasn't a fixable error
                return {
                    "status": "failed", 
                    "error": error_msg, 
                    "pg_error_code": error_code
                }

    def validate_dataset(self):
        return self._validate_dataset_internal(only_targets=None)

    def validate_external_import_only(self, from_manual_review_file):
        targets = self._collect_external_import_targets(from_manual_review_file)
        return self._validate_dataset_internal(only_targets=targets)

    def _collect_external_import_targets(self, from_manual_review_file):
        if not from_manual_review_file or not os.path.exists(from_manual_review_file):
            return {}
        try:
            with open(from_manual_review_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return {}
        targets = {}
        if not isinstance(data, list):
            return targets
        for item in data:
            if not isinstance(item, dict):
                continue
            func_id = item.get("function_id")
            if not func_id:
                continue
            for issue in item.get("issues", []) or []:
                if not isinstance(issue, dict):
                    continue
                if issue.get("issue_type") != "missing_table":
                    continue
                if issue.get("missing_case") != "external_data_import":
                    continue
                ex_id = issue.get("example_id")
                if ex_id is None:
                    continue
                targets.setdefault(func_id, set()).add(ex_id)
        return targets

    def _validate_dataset_internal(self, only_targets=None):
        self.check_environment()
        
        if not os.path.exists(self.input_file):
            print(f"❌ Input file not found: {self.input_file}")
            return

        with open(self.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        validated_results = []
        manual_queue = []
        conn = self.get_db_connection()
        
        # Pre-build dependency indexes to support cross-example CREATE TABLE recovery.
        example_index, table_create_index = self._build_dependency_indexes(data)

        print(f"🚀 Starting Validation & Consistency Check for {len(data)} functions...")

        for index, entry in enumerate(data):
            func_id = entry.get('function_id', 'Unknown')
            examples = entry.get('examples', [])
            
            if not examples: continue

            if isinstance(only_targets, dict) and func_id not in only_targets:
                continue
            
            print(f"[{index+1}/{len(data)}] Testing: {func_id} ({len(examples)} examples)")
            
            # Iterate through Independent Examples
            for ex in examples:
                ex_id = ex.get('example_id')
                steps = ex.get('steps', [])
                step_logs = []
                all_success = True

                if isinstance(only_targets, dict):
                    allowed = only_targets.get(func_id) or set()
                    if ex_id not in allowed:
                        continue
                
                # Flag to track if we've hit a missing table in this example
                missing_table_blocker = False

                try:
                    # Transaction Block for ONE Example
                    with conn.cursor() as cursor:
                        for step_index, step in enumerate(steps):
                            # 1. Check if we should skip due to previous missing table
                            if missing_table_blocker:
                                step_logs.append({
                                    "step_id": step['step_id'],
                                    "status": "skipped_due_to_dependency",
                                    "error": "Skipped because a previous step failed due to missing table.",
                                    "sql": step['sql']
                                })
                                continue

                            # 2. Existing Skip logic
                            if step.get('execution_mode') == 'blocked':
                                step_logs.append({"step_id": step['step_id'], "status": "skipped_blocked", "sql": step['sql']})
                                continue
                            if step.get('execution_mode') == 'chain' and not all_success:
                                step_logs.append({"step_id": step['step_id'], "status": "skipped_chain_broken", "sql": step['sql']})
                                continue

                            # === EXECUTE ===
                            res = self._execute_step(
                                cursor,
                                step,
                                context={
                                    "function_id": func_id,
                                    "example_id": ex_id,
                                    "step_index": step_index,
                                    "steps": steps,
                                    "example_index": example_index,
                                    "table_create_index": table_create_index
                                }
                            )
                            
                            log_item = {
                                "step_id": step['step_id'],
                                "sql": step['sql'], # Include SQL in log
                                "status": res['status'],
                                "error": res.get('error'),
                                "pg_error_code": res.get('pg_error_code'),
                                "was_auto_fixed": res.get('fixed', False),
                                "missing_case": res.get("missing_case"),
                                "missing_case_label_cn": res.get("missing_case_label_cn"),
                                "ddl_source": res.get("ddl_source"),
                                "insert_source": res.get("insert_source"),
                                "dataset_key": res.get("dataset_key"),
                                "source_hint": res.get("source_hint"),
                                "lookup_status": res.get("lookup_status"),
                                "search_keywords": res.get("search_keywords"),
                            }

                            # === CONSISTENCY CHECK ===
                            if res['status'] == 'success':
                                actual_raw = res.get('actual_result_raw')
                                expected_data = step.get('expected_result')
                                
                                log_item['actual_result'] = actual_raw
                                
                                consistency, comment = self._compare_results(actual_raw, expected_data)
                                log_item['consistency_status'] = consistency
                                log_item['consistency_comment'] = comment
                            
                            else:
                                # Execution Failed
                                log_item['consistency_status'] = "execution_failed"
                                
                                # [New Logic] Identify Missing Table (42P01)
                                if res.get('pg_error_code') == '42P01':
                                    missing_table_name = res.get("missing_table_name") or self._extract_missing_table_name(res.get('error'))
                                    missing_info = self._pick_missing_table_info(step, missing_table_name) if missing_table_name else None
                                    missing_features = (missing_info or {}).get("table_features") or {}
                                    
                                    if not log_item.get("missing_case"):
                                        decision = self._decide_missing_case(missing_table_name, missing_info, {
                                            "function_id": func_id,
                                            "example_id": ex_id,
                                            "step_index": step_index,
                                            "steps": steps,
                                            "example_index": example_index,
                                            "table_create_index": table_create_index
                                        })
                                        log_item["missing_case"] = decision.get("missing_case")
                                        log_item["missing_case_label_cn"] = decision.get("missing_case_label_cn")
                                        log_item["ddl_source"] = decision.get("ddl_source")
                                        log_item["dataset_key"] = decision.get("dataset_key")
                                        log_item["source_hint"] = decision.get("source_hint")
                                        log_item["lookup_status"] = decision.get("lookup_status")
                                        log_item["search_keywords"] = decision.get("search_keywords")
                                    
                                    log_item['status'] = "missing_table" # Override status for clarity
                                    missing_table_blocker = True # Trigger blocking for next steps
                                    log_item['error'] = f"[Missing Table] {res.get('error')}"
                                    
                                    # Record missing-table type, dependency, and schema hints for debugging.
                                    log_item["missing_table_name"] = missing_table_name
                                    log_item["missing_table_type"] = (missing_info or {}).get("missing_type") or "unknown"
                                    log_item["missing_table_dep_scope"] = (missing_info or {}).get("dep_scope")
                                    log_item["missing_table_dep_example_id"] = (missing_info or {}).get("dep_example_id")
                                    log_item["missing_table_dep_function_id"] = (missing_info or {}).get("dep_function_id")
                                    log_item["missing_table_features"] = {
                                        "has_geometry": missing_features.get("has_geometry"),
                                        "geometry_column": missing_features.get("geometry_column"),
                                        "primary_key": missing_features.get("primary_key")
                                    }
                                    
                                all_success = False
                            
                            step_logs.append(log_item)

                        # Store logs back to the example object
                        ex['validation_log'] = step_logs
                        
                        # Explicitly rollback to clean up
                        conn.rollback()

                except Exception as e:
                    print(f"  ❌ Error on Example {ex_id}: {e}")
                    conn.rollback()

            # === FILTER FOR REVIEW (Full Details) ===
            # Preserve missing-table classifications while still reporting
            # mismatches and execution failures as explicit issues.
            issues_found = False
            detailed_issues_list = []
            issues_by_missing_case = {"context_ddl_found": 0, "external_data_import": 0, "unknown_todo": 0}
            # Keep failed steps inside the issues list instead of a separate summary.
            
            for ex in examples:
                for s in ex.get('validation_log', []):
                    
                    # Criteria for adding to manual review:
                    # 1. Status is 'failed' (syntax/logic error)
                    # 2. Status is 'missing_table' (crucial to report)
                    # 3. Consistency is 'mismatch'
                    # 4. Status is 'skipped_due_to_dependency' (optional, but helps show impact)
                    
                    is_issue = (
                        s.get("status") in ["failed", "missing_table"] or
                        s.get("consistency_status") == "mismatch"
                    )

                    if is_issue:
                        issues_found = True
                        
                        issue_type = s['status'] if s.get('status') in ['failed', 'missing_table'] else 'mismatch'
                        issue_obj = {
                            "example_id": ex.get('example_id'),
                            "step_id": s['step_id'],
                            "issue_type": issue_type,
                            "sql_executed": s.get('sql'),
                            "error_message": s.get('error'),
                        }
                        if s.get('pg_error_code'):
                            issue_obj['pg_error_code'] = s.get('pg_error_code')
                        
                        # Missing-table details and remediation strategy.
                        if issue_type == "missing_table":
                            issue_obj["missing_table_name"] = s.get("missing_table_name")
                            issue_obj["missing_case"] = s.get("missing_case")
                            issue_obj["missing_case_label_cn"] = s.get("missing_case_label_cn")
                            issue_obj["resolution"] = self._missing_case_resolution_cn(s.get("missing_case"))
                            if s.get("missing_case") == "context_ddl_found":
                                issue_obj["ddl_source"] = s.get("ddl_source")
                            if s.get("missing_case") == "external_data_import":
                                issue_obj["dataset_key"] = s.get("dataset_key")
                                issue_obj["source_hint"] = s.get("source_hint")
                                issue_obj["lookup_status"] = s.get("lookup_status")
                                issue_obj["search_keywords"] = s.get("search_keywords")
                            mc = issue_obj.get("missing_case")
                            if mc in issues_by_missing_case:
                                issues_by_missing_case[mc] += 1
                        
                        # Mismatch details.
                        if issue_type == "mismatch":
                            try:
                                exp = ex.get('steps')[int(s['step_id'])-1].get('expected_result')
                            except Exception:
                                exp = None
                            issue_obj["expected_result"] = exp
                            issue_obj["actual_result"] = s.get("actual_result")
                            issue_obj["mismatch_details"] = s.get("consistency_comment")
                        
                        # Attach root-cause classification to failed execution issues.
                        if issue_type == "failed":
                            cls = self._classify_execution_error(s.get("pg_error_code"), s.get("error"), s.get("sql"))
                            issue_obj["root_cause"] = cls.get("root_cause")
                            issue_obj["root_cause_cn"] = cls.get("root_cause_cn")
                            issue_obj["suggestion"] = cls.get("suggestion")
                        
                        detailed_issues_list.append(issue_obj)
            
            if issues_found:
                manual_queue.append({
                    "function_id": func_id,
                    "source_file": entry.get('source_file'),
                    "issues": detailed_issues_list,
                    "issues_by_missing_case": issues_by_missing_case
                })
                
            validated_results.append(entry)

        conn.close()

        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(validated_results, f, indent=2, ensure_ascii=False, default=str)
        
        with open(self.manual_review_file, 'w', encoding='utf-8') as f:
            json.dump(manual_queue, f, indent=2, ensure_ascii=False, default=str)
        
        self._write_external_import_tables_file(manual_queue)
        
        # Calculate stats for summary
        missing_tbl_count = sum(1 for item in manual_queue for i in item.get("issues", []) if i.get("issue_type") == "missing_table")
        mismatch_count = sum(1 for item in manual_queue for i in item.get("issues", []) if i.get("issue_type") == "mismatch")
        exec_fail_count = sum(1 for item in manual_queue for i in item.get("issues", []) if i.get("issue_type") == "failed")
        
        missing_tbl_by_case = {"context_ddl_found": 0, "external_data_import": 0, "unknown_todo": 0}
        for item in manual_queue:
            by_case = item.get("issues_by_missing_case") or {}
            for mc in missing_tbl_by_case:
                missing_tbl_by_case[mc] += int(by_case.get(mc) or 0)

        print(f"\n✅ Validation Finished!")
        print(f"   Validated File: {self.output_file}")
        print(f"   Review Queue: {self.manual_review_file}")
        print(f"   ----------------------------------------")
        print(f"   📊 Summary of Issues:")
        print(f"      - 📉 Missing Tables: {missing_tbl_count}")
        print(f"        - by case: context_ddl_found={missing_tbl_by_case['context_ddl_found']}, external_data_import={missing_tbl_by_case['external_data_import']}, unknown_todo={missing_tbl_by_case['unknown_todo']}")
        print(f"      - ❌ Data Mismatches: {mismatch_count}")
        print(f"      - 💥 Execution Failures: {exec_fail_count}")

    def _write_external_import_tables_file(self, manual_queue):
        out_dir = os.path.dirname(self.manual_review_file) or "."
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "external_import_tables.json")
        
        occurrences = {}
        for item in manual_queue or []:
            func_id = (item or {}).get("function_id")
            source_file = (item or {}).get("source_file")
            for issue in (item or {}).get("issues", []) or []:
                if not isinstance(issue, dict):
                    continue
                if issue.get("issue_type") != "missing_table":
                    continue
                if issue.get("missing_case") != "external_data_import":
                    continue
                name = issue.get("missing_table_name")
                if not name:
                    continue
                ex_id = issue.get("example_id")
                step_id = issue.get("step_id")
                occurrences.setdefault(name, set()).add((func_id, ex_id, step_id, source_file))
        
        tables = []
        for name in sorted(occurrences.keys()):
            locs = sorted(list(occurrences.get(name) or set()), key=lambda x: (str(x[0]), str(x[1]), str(x[2]), str(x[3])))
            tables.append({
                "table": name,
                "count": len(locs),
                "locations": [
                    {
                        "function_id": func_id,
                        "example_id": ex_id,
                        "step_id": step_id,
                        "source_file": source_file,
                    }
                    for func_id, ex_id, step_id, source_file in locs
                ],
            })
        
        payload = {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "missing_case": "external_data_import",
            "tables": tables,
        }
        
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=str)

if __name__ == "__main__":
    # === Database Configuration ===
    DB_CONFIG = {
        "dbname": "postgis_test_db",     
        "user": "postgres",       
        "password": "1234",       
        "host": "localhost",
        "port": 5432
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["full", "external_only", "import_external"], default="full")
    parser.add_argument("--from-review", dest="from_review", default=None)
    parser.add_argument("--input", default="extract_result/postgis_extracted7.json")
    parser.add_argument("--output", default="validation_result/postgis_validated7.json")
    parser.add_argument("--review", dest="review", default="manual_review/manual_review7.json")
    parser.add_argument("--external-sources", dest="external_sources", default=str(DEFAULT_EXTERNAL_TABLE_SOURCES_FILE))
    parser.add_argument("--if-exists", dest="if_exists", choices=["fail", "replace", "append"], default="append")
    parser.add_argument("--schema", dest="schema", default=None)
    args = parser.parse_args()

    validator = PostGISValidator(
        db_config=DB_CONFIG,
        input_file=args.input,
        output_file=args.output,
        manual_review_file=args.review,
        external_table_sources_file=args.external_sources,
    )

    if args.mode == "import_external":
        if not args.from_review:
            raise SystemExit("--mode import_external requires --from-review <manual_review.json>")
        validator.import_external_tables_via_shp2db(args.from_review, if_exists=args.if_exists, schema=args.schema)
    elif args.mode == "external_only":
        if not args.from_review:
            raise SystemExit("--mode external_only requires --from-review <manual_review.json>")
        validator.validate_external_import_only(args.from_review)
    else:
        validator.validate_dataset()
