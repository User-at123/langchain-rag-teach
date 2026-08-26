"""sql_db.py 的 SQLiteDb 测试：导入 / 查询 / 三道校验 / schema 文本。

覆盖 V9.0 Text-to-SQL 的核心安全防线（单条 SELECT / 只读白名单 / 中文列名校验）。
零 LLM、零外部依赖，只依赖 openpyxl 生成临时 xlsx。
"""

import pytest

from sql_db import SQLiteDb


@pytest.fixture
def db(tmp_path):
    """临时目录建库 + 一个简单 xlsx（3 行数据），返回 SQLiteDb。"""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = "customers"
    ws.append(["客户名称", "行业", "省份"])
    ws.append(["华信银行", "金融", "北京"])
    ws.append(["蓝天航空", "航空", "上海"])
    ws.append(["绿野农业", "农业", "山东"])
    wb.save(docs_dir / "客户.xlsx")

    inst = SQLiteDb(db_path=str(tmp_path / "test.db"), docs_dir=str(docs_dir))
    yield inst
    inst.conn.close()  # 释放句柄，避免 Windows 下临时目录清理失败


def test_import_xlsx_creates_table(db):
    assert db._tables() == ["customers"]
    assert db._columns("customers") == ["客户名称", "行业", "省份"]


def test_simple_select(db):
    cols, rows = db.query("SELECT 行业 FROM customers")
    assert cols == ["行业"]
    assert len(rows) == 3
    assert {r["行业"] for r in rows} == {"金融", "航空", "农业"}


def test_aggregate_count(db):
    cols, rows = db.query("SELECT COUNT(*) AS 总数 FROM customers")
    assert rows[0]["总数"] == 3


def test_reject_multiple_statements(db):
    with pytest.raises(ValueError, match="只允许单条"):
        db.query("SELECT * FROM customers; DROP TABLE customers")


def test_reject_non_select(db):
    with pytest.raises(ValueError, match="只允许 SELECT"):
        db.query("DELETE FROM customers")


def test_reject_fake_chinese_column(db):
    """防 LLM 捏造列名：中文标识符必须是真实表/列名。"""
    with pytest.raises(ValueError, match="标识符"):
        db.query("SELECT 不存在的列 FROM customers")


def test_schema_text(db):
    text = db.schema_text()
    assert "customers" in text
    assert "客户名称" in text
    assert "样例" in text
