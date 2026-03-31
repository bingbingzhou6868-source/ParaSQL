"""
快速验证脚本：不需要训练，直接用基础模型测试 pipeline 是否跑通。

用法：
  python scripts/quick_test.py --model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from inference.engine import HFParaSQLEngine
from inference.sql_assembler import parse_model_output
from eval.metrics import normalize_sql
from data.preprocess_spider import parse_sql_clauses


DEMO_SCHEMA = """
Table employees: (id INTEGER [PK], name TEXT, salary REAL, dept_id INTEGER)
Table departments: (id INTEGER [PK], name TEXT, location TEXT)
Foreign Keys:
  employees.dept_id = departments.id
"""

DEMO_QUESTION = "Find the names of employees in the Engineering department with salary above 80000, ordered by salary descending."

DEMO_GOLD_SQL = """
SELECT e.name
FROM employees e
JOIN departments d ON e.dept_id = d.id
WHERE d.name = 'Engineering' AND e.salary > 80000
ORDER BY e.salary DESC
"""


def test_sql_parser():
    """测试 SQL 子句解析"""
    print("=== 测试 SQL 子句解析 ===")
    test_sqls = [
        "SELECT name FROM employees WHERE salary > 50000",
        "SELECT e.name, d.name FROM employees e JOIN departments d ON e.dept_id = d.id WHERE d.name = 'Engineering' ORDER BY e.salary DESC LIMIT 10",
        "SELECT dept_id, COUNT(*) FROM employees GROUP BY dept_id HAVING COUNT(*) > 5",
    ]
    for sql in test_sqls:
        clauses = parse_sql_clauses(sql)
        print(f"\nSQL: {sql}")
        for k, v in clauses.items():
            if v:
                print(f"  {k:10s}: {v}")
    print("\n✓ SQL 解析测试通过\n")


def test_assembler():
    """测试 SQL 汇总器"""
    print("=== 测试 SQL 汇总器 ===")
    mock_output = """
<clause_select> e.name </clause_select>
<clause_from> employees e </clause_from>
<clause_join> JOIN departments d ON e.dept_id = d.id </clause_join>
<clause_where> d.name = 'Engineering' AND e.salary > 80000 </clause_where>
<clause_group_by>  </clause_group_by>
<clause_having>  </clause_having>
<clause_order_by> e.salary DESC </clause_order_by>
<clause_limit>  </clause_limit>
<summary>
SELECT e.name FROM employees e
JOIN departments d ON e.dept_id = d.id
WHERE d.name = 'Engineering' AND e.salary > 80000
ORDER BY e.salary DESC
</summary>
"""
    sql, clauses = parse_model_output(mock_output)
    print(f"解析出的 SQL:\n{sql}")
    print(f"\n各子句:")
    for k, v in clauses.items():
        if v:
            print(f"  {k:10s}: {v}")
    print("\n✓ 汇总器测试通过\n")


def test_model_inference(model_path: str):
    """测试模型推理"""
    print(f"=== 测试模型推理 ({model_path}) ===")
    try:
        engine = HFParaSQLEngine(model_path, max_new_tokens=512, temperature=0.0)
        pred_sql = engine.generate(DEMO_SCHEMA, DEMO_QUESTION)
        print(f"问题: {DEMO_QUESTION}")
        print(f"预测 SQL:\n{pred_sql}")
        print(f"参考 SQL:\n{DEMO_GOLD_SQL.strip()}")
        print("\n✓ 模型推理测试通过\n")
    except Exception as e:
        print(f"✗ 模型推理失败: {e}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=None, help="模型路径，不指定则跳过模型推理测试")
    args = parser.parse_args()

    test_sql_parser()
    test_assembler()

    if args.model_path:
        test_model_inference(args.model_path)
    else:
        print("未指定 --model_path，跳过模型推理测试")
        print("运行: python scripts/quick_test.py --model_path <your_model_path>")


if __name__ == "__main__":
    main()
