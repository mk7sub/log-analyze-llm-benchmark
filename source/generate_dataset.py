import os
import csv
from typing import List, Tuple

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "log")
TESTCASE_DIR = os.path.join(BASE_DIR, "testcase")
APP_LOG_PATH = os.path.join(LOG_DIR, "app.log")
TESTCASE_CSV_PATH = os.path.join(TESTCASE_DIR, "testcases.csv")
TESTCASE_TABLE_CSV_PATH = os.path.join(TESTCASE_DIR, "testcases_table.csv")

# システム仕様（概要）: テストケースCSVをそのまま仕様書として参照できるようにするため
SYSTEM_SPEC = """対象システムはフードコートの呼び出し・順番待ちで使用される画面表示系アプリケーションである。

【ステータス】
- 受付済み
- 仕掛中
- 完了

【処理の流れ】
1. 受付・採番
2. 表示（お待ち番号）
    - ステータス: 受付済み
    - ステータス: 仕掛中
3. 調理完了
4. 表示（呼び出し番号）
    - ステータス: 完了
5. お渡し完了

【制約条件】
- ステータスは必ず「受付済み」→「仕掛中」→「完了」の順番で遷移する必要がある。
- ログに記録されている内容と、実際の画面表示の動きは一致している前提とする。
- エラーや例外（ERROR, Exception など）がログに出力されている場合、そのテストは正常動作（PASS）とはみなさない。
""".strip()


def ensure_dirs() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(TESTCASE_DIR, exist_ok=True)


def make_normal_logs(target_lines: int, case_prefix: str) -> List[str]:
    lines: List[str] = []
    order_id = 1
    ts_base = "2026-02-05 10:00:00"

    while len(lines) < target_lines:
        lines.append(f"{ts_base} INFO order={case_prefix}-{order_id} status=受付済み message=受付完了")
        if len(lines) >= target_lines:
            break
        lines.append(f"{ts_base} INFO order={case_prefix}-{order_id} status=仕掛中 message=調理開始")
        if len(lines) >= target_lines:
            break
        lines.append(f"{ts_base} INFO order={case_prefix}-{order_id} status=完了 message=調理完了・呼び出し")
        if len(lines) >= target_lines:
            break
        order_id += 1

    return lines[:target_lines]


def make_missing_logs(target_lines: int, case_prefix: str) -> List[str]:
    """一部の注文でステータスが欠損しているログ。"""
    lines: List[str] = []
    order_id = 1
    ts_base = "2026-02-05 11:00:00"

    while len(lines) < target_lines:
        # 正常な注文
        lines.append(f"{ts_base} INFO order={case_prefix}-N{order_id} status=受付済み message=受付完了")
        if len(lines) >= target_lines:
            break
        lines.append(f"{ts_base} INFO order={case_prefix}-N{order_id} status=仕掛中 message=調理開始")
        if len(lines) >= target_lines:
            break
        lines.append(f"{ts_base} INFO order={case_prefix}-N{order_id} status=完了 message=調理完了・呼び出し")
        if len(lines) >= target_lines:
            break

        # ステータス欠損の注文（仕掛中を飛ばす）
        order_id += 1
        lines.append(f"{ts_base} INFO order={case_prefix}-M{order_id} status=受付済み message=受付完了")
        if len(lines) >= target_lines:
            break
        # 本来は仕掛中が必要だが欠損して、いきなり完了になる
        lines.append(f"{ts_base} INFO order={case_prefix}-M{order_id} status=完了 message=調理完了・呼び出し (仕掛中欠損)")
        if len(lines) >= target_lines:
            break

        order_id += 1

    return lines[:target_lines]


def make_reorder_logs(target_lines: int, case_prefix: str) -> List[str]:
    """一部の注文でステータスの順序が逆転しているログ。"""
    lines: List[str] = []
    order_id = 1
    ts_base = "2026-02-05 12:00:00"

    while len(lines) < target_lines:
        # 正常な注文
        lines.append(f"{ts_base} INFO order={case_prefix}-N{order_id} status=受付済み message=受付完了")
        if len(lines) >= target_lines:
            break
        lines.append(f"{ts_base} INFO order={case_prefix}-N{order_id} status=仕掛中 message=調理開始")
        if len(lines) >= target_lines:
            break
        lines.append(f"{ts_base} INFO order={case_prefix}-N{order_id} status=完了 message=調理完了・呼び出し")
        if len(lines) >= target_lines:
            break

        # 順序逆転の注文（仕掛中 -> 受付済み -> 完了）
        order_id += 1
        lines.append(f"{ts_base} INFO order={case_prefix}-R{order_id} status=仕掛中 message=調理開始 (順序逆転)")
        if len(lines) >= target_lines:
            break
        lines.append(f"{ts_base} INFO order={case_prefix}-R{order_id} status=受付済み message=受付完了 (順序逆転)")
        if len(lines) >= target_lines:
            break
        lines.append(f"{ts_base} INFO order={case_prefix}-R{order_id} status=完了 message=調理完了・呼び出し")
        if len(lines) >= target_lines:
            break

        order_id += 1

    return lines[:target_lines]


def make_extra_logs(target_lines: int, case_prefix: str) -> List[str]:
    """仕様外のステータスやERRORが混入しているログ。"""
    lines: List[str] = []
    order_id = 1
    ts_base = "2026-02-05 13:00:00"

    while len(lines) < target_lines:
        # 正常な注文
        lines.append(f"{ts_base} INFO order={case_prefix}-N{order_id} status=受付済み message=受付完了")
        if len(lines) >= target_lines:
            break
        lines.append(f"{ts_base} INFO order={case_prefix}-N{order_id} status=仕掛中 message=調理開始")
        if len(lines) >= target_lines:
            break
        lines.append(f"{ts_base} INFO order={case_prefix}-N{order_id} status=完了 message=調理完了・呼び出し")
        if len(lines) >= target_lines:
            break

        # 仕様外ステータス
        order_id += 1
        lines.append(f"{ts_base} WARN order={case_prefix}-X{order_id} status=キャンセル message=ユーザーキャンセル (仕様外ステータス)")
        if len(lines) >= target_lines:
            break
        # ERROR ログ
        lines.append(f"{ts_base} ERROR order={case_prefix}-X{order_id} status=仕掛中 message=キッチン端末との通信エラー")
        if len(lines) >= target_lines:
            break

        order_id += 1

    return lines[:target_lines]


def build_all_cases() -> Tuple[List[str], List[dict]]:
    """全12ケース分のログ行とテストケースメタ情報を構成する。"""
    all_log_lines: List[str] = []
    testcases: List[dict] = []

    specs = [
        ("normal_small", "normal", "small", 10, make_normal_logs),
        ("normal_medium", "normal", "medium", 50, make_normal_logs),
        ("normal_large", "normal", "large", 200, make_normal_logs),
        ("missing_small", "missing", "small", 10, make_missing_logs),
        ("missing_medium", "missing", "medium", 50, make_missing_logs),
        ("missing_large", "missing", "large", 200, make_missing_logs),
        ("reorder_small", "reorder", "small", 10, make_reorder_logs),
        ("reorder_medium", "reorder", "medium", 50, make_reorder_logs),
        ("reorder_large", "reorder", "large", 200, make_reorder_logs),
        ("extra_small", "extra", "small", 10, make_extra_logs),
        ("extra_medium", "extra", "medium", 50, make_extra_logs),
        ("extra_large", "extra", "large", 200, make_extra_logs),
    ]

    case_id = 1
    current_line = 1

    for name, kind, size, length, maker in specs:
        lines = maker(length, name)
        start_line = current_line
        end_line = current_line + len(lines) - 1

        all_log_lines.extend(lines)
        current_line = end_line + 1

        expected_label = "PASS" if kind == "normal" else "FAIL"

        testcases.append(
            {
                "id": case_id,
                "name": name,
                "kind": kind,
                "size": size,
                "expected_label": expected_label,
                "start_line": start_line,
                "end_line": end_line,
                "description": f"{kind} / {size} / {length} lines",
                "system_spec": SYSTEM_SPEC,
            }
        )

        case_id += 1

    return all_log_lines, testcases


def write_files() -> None:
    ensure_dirs()

    log_lines, testcases = build_all_cases()

    # app.log を書き出し
    with open(APP_LOG_PATH, "w", encoding="utf-8") as f_log:
        for line in log_lines:
            f_log.write(line + "\n")

    # testcases.csv を書き出し（LLMベンチマーク用の機械可読フォーマット）
    fieldnames = [
        "id",
        "name",
        "kind",
        "size",
        "expected_label",
        "start_line",
        "end_line",
        "description",
        "system_spec",
    ]
    with open(TESTCASE_CSV_PATH, "w", newline="", encoding="utf-8") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        writer.writeheader()
        for row in testcases:
            writer.writerow(row)

    # テストケース表 (人間向け仕様書) を書き出し
    table_fieldnames = [
        "テストケースID",
        "テスト項目",
        "パターン名",
        "ログ種別",
        "ログサイズ区分",
        "期待判定",
        "対象ログ開始行",
        "対象ログ終了行",
        "前提条件",
        "確認手順",
        "期待結果",
        "判定",
        "実施者",
        "備考",
    ]

    def kind_to_label(kind: str) -> str:
        if kind == "normal":
            return "正常系ログ"
        if kind == "missing":
            return "異常系ログ（ステップ欠損）"
        if kind == "reorder":
            return "異常系ログ（順序逆転）"
        if kind == "extra":
            return "異常系ログ（仕様外・ERROR混入）"
        return kind

    def size_to_label(size: str, start_line: int, end_line: int) -> str:
        length = end_line - start_line + 1
        if size == "small":
            prefix = "小"
        elif size == "medium":
            prefix = "中"
        elif size == "large":
            prefix = "大"
        else:
            prefix = size
        return f"{prefix}（{length}行）"

    with open(TESTCASE_TABLE_CSV_PATH, "w", newline="", encoding="utf-8") as f_table:
        writer = csv.DictWriter(f_table, fieldnames=table_fieldnames)
        writer.writeheader()

        for row in testcases:
            kind_label = kind_to_label(str(row["kind"]))
            size_label = size_to_label(str(row["size"]), int(row["start_line"]), int(row["end_line"]))

            test_item = f"{kind_label}（{size_label}）"

            precondition = (
                "システム仕様（system_spec）が適用されており、"
                "log/app.log の該当行範囲に対象テストケースのログが記録されていること。"
            )

            steps = (
                "1) log/app.log の当該テストケース範囲（start_line～end_line 行）を抽出する。"
                " 2) 仕様とともにLLMへ入力し、PASS/FAIL 判定を取得する。"
                " 3) LLMの判定結果と期待ラベル（expected_label）を比較する。"
            )

            if row["expected_label"] == "PASS":
                expected = (
                    "ログが仕様どおりに動作しており"
                    "（受付済み→仕掛中→完了の順で遷移し、エラーや例外が出力されていない）、"
                    "LLM が PASS と判定すること。"
                )
            else:
                if row["kind"] == "missing":
                    violation = "ステップ欠損（例: 仕掛中を経由せずに完了になる）"
                elif row["kind"] == "reorder":
                    violation = "ステータス順序の逆転（受付済みと仕掛中の順序入れ替わりなど）"
                elif row["kind"] == "extra":
                    violation = "仕様外ステータスやERRORログの混入"
                else:
                    violation = "仕様違反"

                expected = (
                    f"ログが仕様に違反している（{violation}）ため、"
                    "LLM が FAIL と判定すること。"
                )

            # 備考は空欄（テストパターン側に必要情報を集約）
            remarks = ""

            writer.writerow(
                {
                    "テストケースID": row["id"],
                    "テスト項目": test_item,
                    "パターン名": row["name"],
                    "ログ種別": row["kind"],
                    "ログサイズ区分": row["size"],
                    "期待判定": row["expected_label"],
                    "対象ログ開始行": row["start_line"],
                    "対象ログ終了行": row["end_line"],
                    "前提条件": precondition,
                    "確認手順": steps,
                    "期待結果": expected,
                    "判定": "保留",  # 実行前は保留を想定
                    "実施者": "",
                    "備考": remarks,
                }
            )

    print(f"Generated log: {APP_LOG_PATH}")
    print(f"Generated testcases: {TESTCASE_CSV_PATH}")
    print(f"Generated testcase table: {TESTCASE_TABLE_CSV_PATH}")


if __name__ == "__main__":
    write_files()
