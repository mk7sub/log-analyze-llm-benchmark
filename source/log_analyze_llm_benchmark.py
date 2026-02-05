import ollama
import time
import pandas as pd
import json
from typing import Tuple, Optional, List, Dict

# 比較対象モデル
LLM_MODELS = [
    "phi-4-mini",
    "llama3.2-8b",
    "gpt-oss-20b",
    "gemma:3b",
]


def ensure_models_available(models: List[str]) -> None:
    """Ollamaに対象モデルがなければ事前にpullしておく。"""
    try:
        existing = {m["name"] for m in ollama.list().get("models", [])}
    except Exception as e:
        print(f"[MODEL] モデル一覧取得に失敗しました: {e}")
        existing = set()

    for model in models:
        if model in existing:
            print(f"[MODEL] {model} は既に利用可能です")
            continue
        print(f"[MODEL] {model} が見つからないため、pull を実行します...")
        try:
            # ストリーミング進捗をそのまま標準出力に流す
            for progress in ollama.pull(model=model, stream=True):
                status = progress.get("status")
                percent = progress.get("completed")
                total = progress.get("total")
                if status:
                    if percent is not None and total:
                        print(f"  {status}: {percent}/{total}")
                    else:
                        print(f"  {status}")
            print(f"[MODEL] {model} のpullが完了しました")
        except Exception as e:
            print(f"[MODEL] {model} のpullに失敗しました: {e}")


def ollama_generate(model: str, prompt: str) -> Tuple[str, Optional[int], Optional[int]]:
    """ollama-pythonでプロンプトを送信し、応答とeval情報を返す。

    戻り値:
        response_text: 生成テキスト
        eval_count: 評価トークン数（Noneの可能性あり）
        eval_duration: 評価時間[ns]（Noneの可能性あり）
    """
    response = ollama.generate(model=model, prompt=prompt, options={"temperature": 0})
    response_text = response.get("response", "")
    eval_count = response.get("eval_count")
    eval_duration = response.get("eval_duration")
    return response_text, eval_count, eval_duration

def ollama_unload(model: str):
    """ollama-pythonでモデルをアンロード"""
    ollama.unload(model=model)

def warmup_model(model: str):
    """モデルのウォーミングアップ（初回ロード）"""
    print(f"[WARMUP] {model} ...")
    try:
        ollama_generate(model, "Hello")
        print(f"[WARMUP] {model} done.")
    except Exception as e:
        print(f"[WARMUP] {model} failed: {e}")

def cooldown_model(model: str):
    """モデルのクールダウン（アンロード）"""
    print(f"[COOLDOWN] Unloading {model} ...")
    try:
        ollama_unload(model)
        print(f"[COOLDOWN] {model} unloaded.")
    except Exception as e:
        print(f"[COOLDOWN] {model} unload failed: {e}")


def load_test_cases_from_csv(csv_path: str, log_path: str) -> List[Dict]:
    """テストケースCSVと単一ログファイルからケース一覧を構成する。

    ログファイル: ./log/app.log
    テストケース: ./testcase/testcases.csv

    CSV想定カラム:
        id:           ケースID (int or str)
        start_line:   ログ開始行番号(1始まり)
        end_line:     ログ終了行番号(1始まり, inclusive)
        expected_label: "PASS" or "FAIL"
        kind:         正常/異常の種別 (normal/missing/order/extra など、任意)
        size:         small/middle/large などのサイズ区分
        description:  任意の説明
    """

    # ログファイルを行単位で読み込み
    with open(log_path, encoding="utf-8") as f:
        log_lines = f.read().splitlines()

    # テストケースCSV読み込み
    df = pd.read_csv(csv_path)

    required_cols = {"id", "start_line", "end_line", "expected_label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"testcases.csv に必須カラムが不足しています: {missing}")

    cases: List[Dict] = []
    for _, row in df.iterrows():
        start = int(row["start_line"])
        end = int(row["end_line"])
        # 行番号は1始まりの前提
        snippet_lines = log_lines[start - 1 : end]
        log_text = "\n".join(snippet_lines)

        case = {
            "id": row["id"],
            "log": log_text,
            "expected_label": str(row["expected_label"]).upper(),
            "kind": row.get("kind"),
            "size": row.get("size"),
            "description": row.get("description"),
            "start_line": start,
            "end_line": end,
        }
        cases.append(case)

    return cases


def build_prompt(log_text: str) -> str:
    """システム仕様と評価観点を含んだプロンプトを生成する。"""
    system_description = """
対象システムはフードコートの呼び出し・順番待ちで使用される画面表示系アプリケーションです。
ログと画面表示の動きは常に一致している前提とします。

【ステータス】
- 受付済み
- 仕掛中
- 完了

【処理の流れ】
1. 受付・採番
2. 表示（お待ち番号）: ステータスが「受付済み」「仕掛中」の一覧を表示
3. 調理完了
4. 表示（呼び出し番号）: ステータスが「完了」の一覧を表示
5. お渡し完了

【重要な制約】
- ステータスは必ず「受付済み」→「仕掛中」→「完了」の順番で遷移しなければならない。
- 順番が逆行したり、飛び越えたり、ステップを飛ばしている場合は不正な遷移とみなす。
- ログにエラーや例外が出力されている場合は、テスト結果をPASSと判定してはいけない。
""".strip()

    instruction = """
以下は1件のテストケースに対応するアプリケーションログです。
このログだけを根拠に、テストが期待通りに動作しているかどうかを判定してください。

特に次の点を厳密に確認してください:
1. ログにエラーや例外（"ERROR", "Exception", スタックトレースなど）が含まれていないか。
   → 含まれている場合は必ずFAILと判定すること。
2. ステータスの遷移順が「受付済み」→「仕掛中」→「完了」の順番を守っているか。
   → 順番の入れ替わり、逆行、ステータスの飛び越え・欠落があればFAILと判定すること。

出力フォーマット:
1行目に必ず次のどちらかのみを出力してください:
- "結果: PASS"  (期待通りに動作している場合)
- "結果: FAIL"  (上記のどちらかに違反している場合)

2行目以降に、日本語で簡潔に理由を説明してください。
""".strip()

    return f"{system_description}\n\n{instruction}\n\n--- ログ開始 ---\n{log_text}\n--- ログ終了 ---"


def parse_label(response_text: str) -> str:
    """LLMの応答からPASS/FAILラベルを抽出する。未知の場合は"UNKNOWN"。"""
    first_line = response_text.strip().splitlines()[0] if response_text.strip() else ""
    first_line = first_line.upper()
    if "PASS" in first_line:
        return "PASS"
    if "FAIL" in first_line:
        return "FAIL"
    return "UNKNOWN"


def evaluate_model(model: str, test_cases: List[Dict]) -> List[Dict]:
    """本番計測: モデルでログ解析を実施し、t/sと正答有無を返す。"""
    print(f"[EVAL] {model} ...")
    results: List[Dict] = []

    for case in test_cases:
        case_id = case.get("id")
        log_text = case.get("log", "")
        expected_label = (case.get("expected_label") or "").upper()  # "PASS" or "FAIL"

        prompt = build_prompt(log_text)
        response_text, eval_count, eval_duration = ollama_generate(model, prompt)

        predicted_label = parse_label(response_text)

        # reasoning accuracy: 期待ラベルと一致しているか
        correct = int(predicted_label == expected_label) if expected_label in {"PASS", "FAIL"} else None

        # t/s: eval_count / (eval_duration[ns] / 1e9)
        if eval_count is not None and eval_duration not in (None, 0):
            tps = eval_count / (eval_duration / 1_000_000_000)
        else:
            tps = None

        results.append(
            {
                "Model": model,
                "CaseID": case_id,
                "ExpectedLabel": expected_label,
                "PredictedLabel": predicted_label,
                "Correct": correct,
                "EvalCount": eval_count,
                "EvalDurationNs": eval_duration,
                "TPS": tps,
                "LogLength": len(log_text or ""),
                "RawResponse": response_text,
            }
        )

    print(f"[EVAL] {model} done.")
    return results


def main(
    test_case_csv: str = "./testcase/testcases.csv",
    log_path: str = "./log/app.log",
) -> pd.DataFrame:
    """全モデルについてベンチマークを実行し、結果DataFrameを返す。

    - ① 生成速度 (Tokens/sec): EvalCount / EvalDuration から算出
    - ② 思考の正確性: expected_label との一致率
    - ③ コンテキスト限界: LogLength と Correct / TPS から後で分析可能

    テストケースはCSV、ログは単一ファイルから取得する。
    """
    # ベンチマーク開始前に対象モデルが利用可能か確認し、足りなければpullする
    ensure_models_available(LLM_MODELS)

    test_cases = load_test_cases_from_csv(test_case_csv, log_path)

    all_results: List[Dict] = []

    for i, model in enumerate(LLM_MODELS):
        warmup_model(model)
        model_results = evaluate_model(model, test_cases)
        all_results.extend(model_results)
        cooldown_model(model)
        # モデル切り替え時に少し待つ
        if i < len(LLM_MODELS) - 1:
            time.sleep(3)

    df = pd.DataFrame(all_results)
    return df


if __name__ == "__main__":
    df = main()

    # サマリ表示: モデルごとの正答率と平均TPS
    if not df.empty:
        summary = (
            df.groupby("Model")
            .agg(
                Accuracy=("Correct", lambda x: float(pd.Series(x).mean()) if len(x) > 0 else None),
                AvgTPS=("TPS", lambda x: float(pd.Series(x).mean()) if len(x) > 0 else None),
            )
        )
        print("\n=== Summary (per model) ===")
        print(summary)

    # CSV保存
    df.to_csv("ollama_benchmark_results.csv", index=False)
    print("\nResults saved to 'ollama_benchmark_results.csv'")