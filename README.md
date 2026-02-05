# log-analyze-llm-benchmark
ログ解析に適したLLMモデルを比較評価するためのプロジェクト。

# ファイル説明
## /source/log/app.log
LLMモデル比較評価用のログファイル

## /source/testcase/testcase.csv
LLMがテストケースを見て仕様に沿ったアプリ挙動になっているかを判断する。

## /source/testcase/testcase_table.csv
人間用のテストケース表。
LLMが出した結果を見て実際に仕様に沿って判断できているかを確認するためのテストケース表。

## /source/generate_dataset.py
テストケース作成用スクリプト

## /source/log_analyze_llm_benchmark.py
ベンチマークスクリプト