"""第1回: 素のLLMに会話履歴を保って質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする

ツールは使わず、同じ実行中の会話履歴を引き継ぐ。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse

from google import genai
from google.genai import errors
from google.genai import types

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def ask(question: str, history: list[types.ContentDict] | None = None) -> str:
    client = genai.Client(api_key=get_api_key())
    user_message: types.ContentDict = {
        "role": "user",
        "parts": [{"text": question}],
    }
    contents = [*(history or []), user_message]
    response = client.models.generate_content(model=MODEL, contents=contents)

    usage = response.usage_metadata
    input_tokens = usage.prompt_token_count if usage else "不明"
    output_tokens = usage.candidates_token_count if usage else "不明"
    print(f"[トークン] 入力: {input_tokens}, 出力: {output_tokens}")

    if response.text is None:
        raise RuntimeError("モデルからテキストの回答が返りませんでした。")

    if history is not None:
        history.extend(
            [
                user_message,
                {"role": "model", "parts": [{"text": response.text}]},
            ]
        )

    return response.text


def main() -> None:
    parser = argparse.ArgumentParser(description="会話履歴を保ってLLMに質問する")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    try:
        history: list[types.ContentDict] = []
        question = args.question
        while True:
            print(f"[質問] {question}\n")
            print(ask(question, history))
            try:
                question = input("\n続けて質問（終了するには exit または Ctrl+D）> ").strip()
            except EOFError:
                print()
                break
            if question.lower() in {"exit", "quit"} or question == "終了":
                break
    except errors.APIError as e:
        print(api_error_message(e.code, e.message or ""))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
