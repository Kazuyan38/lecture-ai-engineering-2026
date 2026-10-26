from app import ask


def test_ask_sends_and_updates_conversation_history(monkeypatch, capsys):
    history = [
        {"role": "user", "parts": [{"text": "最初の質問"}]},
        {"role": "model", "parts": [{"text": "最初の回答"}]},
    ]
    sent_contents = []

    class FakeModels:
        def generate_content(self, *, model, contents):
            sent_contents.extend(contents)
            return type("Response", (), {"text": "続きの回答", "usage_metadata": None})()

    class FakeClient:
        models = FakeModels()

        def __init__(self, *, api_key):
            assert api_key == "test-key"

    monkeypatch.setattr(ask, "get_api_key", lambda: "test-key")
    monkeypatch.setattr(ask.genai, "Client", FakeClient)

    result = ask.ask("続きの質問", history)

    assert result == "続きの回答"
    assert [message["role"] for message in sent_contents] == [
        "user",
        "model",
        "user",
    ]
    assert sent_contents[-1]["parts"][0]["text"] == "続きの質問"
    assert [message["role"] for message in history] == [
        "user",
        "model",
        "user",
        "model",
    ]
    assert history[-1]["parts"][0]["text"] == "続きの回答"
    assert "[トークン]" in capsys.readouterr().out


def test_main_accepts_follow_up_questions_until_exit(monkeypatch, capsys):
    questions = iter(["2つ目の質問", "exit"])
    asked = []

    def fake_ask(question, history):
        asked.append(question)
        return f"{question}への回答"

    monkeypatch.setattr("sys.argv", ["app.ask", "最初の質問"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(questions))
    monkeypatch.setattr(ask, "ask", fake_ask)

    ask.main()

    output = capsys.readouterr().out
    assert asked == ["最初の質問", "2つ目の質問"]
    assert "最初の質問への回答" in output
    assert "2つ目の質問への回答" in output
