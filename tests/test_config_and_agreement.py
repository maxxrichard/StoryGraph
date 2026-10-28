from storygraph.config import load_pipeline_config, load_recsys_config, load_tagging_config
from storygraph.evaluation.agreement import field_agreement, overall_agreement
from storygraph.schemas import ContentSignals, Label
from storygraph.training.lora import to_chat_example, write_sft_dataset


def test_configs_load(repo_root):
    p = load_pipeline_config(repo_root / "configs/pipeline.yaml")
    assert p.segmentation.method == "texttiling" and p.embeddings.dim == 1024
    t = load_tagging_config(repo_root / "configs/tagging.yaml")
    assert t.lora.r == 16 and t.model.json_schema_guided
    r = load_recsys_config(repo_root / "configs/recsys.yaml")
    assert r.ranker.objective == "lambdarank" and r.retrieval.candidates.k_total_max == 500


def _sig(*genres):
    return ContentSignals(title_id="t", genres=[Label(name=g, confidence=0.9) for g in genres])


def test_field_agreement():
    pred = [_sig("crime"), _sig("crime", "horror"), _sig("memoir")]
    gold = [_sig("crime"), _sig("crime"), _sig("travel")]
    fa = field_agreement(pred, gold, "genres", ["crime", "horror", "memoir", "travel"])
    assert fa.exact_match == 1 / 3 and 0 < fa.macro_f1 < 1 and 0 < fa.kappa < 1
    perfect = field_agreement(pred, pred, "genres", ["crime", "horror", "memoir", "travel"])
    assert perfect.exact_match == 1 and perfect.macro_f1 == 1 and perfect.kappa == 1
    assert set(overall_agreement(pred, gold)) == {"genres", "themes", "moods"}


def test_sft_example_and_dataset(tmp_path):
    from storygraph.schemas import Segment

    seg = Segment(title_id="t", index=0, start=0, end=1, text="hello world", word_count=2)
    ex = to_chat_example(seg, _sig("crime"))
    assert [m["role"] for m in ex["messages"]] == ["system", "user", "assistant"]
    assert '"crime"' in ex["messages"][2]["content"]
    n = write_sft_dataset([(seg, _sig("crime"))], tmp_path / "sft.jsonl")
    assert n == 1 and (tmp_path / "sft.jsonl").read_text().count("\n") == 1
