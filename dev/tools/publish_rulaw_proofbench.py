"""Prepare and publish RuLaw-ProofBench with readable tables and audit materials."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import tarfile


REPOSITORY = "RefalMachine/RuLaw-ProofBench"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONSTRUCTION_ROOT = PROJECT_ROOT / "dev" / "rulaw_proofbench"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def published_pilot_file(filename):
    from huggingface_hub import hf_hub_download

    spec = json.loads((PROJECT_ROOT / "llmtf/tasks/rulaw_proofbench/manifest.json").read_text())
    path = Path(hf_hub_download(
        spec["dataset_repo"], filename, repo_type="dataset",
        revision=spec["dataset_revision"], token=False,
    ))
    expected = spec["files"].get(filename)
    if expected is not None and sha256(path) != expected["sha256"]:
        raise ValueError(f"Published pilot file hash mismatch: {filename}")
    return path


def write_archive(target, roots):
    """Use stable metadata so unchanged material produces identical archives."""
    with target.open("wb") as destination:
        with gzip.GzipFile(fileobj=destination, mode="wb", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                paths = []
                external_names = {}
                for root in roots:
                    additions = list(root.rglob("*")) if root.is_dir() else [root]
                    paths.extend(additions)
                    if not root.is_relative_to(PROJECT_ROOT):
                        external_names.update({path: "expanded_release/" + str(path.relative_to(root)) for path in additions})
                for path in sorted(set(paths)):
                    if not path.is_file() or "__pycache__" in path.parts:
                        continue
                    if path.suffix not in {
                        ".py", ".json", ".jsonl", ".csv", ".txt", ".md",
                        ".html", ".mht", ".gz",
                    }:
                        continue
                    name = external_names.get(path) or str(path.relative_to(PROJECT_ROOT))
                    info = archive.gettarinfo(str(path), arcname=name)
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = 0
                    with path.open("rb") as source:
                        archive.addfile(info, source)


def export_rows(rows, cards, sources, mcq_rows=None):
    records = []
    for row in rows:
        card = cards[row["rule_id"]]
        fragments = [card["source_fragment"]]
        fragments.extend(
            f"{sources[key.split(':')[0]]['title']}\n{text}"
            for key, text in card["dependency_fragments"].items()
        )
        record = {
            "id": row["id"],
            "article": row["article_key"],
            "domain": row["domain"],
            "as_of": row["as_of"],
            "question": row["question"],
            "answer": row["gold"]["value"],
            "answer_unit": row["gold"]["unit"],
            "context": "\n\n".join(fragments),
            "source_url": sources[card["source_ids"][0]]["url"],
            "source_locator": card["locator"],
            "record_json": json.dumps(row, ensure_ascii=False, sort_keys=True),
        }
        if mcq_rows is not None:
            mcq = mcq_rows[row["id"]]
            record.update(
                question=mcq["question"],
                choices=[option["text"] for option in mcq["options"]],
                answer_label=mcq["gold_label"],
                mcq_json=json.dumps(mcq, ensure_ascii=False, sort_keys=True),
            )
        records.append(record)
    return records


def prepare(output, repo, expanded=None):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from dev.rulaw_proofbench.mcq import validate_mcq
    from dev.rulaw_proofbench.validate import validate

    validation = validate(require_freeze=False)
    if validation["failures"]:
        raise ValueError(validation["failures"])
    if expanded is not None:
        release = json.loads((expanded / "validation_report.json").read_text())
        if release["status"] != "construction_validated" or release["pending"]:
            raise ValueError("Expanded release has not passed acceptance")
        for name, expected in release["files"].items():
            if sha256(expanded / name) != expected:
                raise ValueError(f"Accepted expanded artifact changed: {name}")
        for name, expected in release["implementation_sha256"].items():
            if sha256(CONSTRUCTION_ROOT / name) != expected:
                raise ValueError(f"Expanded implementation changed after acceptance: {name}")
    else:
        validate_mcq()
    output.mkdir(parents=True, exist_ok=True)
    cards = {
        row["rule_id"]: row
        for row in read_rows((expanded or CONSTRUCTION_ROOT) / "rule_cards.jsonl")
    }
    sources = {
        row["id"]: row
        for row in json.loads((CONSTRUCTION_ROOT / "source_manifest.json").read_text())
    }
    if expanded is not None:
        splits = {split: read_rows(expanded / f"{split}.jsonl") for split in ("test", "train")}
        mcq = {row["id"]: row for split in splits for row in read_rows(expanded / f"mcq_{split}.jsonl")}
    else:
        splits = {"test": read_rows(CONSTRUCTION_ROOT / "dataset.jsonl")}
        mcq = {row["id"]: row for row in read_rows(CONSTRUCTION_ROOT / "mcq_dataset.jsonl")}
    rows = splits["test"]
    demonstration_count = len(splits.get("train", []))
    article_count = len({row["article_key"] for row in rows})
    files = {}
    for config in ("open", "mcq"):
        for split, split_rows in splits.items():
            target = output / "data" / config / f"{split}.parquet"
            target.parent.mkdir(parents=True, exist_ok=True)
            exported = export_rows(split_rows, cards, sources, mcq if config == "mcq" else None)
            pq.write_table(pa.Table.from_pylist(exported), target, compression="zstd")
            if pq.read_table(target).to_pylist() != exported:
                raise ValueError(f"Parquet roundtrip failed: {config}/{split}")
            files[str(target.relative_to(output))] = {"sha256": sha256(target), "rows": len(exported)}
    audit = output / "audit"
    audit.mkdir(exist_ok=True)
    write_archive(
        audit / "construction.tar.gz",
        [
            CONSTRUCTION_ROOT,
            PROJECT_ROOT / "llmtf/tasks/rulaw_proofbench/scoring.py",
            PROJECT_ROOT / "llmtf/tasks/rulaw_proofbench/judge.py",
            PROJECT_ROOT / "dev" / "RULAW_PROOFBENCH_PROTOCOL_REVIEW.md",
        ] + ([expanded] if expanded is not None else []),
    )
    (audit / "pilot_results.tar.gz").write_bytes(
        published_pilot_file("audit/pilot_results.tar.gz").read_bytes()
    )
    for path in sorted(audit.iterdir()):
        files[str(path.relative_to(output))] = {"sha256": sha256(path)}
    scoring_target = output / "code" / "scoring.py"
    scoring_target.parent.mkdir(exist_ok=True)
    scoring_target.write_bytes(
        (PROJECT_ROOT / "llmtf/tasks/rulaw_proofbench/scoring.py").read_bytes()
    )
    files["code/scoring.py"] = {"sha256": sha256(scoring_target)}
    for original, name in [
        (CONSTRUCTION_ROOT / "paper_methods.md", "METHODS.md"),
        (CONSTRUCTION_ROOT / "SCORING.md", "SCORING.md"),
        (
            published_pilot_file("PILOT_RESULTS.md"),
            "PILOT_RESULTS.md",
        ),
    ]:
        text = original.read_text()
        # Repository-local links become links to the packaged audit materials.
        text = text.replace("../../rulaw_proofbench/SCORING.md", "SCORING.md")
        text = text.replace("../../llmtf/tasks/rulaw_proofbench/scoring.py", "code/scoring.py")
        (output / name).write_text(text)
    manifest = {
        "dataset_name": "RuLaw-ProofBench",
        "repository": repo,
        "as_of": "2025-01-01",
        "test_count": len(rows),
        "demonstration_count": demonstration_count,
        "article_count": article_count,
        "files": files,
        "expert_review": False,
        "construction_validation": validation["release_status"],
    }
    write_json(output / "manifest.json", manifest)
    train_open = "\n  - split: train\n    path: data/open/train.parquet" if demonstration_count else ""
    train_mcq = "\n  - split: train\n    path: data/mcq/train.parquet" if demonstration_count else ""
    random_accuracy = sum(1 / mcq[row["id"]]["choice_count"] for row in rows) / len(rows)
    card = f"""---
language:
- ru
task_categories:
- question-answering
- text-classification
tags:
- legal
- russian-law
- synthetic
pretty_name: RuLaw-ProofBench
size_categories:
- n<1K
configs:
- config_name: open
  default: true
  data_files:
  - split: test
    path: data/open/test.parquet{train_open}
- config_name: mcq
  data_files:
  - split: test
    path: data/mcq/test.parquet{train_mcq}
---

# RuLaw-ProofBench

Синтетический русскоязычный набор для оценки знания и локального применения
норм российского права на **01.01.2025**. Текущий выпуск: **{len(rows)} тестовых
примеров по {article_count} статьям четырёх актов** и **{demonstration_count} отдельных демонстраций**. Это узкий формальный benchmark,
а не оценка способности разрешать произвольные юридические дела.

## Загрузка

```python
from datasets import load_dataset

open_test = load_dataset("{repo}", "open", split="test")
mcq_test = load_dataset("{repo}", "mcq", split="test")
```

Для воспроизводимого сравнения указывайте `revision` — полный hash Hub-коммита.
LLMTF загружает Parquet из этого репозитория по закреплённой revision и проверяет
SHA-256 файла. Локальные примеры не подменяют опубликованные данные.

## Почему указано 01.01.2025

Это выбранная при сборке **единая дата правового среза**: ответы должны
соответствовать выбранным положениям, действовавшим на этот день. Исходный
протокол требовал зафиксировать дату, но не задавал именно 01.01.2025.
Её выбрал автоматический исполнитель; специального научного обоснования
именно этого дня нет. Это не дата публикации датасета, не утверждение об
актуальности норм сегодня и не предполагаемая граница знаний моделей.

Дата проверялась по источникам, а не только записывалась в вопросы:

1. Сохранены официальные тексты с pravo.gov.ru, номера редакций и точные
   фрагменты, на которых основаны ответы, включая необходимые отсылки.
2. Выбранные фрагменты сопоставлены между редакциями; дополнительно изучены
   применимые законы о поправках, даты вступления в силу и переходные положения.
   Одного совпадения текстов или даты редакции всего кодекса недостаточно.
3. Более поздние изменения не переносились на выбранную дату. Например,
   при проверке 59-ФЗ отдельно учтено, что изменения по 547-ФЗ вступают в силу
   30.03.2025, то есть после даты среза.
4. Источники, цитаты и результаты проверок сохранены в `edition_audit.json`
   и `expansion_source_review.json` внутри `audit/construction.tar.gz`.

По результатам этого аудита все использованные фрагменты приняты как
соответствующие 01.01.2025. Утверждение относится к выбранным положениям,
необходимым отсылкам и условиям вопросов, а не ко всем нормам четырёх актов.

Здесь два уровня контроля. **Содержательный вывод о действии нормы** делали
автоматические агенты, читая документы; юрист его не подтверждал.
**Программные проверки** сверяют дату и ссылки записи с рассмотренным правилом,
точность цитат, хеши источников и наличие завершённых проверок. Они защищают
от подмены и рассогласования материалов, но не умеют заново доказывать
применимость любого закона на заданный день. Поэтому ошибка исходного аудита
всё ещё возможна; совпадение SHA-256 её не исключает.

## Два представления одного теста

- `open`: короткий открытый ответ; `answer` содержит эталон.
- `mcq`: те же ID и факты, варианты в `choices`, правильная буква в `answer_label`.
  У бинарных случаев три варианта, у остальных четыре. Случайный выбор — {random_accuracy:.2%} micro-accuracy.

Это не независимые выборки. `context` содержит норму и необходимые зависимости:
в closed-режиме он **не передаётся модели**, в grounded-режиме добавляется перед
вопросом. `record_json` хранит исходные типизированные факты, proof и связи
минимальных пар без потери структуры. `mcq_json` сохраняет канонические варианты
и привязку к исходному вопросу. Эти поля служат проверке данных, не входу модели.

`test` целиком используется для оценки и не уменьшается при добавлении few-shot.
`train`, если он присутствует, содержит фиксированные демонстрации по отдельным
статьям: нет пересечения ID, вопросов, тестовых статей и их зависимостей.
LLMTF Foundational использует все 5 демонстраций расширенного выпуска; Instruct
использует 0. Запрошенное число демонстраций не сокращается при нехватке контекста:
такой запуск завершается ошибкой. Примеры из test не используются как few-shot.
Общие legal-конфиги LLMTF включают оба режима: с текстом нормы и без него.
В Foundational оценивается MCQ, в Instruct — и открытые ответы, и MCQ.

## Построение и обоснование надёжности

Для каждой нормы составлено правило с явными условиями и исключениями.
Условия вопроса представлены как **набор значений юридически значимых
признаков**: например, возраст, наличие согласия, известность нужного факта.
В технических материалах такой набор назван «вектором фактов»; это не
эмбеддинг и не отдельный вопрос датасета. Если условие неизвестно, проверяются
все допустимые его значения: при разных ответах эталон — «недостаточно данных».

Два отдельных автоматических прохода переводили источники в программные
правила. Их результаты сверены на **886 допустимых комбинациях условий**:
403 для пилота и ещё 483 для расширения. Учитывались как полностью заданные
условия, так и случаи с неизвестными данными. Это проверки правил, а не
886 тестовых вопросов. Оба прохода использовали ту же модель, поэтому
совпадение результатов не исключает общей ошибки интерпретации закона.

В отдельные копии записей намеренно внесли **180 ошибок** (90 в пилоте и 90
в расширении): например, неверный эталон, пропущенное отрицание, неправильную
дату или удалённое исключение. Ранее они назывались «контрольными повреждениями».
Эти копии **не входят в test или train**; они проверяют, умеет ли валидатор
отклонять заведомо ошибочные записи. Все 180 ошибок обнаружены. Это результат
на заданных видах ошибок, а не доказательство отсутствия иных ошибок в наборе.

Что именно менялось в контрольных копиях:

| Вид ошибки | Внесённое изменение | Что должна обнаружить проверка |
|---|---|---|
| Неверный ответ | Эталон заменён на `999` | Ответ расходится с вычислением по правилу |
| Изменённое условие | Один признак заменён с «да» на «нет» или наоборот, остальные поля сохранены | Условия расходятся с текстом и сохранённым выводом |
| Пропущенное отрицание | Из вопроса удалено одно отдельное «не» | Формулировка расходится с исходными условиями |
| Неверная часть статьи | Ссылка заменена на «несуществующая часть 999» | Ссылка расходится с рассмотренным источником |
| Подменённая дата | `as_of` заменено на 2030 год в пилоте или 2099 год в расширении | Дата расходится с зафиксированной датой правила |
| Несуществующий источник | ID источника заменён на отсутствующий | Запись не связана с рассмотренным документом |
| Слишком широкий вопрос | Вывод из названной нормы заменён претензией на итог всего дела | Нарушена допустимая область вопроса |
| Удалённое исключение | Удалена приоритетная ветвь правила; ответ и вывод пересчитаны | Вторая реализация правила даёт другой ответ |
| Сдвинутая граница | Числовой порог правила увеличен на единицу; ответ и вывод пересчитаны | Вторая реализация правила даёт другой ответ |

Подмена даты — **проверка согласованности записи**, а не эксперимент,
подтверждающий правильность редакции на 01.01.2025. Аналогично простая замена
эталона проверяет его связь с программой, а не правильность самой программы.
Поэтому отдельно нужны описанный выше анализ источников и сверка правил.

Затем проверялись формулировки вопросов, исключения, числовые границы и
соответствие источникам; найденные неоднозначности исправлялись с сохранением
журнала проверки. Для каждого ответа сохранён программный вывод (`proof`).
Юристов и независимой человеческой экспертизы не было. Методика делает
построение проверяемым и воспроизводимым, но не гарантирует юридическую
безошибочность или полноту охвата права РФ.

Полная [методика](METHODS.md), [грамматика scorer](SCORING.md),
[пилотное сравнение Qwen3.5](PILOT_RESULTS.md).

## Метрики

Основная детерминированная метрика — normalized exact match с усреднением
accuracy по статьям. Она использует общую грамматику коротких ответов, без
модельных фраз-алиасов. Дополнительная метрика — LLM-судья эквивалентности
фиксированному gold. Судья не создаёт и не исправляет юридические эталоны.
Для MCQ выбирается вариант с максимальной вероятностью буквы и проверяется
его точное совпадение с эталоном; судья не нужен.

Нераспознанный scorer ответ может быть семантически верным. Судья тоже может
ошибаться; в пилоте сохранены его исходные решения и отмечена найденная ошибка.
Открытые и MCQ-результаты нельзя смешивать в один рейтинг.

## Материалы проверки

- `manifest.json`: состав публикации и SHA-256 файлов.
- `audit/construction.tar.gz`: первичные источники, формальные правила, проверки,
  реконструкции, выводы ответов (`proof`), контрольные копии с намеренными
  ошибками и код сборки.
- `audit/pilot_results.tar.gz`: сохранённые модельные ответы, API-запросы без
  авторизационных заголовков, судейские решения и код воспроизведения таблиц.

Архивы содержат пути от корня LLMTF-проекта. Извлекать их следует в отдельный
рабочий каталог. Текущие метрики воспроизводятся из ответов; поля старого
логгера в сырых журналах не являются самостоятельной метрикой.

## Ограничения

Открытый синтетический набор узок и может попасть в обучение будущих моделей.
Правовые квалификации в условиях заданы заранее. Unknown-случаи проверяют также
следование формальным условиям, не только память о норме. Минимальные пары и
случаи одной статьи зависимы; статистические сравнения выполняются по статьям.
Методика развивалась после пилота и не является слепой предварительной регистрацией.
Пилотные результаты на 150 вопросах нельзя переносить на расширенные выпуски без нового
прогона. Отдельная лицензия на производный набор пока не объявлена.
"""
    (output / "README.md").write_text(card)
    return manifest


def publish(output, repo, expected):
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import RepositoryNotFoundError
    import pyarrow.parquet as pq

    api = HfApi()
    account = api.whoami()["name"]
    if repo.split("/")[0] != account:
        raise ValueError("Publication target must belong to the authenticated account")
    try:
        info = api.repo_info(repo, repo_type="dataset")
    except RepositoryNotFoundError:
        info = None
    if info is not None:
        previous_path = hf_hub_download(
            repo, "manifest.json", repo_type="dataset", revision=info.sha
        )
        previous = json.loads(Path(previous_path).read_text())
        if previous.get("dataset_name") != expected["dataset_name"]:
            raise ValueError("Existing repository is not the expected dataset")
    api.create_repo(repo, repo_type="dataset", private=False, exist_ok=True)
    result = api.upload_folder(
        repo_id=repo,
        repo_type="dataset",
        folder_path=output,
        parent_commit=info.sha if info is not None else None,
        commit_message=f"Publish {expected['test_count']} verified RuLaw-ProofBench cases",
    )
    revision = result.oid
    pin = {
        "dataset_repo": repo,
        "dataset_revision": revision,
        "as_of": expected["as_of"],
        "test_count": expected["test_count"],
        "demonstration_count": expected["demonstration_count"],
        "files": expected["files"],
    }
    for filename in (name for name in pin["files"] if name.endswith(".parquet")):
        downloaded = hf_hub_download(
            repo,
            filename,
            repo_type="dataset",
            revision=revision,
            token=False,
            cache_dir=output.parent / "anonymous-cache",
        )
        if sha256(downloaded) != pin["files"][filename]["sha256"]:
            raise ValueError("Anonymous download hash mismatch")
        if pq.read_table(downloaded).to_pylist() != pq.read_table(output / filename).to_pylist():
            raise ValueError("Anonymous Parquet roundtrip mismatch")
    write_json(output.parent / "publication.json", pin)
    print(f"Published and anonymously verified: https://huggingface.co/datasets/{repo}/tree/{revision}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", default=REPOSITORY)
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--expanded", type=Path)
    args = parser.parse_args()
    manifest = prepare(args.output, args.repo, args.expanded)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if args.upload:
        publish(args.output, args.repo, manifest)


if __name__ == "__main__":
    main()
