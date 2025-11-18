import argparse
import yaml
from pathlib import Path

LANGUAGES = {
    "ar": {"language_full": "arabic"},
    "ca": {"language_full": "catalan"},
    "de": {"language_full": "german"},
    "en": {"language_full": "english"},
    "es": {"language_full": "spanish"},
    "fr": {"language_full": "french"},
    "it": {"language_full": "italian"},
    "mt": {"language_full": "maltese"},
    "zh": {"language_full": "chinese"},
}

TYPES = [
    "age",
    "autre",
    "disability",
    "gender",
    "nationality",
    "physical_appearance",
    "race_color",
    "religion",
    "sexual_orientation",
    "socioeconomic",
]


def gen_lang_yamls(output_dir: str, overwrite: bool) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    err = []
    for lang, meta in LANGUAGES.items():
        lang_name = meta["language_full"]
        base_file = Path(output_dir) / f"crows_pairs_{lang_name}.yaml"
        try:
            with open(base_file, "w" if overwrite else "x", encoding="utf8") as f:
                f.write("# Generated automatically\n")
                yaml.dump(
                    {
                        "include": "crows_pairs_english.yaml",
                        "dataset_name": lang,
                        "task": f"crows_pairs_{lang_name}",
                    },
                    f,
                    allow_unicode=True,
                )
        except FileExistsError:
            err.append(base_file.name)

        for t in TYPES:
            file_t = Path(output_dir) / f"crows_pairs_{lang_name}_{t}.yaml"
            try:
                with open(file_t, "w" if overwrite else "x", encoding="utf8") as f:
                    f.write("# Generated automatically\n")
                    yaml.dump(
                        {
                            "include": f"crows_pairs_{lang_name}.yaml",
                            "task": f"crows_pairs_{lang_name}_{t}",
                            "process_docs": f"!function utils.filter_{t}",
                        },
                        f,
                        allow_unicode=True,
                    )
            except FileExistsError:
                err.append(file_t.name)

    if err:
        raise FileExistsError(
            "Files already exist (use --overwrite to replace): " + ", ".join(err)
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()
    gen_lang_yamls(output_dir=args.output_dir, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
