import argparse

from prompt_logger.logger import PromptLogger


def main():
    parser = argparse.ArgumentParser(description="Prompt Logger CLI")
    subparsers = parser.add_subparsers(dest="command")
    export_parser = subparsers.add_parser(
        "export", help="Export prompt records to a JSONL file"
    )
    export_parser.add_argument(
        "output_file", type=str, help="Path to the output JSONL file"
    )
    export_parser.add_argument(
        "--namespace", "-n", type=str, help="Namespace to export", required=True
    )
    export_parser.add_argument(
        "--database",
        "-d",
        type=str,
        default="sqlite:///prompts.db",
        help="Database connection string (default: sqlite:///prompts.db)",
    )
    args = parser.parse_args()

    if args.command == "export":
        logger = PromptLogger(
            args.namespace, database=args.database, create_if_not_exists=False
        )
        logger.export_to_jsonl(args.output_file, namespace=args.namespace)
        print(f"Successfully exported prompts to {args.output_file}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
