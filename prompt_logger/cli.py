import argparse

from prompt_logger.logger import PromptLogger


def create_logger(args):
    return PromptLogger(
        namespace=args.namespace,
        database=args.database,
        create_if_not_exists=False,
    )


def main():
    parser = argparse.ArgumentParser(description="Prompt Logger CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Export command parser
    export_parser = subparsers.add_parser(
        "export", help="Export prompt data to a JSONL file"
    )

    # Export types
    export_type_subparsers = export_parser.add_subparsers(
        dest="export_kind", required=True
    )

    # Common arguments for the export types
    common_args = argparse.ArgumentParser(add_help=False)
    common_args.add_argument(
        "--namespace", "-n", type=str, help="Namespace to export", default="default"
    )
    common_args.add_argument(
        "--models", "-m", type=str, help="Model names to export", default="all"
    )
    common_args.add_argument(
        "--database",
        "-d",
        type=str,
        default="sqlite:///prompts.db",
        help="Database connection string (default: sqlite:///prompts.db)",
    )
    common_args.add_argument(
        "output_file", type=str, help="Path to the output JSONL file"
    )

    # Models export subparser
    _ = export_type_subparsers.add_parser(
        "models",
        help="Export the list of models used for inference",
        parents=[common_args],
    )

    # Prompts export subparser
    export_prompt_parser = export_type_subparsers.add_parser(
        "prompts",
        help="Export the list of executed prompts and recorded completions",
        parents=[common_args],
    )
    export_prompt_parser.add_argument(
        "--type",
        "-t",
        type=str,
        help="Type of models to export",
        default="chat",
        choices=["chat", "text"],
    )
    args = parser.parse_args()

    if args.command == "export":
        if args.export_kind == "models":
            models = args.models.split(",") if args.models != "all" else None
            logger = create_logger(args)
            logger.export_models(args.output_file, models=models)
        elif args.export_kind == "prompts":
            models = args.models.split(",") if args.models != "all" else None
            logger = create_logger(args)
            if args.type == "chat":
                logger.export_chat_prompts(args.output_file, models=models)
            elif args.type == "text":
                logger.export_text_prompts(args.output_file, models=models)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
