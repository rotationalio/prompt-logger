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
        "--namespace", "-n", type=str, help="Namespace to export", default="default"
    )
    export_parser.add_argument(
        "--models", "-m", type=str, help="Model names to export", default="all"
    )
    export_parser.add_argument(
        "--type",
        "-t",
        type=str,
        help="Type of prompts to export",
        default="chat",
        choices=["chat", "text"],
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
            namespace=args.namespace, database=args.database, create_if_not_exists=False
        )
        models = args.models.split(",") if args.models != "all" else None
        if args.type == "chat":
            logger.export_chat_prompts(
                args.output_file, models=models, namespace=args.namespace
            )
        elif args.type == "text":
            logger.export_text_prompts(
                args.output_file, models=models, namespace=args.namespace
            )
        else:
            print(
                f"Unknown type: {args.type}, it should be 'chat' for message-style prompts or 'text' for text-style prompts."
            )
            parser.print_help()
            return
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
